"""CCO champion baseline — matmul (Triton).

A straightforward tiled GEMM with grouped block ordering for L2 reuse and an fp32 accumulator.
This is the *seed* champion (correct, not heavily tuned) that challengers on the `matmul` track
must beat. It is intentionally a hand-written Triton kernel, not a cuBLAS/`torch.matmul` wrapper
(which the no-delegation guard + dispatch trap would reject).

fp32 inputs use `input_precision="ieee"` so the tight 1e-4 tolerance is met (TF32 would not be
accurate enough); for fp16/bf16 this is a no-op and the native tensor-core path is used with fp32
accumulate.

CCO artifact contract: exports KERNEL_TYPE + kernel_fn ONLY (flops/bytes/inputs live in the
locked kernel_configs).
"""

import torch
import triton
import triton.language as tl

KERNEL_TYPE = "matmul"


@triton.jit
def _matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FP32_IEEE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # `% M` / `% N` keep loads in-bounds for ragged tiles; the store mask drops the wrap-around.
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        # fp16/bf16 use the default tensor-core path (fp32 accumulate). Only fp32 needs the
        # higher-precision "ieee" path; forcing "ieee" on fp16 corrupts large-K accumulation.
        if FP32_IEEE:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def kernel_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 64, 64, 32, 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
        FP32_IEEE=(a.dtype == torch.float32),
    )
    return c
