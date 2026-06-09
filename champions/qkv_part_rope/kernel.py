"""CCO champion baseline — qkv_part_rope (Triton).

Partial rotary position embedding. `qkv` is (batch, seq, num_heads, head_dim) with
num_heads = q_heads + 2*kv_heads. RoPE rotates only the trailing rope_dim = head_dim - nope_dim
dims of the first nqk_heads = q_heads + kv_heads heads (Q and K); the nope prefix and the V
head(s) pass through unchanged. The rotation pairs x0 = rope[:half] with x1 = rope[half:]:

    out0 = x0*cos - x1*sin   ->  [nope : nope+half]
    out1 = x0*sin + x1*cos   ->  [nope+half : head_dim]

computed in fp32 (cos/sin are fp32), stored back in the input dtype. cos[s,f]/sin[s,f] are
indexed by sequence position s.

One Triton program per (batch, seq, head) row. The whole op (copy of the unrotated parts +
rotation) is in the kernel — no torch.clone — so it stays delegation-free. Naive but real;
miners can fuse/vectorize. CCO artifact contract: KERNEL_TYPE + kernel_fn only.
"""

import torch
import triton
import triton.language as tl

KERNEL_TYPE = "qkv_part_rope"


@triton.jit
def _qkv_rope_kernel(
    QKV, COS, SIN, OUT,
    num_heads, seq_len, nqk_heads,
    HEAD_DIM: tl.constexpr, NOPE: tl.constexpr, HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid % num_heads
    s = (pid // num_heads) % seq_len
    base = pid * HEAD_DIM

    # 1) copy the nope dims [0:NOPE] unchanged
    d = tl.arange(0, HEAD_DIM)
    nope_mask = d < NOPE
    xn = tl.load(QKV + base + d, mask=nope_mask, other=0.0)
    tl.store(OUT + base + d, xn, mask=nope_mask)

    # 2) rope dims [NOPE:NOPE+2*HALF]: rotate for Q/K heads, copy for V
    f = tl.arange(0, HALF)
    x0 = tl.load(QKV + base + NOPE + f).to(tl.float32)
    x1 = tl.load(QKV + base + NOPE + HALF + f).to(tl.float32)
    cos = tl.load(COS + s * HALF + f)
    sin = tl.load(SIN + s * HALF + f)

    rot0 = x0 * cos - x1 * sin
    rot1 = x0 * sin + x1 * cos
    cond = f < tl.where(h < nqk_heads, HALF, 0)   # all-true for Q/K heads, all-false for V
    out0 = tl.where(cond, rot0, x0)
    out1 = tl.where(cond, rot1, x1)

    tl.store(OUT + base + NOPE + f, out0.to(OUT.dtype.element_ty))
    tl.store(OUT + base + NOPE + HALF + f, out1.to(OUT.dtype.element_ty))


def kernel_fn(qkv, cos, sin, q_heads=10, kv_heads=1, nope_dim=192):
    if qkv.dim() == 3:
        qkv = qkv.unsqueeze(0)
    batch, seq_len, num_heads, head_dim = qkv.shape
    half = (head_dim - nope_dim) // 2
    nqk_heads = q_heads + kv_heads

    qkv = qkv.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    out = torch.empty_like(qkv)

    grid = (batch * seq_len * num_heads,)
    _qkv_rope_kernel[grid](
        qkv, cos, sin, out,
        num_heads, seq_len, nqk_heads,
        HEAD_DIM=head_dim, NOPE=nope_dim, HALF=half,
    )
    return out
