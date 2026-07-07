"""Sequential GPU batch runner for queued PRs.

This is the Phase 2 bridge between the always-on PR bot and the later live GPU
scorer. It consumes ``dashboard/data.json`` (written by ``eval.pr_bot``), takes
queued PRs in oldest-first order, and either prints or executes the exact steps
for a maintainer-controlled GPU window.

Default mode is dry-run. Use ``--run`` only on a disposable GPU machine or a
properly isolated self-hosted runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QUEUE = "dashboard/data.json"
DEFAULT_WORKDIR = "_gpu_batch_work"
DEFAULT_RESULTS_DIR = "gpu-results"
MOCK_GPU_NAME = "RTX 5090 (mock)"


@dataclass(frozen=True)
class QueueItem:
    pr: int
    title: str
    author: str
    head_sha: str
    position: int | None = None
    url: str = ""


@dataclass(frozen=True)
class EvalSpec:
    n: int = 12000
    pairs: int = 3
    dtype: str = "fp32"
    rank_m: int | None = None
    fill: str = "random"
    data_rank: int | None = None
    transforms: str | None = None
    seed: int | None = None
    device: int = 0


def load_queue(path: str | Path) -> list[QueueItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for raw in data.get("queue", []):
        items.append(
            QueueItem(
                pr=int(raw["pr"]),
                title=raw.get("title", ""),
                author=raw.get("author", ""),
                head_sha=raw.get("head_sha", ""),
                position=raw.get("position"),
                url=raw.get("url", ""),
            )
        )
    return sorted(items, key=lambda item: item.position or item.pr)


def select_batch(queue: list[QueueItem], limit: int | None) -> list[QueueItem]:
    if limit is None or limit <= 0:
        return queue
    return queue[:limit]


def eval_args(spec: EvalSpec) -> list[str]:
    args = [
        "uv", "run", "python", "-m", "eval",
        "--n", str(spec.n),
        "--pairs", str(spec.pairs),
        "--dtype", spec.dtype,
        "--fill", spec.fill,
        "--device", str(spec.device),
        "--json",
    ]
    if spec.rank_m is not None:
        args += ["--rank-m", str(spec.rank_m)]
    if spec.data_rank is not None:
        args += ["--data-rank", str(spec.data_rank)]
    if spec.transforms:
        args += ["--transforms", spec.transforms]
    if spec.seed is not None:
        args += ["--seed", str(spec.seed)]
    return args


def result_path(item: QueueItem, results_dir: str | Path) -> Path:
    return Path(results_dir) / f"pr-{item.pr}-{item.head_sha[:12] or 'unknown'}.json"


def _mock_seed(item: QueueItem, spec: EvalSpec) -> int:
    if spec.seed is not None:
        return spec.seed
    material = f"{item.pr}:{item.head_sha}:{spec.n}:{spec.pairs}:{spec.fill}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16)


def mock_result(item: QueueItem, spec: EvalSpec) -> dict:
    """Produce an evaluate()-shaped result without requiring GPU hardware."""
    transform = (spec.transforms.split(",")[0].strip() if spec.transforms else "mock_transform")
    seed = _mock_seed(item, spec)
    score = 10.0 + (item.pr % 7) / 10.0
    result = {
        "accuracy": 0.94,
        "rel_frobenius_error": 0.06,
        "latency_s": 0.021 + (item.pr % 3) * 0.002,
        "peak_vram_bytes": 1_610_612_736,
        "peak_vram_mib": 1536.0,
        "flop_ratio_vs_exact": 2.5,
        "faster_than_exact": True,
        "less_vram_than_exact": True,
        "fewer_flops_than_exact": True,
        "gated": False,
        "improvement": True,
        "perf_score": score,
        "score": score,
    }
    return {
        "pr": item.pr,
        "title": item.title,
        "author": item.author,
        "head_sha": item.head_sha,
        "url": item.url,
        "mock": True,
        "eval": {
            "config": {
                "n": spec.n,
                "pairs": spec.pairs,
                "dtype": spec.dtype,
                "rank_m": spec.rank_m,
                "fill": spec.fill,
                "accuracy_floor": 0.8,
                "vram_unit": "gib",
                "device": MOCK_GPU_NAME,
                "seed": seed,
            },
            "complexity": {"normal": "O(N^3)", "smart": "O(N^2 * M)"},
            "exact": {
                "latency_s": 0.052,
                "peak_vram_bytes": 4_294_967_296,
                "peak_vram_mib": 4096.0,
            },
            "transforms": {transform: result},
            "ranking": [transform],
            "best": transform,
        },
    }


def wrap_result(item: QueueItem, eval_output: str, *, mock: bool = False) -> dict:
    return {
        "pr": item.pr,
        "title": item.title,
        "author": item.author,
        "head_sha": item.head_sha,
        "url": item.url,
        "mock": mock,
        "eval": json.loads(eval_output),
    }


def plan_item(
    item: QueueItem,
    *,
    repo: str,
    workdir: str | Path,
    results_dir: str | Path,
    spec: EvalSpec,
) -> list[str]:
    checkout = Path(workdir) / f"pr-{item.pr}"
    result = (Path.cwd() / result_path(item, results_dir))
    return [
        f"gh repo clone {repo} {checkout}",
        f"cd {checkout} && gh pr checkout {item.pr}",
        f"cd {checkout} && test \"$(git rev-parse HEAD)\" = \"{item.head_sha}\"",
        f"cd {checkout} && uv sync --extra test --extra gpu",
        f"cd {checkout} && uv run --extra test python -m py_compile $(find matmul strategy eval tests examples -name '*.py')",
        f"cd {checkout} && uv run --extra test python -m pytest tests/ strategy/tests/ eval/tests/ -v",
        f"cd {checkout} && uv run python -m strategy.smoke",
        "cd "
        + str(checkout)
        + " && "
        + " ".join(eval_args(spec))
        + " > "
        + str(result),
    ]


def _run(cmd: list[str] | str, *, cwd: str | Path | None = None, capture: bool = False):
    return subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        text=True,
        capture_output=capture,
        check=True,
    )


def run_item(
    item: QueueItem,
    *,
    repo: str,
    workdir: str | Path,
    results_dir: str | Path,
    spec: EvalSpec,
    clean: bool = False,
    mock: bool = False,
) -> Path:
    """Execute one queued PR sequentially and return the JSON result path."""
    workdir = Path(workdir)
    results_dir = Path(results_dir)
    checkout = workdir / f"pr-{item.pr}"
    result = result_path(item, results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if mock:
        result.write_text(json.dumps(mock_result(item, spec), indent=2) + "\n",
                          encoding="utf-8")
        return result

    if checkout.exists():
        if not clean:
            raise FileExistsError(f"{checkout} already exists; pass --clean to replace it")
        shutil.rmtree(checkout)

    workdir.mkdir(parents=True, exist_ok=True)

    _run(["gh", "repo", "clone", repo, str(checkout)])
    _run(["gh", "pr", "checkout", str(item.pr)], cwd=checkout)
    actual_sha = _run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).stdout.strip()
    if item.head_sha and actual_sha != item.head_sha:
        raise RuntimeError(
            f"PR #{item.pr} checked out {actual_sha}, expected queued SHA {item.head_sha}"
        )

    _run(["uv", "sync", "--extra", "test", "--extra", "gpu"], cwd=checkout)
    _run("uv run --extra test python -m py_compile $(find matmul strategy eval tests examples -name '*.py')",
         cwd=checkout)
    _run(["uv", "run", "--extra", "test", "python", "-m", "pytest",
          "tests/", "strategy/tests/", "eval/tests/", "-v"], cwd=checkout)
    _run(["uv", "run", "python", "-m", "strategy.smoke"], cwd=checkout)

    completed = _run(eval_args(spec), cwd=checkout, capture=True)
    result.write_text(json.dumps(wrap_result(item, completed.stdout), indent=2) + "\n",
                      encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.gpu_batch",
        description="Run or print the next sequential GPU evaluation batch.",
    )
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--repo", default="zeokin/Cuda-Compute-OSS")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=1,
                        help="number of queued PRs to evaluate; <=0 means all")
    parser.add_argument("--run", action="store_true",
                        help="execute the batch. Omit for a dry-run plan.")
    parser.add_argument("--mock", action="store_true",
                        help="with --run, write mock RTX 5090 result JSON without gh/GPU")
    parser.add_argument("--clean", action="store_true",
                        help="replace existing per-PR checkout directories")
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--dtype", choices=("fp16", "fp32", "fp64"), default="fp32")
    parser.add_argument("--rank-m", type=int, default=None)
    parser.add_argument("--fill", choices=("random", "lowrank", "decaying-spectrum", "iota"),
                        default="random")
    parser.add_argument("--data-rank", type=int, default=None)
    parser.add_argument("--transforms", default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="omit for fresh unseen inputs; pass only to reproduce a run")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)

    spec = EvalSpec(
        n=args.n,
        pairs=args.pairs,
        dtype=args.dtype,
        rank_m=args.rank_m,
        fill=args.fill,
        data_rank=args.data_rank,
        transforms=args.transforms,
        seed=args.seed,
        device=args.device,
    )
    batch = select_batch(load_queue(args.queue), args.limit)
    if not batch:
        print("No queued PRs found.")
        return 0

    for item in batch:
        print(f"PR #{item.pr} ({item.author}): {item.title}")
        if args.run:
            result = run_item(
                item,
                repo=args.repo,
                workdir=args.workdir,
                results_dir=args.results_dir,
                spec=spec,
                clean=args.clean,
                mock=args.mock,
            )
            print(f"  wrote {result}")
        else:
            for command in plan_item(
                item,
                repo=args.repo,
                workdir=args.workdir,
                results_dir=args.results_dir,
                spec=spec,
            ):
                print(f"  {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
