"""Tests for eval.gpu_batch's queue planning logic -- no GPU or gh calls."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path

from eval.gpu_batch import (
    EvalSpec,
    eval_args,
    load_queue,
    mock_result,
    plan_item,
    run_item,
    select_batch,
    wrap_result,
)


def _queue_file(items):
    d = tempfile.TemporaryDirectory()
    path = os.path.join(d.name, "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"queue": items}, f)
    return d, path


def test_load_queue_orders_by_position():
    tmp, path = _queue_file([
        {"pr": 7, "title": "later", "author": "bob", "head_sha": "b" * 40, "position": 2},
        {"pr": 3, "title": "first", "author": "alice", "head_sha": "a" * 40, "position": 1},
    ])
    try:
        queue = load_queue(path)
        assert [item.pr for item in queue] == [3, 7]
    finally:
        tmp.cleanup()


def test_select_batch_limit():
    tmp, path = _queue_file([
        {"pr": 1, "title": "a", "author": "a", "head_sha": "1", "position": 1},
        {"pr": 2, "title": "b", "author": "b", "head_sha": "2", "position": 2},
    ])
    try:
        assert [item.pr for item in select_batch(load_queue(path), 1)] == [1]
        assert [item.pr for item in select_batch(load_queue(path), 0)] == [1, 2]
    finally:
        tmp.cleanup()


def test_eval_args_omit_seed_by_default():
    args = eval_args(EvalSpec(transforms="mine", rank_m=128))
    assert "--seed" not in args
    assert args[args.index("--transforms") + 1] == "mine"
    assert args[args.index("--rank-m") + 1] == "128"


def test_eval_args_include_seed_when_reproducing():
    args = eval_args(EvalSpec(seed=123))
    assert args[args.index("--seed") + 1] == "123"


def test_plan_contains_sha_check_and_json_output():
    tmp, path = _queue_file([
        {"pr": 4, "title": "mine", "author": "alice", "head_sha": "abcdef1234567890",
         "position": 1},
    ])
    try:
        item = load_queue(path)[0]
        commands = plan_item(
            item,
            repo="owner/repo",
            workdir="_work",
            results_dir="_results",
            spec=EvalSpec(transforms="mine"),
        )
        joined = "\n".join(commands)
        assert "gh pr checkout 4" in joined
        assert "abcdef1234567890" in joined
        assert "python -m eval" in joined
        assert "--json" in joined
        assert "_results/pr-4-abcdef123456.json" in joined
    finally:
        tmp.cleanup()


def test_mock_result_has_wrapped_eval_shape():
    tmp, path = _queue_file([
        {"pr": 9, "title": "mock me", "author": "alice", "head_sha": "f" * 40,
         "position": 1},
    ])
    try:
        payload = mock_result(load_queue(path)[0], EvalSpec(transforms="mine"))
        assert payload["mock"] is True
        assert payload["eval"]["best"] == "mine"
        assert payload["eval"]["config"]["device"] == "RTX 5090 (mock)"
        assert payload["eval"]["transforms"]["mine"]["improvement"] is True
    finally:
        tmp.cleanup()


def test_wrap_result_adds_pr_metadata():
    tmp, path = _queue_file([
        {"pr": 5, "title": "real", "author": "bob", "head_sha": "a" * 40,
         "position": 1},
    ])
    try:
        item = load_queue(path)[0]
        payload = wrap_result(item, '{"config": {}, "transforms": {}, "best": null}')
        assert payload["pr"] == 5
        assert payload["mock"] is False
        assert "eval" in payload
    finally:
        tmp.cleanup()


def test_run_item_mock_writes_result_without_checkout():
    tmp, path = _queue_file([
        {"pr": 8, "title": "mock run", "author": "carol", "head_sha": "b" * 40,
         "position": 1},
    ])
    try:
        with tempfile.TemporaryDirectory() as d:
            out = run_item(
                load_queue(path)[0],
                repo="owner/repo",
                workdir=Path(d) / "work",
                results_dir=Path(d) / "results",
                spec=EvalSpec(transforms="mine"),
                mock=True,
            )
            data = json.loads(out.read_text())
            assert data["pr"] == 8
            assert data["eval"]["best"] == "mine"
            assert not (Path(d) / "work").exists()
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
