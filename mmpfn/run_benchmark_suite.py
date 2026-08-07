"""One-command, resumable MMPFN evaluation across registered datasets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from mmpfn.benchmarking.registry import get_dataset_spec, registry_rows, select_dataset_specs
from mmpfn.benchmarking.summary import summarize_results


DEFAULT_BASE = Path("/mnt/hdd/zhangyg/projects/tab")


def _mapping(values: Iterable[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH, got: {value}")
        key, path = value.split("=", 1)
        try:
            key = get_dataset_spec(key.strip()).key
        except ValueError:
            key = key.strip()
        output[key] = Path(path).expanduser().resolve()
    return output


def _print_registry() -> None:
    rows = registry_rows()
    headers = ("key", "benchmark", "task", "display_name", "primary_metric", "automatic_download")
    widths = {header: max(len(header), *(len(str(row[header])) for row in rows)) for header in headers}
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def _run_logged(command: list[str], log_path: Path, env: dict[str, str], *, dry_run: bool) -> int:
    rendered = subprocess.list2cmdline(command)
    print(f"\n$ {rendered}", flush=True)
    print(f"log: {log_path}", flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["multabench_text0"])
    parser.add_argument("--modes", nargs="+", choices=("full", "image_only", "tabular_only"), default=["full", "image_only", "tabular_only"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="1", help="Physical GPU id exposed to each child process.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_BASE / "benchmark_data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASE / "results" / "mmpfn_benchmark_suite")
    parser.add_argument("--dino-checkpoint", type=Path, default=DEFAULT_BASE / "models" / "dinov2_vitb14_pretrain.pth")
    parser.add_argument("--dataset-root", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--source-root", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--download-multabench", action="store_true")
    parser.add_argument("--max-train-context", type=int, default=None)
    parser.add_argument("--max-val-context", type=int, default=0)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--validation-chunk-size", type=int, default=512)
    parser.add_argument("--finetune-steps", type=int, default=100)
    parser.add_argument("--time-limit", type=int, default=43_200)
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun completed metrics and re-prepare data.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_datasets:
        _print_registry()
        return

    specs = select_dataset_specs(args.datasets)
    if not specs:
        raise ValueError("No datasets selected.")
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    overrides = _mapping(args.dataset_root)
    source_roots = _mapping(args.source_root)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cache_root = data_root / "_cache"
    env.update(
        {
            "HF_HOME": str(cache_root / "huggingface"),
            "TORCH_HOME": str(cache_root / "torch"),
            "KAGGLEHUB_CACHE": str(cache_root / "kagglehub"),
            "XDG_CACHE_HOME": str(cache_root),
        }
    )
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    for spec in specs:
        for fold in args.folds:
            if spec.key in overrides:
                dataset_root = overrides[spec.key]
            elif spec.benchmark == "multabench":
                dataset_root = data_root / "multabench" / spec.key / f"fold_{fold}"
            else:
                dataset_root = data_root / "vtbench" / spec.key / f"fold_{fold}"

            prepared = (dataset_root / "metadata.json").is_file() and (dataset_root / "train.npz").is_file()
            legacy_available = spec.legacy_vtbench_name is not None and dataset_root.is_dir()
            if spec.benchmark == "multabench" and (args.force or not prepared):
                if not args.download_multabench and spec.key not in source_roots:
                    message = (
                        f"{spec.key} is not prepared at {dataset_root}. Add --download-multabench "
                        "or --source-root DATASET=/path/to/official/release."
                    )
                    print(f"ERROR: {message}")
                    failures.append({"dataset": spec.key, "fold": fold, "stage": "prepare", "error": message})
                    if args.fail_fast:
                        raise RuntimeError(message)
                    continue
                prepare_command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "mmpfn.prepare_multabench",
                    "--datasets",
                    spec.key,
                    "--data-root",
                    str(data_root / "multabench"),
                    "--folds",
                    str(fold),
                ]
                if spec.key in source_roots:
                    prepare_command += ["--source-root", f"{spec.key}={source_roots[spec.key]}"]
                if args.force:
                    prepare_command.append("--force")
                code = _run_logged(
                    prepare_command,
                    output_root / "logs" / spec.key / f"fold_{fold}" / "prepare.log",
                    env,
                    dry_run=args.dry_run,
                )
                if code:
                    failures.append({"dataset": spec.key, "fold": fold, "stage": "prepare", "returncode": code})
                    if args.fail_fast:
                        raise RuntimeError(f"Preparing {spec.key} failed with exit code {code}")
                    continue
            elif not prepared and not legacy_available and not args.dry_run:
                message = f"No prepared/legacy data found for {spec.key}: {dataset_root}"
                print(f"ERROR: {message}")
                failures.append({"dataset": spec.key, "fold": fold, "stage": "load", "error": message})
                if args.fail_fast:
                    raise RuntimeError(message)
                continue

            for mode in args.modes:
                metrics_path = output_root / spec.key / f"fold_{fold}" / mode / f"seed_{args.seed}" / "metrics.json"
                if metrics_path.is_file() and not args.force:
                    print(f"SKIP completed: {spec.key} fold={fold} mode={mode}")
                    continue
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "mmpfn.run_benchmark_dataset",
                    "--dataset",
                    spec.key,
                    "--mode",
                    mode,
                    "--data-root",
                    str(dataset_root),
                    "--dino-checkpoint",
                    str(args.dino_checkpoint.expanduser().resolve()),
                    "--output-dir",
                    str(output_root),
                    "--fold",
                    str(fold),
                    "--seed",
                    str(args.seed),
                    "--embedding-batch-size",
                    str(args.embedding_batch_size),
                    "--prediction-batch-size",
                    str(args.prediction_batch_size),
                    "--validation-chunk-size",
                    str(args.validation_chunk_size),
                    "--max-val-context",
                    str(args.max_val_context),
                    "--finetune-steps",
                    str(args.finetune_steps),
                    "--time-limit",
                    str(args.time_limit),
                ]
                if args.max_train_context is not None:
                    command += ["--max-train-context", str(args.max_train_context)]
                if args.save_all_checkpoints:
                    command.append("--save-all-checkpoints")
                code = _run_logged(
                    command,
                    output_root / "logs" / spec.key / f"fold_{fold}" / f"{mode}_seed_{args.seed}.log",
                    env,
                    dry_run=args.dry_run,
                )
                if code:
                    failures.append(
                        {"dataset": spec.key, "fold": fold, "mode": mode, "stage": "evaluate", "returncode": code}
                    )
                    if args.fail_fast:
                        raise RuntimeError(f"{spec.key}/{mode} failed with exit code {code}")

    if not args.dry_run:
        results_path, transfer_path = summarize_results(output_root)
        failure_path = output_root / "summary" / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"\nResults:           {results_path}")
        print(f"Negative transfer: {transfer_path}")
        print(f"Failures:          {failure_path} ({len(failures)})")
    elif failures:
        print(f"Dry run found {len(failures)} unavailable dataset preparations.")


if __name__ == "__main__":
    main()
