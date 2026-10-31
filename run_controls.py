import argparse
import subprocess
import sys
from pathlib import Path

from run_experiments import MODELS


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["all", *MODELS], default="all")
    p.add_argument("--results-root", default="results/main")
    p.add_argument("--control-root", default="results/control_csqa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    families = list(MODELS) if args.family == "all" else [args.family]
    main_root = Path(args.results_root)
    control_root = Path(args.control_root)
    manifest = control_root / "csqa_1200.json"

    for family in families:
        for variant in ("base", "instruct"):
            model = MODELS[family][variant]
            items = main_root / "commonsense_qa" / f"{family}_{variant}" / "items.jsonl"
            out = control_root / f"{family}_{variant}"
            run([
                sys.executable,
                "evaluate.py",
                "--model", model,
                "--task", "commonsense_qa",
                "--output-dir", str(out),
                "--stage", "rationales",
                "--items-path", str(items),
                "--generation-items", "1200",
                "--generation-manifest", str(manifest),
                "--seed", str(args.seed),
                "--k", "5",
                "--temperature", "0.7",
                "--top-p", "1.0",
                "--max-new-tokens", "100",
                "--rationale-prompt-mode", "plain",
                "--score-rationale-answers",
                "--dtype", args.dtype,
                "--device", args.device,
            ])

        run([
            sys.executable,
            "controlled_analysis.py",
            "--base-items", str(main_root / "commonsense_qa" / f"{family}_base" / "items.jsonl"),
            "--instruct-items", str(main_root / "commonsense_qa" / f"{family}_instruct" / "items.jsonl"),
            "--base-rationales", str(control_root / f"{family}_base" / "rationale_samples.jsonl"),
            "--instruct-rationales", str(control_root / f"{family}_instruct" / "rationale_samples.jsonl"),
            "--output-dir", str(control_root / family),
        ])


if __name__ == "__main__":
    main()
