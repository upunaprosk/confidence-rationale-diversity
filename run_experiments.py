import argparse
import subprocess
import sys
from pathlib import Path


MODELS = {
    "qwen": {
        "base": "Qwen/Qwen2.5-7B",
        "instruct": "Qwen/Qwen2.5-7B-Instruct",
    },
    "mistral": {
        "base": "mistralai/Mistral-7B-v0.3",
        "instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "llama": {
        "base": "meta-llama/Llama-3.1-8B",
        "instruct": "meta-llama/Llama-3.1-8B-Instruct",
    },
}

TASKS = ["arc_easy", "mmlu", "commonsense_qa"]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["all", *MODELS], default="all")
    p.add_argument("--variant", choices=["both", "base", "instruct"], default="both")
    p.add_argument("--task", choices=["all", *TASKS], default="all")
    p.add_argument("--results-root", default="results/main")
    p.add_argument("--generation-items", type=int, default=1000)
    p.add_argument("--n-items", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    families = list(MODELS) if args.family == "all" else [args.family]
    variants = ["base", "instruct"] if args.variant == "both" else [args.variant]
    tasks = TASKS if args.task == "all" else [args.task]
    root = Path(args.results_root)

    for task in tasks:
        manifest = root / "manifests" / f"{task}_{args.generation_items}.json"
        for family in families:
            for variant in variants:
                model = MODELS[family][variant]
                out = root / task / f"{family}_{variant}"
                cmd = [
                    sys.executable,
                    "evaluate.py",
                    "--model", model,
                    "--task", task,
                    "--output-dir", str(out),
                    "--generation-items", str(args.generation_items),
                    "--generation-manifest", str(manifest),
                    "--seed", str(args.seed),
                    "--k", str(args.k),
                    "--temperature", "0.7",
                    "--top-p", "1.0",
                    "--max-new-tokens", str(args.max_new_tokens),
                    "--dtype", args.dtype,
                    "--device", args.device,
                    "--rationale-prompt-mode", "plain",
                    "--verbal-prompt-mode", "chat" if variant == "instruct" else "plain",
                    "--stage", "all",
                ]
                if args.n_items is not None:
                    cmd.extend(["--n-items", str(args.n_items)])
                run(cmd)


if __name__ == "__main__":
    main()
