import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from run_experiments import MODELS, TASKS


MODEL_NAMES = {"qwen": "Qwen2.5-7B", "mistral": "Mistral-7B", "llama": "Llama-3.1-8B"}
TASK_NAMES = {"arc_easy": "ARC-Easy", "mmlu": "MMLU", "commonsense_qa": "CommonsenseQA"}


def read_run(root: Path, task: str, family: str, variant: str) -> tuple[pd.DataFrame, dict]:
    path = root / task / f"{family}_{variant}"
    return pd.read_csv(path / "per_item_metrics.csv"), json.loads((path / "summary.json").read_text())


def paired_test(base: pd.DataFrame, instruct: pd.DataFrame, column: str) -> tuple[int, float, float]:
    x = base[["item_id", column]].merge(
        instruct[["item_id", column]], on="item_id", suffixes=("_base", "_instruct")
    ).dropna()
    if len(x) < 2:
        return len(x), float("nan"), float("nan")
    delta = x[f"{column}_instruct"] - x[f"{column}_base"]
    p = ttest_rel(x[f"{column}_instruct"], x[f"{column}_base"]).pvalue
    return len(x), float(delta.mean()), float(p)


def direction_table(base: pd.DataFrame, instruct: pd.DataFrame, family: str, task: str) -> list[dict]:
    cols = ["item_id", "choice_entropy", "unique_2", "one_minus_self_bleu"]
    x = base[cols].merge(instruct[cols], on="item_id", suffixes=("_base", "_instruct")).dropna()
    rows = []
    for metric, label in (("unique_2", "U2"), ("one_minus_self_bleu", "1-SB")):
        dh = x["choice_entropy_instruct"] - x["choice_entropy_base"]
        dd = x[f"{metric}_instruct"] - x[f"{metric}_base"]
        n = len(x)
        rows.append({
            "Dataset": TASK_NAMES[task],
            "Model": MODEL_NAMES[family],
            "Diversity": label,
            "N": n,
            "H_down_D_down": 100 * float(((dh < 0) & (dd < 0)).sum()) / n,
            "H_down_D_up": 100 * float(((dh < 0) & (dd >= 0)).sum()) / n,
            "H_up_D_down": 100 * float(((dh >= 0) & (dd < 0)).sum()) / n,
            "H_up_D_up": 100 * float(((dh >= 0) & (dd >= 0)).sum()) / n,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results/main")
    p.add_argument("--output-dir", default="results/tables")
    args = p.parse_args()

    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    table1 = []
    tests = []
    table4 = []
    directional = []
    lengths = []

    for task in TASKS:
        for family in MODELS:
            base, base_summary = read_run(root, task, family, "base")
            instruct, instruct_summary = read_run(root, task, family, "instruct")

            for variant, df, summary in (
                ("Base", base, base_summary),
                ("Instruct", instruct, instruct_summary),
            ):
                table1.append({
                    "Dataset": TASK_NAMES[task],
                    "Model": MODEL_NAMES[family],
                    "Variant": variant,
                    "Accuracy": 100 * float(df["correct"].mean()),
                    "Choice entropy": float(df["choice_entropy"].mean()),
                    "Verbal confidence": 100 * float(df["verbalized_confidence"].dropna().mean()),
                    "Unique-2": float(df["unique_2"].dropna().mean()),
                    "1-SelfBLEU": float(df["one_minus_self_bleu"].dropna().mean()),
                })
                table4.append({
                    "Dataset": TASK_NAMES[task],
                    "Model": MODEL_NAMES[family],
                    "Variant": variant,
                    "Likelihood ECE": 100 * float(summary["likelihood"]["ece"]),
                    "Verbal ECE": 100 * float(summary["verbalized"]["ece"]),
                })

            for column in ("correct", "choice_entropy", "verbalized_confidence", "unique_2", "one_minus_self_bleu"):
                n, delta, p_value = paired_test(base, instruct, column)
                tests.append({
                    "Dataset": TASK_NAMES[task],
                    "Model": MODEL_NAMES[family],
                    "Metric": column,
                    "N": n,
                    "Delta": delta,
                    "p": p_value,
                })

            directional.extend(direction_table(base, instruct, family, task))

            length_pairs = base[["item_id", "mean_rationale_length"]].merge(
                instruct[["item_id", "mean_rationale_length"]],
                on="item_id",
                suffixes=("_base", "_instruct"),
            ).dropna()
            lengths.append({
                "Dataset": TASK_NAMES[task],
                "Model": MODEL_NAMES[family],
                "Base": float(length_pairs["mean_rationale_length_base"].mean()),
                "Instruct": float(length_pairs["mean_rationale_length_instruct"].mean()),
                "Delta": float((length_pairs["mean_rationale_length_instruct"] - length_pairs["mean_rationale_length_base"]).mean()),
            })

    pd.DataFrame(table1).to_csv(out / "table1.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "table1_tests.csv", index=False)
    pd.DataFrame(table4).to_csv(out / "table4_ece.csv", index=False)
    direction_df = pd.DataFrame(directional)
    direction_df[direction_df["Dataset"] == "CommonsenseQA"].to_csv(out / "table2_csqa.csv", index=False)
    direction_df[direction_df["Dataset"] != "CommonsenseQA"].to_csv(out / "table7_arc_mmlu.csv", index=False)
    pd.DataFrame(lengths).to_csv(out / "rationale_length.csv", index=False)


if __name__ == "__main__":
    main()
