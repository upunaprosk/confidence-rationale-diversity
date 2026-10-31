import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from evaluate import diversity, words


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["item_id"])] = row
    return rows


def support(record: dict, answer: int) -> list[str]:
    answers = record.get("answer_indices", [])
    return [text for text, idx in zip(record["rationales"], answers, strict=True) if int(idx) == answer]


def match_lengths(base: list[str], instruct: list[str]) -> tuple[list[str], list[str]]:
    n = min(len(base), len(instruct))
    if n < 2:
        return [], []

    base = sorted(base, key=lambda text: len(words(text)))[:n]
    instruct = sorted(instruct, key=lambda text: len(words(text)))[:n]
    matched_base = []
    matched_instruct = []
    for base_text, instruct_text in zip(base, instruct, strict=True):
        base_tokens = words(base_text)
        instruct_tokens = words(instruct_text)
        length = min(len(base_tokens), len(instruct_tokens))
        matched_base.append(" ".join(base_tokens[:length]))
        matched_instruct.append(" ".join(instruct_tokens[:length]))
    return matched_base, matched_instruct


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-items", required=True)
    p.add_argument("--instruct-items", required=True)
    p.add_argument("--base-rationales", required=True)
    p.add_argument("--instruct-rationales", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    base_items = load_jsonl(Path(args.base_items))
    instruct_items = load_jsonl(Path(args.instruct_items))
    base_rationales = load_jsonl(Path(args.base_rationales))
    instruct_rationales = load_jsonl(Path(args.instruct_rationales))

    rows = []
    common = sorted(set(base_items) & set(instruct_items) & set(base_rationales) & set(instruct_rationales))
    for item_id in common:
        base_item = base_items[item_id]
        instruct_item = instruct_items[item_id]
        selected = int(base_item["selected_index"])
        if selected != int(instruct_item["selected_index"]):
            continue

        base_support = support(base_rationales[item_id], selected)
        instruct_support = support(instruct_rationales[item_id], selected)
        base_texts, instruct_texts = match_lengths(base_support, instruct_support)
        if len(base_texts) < 2:
            continue

        base_scores = diversity(base_texts)
        instruct_scores = diversity(instruct_texts)
        rows.append({
            "item_id": item_id,
            "n_rationales": len(base_texts),
            "base_unique_2": base_scores["unique_2"],
            "instruct_unique_2": instruct_scores["unique_2"],
            "base_one_minus_self_bleu": base_scores["one_minus_self_bleu"],
            "instruct_one_minus_self_bleu": instruct_scores["one_minus_self_bleu"],
        })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "controlled_items.csv", index=False)

    summary = {"n": int(len(df))}
    for metric in ("unique_2", "one_minus_self_bleu"):
        base = df[f"base_{metric}"].to_numpy(dtype=float)
        instruct = df[f"instruct_{metric}"].to_numpy(dtype=float)
        test = ttest_rel(instruct, base, nan_policy="omit")
        summary[f"base_{metric}"] = float(np.nanmean(base))
        summary[f"instruct_{metric}"] = float(np.nanmean(instruct))
        summary[f"delta_{metric}"] = float(np.nanmean(instruct - base))
        summary[f"p_{metric}"] = float(test.pvalue)

    pd.DataFrame([summary]).to_csv(out / "controlled_summary.csv", index=False)


if __name__ == "__main__":
    main()
