from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def softmax(scores: list[float]) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()


def choice_entropy(probs: list[float] | np.ndarray) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(probs)))


def ece(correct: np.ndarray, confidence: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    keep = np.isfinite(confidence)
    correct = correct[keep]
    confidence = confidence[keep]

    edges = np.linspace(0, 1, bins + 1)
    ids = np.digitize(confidence, edges[1:-1])
    total = 0.0
    rows = []
    for b in range(bins):
        mask = ids == b
        n = int(mask.sum())
        if not n:
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        weight = n / len(correct)
        total += weight * abs(acc - conf)
        rows.append({"bin": b, "n": n, "accuracy": acc, "confidence": conf})
    return float(total), pd.DataFrame(rows)


def calibration(df: pd.DataFrame, confidence_col: str, bins: int = 10) -> tuple[dict[str, Any], pd.DataFrame]:
    x = df[["correct", confidence_col]].dropna()
    if x.empty:
        return {"n": 0}, pd.DataFrame()
    value, bin_df = ece(x["correct"].to_numpy(), x[confidence_col].to_numpy(), bins)
    return {
        "n": int(len(x)),
        "accuracy": float(x["correct"].mean()),
        "mean_confidence": float(x[confidence_col].mean()),
        "ece": value,
    }, bin_df


def answer_labels(n: int) -> list[str]:
    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n])


def question_text(doc: dict[str, Any]) -> str:
    for key in ("question", "query", "stem"):
        if doc.get(key) is not None:
            return str(doc[key])
    raise ValueError("Question text not found")


def get_choices(task: str, doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    if task.startswith("mmlu_") or task == "mmlu":
        choices = [str(x) for x in doc.get("choices", [])]
        return answer_labels(len(choices)), choices

    choices = doc.get("choices")
    if isinstance(choices, dict):
        texts = [str(x) for x in choices.get("text", [])]
        labels = [str(x) for x in choices.get("label", [])]
        if len(texts) == len(labels) and texts:
            return labels, texts
    if isinstance(choices, (list, tuple)):
        texts = [str(x) for x in choices]
        return answer_labels(len(texts)), texts
    raise ValueError(f"Unsupported choices for {task}")


def get_gold(task: str, doc: dict[str, Any], labels: list[str]) -> int:
    raw = doc.get("answer") if task.startswith("mmlu") else doc.get("answerKey", doc.get("answer"))
    if isinstance(raw, (int, np.integer)):
        idx = int(raw)
        if 0 <= idx < len(labels):
            return idx
    value = str(raw).strip()
    if value in labels:
        return labels.index(value)
    if len(value) == 1 and value.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(value.upper()) - ord("A")
        if 0 <= idx < len(labels):
            return idx
    if value.isdigit():
        idx = int(value)
        if 0 <= idx < len(labels):
            return idx
        if 1 <= idx <= len(labels):
            return idx - 1
    raise ValueError(f"Could not map answer {raw!r}")


def first_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, dict):
        for key in ("loglikelihood", "logprob", "score"):
            if key in value:
                found = first_number(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = first_number(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_number(item)
            if found is not None:
                return found
    return None


def parse_sample(task: str, sample: dict[str, Any]) -> dict[str, Any]:
    doc = sample["doc"]
    labels, choices = get_choices(task, doc)
    responses = sample.get("resps", [])
    if len(responses) != len(choices):
        raise ValueError(f"Unexpected number of responses for {task}: {len(responses)}")

    scores = []
    for response in responses:
        score = first_number(response)
        if score is None:
            raise ValueError(f"Could not read log-likelihood: {response!r}")
        scores.append(score)

    probs = softmax(scores)
    selected = int(np.argmax(scores))
    gold = get_gold(task, doc, labels)
    doc_id = int(sample.get("doc_id", -1))
    native_id = next((str(doc[k]) for k in ("id", "question_id", "item_id") if doc.get(k) is not None), None)
    item_id = f"{task}::{native_id if native_id is not None else doc_id}"

    return {
        "item_id": item_id,
        "task_name": task,
        "doc_id": doc_id,
        "question": question_text(doc),
        "choices": choices,
        "gold_index": gold,
        "selected_index": selected,
        "choice_loglikelihoods": scores,
        "choice_probabilities": probs.tolist(),
        "likelihood_confidence": float(probs[selected]),
        "choice_entropy": choice_entropy(probs),
        "correct": int(selected == gold),
    }


def task_size(task: Any) -> int:
    try:
        return len(task.eval_docs)
    except TypeError:
        return sum(1 for _ in task.eval_docs)


def eval_subset(task_name: str, manager: Any, n_items: int | None, seed: int):
    if n_items is None:
        return None
    loaded = manager.load([task_name])["tasks"]
    names = sorted(loaded)
    rng = np.random.default_rng(seed)
    if len(names) == 1:
        n = task_size(loaded[names[0]])
        take = min(n_items, n)
        return {names[0]: np.sort(rng.choice(n, take, replace=False)).astype(int).tolist()}

    base, extra = divmod(n_items, len(names))
    order = list(names)
    rng.shuffle(order)
    extra_names = set(order[:extra])
    out = {}
    for name in names:
        n = task_size(loaded[name])
        take = min(base + int(name in extra_names), n)
        if take:
            out[name] = np.sort(rng.choice(n, take, replace=False)).astype(int).tolist()
    return out


def logged_samples(results: dict[str, Any], requested_task: str) -> list[tuple[str, dict[str, Any]]]:
    samples = results.get("samples", {})
    rows = []
    for task, task_rows in samples.items():
        if requested_task == "mmlu":
            if not task.startswith("mmlu_"):
                continue
        elif task != requested_task and requested_task not in task:
            continue
        rows.extend((task, row) for row in task_rows)
    if not rows:
        raise RuntimeError(f"No logged samples found for {requested_task}")
    return rows


def run_likelihood(args: argparse.Namespace, out: Path) -> None:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    manager = TaskManager()
    subset = eval_subset(args.task, manager, args.n_items, args.seed)
    result = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model},trust_remote_code=True,dtype={args.dtype}",
        tasks=[args.task],
        num_fewshot=0,
        batch_size=args.eval_batch_size,
        device=args.device,
        samples=subset,
        bootstrap_iters=0,
        log_samples=True,
        apply_chat_template=False,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
        manager=manager,
    )
    if result is None:
        raise RuntimeError("lm-eval returned no results")

    rows = [parse_sample(task, sample) for task, sample in logged_samples(result, args.task)]
    rows.sort(key=lambda x: (x["task_name"], x["doc_id"]))
    save_jsonl(rows, out / "items.jsonl")

    df = pd.DataFrame(rows)
    summary, bins = calibration(df, "likelihood_confidence", args.n_bins)
    bins.to_csv(out / "likelihood_ece_bins.csv", index=False)
    save_json(summary, out / "likelihood_summary.json")

    compact = dict(result)
    compact.pop("samples", None)
    save_json(compact, out / "lm_eval_results.json")

    del result
    gc.collect()
    _empty_cache()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_model(name: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model, next(model.parameters()).device


def format_question(row: dict[str, Any]) -> str:
    labels = answer_labels(len(row["choices"]))
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(labels, row["choices"], strict=True))
    return f"Question: {row['question']}\n{options}"


def render_prompt(tokenizer, text: str, mode: str) -> str:
    if mode == "plain":
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


WORD_RE = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.UNICODE)


def words(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def unique_2(texts: list[str]) -> float:
    bigrams = []
    for text in texts:
        toks = words(text)
        bigrams.extend(zip(toks, toks[1:]))
    return float(len(set(bigrams)) / len(bigrams)) if bigrams else float("nan")


def self_bleu(texts: list[str]) -> float:
    if len(texts) < 2:
        return float("nan")
    from sacrebleu.metrics import BLEU

    bleu = BLEU(effective_order=True, tokenize="none", smooth_method="exp")
    normalized = [" ".join(words(text)) for text in texts]
    scores = []
    for i, hypothesis in enumerate(normalized):
        refs = [text for j, text in enumerate(normalized) if j != i]
        if hypothesis and any(refs):
            scores.append(bleu.sentence_score(hypothesis, refs).score / 100.0)
    return float(np.mean(scores)) if scores else float("nan")


def diversity(texts: list[str]) -> dict[str, float]:
    texts = [str(text).strip() for text in texts if str(text).strip()]
    sb = self_bleu(texts)
    lengths = [len(words(text)) for text in texts]
    return {
        "mean_rationale_length": float(np.mean(lengths)) if lengths else float("nan"),
        "unique_2": unique_2(texts),
        "one_minus_self_bleu": 1.0 - sb if np.isfinite(sb) else float("nan"),
    }


def make_generation_manifest(rows: list[dict[str, Any]], n_items: int, seed: int) -> dict[str, Any]:
    if n_items >= len(rows):
        selected = rows
    else:
        rng = np.random.default_rng(seed)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row["task_name"], []).append(row)
        names = sorted(groups)
        base, extra = divmod(n_items, len(names))
        order = list(names)
        rng.shuffle(order)
        extra_names = set(order[:extra])
        selected = []
        leftovers = []
        for name in names:
            group = groups[name]
            take = min(base + int(name in extra_names), len(group))
            chosen = set(rng.choice(len(group), take, replace=False).astype(int).tolist()) if take else set()
            selected.extend(group[i] for i in sorted(chosen))
            leftovers.extend(row for i, row in enumerate(group) if i not in chosen)
        if len(selected) < n_items:
            need = min(n_items - len(selected), len(leftovers))
            selected.extend(leftovers[i] for i in rng.choice(len(leftovers), need, replace=False).astype(int))

    return {
        "seed": seed,
        "n_items": n_items,
        "item_ids": sorted(str(row["item_id"]) for row in selected),
    }


def generation_rows(rows: list[dict[str, Any]], manifest_path: Path, n_items: int, seed: int) -> list[dict[str, Any]]:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["seed"]) != seed or int(manifest["n_items"]) != n_items:
            raise RuntimeError(f"Manifest settings do not match: {manifest_path}")
    else:
        manifest = make_generation_manifest(rows, n_items, seed)
        save_json(manifest, manifest_path)

    by_id = {str(row["item_id"]): row for row in rows}
    missing = [item for item in manifest["item_ids"] if item not in by_id]
    if missing:
        raise RuntimeError(f"Manifest contains items missing from this run: {missing[:5]}")
    return [by_id[item] for item in manifest["item_ids"]]


def score_sequences(tokenizer, model, device, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    import torch

    scores = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        encoded = []
        starts = []
        for prompt, continuation in batch:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
            if not prompt_ids:
                prompt_ids = [tokenizer.bos_token_id or tokenizer.eos_token_id]
            encoded.append(prompt_ids + continuation_ids)
            starts.append(len(prompt_ids))

        width = max(map(len, encoded))
        input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(encoded):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, :len(ids)] = 1

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)

        for i, ids in enumerate(encoded):
            score = 0.0
            for pos in range(starts[i], len(ids)):
                score += float(log_probs[i, pos - 1, ids[pos]].item())
            scores.append(score)
    return scores


def rationale_answer(tokenizer, model, device, row: dict[str, Any], rationale: str, batch_size: int) -> int:
    prompt = f"{format_question(row)}\n\nAnswer: Let's think step by step.\n{rationale}\nTherefore, the answer is"
    pairs = [(prompt, f" {label}") for label in answer_labels(len(row["choices"]))]
    scores = score_sequences(tokenizer, model, device, pairs, batch_size)
    return int(np.argmax(scores))


def run_rationales(args: argparse.Namespace, out: Path) -> None:
    import torch

    items_path = Path(args.items_path) if args.items_path else out / "items.jsonl"
    rows = load_jsonl(items_path)
    if not rows:
        raise RuntimeError(f"No items found at {items_path}")

    manifest_path = Path(args.generation_manifest) if args.generation_manifest else out / "generation_manifest.json"
    rows = generation_rows(rows, manifest_path, args.generation_items, args.seed)
    tokenizer, model, device = load_model(args.model, args.dtype)
    samples_path = out / "rationale_samples.jsonl"
    existing = {str(row["item_id"]): row for row in load_jsonl(samples_path)}

    for i, row in enumerate(rows, 1):
        if str(row["item_id"]) in existing:
            continue
        prompt = f"{format_question(row)}\n\nAnswer: Let's think step by step."
        prompt = render_prompt(tokenizer, prompt, args.rationale_prompt_mode)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.k,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        rationales = [text.strip() for text in tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)]
        record = {"item_id": row["item_id"], "doc_id": row["doc_id"], "rationales": rationales}
        if args.score_rationale_answers:
            record["answer_indices"] = [
                rationale_answer(tokenizer, model, device, row, text, args.answer_batch_size)
                for text in rationales
            ]
        existing[str(row["item_id"])] = record

        if i % args.checkpoint_every == 0 or i == len(rows):
            ordered = [existing[str(r["item_id"])] for r in rows if str(r["item_id"]) in existing]
            save_jsonl(ordered, samples_path)

    records = {str(row["item_id"]): row for row in load_jsonl(samples_path)}
    metric_rows = []
    for row in rows:
        record = records.get(str(row["item_id"]))
        if record:
            metric_rows.append({"item_id": row["item_id"], "doc_id": row["doc_id"], **diversity(record["rationales"])})
    pd.DataFrame(metric_rows).to_csv(out / "rationale_metrics.csv", index=False)

    del model, tokenizer
    gc.collect()
    _empty_cache()


PROB_RE = re.compile(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")
PERCENT_RE = re.compile(r"(?<!\d)(100|\d{1,2}(?:\.\d+)?)\s*%")


def parse_probability(text: str) -> float | None:
    text = str(text).strip()
    match = PROB_RE.search(text)
    if match:
        return float(match.group(1))
    match = PERCENT_RE.search(text)
    if match:
        return float(match.group(1)) / 100.0
    try:
        value = float(text.split()[0])
        if 0 <= value <= 1:
            return value
    except (ValueError, IndexError):
        pass
    return None


def confidence_prompt(row: dict[str, Any]) -> str:
    selected = int(row["selected_index"])
    label = answer_labels(len(row["choices"]))[selected]
    answer = row["choices"][selected]
    return (
        f"{format_question(row)}\n\n"
        f"Selected answer: {label}. {answer}\n\n"
        "What is the probability that the selected answer is correct?\n"
        "Give only a number between 0 and 1."
    )


def run_verbal(args: argparse.Namespace, out: Path) -> None:
    import torch

    items_path = Path(args.items_path) if args.items_path else out / "items.jsonl"
    rows = load_jsonl(items_path)
    if not rows:
        raise RuntimeError(f"No items found at {items_path}")

    tokenizer, model, device = load_model(args.model, args.dtype)
    path = out / "verbal_confidence.jsonl"
    existing = {str(row["item_id"]): row for row in load_jsonl(path)}
    pending = [row for row in rows if str(row["item_id"]) not in existing]

    for start in range(0, len(pending), args.verbal_batch_size):
        batch = pending[start:start + args.verbal_batch_size]
        prompts = [render_prompt(tokenizer, confidence_prompt(row), args.verbal_prompt_mode) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.verbal_max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        for row, output in zip(batch, outputs, strict=True):
            value = parse_probability(output)
            existing[str(row["item_id"])] = {
                "item_id": row["item_id"],
                "doc_id": row["doc_id"],
                "raw_output": output,
                "verbalized_confidence": value,
            }
        save_jsonl([existing[str(row["item_id"])] for row in rows if str(row["item_id"]) in existing], path)

    del model, tokenizer
    gc.collect()
    _empty_cache()


def summarize(args: argparse.Namespace, out: Path) -> None:
    items_path = Path(args.items_path) if args.items_path else out / "items.jsonl"
    df = pd.DataFrame(load_jsonl(items_path))
    if df.empty:
        raise RuntimeError(f"No items found at {items_path}")

    rationale_path = out / "rationale_metrics.csv"
    if rationale_path.exists():
        df = df.merge(pd.read_csv(rationale_path), on=["item_id", "doc_id"], how="left", validate="one_to_one")

    verbal_path = out / "verbal_confidence.jsonl"
    if verbal_path.exists():
        verbal = pd.DataFrame(load_jsonl(verbal_path))
        df = df.merge(verbal[["item_id", "doc_id", "verbalized_confidence"]], on=["item_id", "doc_id"], how="left", validate="one_to_one")

    df.to_csv(out / "per_item_metrics.csv", index=False)

    likelihood, likelihood_bins = calibration(df, "likelihood_confidence", args.n_bins)
    likelihood_bins.to_csv(out / "likelihood_ece_bins.csv", index=False)
    verbal = {"n": 0}
    if "verbalized_confidence" in df:
        verbal, verbal_bins = calibration(df, "verbalized_confidence", args.n_bins)
        verbal_bins.to_csv(out / "verbal_ece_bins.csv", index=False)

    summary = {
        "model": args.model,
        "task": args.task,
        "n_items": int(len(df)),
        "accuracy": float(df["correct"].mean()),
        "choice_entropy": float(df["choice_entropy"].mean()),
        "likelihood": likelihood,
        "verbalized": verbal,
    }
    for col in ("verbalized_confidence", "mean_rationale_length", "unique_2", "one_minus_self_bleu"):
        if col in df:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            summary[col] = float(values.mean()) if len(values) else float("nan")
    save_json(summary, out / "summary.json")


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", choices=["arc_easy", "mmlu", "commonsense_qa"], required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stage", choices=["likelihood", "rationales", "verbal", "summarize", "all"], default="all")
    p.add_argument("--items-path")
    p.add_argument("--n-items", type=int)
    p.add_argument("--generation-items", type=int, default=1000)
    p.add_argument("--generation-manifest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32", "auto"], default="bfloat16")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--eval-batch-size", default="auto")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--rationale-prompt-mode", choices=["plain", "chat"], default="plain")
    p.add_argument("--score-rationale-answers", action="store_true")
    p.add_argument("--answer-batch-size", type=int, default=32)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--verbal-prompt-mode", choices=["plain", "chat"], default="plain")
    p.add_argument("--verbal-batch-size", type=int, default=8)
    p.add_argument("--verbal-max-new-tokens", type=int, default=8)
    return p


def main() -> None:
    args = parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    save_json(vars(args), out / "config.json")

    if args.stage in {"likelihood", "all"}:
        run_likelihood(args, out)
    if args.stage in {"rationales", "all"}:
        run_rationales(args, out)
    if args.stage in {"verbal", "all"}:
        run_verbal(args, out)
    if args.stage in {"summarize", "all"}:
        summarize(args, out)


if __name__ == "__main__":
    main()
