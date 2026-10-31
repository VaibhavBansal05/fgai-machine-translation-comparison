"""Small, reproducible English-to-French comparison for FGAI Assignment 3."""

import argparse
import csv
import gc
import hashlib
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import sacrebleu
import torch
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {
    "mt5": "google/mt5-small",
    "llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gemma": "google/gemma-3-270m-it",
}
PAD, BOS, EOS, UNK = 0, 1, 2, 3
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def seed_all():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)


def prepare_data(train_n, val_n, test_n):
    DetectorFactory.seed = SEED
    raw = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train")
    pairs = []
    seen = set()
    for row in raw:
        en = " ".join(row["translation"]["en"].split())
        fr = " ".join(row["translation"]["fr"].split())
        key = en.casefold()
        if 5 <= len(en) <= 120 and 5 <= len(fr) <= 120 and key not in seen:
            seen.add(key)
            pairs.append((en, fr))
    random.Random(SEED).shuffle(pairs)
    total = train_n + val_n + test_n
    clean = []
    for en, fr in pairs:
        try:
            if detect(en) == "en" and detect(fr) == "fr":
                clean.append((en, fr))
        except LangDetectException:
            pass
        if len(clean) == total:
            break
    if len(clean) < total:
        raise ValueError(f"Only {len(clean)} language-checked pairs; requested {total}")
    pairs = clean
    test = pairs[train_n + val_n:]
    digest = hashlib.sha256(json.dumps(test, ensure_ascii=False).encode()).hexdigest()
    return pairs[:train_n], pairs[train_n:train_n + val_n], test, digest


def words(s):
    return TOKEN_RE.findall(s.lower())


def vocabulary(train):
    counts = Counter(t for en, fr in train for s in (en, fr) for t in words(s))
    return {t: i + 4 for i, (t, _) in enumerate(counts.most_common(7996))}


def ids(s, vocab, max_len=40):
    return [BOS] + [vocab.get(t, UNK) for t in words(s)[:max_len - 2]] + [EOS]


def detok(tokens):
    s = " ".join(tokens)
    return re.sub(r"\s+([.,!?;:’'])", r"\1", s).replace("' ", "'")


def clean_prompt_output(text):
    first = text.strip().splitlines()[0] if text.strip() else ""
    return re.sub(r"^(?:French|Fran[cç]ais|Translation|Traduction)\s*:\s*", "", first,
                  flags=re.IGNORECASE).strip()


def batch(pairs, vocab):
    src = [torch.tensor(ids(en, vocab)) for en, _ in pairs]
    tgt = [torch.tensor(ids(fr, vocab)) for _, fr in pairs]
    return (nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD),
            nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD))


class BasicTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=PAD)
        self.position = nn.Embedding(64, 128)
        self.net = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=2,
                                  num_decoder_layers=2, dim_feedforward=256,
                                  dropout=0.1, batch_first=True)
        self.head = nn.Linear(128, vocab_size)

    def embed(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.embedding(x) * (128 ** 0.5) + self.position(pos)

    def forward(self, src, tgt):
        mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device,
                                     dtype=torch.bool), diagonal=1)
        h = self.net(self.embed(src), self.embed(tgt), tgt_mask=mask,
                     src_key_padding_mask=src.eq(PAD),
                     tgt_key_padding_mask=tgt.eq(PAD),
                     memory_key_padding_mask=src.eq(PAD))
        return self.head(h)


@torch.inference_mode()
def basic_translate(model, source, vocab, inverse):
    src = torch.tensor([ids(source, vocab)], device=DEVICE)
    out = torch.tensor([[BOS]], device=DEVICE)
    for _ in range(40):
        next_id = model(src, out)[:, -1].argmax(-1, keepdim=True)
        out = torch.cat((out, next_id), dim=1)
        if next_id.item() == EOS:
            break
    return detok([inverse.get(i, "<unk>") for i in out[0].tolist()[1:] if i not in (PAD, EOS)])


def run_basic(train, val, test, epochs):
    vocab = vocabulary(train)
    inverse = {i: t for t, i in vocab.items()}
    model = BasicTransformer(max(vocab.values()) + 1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    loader = DataLoader(train, batch_size=32, shuffle=True,
                        collate_fn=lambda x: batch(x, vocab))
    sync(); start = time.perf_counter()
    losses = []
    for epoch in range(epochs):
        model.train()
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_losses = []
            for i in range(0, len(val), 32):
                src, tgt = batch(val[i:i + 32], vocab)
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                logits = model(src, tgt[:, :-1])
                val_losses.append(loss_fn(logits.reshape(-1, logits.size(-1)),
                                          tgt[:, 1:].reshape(-1)).item())
        losses.append(sum(val_losses) / len(val_losses))
        print(f"basic epoch {epoch + 1}: validation loss {losses[-1]:.3f}", flush=True)
    sync(); train_seconds = time.perf_counter() - start
    predictions, infer_seconds = evaluate(lambda s: basic_translate(model, s, vocab, inverse), test)
    count = sum(p.numel() for p in model.parameters())
    del model
    cleanup()
    return predictions, train_seconds, infer_seconds, count, {"validation_loss": losses}


def run_mt5(train, val, test, epochs):
    name = MODELS["mt5"]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    count = sum(p.numel() for p in model.parameters())
    def encode(examples):
        inputs = tokenizer(["translate English to French: " + en for en, _ in examples],
                           padding=True, truncation=True, max_length=64, return_tensors="pt")
        labels = tokenizer([fr for _, fr in examples], padding=True,
                           truncation=True, max_length=48, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return {k: v.to(DEVICE) for k, v in inputs.items()}, labels.to(DEVICE)
    sync(); start = time.perf_counter()
    losses = []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        for j in range(0, len(order), 4):
            x, y = encode([train[i] for i in order[j:j + 4]])
            optimizer.zero_grad()
            loss = model(**x, labels=y).loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_losses = []
            for j in range(0, len(val), 4):
                x, y = encode(val[j:j + 4])
                val_losses.append(model(**x, labels=y).loss.item())
        losses.append(sum(val_losses) / len(val_losses))
        print(f"mt5 epoch {epoch + 1}: validation loss {losses[-1]:.3f}", flush=True)
    sync(); train_seconds = time.perf_counter() - start
    @torch.inference_mode()
    def translate(s):
        x = tokenizer("translate English to French: " + s, return_tensors="pt",
                      truncation=True, max_length=64).to(DEVICE)
        return tokenizer.decode(model.generate(**x, max_new_tokens=40, do_sample=False)[0],
                                skip_special_tokens=True).strip()
    predictions, infer_seconds = evaluate(translate, test)
    del model, optimizer
    cleanup()
    return predictions, train_seconds, infer_seconds, count, {"validation_loss": losses}


def run_prompt(model_key, test):
    name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(DEVICE)
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    @torch.inference_mode()
    def translate(s, temperature=0):
        prompt = [{"role": "user", "content":
                   "Translate this English sentence into French. Output only the French translation.\nEnglish: " + s}]
        x = tokenizer.apply_chat_template(prompt, add_generation_prompt=True,
                                           tokenize=True, return_dict=True,
                                           return_tensors="pt").to(DEVICE)
        options = {"max_new_tokens": 40, "do_sample": temperature > 0,
                   "pad_token_id": tokenizer.eos_token_id}
        if temperature > 0:
            options["temperature"] = temperature
        output = model.generate(**x, **options)[0][x["input_ids"].shape[-1]:]
        return clean_prompt_output(tokenizer.decode(output, skip_special_tokens=True))
    predictions, infer_seconds = evaluate(translate, test)
    temperature_rows = []
    for source, reference in test[:2]:
        for temperature in (0, 0.7, 1.0):
            temperature_rows.append({"model": model_key, "temperature": temperature,
                                     "source": source, "reference": reference,
                                     "translation": translate(source, temperature)})
    del model
    cleanup()
    return predictions, None, infer_seconds, count, {"temperature_samples": temperature_rows}


def cleanup():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def evaluate(translate, test):
    sync(); start = time.perf_counter()
    predictions = [translate(en) for en, _ in test]
    sync(); elapsed = time.perf_counter() - start
    return predictions, elapsed


def save_result(folder, key, test, digest, result, settings):
    predictions, train_seconds, infer_seconds, count, extra = result
    refs = [fr for _, fr in test]
    metrics = {
        "model": key, "model_id": MODELS.get(key, "scratch Transformer"),
        "adaptation": "prompt only" if key in ("llama", "gemma") else "trained",
        "bleu": sacrebleu.corpus_bleu(predictions, [refs]).score,
        "chrf": sacrebleu.corpus_chrf(predictions, [refs]).score,
        "train_seconds": train_seconds, "infer_seconds": infer_seconds,
        "seconds_per_sentence": infer_seconds / len(test),
        "parameters": count, "test_sha256": digest,
        "test_size": len(test), "device": str(DEVICE), "settings": settings,
        **{k: v for k, v in extra.items() if k != "temperature_samples"},
    }
    (folder / f"{key}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (folder / f"{key}_translations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "reference", "translation"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows({"source": en, "reference": fr, "translation": pred}
                         for (en, fr), pred in zip(test, predictions))
    if "temperature_samples" in extra:
        with (folder / f"{key}_temperatures.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(extra["temperature_samples"][0]),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(extra["temperature_samples"])
    return metrics


def summarize(folder):
    rows = [json.loads((folder / f"{key}.json").read_text()) for key in
            ("basic", "mt5", "llama", "gemma") if (folder / f"{key}.json").exists()]
    if not rows:
        return
    hashes = {row["test_sha256"] for row in rows}
    if len(hashes) != 1:
        raise ValueError("Results use different test sets; do not compare them")
    lines = ["# Measured results", "", "All models use the same English-to-French test set and greedy decoding (temperature 0).", "",
             "| Model | Adaptation | BLEU | chrF | Train seconds | Seconds/sentence | Parameters |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        train_time = "N/A" if r["train_seconds"] is None else f'{r["train_seconds"]:.1f}'
        lines.append(f'| {r["model"]} | {r["adaptation"]} | {r["bleu"]:.2f} | {r["chrf"]:.2f} | {train_time} | {r["seconds_per_sentence"]:.3f} | {r["parameters"]:,} |')
    if len(rows) == 4:
        winner = max(rows, key=lambda r: r["chrf"])
        lines += ["", f'Best chrF on this test set: **{winner["model"]}**. BLEU, speed, and parameter count may favor other models.']
    lines += ["", "Prompt-only models have no task training time. Timing depends on this run's hardware and tokenizer. Parameter count and pretraining differ, so architectural effects cannot be isolated."]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="390" viewBox="0 0 720 390">',
           '<rect width="720" height="390" fill="white"/>',
           '<text x="24" y="32" font-family="sans-serif" font-size="20">Translation scores (higher is better)</text>']
    for col, metric, color in ((0, "bleu", "#2563eb"), (1, "chrf", "#16a34a")):
        x0 = 24 + col * 350
        maximum = max(1, *(r[metric] for r in rows))
        svg.append(f'<text x="{x0}" y="65" font-family="sans-serif" font-size="16">{metric.upper()}</text>')
        for i, r in enumerate(rows):
            y = 85 + i * 68
            width = 220 * r[metric] / maximum
            svg.extend((f'<text x="{x0}" y="{y + 15}" font-family="sans-serif" font-size="13">{r["model"]}</text>',
                        f'<rect x="{x0 + 75}" y="{y}" width="{width:.1f}" height="22" fill="{color}"/>',
                        f'<text x="{x0 + 82 + width:.1f}" y="{y + 16}" font-family="sans-serif" font-size="12">{r[metric]:.2f}</text>'))
    svg.append('</svg>')
    (folder / "scores.svg").write_text("\n".join(svg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["basic", "mt5", "llama", "gemma"],
                        default=["basic", "mt5", "llama", "gemma"])
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--basic-epochs", type=int, default=2)
    parser.add_argument("--mt5-epochs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if min(args.train, args.val, args.test, args.basic_epochs, args.mt5_epochs) < 1:
        parser.error("All sizes and epochs must be positive")
    seed_all()
    args.output.mkdir(parents=True, exist_ok=True)
    train, val, test, digest = prepare_data(args.train, args.val, args.test)
    settings = {"seed": SEED, "train": len(train), "val": len(val), "test": len(test),
                "basic_epochs": args.basic_epochs, "mt5_epochs": args.mt5_epochs,
                "dataset": "Helsinki-NLP/opus_books en-fr"}
    (args.output / "run_config.json").write_text(json.dumps({**settings, "test_sha256": digest}, indent=2))
    print(f"device={DEVICE}, split={len(train)}/{len(val)}/{len(test)}", flush=True)
    for key in args.models:
        print(f"running {key}", flush=True)
        if key == "basic":
            result = run_basic(train, val, test, args.basic_epochs)
        elif key == "mt5":
            result = run_mt5(train, val, test, args.mt5_epochs)
        else:
            result = run_prompt(key, test)
        metrics = save_result(args.output, key, test, digest, result, settings)
        print(f'{key}: BLEU {metrics["bleu"]:.2f}, chrF {metrics["chrf"]:.2f}', flush=True)
        summarize(args.output)


if __name__ == "__main__":
    main()
