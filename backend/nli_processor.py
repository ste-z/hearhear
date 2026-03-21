from functools import lru_cache


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_LENGTH = 512
CLAIMNESS_HYPOTHESIS = "This sentence is the author's main claim."


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for stance reranking. Install the 'torch' package in the active environment."
        ) from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "transformers is required for stance reranking. Install the 'transformers' package in the active environment."
        ) from exc
    return AutoModelForSequenceClassification, AutoTokenizer


def _default_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_nli_bundle(model_name=MODEL_NAME):
    torch = _import_torch()
    AutoModelForSequenceClassification, AutoTokenizer = _import_transformers()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = _default_device(torch)
    model.to(device)
    model.eval()

    id2label = {
        int(idx): str(label).lower()
        for idx, label in model.config.id2label.items()
    }

    def label_index(fragment):
        for idx, label in id2label.items():
            if fragment in label:
                return idx
        raise KeyError(f"Could not find a label containing {fragment!r} in {id2label}.")

    label_map = {
        "contradiction": label_index("contrad"),
        "neutral": label_index("neutral"),
        "entailment": label_index("entail"),
    }

    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "id2label": id2label,
        "label_map": label_map,
        "model_name": model_name,
    }


def score_nli_pairs(
    premises,
    hypothesis,
    model_name=MODEL_NAME,
    batch_size=DEFAULT_BATCH_SIZE,
    max_length=DEFAULT_MAX_LENGTH,
):
    cleaned_premises = [str(premise or "").strip() for premise in premises]
    cleaned_hypothesis = str(hypothesis or "").strip()
    if not cleaned_premises or not cleaned_hypothesis:
        return []

    bundle = load_nli_bundle(model_name=model_name)
    torch = bundle["torch"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    label_map = bundle["label_map"]

    rows = []
    for start_idx in range(0, len(cleaned_premises), int(batch_size)):
        batch_premises = cleaned_premises[start_idx:start_idx + int(batch_size)]
        batch_hypotheses = [cleaned_hypothesis] * len(batch_premises)
        encoded = tokenizer(
            batch_premises,
            batch_hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()

        for row in probs:
            entailment_prob = float(row[label_map["entailment"]])
            neutral_prob = float(row[label_map["neutral"]])
            contradiction_prob = float(row[label_map["contradiction"]])
            rows.append(
                {
                    "entailment_prob": entailment_prob,
                    "neutral_prob": neutral_prob,
                    "contradiction_prob": contradiction_prob,
                    "stance_score": stance_score_from_probs(
                        entailment_prob=entailment_prob,
                        contradiction_prob=contradiction_prob,
                    ),
                }
            )
    return rows


def stance_score_from_probs(entailment_prob, contradiction_prob):
    return float(entailment_prob) - float(contradiction_prob)


def normalize_stance_score(stance_score):
    bounded = max(-1.0, min(1.0, float(stance_score)))
    return (bounded + 1.0) / 2.0


def stance_label_from_probs(entailment_prob, neutral_prob, contradiction_prob):
    scores = {
        "entailment": float(entailment_prob),
        "neutral": float(neutral_prob),
        "contradiction": float(contradiction_prob),
    }
    return max(scores, key=scores.get)


def score_claim_sentences(
    sentence_rows,
    hypothesis=CLAIMNESS_HYPOTHESIS,
    top_n=5,
    model_name=MODEL_NAME,
    batch_size=DEFAULT_BATCH_SIZE,
    max_length=DEFAULT_MAX_LENGTH,
):
    rows = list(sentence_rows)
    if not rows:
        return []

    scores = score_nli_pairs(
        premises=[row.get("sentence", "") for row in rows],
        hypothesis=hypothesis,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    ranked = []
    for row, score in zip(rows, scores):
        claim_score = stance_score_from_probs(
            entailment_prob=score["entailment_prob"],
            contradiction_prob=score["contradiction_prob"],
        )
        ranked.append(
            {
                "sentence_id": row.get("sentence_id"),
                "sentence": row.get("sentence"),
                "entailment_prob": score["entailment_prob"],
                "neutral_prob": score["neutral_prob"],
                "contradiction_prob": score["contradiction_prob"],
                "claim_score": claim_score,
                "claim_score_normalized": normalize_stance_score(claim_score),
                "claim_label": stance_label_from_probs(
                    entailment_prob=score["entailment_prob"],
                    neutral_prob=score["neutral_prob"],
                    contradiction_prob=score["contradiction_prob"],
                ),
            }
        )

    ranked.sort(
        key=lambda row: (
            float(row["claim_score"]),
            float(row["entailment_prob"]),
        ),
        reverse=True,
    )
    return ranked[:max(1, int(top_n))]
