# train_module.py

import os
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim import AdamW
from tqdm import tqdm

from model import TokenNormClassifier
from utils import mfr, evaluate


# ============================================================
# 1. Label vocab
# ============================================================

def build_label_vocab(dataset, min_freq=1, add_unk=True):
    """
    norm token 전체를 label vocabulary로 만든다.

    dataset item 형태:
    {
        "raw": [...],
        "norm": [...],
        "lang": "..."
    }
    """
    counter = Counter()

    for item in dataset:
        for tok in item["norm"]:
            counter[tok] += 1

    labels = []

    if add_unk:
        labels.append("<UNK>")

    for tok, freq in counter.most_common():
        if freq >= min_freq:
            labels.append(tok)

    label2id = {tok: i for i, tok in enumerate(labels)}
    id2label = {i: tok for tok, i in label2id.items()}

    return label2id, id2label


# ============================================================
# 2. Dataset
# ============================================================

class TokenNormDataset(Dataset):
    """
    raw token sequence를 입력으로 받고,
    각 raw token 위치의 gold norm token id를 label로 만든다.

    tokenizer는 subword 단위로 쪼갤 수 있으므로,
    각 원래 word의 첫 번째 subword에만 label을 부여하고
    나머지는 -100으로 둬서 loss에서 제외한다.
    """

    def __init__(self, data, tokenizer, label2id, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

        self.unk_id = self.label2id.get("<UNK>", 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        raw_tokens = item["raw"]
        norm_tokens = item["norm"]

        if len(raw_tokens) != len(norm_tokens):
            raise ValueError(
                f"raw/norm length mismatch at idx={idx}: "
                f"{len(raw_tokens)} vs {len(norm_tokens)}"
            )

        encoding = self.tokenizer(
            raw_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)

        labels = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)

            elif word_id != prev_word_id:
                gold_tok = norm_tokens[word_id]
                label_id = self.label2id.get(gold_tok, self.unk_id)
                labels.append(label_id)

            else:
                labels.append(-100)

            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================
# 3. Checkpoint
# ============================================================

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pt"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer=None, path="checkpoint.pt", device="cpu"):
    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"🔥 checkpoint 로드: epoch {start_epoch}부터 시작")

    return start_epoch


# ============================================================
# 4. Train / validation loss
# ============================================================

def compute_val_loss(model, val_loader, criterion, num_labels, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation loss"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(input_ids, attention_mask)

                loss = criterion(
                    logits.view(-1, num_labels),
                    labels.view(-1),
                )

            total_loss += loss.item()

    return total_loss / max(len(val_loader), 1)


DEBUG = True
DEBUG_FIRST_N_BATCHES = 3      # 처음 몇 batch 상세 출력
DEBUG_EVERY_N_BATCHES = 100    # 이후 몇 batch마다 출력

def train_model(
    train_loader,
    val_loader=None,
    model_name="xlm-roberta-base",
    num_labels=None,
    epochs=3,
    lr=2e-5,
    ckpt_path="token_norm_checkpoint.pt",
    weight_decay=0.01,
):
    """
    token-level normalization classifier 학습.

    train_loader:
        TokenNormDataset 기반 DataLoader

    val_loader:
        선택 사항. 있으면 epoch마다 validation loss 출력.

    return:
        trained model
    """
    if num_labels is None:
        raise ValueError("num_labels must be provided")

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("device:", device)

    model = TokenNormClassifier(
        model_name=model_name,
        num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # CUDA일 때 mixed precision 사용
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=ckpt_path,
        device=device,
    )

    last_completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(input_ids, attention_mask)

                    loss = criterion(
                        logits.view(-1, num_labels),
                        labels.view(-1),
                    )

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            train_loss = total_loss / max(len(train_loader), 1)
            print(f"[Epoch {epoch}] train loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss = compute_val_loss(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    num_labels=num_labels,
                    device=device,
                )
                print(f"[Epoch {epoch}] val loss:   {val_loss:.4f}")

            save_checkpoint(model, optimizer, epoch, ckpt_path)
            last_completed_epoch = epoch

    except KeyboardInterrupt:
        print("⛔ 중단 감지")

        if last_completed_epoch >= start_epoch:
            print(f"마지막으로 완료된 epoch {last_completed_epoch} 기준 checkpoint 유지")
        else:
            print("현재 epoch가 완료되지 않았으므로 새 checkpoint를 저장하지 않음")

    except torch.cuda.OutOfMemoryError:
        print("💥 CUDA OOM 발생")
        print("추천: batch_size를 줄이거나 max_len을 줄이세요.")
        print("현재 GPU 캐시를 비웁니다.")
        torch.cuda.empty_cache()
        raise

    return model


# ============================================================
# 5. MFR ratio helpers
# ============================================================

def get_mfr_ratio(raw_tok, counts):
    """
    train에서 raw_tok가 특정 norm으로 얼마나 일관되게 바뀌었는지 계산.

    예상 counts 구조:
        counts[raw_tok] = Counter({norm_tok: freq, ...})
    또는 dict 유사 구조.

    반환:
        max_count / total_count

    unknown token이면 0.0 반환.
    """
    if raw_tok not in counts:
        return 0.0

    entry = counts[raw_tok]

    if isinstance(entry, Counter) or isinstance(entry, dict):
        values = list(entry.values())
    elif isinstance(entry, list):
        # 혹시 [(norm, count), ...] 구조인 경우 대응
        if len(entry) == 0:
            return 0.0

        if isinstance(entry[0], tuple) and len(entry[0]) == 2:
            values = [x[1] for x in entry]
        else:
            return 0.0
    else:
        return 0.0

    total = sum(values)

    if total == 0:
        return 0.0

    return max(values) / total


def safe_mfr(raw_tokens, counts):
    """
    utils.mfr를 사용하되, 혹시 실패하면 raw를 그대로 반환.
    """
    try:
        return mfr(raw_tokens, counts)
    except Exception:
        return list(raw_tokens)


# ============================================================
# 6. Prediction
# ============================================================

def predict_model_only(
    model,
    tokenizer,
    data,
    id2label,
    max_len=128,
):
    """
    MFR fallback 없이 classifier 예측만 사용.
    """
    device = next(model.parameters()).device
    model.eval()

    all_pred = []
    all_conf = []

    with torch.no_grad():
        for item in tqdm(data, desc="Predicting model-only"):
            raw_tokens = item["raw"]

            encoding = tokenizer(
                raw_tokens,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )

            word_ids = encoding.word_ids(batch_index=0)

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)

            pred_ids = probs.argmax(dim=-1).squeeze(0).cpu().tolist()
            confs = probs.max(dim=-1).values.squeeze(0).cpu().tolist()

            sent_pred = []
            sent_conf = []
            seen_word_ids = set()

            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue

                if word_id in seen_word_ids:
                    continue

                seen_word_ids.add(word_id)

                if word_id >= len(raw_tokens):
                    continue

                pred_id = pred_ids[token_idx]
                pred_tok = id2label.get(pred_id, "<UNK>")

                if pred_tok == "<UNK>":
                    pred_tok = raw_tokens[word_id]

                sent_pred.append(pred_tok)
                sent_conf.append(confs[token_idx])

            # truncation으로 누락된 token은 raw로 복원
            if len(sent_pred) < len(raw_tokens):
                missing_start = len(sent_pred)
                sent_pred.extend(raw_tokens[missing_start:])
                sent_conf.extend([0.0] * (len(raw_tokens) - missing_start))

            sent_pred = sent_pred[:len(raw_tokens)]
            sent_conf = sent_conf[:len(raw_tokens)]

            all_pred.append(sent_pred)
            all_conf.append(sent_conf)

    return {
        "pred": all_pred,
        "confidence": all_conf,
    }


def predict_with_mfr_fallback(
    model,
    tokenizer,
    data,
    counts,
    id2label,
    max_len=128,
    use_mfr_fallback=True,
    conf_threshold=0.80,
    mfr_ratio_threshold=0.95,
):
    """
    최종 예측 생성.

    use_mfr_fallback=False:
        classifier 예측만 사용.

    use_mfr_fallback=True:
        token별로 아래 규칙 적용.

        1. raw token의 MFR ratio가 mfr_ratio_threshold 이상이면 MFR 사용
        2. 아니고 classifier confidence가 conf_threshold 이상이면 classifier 사용
        3. 그 외에는 MFR 사용

    반환:
        {
            "pred": 최종 예측,
            "model_pred": classifier 단독 예측,
            "model_confidence": classifier confidence,
            "mfr_pred": MFR 예측
        }
    """
    model_result = predict_model_only(
        model=model,
        tokenizer=tokenizer,
        data=data,
        id2label=id2label,
        max_len=max_len,
    )

    model_preds = model_result["pred"]
    model_confs = model_result["confidence"]

    final_preds = []
    mfr_preds = []

    for item, model_sent, conf_sent in tqdm(
        zip(data, model_preds, model_confs),
        total=len(data),
        desc="Applying MFR fallback",
    ):
        raw_tokens = item["raw"]
        mfr_sent = safe_mfr(raw_tokens, counts)

        # 혹시 mfr 길이가 이상하면 raw로 복구
        if len(mfr_sent) != len(raw_tokens):
            mfr_sent = list(raw_tokens)

        if not use_mfr_fallback:
            final_sent = model_sent
        else:
            final_sent = []

            for raw_tok, mfr_tok, model_tok, model_conf in zip(
                raw_tokens,
                mfr_sent,
                model_sent,
                conf_sent,
            ):
                ratio = get_mfr_ratio(raw_tok, counts)

                if ratio >= mfr_ratio_threshold:
                    final_tok = mfr_tok

                elif model_conf >= conf_threshold:
                    final_tok = model_tok

                else:
                    final_tok = mfr_tok

                final_sent.append(final_tok)

        # 길이 안전장치
        if len(final_sent) < len(raw_tokens):
            final_sent.extend(raw_tokens[len(final_sent):])

        final_sent = final_sent[:len(raw_tokens)]

        final_preds.append(final_sent)
        mfr_preds.append(mfr_sent)

    return {
        "pred": final_preds,
        "model_pred": model_preds,
        "model_confidence": model_confs,
        "mfr_pred": mfr_preds,
    }


# ============================================================
# 7. Evaluation wrapper
# ============================================================

def evaluate_predictions(
    data,
    pred,
    ignCaps=False,
    verbose=False,
    info=True,
):
    """
    개최측 utils.evaluate를 그대로 사용.
    """
    raw = [item["raw"] for item in data]
    gold = [item["norm"] for item in data]

    lai, accuracy, err = evaluate(
        raw,
        gold,
        pred,
        ignCaps=ignCaps,
        verbose=verbose,
        info=info,
    )

    return {
        "lai": lai,
        "accuracy": accuracy,
        "err": err,
    }