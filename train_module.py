import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm  # 🔥 추가

from model import SimpleEncoder
from eer import compute_eer
import os

def save_checkpoint(model, optimizer, epoch, path="checkpoint.pt"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"🔥 checkpoint 로드: epoch {start_epoch}부터 시작")
        return start_epoch
    else:
        return 0


def train_and_eval(train_loader, val_loader, model_name="distilbert-base-uncased", epochs=3, ckpt_path="checkpoint.pt"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleEncoder(model_name).to(device)
    optimizer = Adam(model.parameters(), lr=2e-5)


    start_epoch = load_checkpoint(model, optimizer, ckpt_path)
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0

            # 🔥 tqdm 적용
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

                ids1 = batch["input_ids1"].to(device)
                mask1 = batch["attention_mask1"].to(device)

                ids2 = batch["input_ids2"].to(device)
                mask2 = batch["attention_mask2"].to(device)

                labels = batch["labels"].float().to(device)

                emb1 = model(ids1, mask1)
                emb2 = model(ids2, mask2)

                # normalize
                emb1 = F.normalize(emb1, dim=1)
                emb2 = F.normalize(emb2, dim=1)

                # cosine similarity
                sim = F.cosine_similarity(emb1, emb2)

                # loss
                loss = ((sim - labels) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[Epoch {epoch}] loss: {total_loss / len(train_loader):.4f}")
            save_checkpoint(model, optimizer, epoch, ckpt_path)

    except KeyboardInterrupt:
        print("⛔ 중단 감지 → checkpoint 저장")
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        return model, None
    
    # evaluation
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):  # 🔥 eval에도 추가

            ids1 = batch["input_ids1"].to(device)
            mask1 = batch["attention_mask1"].to(device)

            ids2 = batch["input_ids2"].to(device)
            mask2 = batch["attention_mask2"].to(device)

            labels = batch["labels"].cpu().numpy()

            emb1 = model(ids1, mask1)
            emb2 = model(ids2, mask2)

            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)

            sim = F.cosine_similarity(emb1, emb2).cpu().numpy()

            all_scores.extend(sim)
            all_labels.extend(labels)

    eer, th = compute_eer(all_labels, all_scores)

    print(f"\n🔥 REAL EER: {eer:.4f} (threshold={th:.4f})")

    return model, eer