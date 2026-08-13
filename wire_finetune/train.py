import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

torch.manual_seed(42)
random.seed(42)

BASE = Path(r"C:\Users\Admin\unoq\wire_finetune")
DATASET = BASE / "dataset"
OUT = BASE / "runs"
OUT.mkdir(exist_ok=True)

LABELS = ["lanh", "dut"]  # index 0 = lanh (intact), 1 = dut (broken)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


def collect_samples():
    samples = []  # (path, label_idx, subgroup)
    for label_idx, label in enumerate(LABELS):
        for subdir in sorted((DATASET / label).iterdir()):
            if not subdir.is_dir():
                continue
            subgroup = subdir.name  # e.g. "vang_lanh_vang", "den_dut_den" etc
            for f in sorted(subdir.glob("*.jpg")):
                samples.append((f, label_idx, subgroup))
    return samples


def stratified_split(samples, val_frac=0.15):
    from collections import defaultdict
    by_group = defaultdict(list)
    for s in samples:
        by_group[s[2]].append(s)
    train, val = [], []
    for group, items in by_group.items():
        items = items[:]
        random.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


class WireDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, group = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.3, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.05),
    transforms.RandomGrayscale(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features.parameters():
        p.requires_grad = False
    m.classifier[1] = nn.Linear(m.last_channel, 2)
    return m.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return loss_sum / total, correct / total, all_preds, all_labels


def confusion(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    return tp, tn, fp, fn


def main():
    samples = collect_samples()
    print(f"Tong so anh: {len(samples)}")
    for label_idx, label in enumerate(LABELS):
        n = sum(1 for s in samples if s[1] == label_idx)
        print(f"  {label}: {n}")

    train_samples, val_samples = stratified_split(samples, val_frac=0.15)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = WireDataset(train_samples, train_transform)
    val_ds = WireDataset(val_samples, val_transform)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    N_EPOCHS = 40
    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader)
        scheduler.step(val_acc)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch+1}/{N_EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val acc: {best_val_acc:.4f}")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / "mobilenet_v2_wire_best.pt")

    # Final eval with confusion matrix
    val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader)
    tp, tn, fp, fn = confusion(val_preds, val_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n=== Confusion matrix (toan bo val set) ===")
    print(f"TP(dut->dut)={tp} TN(lanh->lanh)={tn} FP(lanh->dut)={fp} FN(dut->lanh)={fn}")
    print(f"Val acc={val_acc:.4f} Precision={precision:.4f} Recall={recall:.4f}")

    # Bias check: black-wire-only subset (background held constant across classes)
    black_val = [s for s in val_samples if s[2].startswith("den_")]
    if black_val:
        black_ds = WireDataset(black_val, val_transform)
        black_loader = DataLoader(black_ds, batch_size=16, shuffle=False)
        b_loss, b_acc, b_preds, b_labels = evaluate(model, black_loader)
        b_tp, b_tn, b_fp, b_fn = confusion(b_preds, b_labels)
        print(f"\n=== BIAS CHECK: chi tap day DEN (nen trang co dinh ca 2 lop) ===")
        print(f"So mau: {len(black_val)}")
        print(f"TP={b_tp} TN={b_tn} FP={b_fp} FN={b_fn}")
        print(f"Accuracy (den only) = {b_acc:.4f}")

    # Yellow-only subset (background CHANGES with class - confounded, expect possibly inflated or different acc)
    yellow_val = [s for s in val_samples if s[2].startswith("vang_")]
    if yellow_val:
        yellow_ds = WireDataset(yellow_val, val_transform)
        yellow_loader = DataLoader(yellow_ds, batch_size=16, shuffle=False)
        y_loss, y_acc, y_preds, y_labels = evaluate(model, yellow_loader)
        print(f"\n=== Tap day VANG (nen thay doi theo lop - co the bi anh huong boi bias) ===")
        print(f"So mau: {len(yellow_val)}, Accuracy = {y_acc:.4f}")

    report = {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "precision": precision,
        "recall": recall,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "black_only_acc": b_acc if black_val else None,
        "yellow_only_acc": y_acc if yellow_val else None,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "history": history,
    }
    with open(OUT / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDa luu model -> {OUT / 'mobilenet_v2_wire_best.pt'}")
    print(f"Da luu report -> {OUT / 'training_report.json'}")


if __name__ == "__main__":
    main()
