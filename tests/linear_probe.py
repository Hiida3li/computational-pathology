"""
Linear probing benchmark for pathology foundation models.

Train on NCT-CRC-HE-100K, test on CRC-VAL-HE-7K (different patients, different cohort).
Frozen encoder, embeddings, logistic regression, balanced accuracy.

Data (download and unzip these two, NONORM versions):
    https://zenodo.org/records/1214456
      NCT-CRC-HE-100K-NONORM.zip   ./data/NCT-CRC-HE-100K-NONORM/
      CRC-VAL-HE-7K.zip          ./data/CRC-VAL-HE-7K/

Both unzip into class-named subfolders (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM),
which is exactly the layout torch-vision's ImageFolder expects.

Run:
    python linear_probe.py --model resnet50
    python linear_probe.py --model phikon
    python linear_probe.py --model uni          # needs HF access approval
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_encoder(name):
    if name == "resnet50":
        import timm
        m = timm.create_model("resnet50", pretrained=True, num_classes=0)
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return m, tf, 2048

    if name == "phikon":
        from transformers import AutoModel
        hf = AutoModel.from_pretrained("owkin/phikon")

        class Wrap(nn.Module):
            def __init__(self, hf):
                super().__init__()
                self.hf = hf

            def forward(self, x):
                # Phikon uses the CLS token as the image representation.
                return self.hf(pixel_values=x).last_hidden_state[:, 0, :]

        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return Wrap(hf), tf, 768

    if name == "uni":
        import timm
        m = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True, init_values=1e-5, dynamic_img_size=True,
        )
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        return m, tf, 1024

    if name == "h-optimus-0":
        import timm
        m = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True, init_values=1e-5, dynamic_img_size=False,
        )
        # H-optimus-0 uses its own normalisation statistics, not ImageNet's.
        tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ])
        return m, tf, 1536

    raise ValueError(f"unknown model: {name}")



def subsample(dataset, per_class, seed=0):
    """Take at most `per_class` images from each class. Keeps runtime sane."""
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    keep = []
    for c in np.unique(targets):
        idx = np.flatnonzero(targets == c)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.extend(idx.tolist())
    return Subset(dataset, sorted(keep))


@torch.no_grad()
def extract(model, loader):
    model.eval().to(DEVICE)
    feats, labels = [], []
    for x, y in tqdm(loader, desc="extracting"):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            f = model(x.to(DEVICE, non_blocking=True))
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50",
                    choices=["resnet50", "phikon", "uni", "h-optimus-0"])
    ap.add_argument("--train-dir", default="./data/NCT-CRC-HE-100K-NONORM")
    ap.add_argument("--test-dir", default="./data/CRC-VAL-HE-7K")
    ap.add_argument("--per-class", type=int, default=500,
                    help="training tiles per class; raise once it runs end to end")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--umap", action="store_true", help="also save a UMAP plot")
    args = ap.parse_args()

    print(f"model={args.model}  device={DEVICE}")
    model, tf, dim = build_encoder(args.model)

    train_ds = subsample(datasets.ImageFolder(args.train_dir, tf), args.per_class)
    test_ds = datasets.ImageFolder(args.test_dir, tf)
    classes = test_ds.classes
    print(f"train={len(train_ds)}  test={len(test_ds)}  dim={dim}  classes={classes}")

    mk = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)
    Xtr, ytr = extract(model, mk(train_ds))
    Xte, yte = extract(model, mk(test_ds))

    os.makedirs("features", exist_ok=True)
    np.savez_compressed(f"features/{args.model}.npz",
                        Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

    # L2-normalise before the probe: standard for frozen-feature evaluation,
    # and it stops embedding-norm differences between models from confounding

    Xtr = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    Xte = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)

    clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    bal = balanced_accuracy_score(yte, pred)
    print(f"\n=== {args.model} ===")
    print(f"balanced accuracy (external cohort): {bal:.4f}\n")
    print(classification_report(yte, pred, target_names=classes, digits=3))
    print("confusion matrix (rows = true):")
    print(confusion_matrix(yte, pred))

    if args.umap:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Xte)
        plt.figure(figsize=(8, 7))
        for i, c in enumerate(classes):
            m = yte == i
            plt.scatter(emb[m, 0], emb[m, 1], s=2, label=c, alpha=0.6)
        plt.legend(markerscale=5, fontsize=8)
        plt.title(f"UMAP of {args.model} embeddings (CRC-VAL-HE-7K)")
        plt.tight_layout()
        plt.savefig(f"umap_{args.model}.png", dpi=150)
        print(f"\nsaved umap_{args.model}.png")


if __name__ == "__main__":
    main()
