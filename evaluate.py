import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_curve, auc


def load_embeddings(emb_dir, name):
    base = Path(emb_dir) / name
    data = {}
    for split in ["train", "eval"]:
        data[f"{split}_embs"]   = np.load(base / f"{split}_embeddings.npy")
        data[f"{split}_labels"] = np.load(base / f"{split}_labels.npy")
    print(f"{name} — gallery: {data['train_embs'].shape}, queries: {data['eval_embs'].shape}")
    return data


def cosine_sim(query_embs, gallery_embs):
    
    return query_embs @ gallery_embs.T


def rank1(sim_matrix, query_labels, gallery_labels):
    top1 = sim_matrix.argmax(axis=1)
    return (gallery_labels[top1] == query_labels).mean()


def top5(sim_matrix, query_labels, gallery_labels):
    k       = min(5, sim_matrix.shape[1])
    top_idx = np.argsort(sim_matrix, axis=1)[:, -k:]
    matches = gallery_labels[top_idx] == query_labels[:, None]
    return matches.any(axis=1).mean()


def compute_roc(sim_matrix, query_labels, gallery_labels, target_far=0.01):
    scores = sim_matrix.flatten()

    # build genuine/impostor ground truth for every query-gallery pair
    q_mat = np.repeat(query_labels[:, None],   gallery_labels.shape[0], axis=1)
    g_mat = np.repeat(gallery_labels[None, :], query_labels.shape[0],   axis=0)
    is_same = (q_mat == g_mat).flatten().astype(int)

    n_genuine  = is_same.sum()
    n_impostor = len(is_same) - n_genuine
    print(f"  genuine pairs: {n_genuine:,}  impostor pairs: {n_impostor:,}")

    if n_genuine == 0 or n_impostor == 0:
        print("  Warning: need both genuine and impostor pairs for ROC")
        return None

    fpr, tpr, _ = roc_curve(is_same, scores)
    roc_auc     = auc(fpr, tpr)

    # TAR at the given FAR operating point
    idx = np.searchsorted(fpr, target_far, side="right")
    idx = min(idx, len(tpr) - 1)

    return {
        "fpr"    : fpr,
        "tpr"    : tpr,
        "auc"    : roc_auc,
        "tar"    : float(tpr[idx]),
        "far"    : float(fpr[idx]),
    }


def save_roc_plot(roc, name, target_far, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(roc["fpr"], roc["tpr"], color="steelblue", lw=2,
            label=f"AUC = {roc['auc']:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.scatter([target_far], [roc["tar"]], color="red", zorder=5,
               label=f"TAR={roc['tar']:.3f} @ FAR={target_far}")
    ax.set_xlabel("False Acceptance Rate")
    ax.set_ylabel("True Acceptance Rate")
    ax.set_title(f"ROC — {name.upper()}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC saved: {save_path}")


def evaluate(name, data, out_dir, target_far=0.01):
    print(f"\n{name.upper()} ")

    gallery_embs   = data["train_embs"]
    gallery_labels = data["train_labels"]
    query_embs     = data["eval_embs"]
    query_labels   = data["eval_labels"]

    sim = cosine_sim(query_embs, gallery_embs)

    r1  = rank1(sim, query_labels, gallery_labels)
    r5  = top5(sim, query_labels, gallery_labels)
    roc = compute_roc(sim, query_labels, gallery_labels, target_far)

    print(f"  Rank-1   : {r1*100:.2f}%")
    print(f"  Top-5    : {r5*100:.2f}%")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tar, far, roc_auc = None, None, None
    if roc:
        tar     = roc["tar"]
        far     = roc["far"]
        roc_auc = roc["auc"]
        print(f"  TAR@{target_far} : {tar:.4f}")
        print(f"  FAR      : {far:.4f}")
        print(f"  AUC      : {roc_auc:.4f}")
        save_roc_plot(roc, name, target_far, f"{out_dir}/{name}_roc.png")

    return {
        "dataset"    : name,
        "identities" : int(len(np.unique(gallery_labels))),
        "reference"    : int(len(gallery_embs)),
        "test"    : int(len(query_embs)),
        "rank1"      : round(float(r1), 4),
        "top5"       : round(float(r5), 4),
        "tar"        : round(tar, 4)     if tar     is not None else None,
        "far"        : round(far, 4)     if far     is not None else None,
        "auc"        : round(roc_auc, 4) if roc_auc is not None else None,
    }


def print_summary(results):
    
    print(f"{'Dataset':<12} {'IDs':>6} {'Reference':>8} {'Test':>8} "
          f"{'Rank-1':>8} {'Top-5':>8} {'TAR':>8} {'AUC':>8}")
    
    for r in results:
        def pct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
        print(f"{r['dataset']:<12} {r['identities']:>6} {r['reference']:>8} "
              f"{r['test']:>8} {pct(r['rank1']):>8} {pct(r['top5']):>8} "
              f"{pct(r['tar']):>8} {pct(r['auc']):>8}")
    


if __name__ == "__main__":

    BASE = "C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition"

    EMBEDDINGS_DIR = f"{BASE}/data/embeddings_facerec"
    OUTPUT_DIR     = f"{BASE}/data/results_facerec"

    # EMBEDDINGS_DIR = "data/embeddings"
    # OUTPUT_DIR     = "results"
    DATASETS       = ["att", "imfdb", "imdbwiki"]
    TARGET_FAR     = 0.01

    all_results = []

    for name in DATASETS:
        emb_path = Path(EMBEDDINGS_DIR) / name
        if not emb_path.exists():
            print(f"Skipping {name} — run extract_embeddings.py first")
            continue

        data   = load_embeddings(EMBEDDINGS_DIR, name)
        result = evaluate(name, data, OUTPUT_DIR, TARGET_FAR)
        all_results.append(result)

    if all_results:
        print_summary(all_results)

        out_file = Path(OUTPUT_DIR) / "results.json"
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved: {out_file}")

    print("\nDone.")