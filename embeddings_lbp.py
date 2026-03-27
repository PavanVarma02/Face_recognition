import os
import json
import numpy as np
import cv2
from pathlib import Path


def get_lbp_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    # convert to grayscale — LBP works on grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize to fixed size so all feature vectors are same length
    img = cv2.resize(img, (128, 128))

    # compute LBP features
    lbp = compute_lbp(img)

    # normalize
    norm = np.linalg.norm(lbp)
    if norm == 0:
        return None
    return lbp / norm


def compute_lbp(gray_img, num_points=8, radius=1, grid_x=8, grid_y=8):
    """
    Divides image into grid_x * grid_y cells.
    Computes LBP histogram for each cell.
    Concatenates all histograms into one feature vector.
    """
    h, w   = gray_img.shape
    cell_h = h // grid_y
    cell_w = w // grid_x

    histograms = []

    for i in range(grid_y):
        for j in range(grid_x):
            # crop one cell
            cell = gray_img[i*cell_h:(i+1)*cell_h,
                            j*cell_w:(j+1)*cell_w]

            # compute LBP for this cell manually
            lbp_cell = np.zeros_like(cell, dtype=np.uint8)
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dx == 0 and dy == 0:
                        continue
                    # shift image and compare with center
                    shifted = np.roll(np.roll(cell, dy, axis=0), dx, axis=1)
                    lbp_cell += (shifted >= cell).astype(np.uint8)

            # histogram of LBP values for this cell
            hist, _ = np.histogram(lbp_cell.ravel(), bins=num_points+1,
                                   range=(0, num_points+1))
            histograms.append(hist.astype(np.float32))

    return np.concatenate(histograms)


def collect_paths(split_dir, label_map):
    split_dir = Path(split_dir)
    paths, labels = [], []

    for person in sorted(os.listdir(split_dir)):
        folder = split_dir / person
        if not folder.is_dir() or person not in label_map:
            continue
        for img in sorted(os.listdir(folder)):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
                paths.append(str(folder / img))
                labels.append(label_map[person])

    return paths, labels


def process_dataset(name, splits_root, output_root):
    splits_dir = Path(splits_root) / name
    out_dir    = Path(output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "label_map.json") as f:
        label_map = json.load(f)

    print(f"\n--- {name.upper()} ({len(label_map)} identities) ---")

    for split in ["train", "eval"]:
        paths, labels = collect_paths(splits_dir / split, label_map)
        print(f"  {split}: {len(paths)} images")

        embs         = []
        valid_labels = []

        for i, (path, label) in enumerate(zip(paths, labels)):
            emb = get_lbp_embedding(path)
            if emb is not None:
                embs.append(emb)
                valid_labels.append(label)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(paths)} done...")

        embs_arr   = np.array(embs)
        labels_arr = np.array(valid_labels)

        np.save(out_dir / f"{split}_embeddings.npy", embs_arr)
        np.save(out_dir / f"{split}_labels.npy",     labels_arr)
        print(f"  Saved {split} embeddings: {embs_arr.shape}")


if __name__ == "__main__":

    BASE = "C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition"

    SPLITS_ROOT = f"{BASE}/data/splits"
    OUTPUT_ROOT = f"{BASE}/data/embeddings_lbp"   # separate folder from arcface
    DATASETS    = ["att", "imfdb", "imdbwiki"]

    for name in DATASETS:
        if not (Path(SPLITS_ROOT) / name).exists():
            print(f"Skipping {name} — run dataset_prep.py first")
            continue
        process_dataset(name, SPLITS_ROOT, OUTPUT_ROOT)

    print("\nDone. Next: python evaluate.py")