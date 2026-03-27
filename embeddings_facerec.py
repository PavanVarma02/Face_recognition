import os
import json
import numpy as np
import cv2
from pathlib import Path

try:
    import face_recognition
except ImportError:
    raise ImportError("Run: pip install face_recognition")


def get_embedding(img_path):
    # face_recognition expects RGB, cv2 loads BGR
    img = cv2.imread(img_path)
    if img is None:
        return None

    # images from gray to RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    embeddings = face_recognition.face_encodings(img)

    if len(embeddings) == 0:
        # retry with a larger image — sometimes small faces are missed
        img = cv2.resize(img, (300, 300))
        embeddings = face_recognition.face_encodings(img)

    if len(embeddings) == 0:
        return None

    
    emb = np.array(embeddings[0])
    return emb / np.linalg.norm(emb)   # L2 normalize


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
        skipped      = 0

        for i, (path, label) in enumerate(zip(paths, labels)):
            emb = get_embedding(path)
            if emb is not None:
                embs.append(emb)
                valid_labels.append(label)
            else:
                skipped += 1

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(paths)} done  (skipped so far: {skipped})")

        embs_arr   = np.array(embs)
        labels_arr = np.array(valid_labels)

        np.save(out_dir / f"{split}_embeddings.npy", embs_arr)
        np.save(out_dir / f"{split}_labels.npy",     labels_arr)
        print(f"  Saved {split} embeddings: {embs_arr.shape}  (skipped {skipped})")


if __name__ == "__main__":

    BASE = "C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition"

    SPLITS_ROOT = f"{BASE}/data/splits"
    OUTPUT_ROOT = f"{BASE}/data/embeddings_facerec"   
    DATASETS    = ["att", "imfdb", "imdbwiki"]

    for name in DATASETS:
        if not (Path(SPLITS_ROOT) / name).exists():
            print(f"Skipping {name} — run dataset_prep.py first")
            continue
        process_dataset(name, SPLITS_ROOT, OUTPUT_ROOT)

    print("\nDone. Next: python evaluate.py")