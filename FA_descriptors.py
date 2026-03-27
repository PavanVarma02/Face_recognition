import os
import json
import numpy as np
import cv2
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis


def load_model():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    return app


def get_embedding(app, img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Could not read: {img_path}")
            return None

        # grayscale to BGR (AT&T pgm files)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        faces = app.get(img)

        if len(faces) == 0:
            # no face detected resize 
            img = cv2.resize(img, (160, 160))
            faces = app.get(img)

        if len(faces) == 0:
            print(f"  No face found: {img_path}")
            return None

        
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return face.embedding / np.linalg.norm(face.embedding)   # L2 normalize

    except Exception as e:
        print(f"  Error on {img_path}: {e}")
        return None


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


def process_dataset(name, splits_root, output_root, app):
    splits_dir = Path(splits_root) / name
    out_dir    = Path(output_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / "label_map.json") as f:
        label_map = json.load(f)

    print(f"\n {name.upper()} ({len(label_map)} identities) ")

    for split in ["train", "eval"]:
        paths, labels = collect_paths(splits_dir / split, label_map)
        print(f"  {split}: {len(paths)} images")

        embs         = []
        valid_labels = []

        for i, (path, label) in enumerate(zip(paths, labels)):
            emb = get_embedding(app, path)
            if emb is not None:
                embs.append(emb)
                valid_labels.append(label)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(paths)} done...")

        embs_arr   = np.array(embs)
        labels_arr = np.array(valid_labels)

        np.save(out_dir / f"{split}_embeddings.npy", embs_arr)
        np.save(out_dir / f"{split}_labels.npy",     labels_arr)
        print(f"  Saved {split} embeddings: {embs_arr.shape}")


if __name__ == "__main__":

    BASE = "C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition"

    SPLITS_ROOT = f"{BASE}/data/splits"
    OUTPUT_ROOT = f"{BASE}/data/embeddings"

    # SPLITS_ROOT = "data/splits"
    # OUTPUT_ROOT = "data/embeddings"
    # DATASETS    = ["att", "imfdb", "imdbwiki"]
    DATASETS    = ["att",  "imdbwiki"]

    app = load_model()

    for name in DATASETS:
        if not (Path(SPLITS_ROOT) / name).exists():
            print(f"Skipping {name} — run dataset_prep.py first")
            continue
        process_dataset(name, SPLITS_ROOT, OUTPUT_ROOT, app)

    print("\nDone.")
