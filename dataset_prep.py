import os
import json
import shutil
import random
from pathlib import Path

random.seed(42)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.pgm')


def get_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]


def make_split(dataset_path, output_path, train_ratio=0.4, min_imgs=2, max_imgs=20):
    dataset_path = Path(dataset_path)
    output_path  = Path(output_path)

    (output_path / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "eval").mkdir(parents=True, exist_ok=True)

    label_map = {}
    summary   = {}
    label_id  = 0
    skipped   = 0

    for person in sorted(os.listdir(dataset_path)):
        person_dir = dataset_path / person
        if not person_dir.is_dir():
            continue

        imgs = get_images(person_dir)
        if len(imgs) < min_imgs:
            skipped += 1
            continue

        if len(imgs) > max_imgs:
            random.shuffle(imgs)
            imgs = imgs[:max_imgs]

        n_train = max(1, int(len(imgs) * train_ratio))
        n_eval  = len(imgs) - n_train

        # to make eval images greater than train 
        if n_eval <= n_train:
            n_train -= 1
            n_eval  = len(imgs) - n_train

        random.shuffle(imgs)
        train_imgs = imgs[:n_train]
        eval_imgs  = imgs[n_train:]

        for img in train_imgs:
            dst = output_path / "train" / person
            dst.mkdir(exist_ok=True)
            shutil.copy2(person_dir / img, dst / img)

        for img in eval_imgs:
            dst = output_path / "eval" / person
            dst.mkdir(exist_ok=True)
            shutil.copy2(person_dir / img, dst / img)

        label_map[person] = label_id
        summary[person]   = {"train": n_train, "eval": n_eval, "total": len(imgs)}
        label_id += 1

    with open(output_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    with open(output_path / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    total_train = sum(v["train"] for v in summary.values())
    total_eval  = sum(v["eval"]  for v in summary.values())

    print(f"\n{dataset_path.name}")
    print(f"  Identities : {label_id}  (skipped {skipped} with too few images)")
    print(f"  Train imgs : {total_train}")
    print(f"  Eval  imgs : {total_eval}")
    print(f"  Eval > Train check : {total_eval > total_train}")


def prepare_imdbwiki(dataset_path, output_path, max_identities=500, train_ratio=0.4):
    dataset_path = Path(dataset_path)
    
    
    person_images = {}
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = dataset_path / folder
        if not folder_path.is_dir():
            continue
        for img in os.listdir(folder_path):
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            person_id = img.split('_')[0]    
            if person_id not in person_images:
                person_images[person_id] = []
            person_images[person_id].append(str(folder_path / img))

    # keep only persons with at least 2 images
    valid = {p: imgs for p, imgs in person_images.items() if len(imgs) >= 2}
    print(f"IMDB-WIKI: {len(valid)} valid identities found")
    
    selected_ids = random.sample(list(valid.keys()), min(max_identities, len(valid)))

    
    temp_dir = Path(output_path) / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for person_id in selected_ids:
        person_dir = temp_dir / person_id
        person_dir.mkdir(exist_ok=True)
        for img_path in valid[person_id]:
            shutil.copy2(img_path, person_dir / Path(img_path).name)

    print(f"IMDB-WIKI: copied {len(selected_ids)} identities to temp folder")
    make_split(str(temp_dir), output_path, train_ratio)

if __name__ == "__main__":

   

    datasets = {
            "att"      : ("C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition/Datasets/att_faces/Training",  "C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition/data/splits/att",      0.4),
            "imfdb"    : ("C:/Users/ARDB/anaconda3/envs/venv/Assignment/IMFDB FR dataset",                               "C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition/data/splits/imfdb",    0.4),
            "imdbwiki" : ("C:/Users/ARDB/anaconda3/envs/venv/Assignment/imdb_0/imdb", "C:/Users/ARDB/anaconda3/envs/venv/Assignment/Face-Recognition/data/splits/imdbwiki", 0.4),
        }

    for name, (inp, out, ratio) in datasets.items():
        
        inp_path = Path(inp)
        if not inp_path.exists():
            print(f"  Skipping {name} — folder not found at {inp}")
            continue
        if name == "imdbwiki":
            prepare_imdbwiki(inp, out, max_identities=500, train_ratio=ratio)
        else:
            make_split(inp, out, train_ratio=ratio)

 