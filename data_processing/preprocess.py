import os
import re
import json
import random


SUPPORTED_IMG = {".jpg", ".JPG", ".png", ".PNG"}


def read_split_data(video_root: str, audio_root: str, val_rate: float = 0.2):
    """
    Walk the extract.py output tree:
      {video_root}/{category}/{race}/{gender}/{identity_stem}/frame_*.jpg
      {audio_root}/{category}/{race}/{gender}/{identity_stem}/chunk_*.png

    Classes = top-level category directories (e.g. RealVideo-RealAudio).
    """
    random.seed(0)

    assert os.path.exists(video_root), f"video root not found: {video_root}"
    assert os.path.exists(audio_root), f"audio root not found: {audio_root}"

    categories = sorted(
        d for d in os.listdir(video_root)
        if os.path.isdir(os.path.join(video_root, d))
    )
    class_indices = {cat: idx for idx, cat in enumerate(categories)}

    with open("class_indices.json", "w") as f:
        json.dump({str(v): k for k, v in class_indices.items()}, f, indent=4)

    train_images_path, train_images_label = [], []
    train_audio_path,  train_audio_label  = [], []
    val_images_path,   val_images_label   = [], []
    val_audio_path,    val_audio_label    = [], []

    for category in categories:
        label = class_indices[category]
        cat_video = os.path.join(video_root, category)
        cat_audio = os.path.join(audio_root, category)

        # Walk every leaf directory under this category
        for dirpath, dirnames, filenames in os.walk(cat_video):
            frames = sorted(
                os.path.join(dirpath, f) for f in filenames
                if os.path.splitext(f)[1] in SUPPORTED_IMG
            )
            if not frames:
                continue

            # Mirror path into audio tree
            rel = os.path.relpath(dirpath, cat_video)
            audio_dir = os.path.join(cat_audio, rel)
            if not os.path.isdir(audio_dir):
                continue

            chunks = sorted(
                (os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
                 if "chunk" in f and os.path.splitext(f)[1] in SUPPORTED_IMG),
                key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[-1])
            )
            if not chunks:
                continue

            # Align lengths
            n = min(len(frames), len(chunks))
            frames = frames[:n]
            chunks = chunks[:n]

            # Train / val split (paired)
            val_idx = set(random.sample(range(n), k=max(1, int(n * val_rate))))
            for i, (f, a) in enumerate(zip(frames, chunks)):
                if i in val_idx:
                    val_images_path.append(f);  val_images_label.append(label)
                    val_audio_path.append(a);   val_audio_label.append(label)
                else:
                    train_images_path.append(f); train_images_label.append(label)
                    train_audio_path.append(a);  train_audio_label.append(label)

    total = len(train_images_path) + len(val_images_path)
    print(f"{total} paired samples found across {len(categories)} classes.")
    print(f"  training: {len(train_images_path)}  validation: {len(val_images_path)}")

    return (train_images_path, train_images_label,
            train_audio_path,  train_audio_label,
            val_images_path,   val_images_label,
            val_audio_path,    val_audio_label)
