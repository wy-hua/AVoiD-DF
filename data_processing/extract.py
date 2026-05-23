import os
import argparse
import subprocess
import cv2
import numpy as np
import librosa
from PIL import Image


def extract_frames(mp4_path, out_dir, fps=1):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(video_fps / fps))
    i, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved:04d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()


def extract_audio_chunks(mp4_path, out_dir, chunk_sec=1.0, sr=22050):
    os.makedirs(out_dir, exist_ok=True)
    wav_path = mp4_path.replace(".mp4", "_tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-ar", str(sr), "-ac", "1", wav_path],
        capture_output=True,
    )
    if not os.path.exists(wav_path):
        print(f"  [warn] audio extraction failed: {mp4_path}")
        return
    y, sr = librosa.load(wav_path, sr=sr)
    os.remove(wav_path)
    chunk_len = int(chunk_sec * sr)
    for idx, start in enumerate(range(0, len(y) - chunk_len, chunk_len)):
        chunk = y[start : start + chunk_len]
        S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (
            (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8) * 255
        ).astype(np.uint8)
        img = Image.fromarray(np.stack([S_norm] * 3, axis=-1))
        img.save(os.path.join(out_dir, f"chunk_{idx:04d}.png"))


def process_category(category_path, video_out, audio_out, category_name, fps, chunk_sec):
    for race in sorted(os.listdir(category_path)):
        race_path = os.path.join(category_path, race)
        if not os.path.isdir(race_path):
            continue
        for gender in sorted(os.listdir(race_path)):
            gender_path = os.path.join(race_path, gender)
            if not os.path.isdir(gender_path):
                continue
            for identity in sorted(os.listdir(gender_path)):
                id_path = os.path.join(gender_path, identity)
                if not os.path.isdir(id_path):
                    continue
                for f in os.listdir(id_path):
                    if not f.endswith(".mp4"):
                        continue
                    mp4 = os.path.join(id_path, f)
                    stem = f.replace(".mp4", "")
                    rel = os.path.join(race, gender, identity + "_" + stem)
                    v_out = os.path.join(video_out, category_name, race, gender, identity + "_" + stem)
                    a_out = os.path.join(audio_out, category_name, race, gender, identity + "_" + stem)
                    print(f"Processing {category_name}/{race}/{gender}/{identity}/{f}")
                    extract_frames(mp4, v_out, fps=fps)
                    extract_audio_chunks(mp4, a_out, chunk_sec=chunk_sec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to FakeAVCeleb_v1.2/")
    parser.add_argument("--video_out", type=str, required=True,
                        help="Output directory for extracted frames")
    parser.add_argument("--audio_out", type=str, required=True,
                        help="Output directory for mel-spectrogram chunks")
    parser.add_argument("--categories", nargs="+",
                        default=["RealVideo-RealAudio", "FakeVideo-FakeAudio",
                                 "RealVideo-FakeAudio", "FakeVideo-RealAudio"],
                        help="Which FakeAVCeleb categories to process")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second to extract")
    parser.add_argument("--chunk_sec", type=float, default=1.0,
                        help="Audio chunk duration in seconds")
    args = parser.parse_args()

    for category in args.categories:
        cat_path = os.path.join(args.data_root, category)
        if not os.path.isdir(cat_path):
            print(f"[skip] {cat_path} not found")
            continue
        process_category(cat_path, args.video_out, args.audio_out, category,
                         args.fps, args.chunk_sec)

    print("Extraction complete.")


if __name__ == "__main__":
    main()
