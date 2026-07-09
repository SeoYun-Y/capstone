"""
03_balance_dataset.py

02_split_dataset.py로 만든 AIhub_split_dataset(train/val/test)은 클래스(무색/유색) 간
이미지 수 비율이 원본 데이터 분포를 그대로 따르기 때문에 불균형이 있을 수 있다.
이 스크립트는 각 split에서 단일 클래스(무색만 또는 유색만 포함된 라벨)로 구성된
이미지를 골라, 클래스별로 정해진 개수만큼 균등하게 샘플링해
최종 학습에 사용한 data_balanced 데이터셋을 만든다.

최종 구성: train 20,000장(무색 10,000 + 유색 10,000)
          val   3,000장(무색 1,500 + 유색 1,500)
          test  3,000장(무색 1,500 + 유색 1,500)
          총 26,000장
"""

import os
import random
import shutil
from glob import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

SPLIT_DATASET_ROOT = "/content/drive/MyDrive/data/AIhub_split_dataset"
BALANCED_ROOT = "/content/drive/MyDrive/data/data_balanced"

# split별 클래스당 샘플링 개수
SAMPLE_SIZE = {
    "train": 10000,
    "val": 1500,
    "test": 1500,
}


def classify_labels_by_class(label_dir: str):
    """라벨(.txt) 파일을 읽어 단일 클래스(0=무색 / 1=유색)로만 구성된 파일을 분류"""
    class_0, class_1 = [], []
    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))
    for label_path in tqdm(label_files, desc=f"📂 클래스 분류 중: {label_dir}"):
        with open(label_path, "r") as f:
            lines = f.read().splitlines()
            class_ids = [int(line.split()[0]) for line in lines if line.strip()]
        if class_ids and all(c == 0 for c in class_ids):
            class_0.append(label_path)
        elif class_ids and all(c == 1 for c in class_ids):
            class_1.append(label_path)
    print(f"✅ 무색: {len(class_0)}개 / 유색: {len(class_1)}개")
    return class_0, class_1


def sample_and_copy(class_0, class_1, split: str, n_per_class: int,
                     src_img_dir: str, dst_img_dir: str, dst_lbl_dir: str) -> None:
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    sample_0 = random.sample(class_0, min(n_per_class, len(class_0)))
    sample_1 = random.sample(class_1, min(n_per_class, len(class_1)))
    selected = sample_0 + sample_1

    for label_path in tqdm(selected, desc=f"🚀 {split} 복사 중"):
        fname = os.path.basename(label_path).replace(".txt", ".jpg")
        src_img = os.path.join(src_img_dir, fname)
        dst_img = os.path.join(dst_img_dir, fname)
        dst_lbl = os.path.join(dst_lbl_dir, os.path.basename(label_path))
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)
            shutil.copy(label_path, dst_lbl)


if __name__ == "__main__":
    random.seed(42)

    for split, n_per_class in SAMPLE_SIZE.items():
        label_dir = os.path.join(SPLIT_DATASET_ROOT, split, "labels")
        img_dir = os.path.join(SPLIT_DATASET_ROOT, split, "images")

        class_0, class_1 = classify_labels_by_class(label_dir)

        dst_img_dir = os.path.join(BALANCED_ROOT, "images", split)
        dst_lbl_dir = os.path.join(BALANCED_ROOT, "labels", split)

        sample_and_copy(class_0, class_1, split, n_per_class, img_dir, dst_img_dir, dst_lbl_dir)

    # data_balanced용 data.yaml 저장
    data_yaml = f"""
train: {BALANCED_ROOT}/images/train
val: {BALANCED_ROOT}/images/val
test: {BALANCED_ROOT}/images/test

nc: 2
names: ['투명_pet', '불투명_pet']
""".strip()
    yaml_path = os.path.join(BALANCED_ROOT, "data.yaml")
    os.makedirs(BALANCED_ROOT, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(data_yaml)

    print(f"✅ 클래스 균형 데이터셋 구성 완료 → {BALANCED_ROOT}")
    print(f"✅ data.yaml 저장: {yaml_path}")

