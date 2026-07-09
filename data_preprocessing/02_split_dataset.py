"""
02_split_dataset.py

01_extract_and_merge.py에서 생성한 full_coco.json을 읽어
클래스(무색/유색) 비율을 유지한 stratified train/val/test 분할을 수행하고,
YOLO 학습 포맷(images/, labels/*.txt)으로 저장한다.

분할 비율: train 80% / val 10% / test 10%
"""

import os
import json
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ── 경로 설정 ─────────────────────────────────────────────
FULL_COCO_PATH = "full_coco.json"
SAVE_ROOT = "/content/drive/MyDrive/data/AIhub_split_dataset"

# 원천 이미지가 실제로 위치한 폴더들 (01번 스크립트의 압축 해제 경로와 일치해야 함)
IMAGE_SOURCE_DIRS = [
    "/content/drive/MyDrive/캡스톤디자인 2조/Colab Notebooks/AIhub_dataset/무색_training/무색단일1_원천데이터",
    "/content/drive/MyDrive/캡스톤디자인 2조/Colab Notebooks/AIhub_dataset/무색_training/무색단일2_원천데이터",
    "/content/drive/MyDrive/캡스톤디자인 2조/Colab Notebooks/AIhub_dataset/무색_training/무색단일3_원천데이터",
    "/content/drive/MyDrive/캡스톤디자인 2조/Colab Notebooks/AIhub_dataset/유색_training/유색단일1_원천데이터",
    "/content/drive/MyDrive/캡스톤디자인 2조/Colab Notebooks/AIhub_dataset/유색_training/유색단일2_원천데이터",
]


def extract_image_id_to_category(coco: dict) -> dict:
    """image_id -> category_id(대표 클래스) 매핑 (stratify 기준용)"""
    id2cat = {}
    for ann in coco["annotations"]:
        id2cat[ann["image_id"]] = ann["category_id"]
    return id2cat


def stratified_split(id2cat: dict):
    """무색/유색 비율을 유지하며 train(80%) / val(10%) / test(10%)로 분할"""
    image_ids = list(id2cat.keys())
    labels = [id2cat[i] for i in image_ids]

    train_ids, temp_ids, _, y_temp = train_test_split(
        image_ids, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    return train_ids, val_ids, test_ids


def save_yolo_format(coco: dict, image_ids: list, split_name: str,
                      image_dirs: list, save_root: str = SAVE_ROOT) -> None:
    """선택된 image_id들의 이미지를 복사하고, 어노테이션을 YOLO txt로 저장"""
    image_save_dir = os.path.join(save_root, split_name, "images")
    label_save_dir = os.path.join(save_root, split_name, "labels")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    id2img = {img["id"]: img for img in coco["images"]}
    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    for img_id in tqdm(image_ids, desc=f"📦 저장 중: {split_name}"):
        img = id2img[img_id]
        fname = img["file_name"]
        w, h = img["width"], img["height"]
        anns = ann_map.get(img_id, [])

        if not fname or not w or not h:
            continue

        img_dst_path = os.path.join(image_save_dir, fname)
        label_dst_path = os.path.join(label_save_dir, os.path.splitext(fname)[0] + ".txt")

        if os.path.exists(img_dst_path) and os.path.exists(label_dst_path):
            continue  # 이미 처리됨

        # 이미지 복사
        if not os.path.exists(img_dst_path):
            found = False
            for dir_path in image_dirs:
                src = os.path.join(dir_path, fname)
                if os.path.exists(src):
                    shutil.copy(src, img_dst_path)
                    found = True
                    break
            if not found:
                print(f"[경고 - {split_name}] 이미지 못 찾음: {fname}")
                continue

        # YOLO 라벨(txt) 저장: class_id x_center y_center width height (모두 0~1 정규화)
        with open(label_dst_path, "w") as f:
            for ann in anns:
                x_min, y_min, box_w, box_h = ann["bbox"]
                if box_w < 5 or box_h < 5:
                    continue
                x_c = (x_min + box_w / 2) / w
                y_c = (y_min + box_h / 2) / h
                norm_w = box_w / w
                norm_h = box_h / h
                f.write(f"{ann['category_id']} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}\n")


def write_data_yaml(save_root: str = SAVE_ROOT) -> None:
    data_yaml = f"""
train: {save_root}/train/images
val: {save_root}/val/images
test: {save_root}/test/images

nc: 2
names: ['투명_pet', '불투명_pet']
""".strip()
    yaml_path = os.path.join(save_root, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(data_yaml)
    print(f"✅ data.yaml 저장: {yaml_path}")


if __name__ == "__main__":
    with open(FULL_COCO_PATH, "r", encoding="utf-8") as f:
        full_coco = json.load(f)

    id2cat = extract_image_id_to_category(full_coco)
    train_ids, val_ids, test_ids = stratified_split(id2cat)

    save_yolo_format(full_coco, train_ids, "train", IMAGE_SOURCE_DIRS)
    save_yolo_format(full_coco, val_ids, "val", IMAGE_SOURCE_DIRS)
    save_yolo_format(full_coco, test_ids, "test", IMAGE_SOURCE_DIRS)

    write_data_yaml()

    print(f"✅ 분할 완료 — train {len(train_ids)} / val {len(val_ids)} / test {len(test_ids)}")

