
"""
01_extract_and_merge.py

AI Hub "재활용품 분류 및 선별 데이터" 중 페트병(무색단일 / 유색단일) 원천 이미지와
라벨링 데이터(.json)를 압축 해제하고, 하나의 COCO 스타일 통합 어노테이션으로 병합한다.

- 원천 이미지: 무색단일(투명 PET), 유색단일(불투명 PET) zip 압축 해제
- 라벨링(.json): AI Hub 포맷(IMAGE_INFO / ANNOTATION_INFO)을 COCO 스타일로 변환
  - category_id: 0 = 무색(투명) PET, 1 = 유색(불투명) PET
  - SHAPE_TYPE == BOX / POLYGON 모두 bounding box로 변환
  - 너무 작은 박스(min_box_size 이하)는 제외
"""

import os
import json
import zipfile
from glob import glob
from tqdm import tqdm

# ── 경로 설정 (환경에 맞게 수정) ─────────────────────────────
BASE_DIR = "/content/drive/MyDrive/캡스톤디자인 2조/Colab Notebooks/AIhub_dataset"

CLEAR_IMAGE_ZIPS = [
    f"{BASE_DIR}/무색_training/TS_2.직접촬영_03.페트병_001.무색단일_1.zip",
]
COLORED_IMAGE_ZIPS = [
    f"{BASE_DIR}/유색_training/TS_2.직접촬영_03.페트병_002.유색단일_1.zip",
]

CLEAR_LABEL_ZIP = f"{BASE_DIR}/무색_training/TL_2.직접촬영_03.페트병_001.무색단일.zip"
COLORED_LABEL_ZIP = f"{BASE_DIR}/유색_training/TL_2.직접촬영_03.페트병_002.유색단일.zip"

CLEAR_LABEL_EXTRACT_DIR = "/content/무"
COLORED_LABEL_EXTRACT_DIR = "/content/유"


def extract_zip(zip_path: str, extract_to: str) -> None:
    """이미 추출된 파일은 건너뛰며 zip을 해제한다."""
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        extracted, skipped = 0, 0
        for file in tqdm(file_list, desc=f"📂 압축 해제 중: {os.path.basename(zip_path)}"):
            clean_name = file.lstrip("/")
            target_path = os.path.join(extract_to, clean_name)
            if os.path.exists(target_path):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with zip_ref.open(file) as src, open(target_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted += 1
        print(f"✅ {os.path.basename(zip_path)}: {extracted}개 추출, {skipped}개 생략")


def merge_aihub_jsons(local_json_path: str, image_files_set: set | None = None,
                       start_image_id: int = 0, start_ann_id: int = 0,
                       min_box_size: int = 1) -> dict:
    """
    AI Hub 라벨(.json, IMAGE_INFO/ANNOTATION_INFO 포맷)을 읽어
    COCO 스타일 dict({images, annotations, categories})로 병합한다.
    """
    json_files = sorted(glob(os.path.join(local_json_path, "**/*.json"), recursive=True))
    full_coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "무색", "supercategory": "PET"},
            {"id": 1, "name": "유색", "supercategory": "PET"},
        ],
    }

    image_id, ann_id = start_image_id, start_ann_id

    for file_path in tqdm(json_files, desc=f"📦 병합 중: {os.path.basename(local_json_path)}"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ JSON 파일 오류: {file_path} ({e})")
            continue

        img_info = data.get("IMAGE_INFO", {})
        anns = data.get("ANNOTATION_INFO", [])
        fname = img_info.get("FILE_NAME")
        w, h = img_info.get("IMAGE_WIDTH"), img_info.get("IMAGE_HEIGHT")

        if not fname or not w or not h:
            continue
        if image_files_set and fname not in image_files_set:
            continue

        full_coco["images"].append({"id": image_id, "file_name": fname, "width": w, "height": h})

        for ann in anns:
            points = ann.get("POINTS", [])
            shape_type = ann.get("SHAPE_TYPE", "BOX").upper()
            if not points:
                continue

            if shape_type == "BOX":
                if len(points) == 1 and len(points[0]) == 4:
                    # AI Hub: [x, y, width, height]
                    x_min, y_min, box_w, box_h = points[0]
                elif len(points) == 2 and all(len(p) == 2 for p in points):
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    x_min, y_min = min(x1, x2), min(y1, y2)
                    box_w, box_h = abs(x2 - x1), abs(y2 - y1)
                else:
                    print(f"⚠️ BOX 포맷 오류: {points}")
                    continue
                segmentation = [[x_min, y_min, x_min + box_w, y_min,
                                  x_min + box_w, y_min + box_h, x_min, y_min + box_h]]

            elif shape_type == "POLYGON":
                if not all(len(p) == 2 for p in points):
                    continue
                x_coords = [pt[0] for pt in points]
                y_coords = [pt[1] for pt in points]
                x_min, y_min = min(x_coords), min(y_coords)
                box_w, box_h = max(x_coords) - x_min, max(y_coords) - y_min
                segmentation = [[coord for pt in points for coord in pt]]

            else:
                continue  # 정의되지 않은 SHAPE_TYPE 무시

            if box_w <= min_box_size or box_h <= min_box_size:
                continue

            # DETAILS 필드에 "유색"이 포함되면 category_id=1, 아니면 0(무색)
            category_id = 1 if "유색" in ann.get("DETAILS", "") else 0

            full_coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x_min), float(y_min), float(box_w), float(box_h)],
                "area": float(box_w * box_h),
                "iscrowd": 0,
                "segmentation": segmentation,
            })
            ann_id += 1

        image_id += 1

    return full_coco


if __name__ == "__main__":
    # 1) 원천 이미지 zip 압축 해제
    for zp in CLEAR_IMAGE_ZIPS + COLORED_IMAGE_ZIPS:
        extract_zip(zp, extract_to=os.path.dirname(zp))

    # 2) 라벨(.json) zip 압축 해제
    os.system(f"unzip -o '{CLEAR_LABEL_ZIP}' -d '{CLEAR_LABEL_EXTRACT_DIR}'")
    os.system(f"unzip -o '{COLORED_LABEL_ZIP}' -d '{COLORED_LABEL_EXTRACT_DIR}'")

    # 3) 무색 / 유색 라벨을 각각 COCO 포맷으로 변환 후 병합
    full_clear = merge_aihub_jsons(CLEAR_LABEL_EXTRACT_DIR)
    full_colored = merge_aihub_jsons(
        COLORED_LABEL_EXTRACT_DIR,
        start_image_id=len(full_clear["images"]),
        start_ann_id=len(full_clear["annotations"]),
    )

    full_coco = {
        "images": full_clear["images"] + full_colored["images"],
        "annotations": full_clear["annotations"] + full_colored["annotations"],
        "categories": full_clear["categories"],
    }

    with open("full_coco.json", "w", encoding="utf-8") as f:
        json.dump(full_coco, f, ensure_ascii=False)

    print(f"✅ 통합 완료 — 이미지 {len(full_coco['images'])}장, "
          f"어노테이션 {len(full_coco['annotations'])}개 → full_coco.json 저장")
