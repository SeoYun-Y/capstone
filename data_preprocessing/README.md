# 데이터 수집 및 전처리

AI Hub "재활용품 분류 및 선별 데이터"에서 페트병(무색단일/유색단일) 이미지와 라벨을 수집해
YOLO 학습용 데이터셋(`data_balanced`)으로 만드는 파이프라인입니다. 원본 Colab 노트북(`train_val_test 나누기`)의 실제 실행 코드를 스크립트 단위로 정리했습니다.

## 실행 순서

| 순서 | 스크립트 | 설명 |
|---|---|---|
| 1 | `01_extract_and_merge.py` | 무색/유색 PET 이미지·라벨 zip 압축 해제 → AI Hub 라벨(.json)을 COCO 스타일 통합 어노테이션(`full_coco.json`)으로 병합 |
| 2 | `02_split_dataset.py` | `full_coco.json`을 클래스 비율 유지(stratified) 방식으로 train(80%)/val(10%)/test(10%) 분할 → YOLO 포맷(images/, labels/*.txt)으로 저장 |
| 3 | `03_balance_dataset.py` | 단일 클래스로만 구성된 이미지를 골라 무색/유색 각각 정해진 개수(train 10,000장·val/test 각 1,500장)만큼 균등 샘플링 → 최종 `data_balanced` 데이터셋 구성 |

## 클래스 정의

| class_id | 라벨 |
|---|---|
| 0 | 무색(투명) PET |
| 1 | 유색(불투명) PET |

- AI Hub 원본 라벨은 `ANNOTATION_INFO`의 `DETAILS` 필드에 "유색" 포함 여부로 구분
- `SHAPE_TYPE`이 BOX/POLYGON 둘 다 지원하며, 학습 시에는 모두 bounding box로 통일하여 사용

## 최종 데이터셋 규모

- train: 20,000장 (무색 10,000 + 유색 10,000)
- val: 3,000장 (무색 1,500 + 유색 1,500)
- test: 3,000장 (무색 1,500 + 유색 1,500)
- **총 26,000장**

## 참고

- 각 스크립트 상단의 경로 변수(`BASE_DIR`, `SAVE_ROOT`, `BALANCED_ROOT` 등)는 실제 실행 시 Google Drive 마운트 경로에 맞춰 수정해서 사용하면 됩니다.
- Colab 환경(Google Drive 마운트, GPU 런타임)에서의 실행을 전제로 작성되었습니다.

