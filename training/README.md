# 모델 학습 및 변환

`data_preprocessing/`으로 구성한 `data_balanced` 데이터셋으로 YOLOv11-nano를 전이학습하고, iOS 앱에 탑재할 수 있도록 Core ML로 변환하는 코드입니다. 원본 Colab 노트북(`Untitled0.ipynb`)의 최종 학습 셀을 스크립트 단위로 정리했습니다.

## 실행 순서

| 순서 | 스크립트 | 설명 |
|---|---|---|
| 1 | `train.py` | YOLOv11-nano(`yolo11n.pt`) 전이학습. epochs 50 / imgsz 640 / batch 32 |
| 2 | `evaluate.py` | test split 기준 mAP50, mAP50-95, Precision, Recall 평가 |
| 3 | `export_coreml.py` | `best.pt` → Core ML(`.mlpackage`) 변환 (NMS 후처리 포함) |

`data.yaml`은 최종 학습에 사용한 데이터 설정 예시입니다.

## 학습 결과 요약

| 항목 | 값 |
|---|---|
| 베이스 모델 | YOLOv11-nano (COCO 사전학습) |
| 학습 환경 | Google Colab Pro, NVIDIA A100 GPU |
| Epochs | 50 |
| 입력 크기 | 640×640 |
| Batch size | 32 |
| Precision | 0.97+ |
| Recall | 0.98+ |
| mAP50 | 0.99 |
| mAP50-95 | 0.87 |
| 온디바이스 추론 속도 | iPhone 15 Pro 기준 평균 50~60ms/inference |

## Core ML 변환 시 주의사항

`model.export(format="coreml")`만 실행하면 후처리(NMS)가 빠진 채 변환되어, 변환된 모델의 objectness score / class confidence 값이 매우 낮게(0.0002 등) 출력되는 문제가 있었습니다. `nms=True` 옵션을 명시적으로 포함해 재변환하면서 신뢰도 값이 정상적으로(0.98 등) 출력되도록 해결했습니다. 최종 변환된 모델은 `final/best_plastic.mlpackage`로 iOS 앱에 포함되어 있습니다.

