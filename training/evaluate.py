"""
evaluate.py

학습된 YOLOv11n 모델(best.pt)을 test split으로 평가하여
mAP50, mAP50-95, Precision, Recall 등 성능 지표를 확인한다.
"""

from ultralytics import YOLO

MODEL_PATH = "/content/drive/MyDrive/runs/train/yolo11n_custom/weights/best.pt"
DATA_YAML = "/content/drive/MyDrive/data/data_balanced/data.yaml"


def main():
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATA_YAML, split="test")
    print(metrics)  # mAP50, mAP50-95, Precision, Recall 등 출력


if __name__ == "__main__":
    main()

# CLI로 직접 실행하려면:
# yolo val model=/content/drive/MyDrive/runs/train/yolo11n_custom/weights/best.pt \
#          data=/content/drive/MyDrive/data/data_balanced/data.yaml split=test

