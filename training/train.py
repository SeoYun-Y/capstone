"""
train.py

YOLOv11-nano(yolo11n)를 data_balanced 데이터셋으로 전이학습(fine-tuning)한다.
COCO 사전학습 가중치를 기반으로 backbone/detection head를 유지한 채,
무색(투명)/유색(불투명) PET 2-클래스 분류에 맞게 추가 학습한다.

실행 환경: Google Colab Pro, NVIDIA A100 GPU
"""

from ultralytics import YOLO

DATA_YAML = "/content/drive/MyDrive/data/data_balanced/data.yaml"
PROJECT_DIR = "/content/drive/MyDrive/runs/train"
RUN_NAME = "yolo11n_custom"


def main():
    # COCO 사전학습 YOLOv11-nano 가중치 로드
    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=32,
        save_period=10,      # 10 에폭마다 체크포인트 저장
        patience=10,         # 10 에폭 동안 개선 없으면 Early Stopping
        workers=4,           # Colab 환경 안정화를 위해 낮게 설정
        save=True,
        cache=True,          # 데이터 메모리 캐싱으로 I/O 병목 최소화
        augment=True,        # RandAugment, Mosaic 등 데이터 증강 사용
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        half=True,           # FP16 학습으로 GPU 메모리 절감 + 추론 속도 향상
    )

    print(f"✅ 학습 완료 → 결과: {PROJECT_DIR}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()

