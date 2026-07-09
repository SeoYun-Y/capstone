"""
export_coreml.py

학습된 YOLOv11n(best.pt)을 iOS 앱에서 사용할 수 있도록 Core ML(.mlpackage) 포맷으로 변환한다.

⚠️ 주의: export 시 nms=True를 반드시 포함해야 한다.
   NMS(Non-Maximum Suppression) 후처리를 포함하지 않고 변환하면 objectness score /
   class confidence 값이 비정상적으로 낮게(0.0002 등) 출력되는 문제가 있었다.
   (자세한 내용은 상위 폴더 final/의 README 및 프로젝트 보고서 "모델 변환 과정 중
   발생한 문제점 및 해결 방안" 참고)
"""

from ultralytics import YOLO

MODEL_PATH = "/content/drive/MyDrive/runs/train/yolo11n_custom/weights/best.pt"


def main():
    model = YOLO(MODEL_PATH)
    model.export(
        format="coreml",
        nms=True,  # 후처리(NMS) 포함하여 export — 신뢰도 값 정상화에 필수
    )
    print("✅ Core ML(.mlpackage) 변환 완료 → best.mlpackage")
    print("   Xcode 프로젝트에 추가 후, final/best_plastic.mlpackage로 사용됨")


if __name__ == "__main__":
    main()

