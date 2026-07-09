# 시각장애인을 위한 YOLOv11 기반 폐플라스틱 분류 및 음성 안내 모바일 앱

투명/불투명 PET 병을 실시간으로 탐지·분류하고, 라벨 부착 여부까지 판별하여 **음성(TTS)**으로 분리배출 방법을 안내하는 iOS 애플리케이션입니다. 시각장애인이 촉감만으로는 구분하기 어려운 플라스틱 색상·투명도 정보를 카메라와 온디바이스 AI로 대신 인식해 알려주는 것을 목표로 합니다.

📺 시연 영상: https://youtube.com/shorts/-tTtCNdwBeA?feature=shared

<br>

## 배경

2022년부터 국내에서는 투명 PET병과 유색(불투명) PET병의 분리배출이 의무화되었지만, 점자·색상 라벨 등 기존 보조 수단만으로는 시각장애인이 이를 구분하기 어렵습니다. 시중의 분리배출 안내 앱 대부분은 수거 대행 서비스이거나 일반 객체 인식에 그쳐, 페트병 투명도 판별에 특화된 앱은 부재한 상황입니다. 이 프로젝트는 이 공백을 채우기 위해 시작되었습니다.

<br>

## 주요 기능

- **실시간 PET 병 탐지 및 분류**: 온디바이스 YOLOv11n 모델로 투명/불투명 PET 병을 실시간 탐지
- **동일 객체 추적**: ByteTrack 기반 커스텀 트래커로 프레임마다 ID가 바뀌는 문제와 중복 음성 안내를 방지
- **라벨 유무 판단**: 탐지된 영역을 crop하여 OCR(Vision `VNRecognizeTextRequest`)로 텍스트(라벨) 존재 여부 판별
- **음성 안내(TTS)**: 분류 결과 + 라벨 유무를 조합해 상황에 맞는 한국어 음성 안내 자동 출력, 중복 발화 방지 로직 포함

<br>

## 시스템 흐름

```
카메라 프레임 수신 (AVCaptureSession)
        ↓
YOLOv11n(.mlmodelc) 추론 → 객체 탐지 (label, bbox, confidence)
        ↓
ByteTrack 기반 추적 → 동일 객체에 트랙 ID 유지
        ↓
일정 프레임 이상 유지된 객체 → OCR 수행 (crop된 영역)
        ↓
분류 결과 + 라벨 유무 → 음성 메시지 생성
        ↓
TTS 출력 (AVSpeechSynthesizer, 중복 방지 처리)
```

<br>

## 기술 스택

| 구분 | 내용 |
|---|---|
| 언어 | Swift 5.0 |
| UI | SwiftUI |
| 객체 탐지 | YOLOv11-nano → Core ML(.mlpackage) 변환, NMS 후처리 포함 |
| 추론 | Core ML, Vision (`VNCoreMLRequest`) |
| 객체 추적 | ByteTrack 알고리즘 기반 커스텀 구현 (칼만 필터 + IoU + 헝가리안 알고리즘) |
| OCR | Vision (`VNRecognizeTextRequest`) |
| 음성 안내 | AVFoundation (`AVSpeechSynthesizer`) |
| 카메라 입력 | AVFoundation (`AVCaptureSession`) |
| 모델 학습 | Google Colab (NVIDIA A100), PyTorch, Ultralytics |
| 학습 데이터 | AI Hub 재활용품 분류 및 선별 데이터셋 (총 26,000장, train/val/test = 80/10/10) |

<br>

## 프로젝트 구조

```
final/
├── finalApp.swift              # 앱 진입점
├── ContentView.swift           # 최상위 뷰 (CameraView 표시)
├── CameraView.swift            # 카메라 프리뷰 + 탐지 결과 시각화 UI
├── CameraPreview.swift         # AVCaptureVideoPreviewLayer 래핑
├── CameraSessionManager.swift  # 카메라 세션 구성, 프레임별 Core ML 추론
├── MyModel.swift               # Core ML 모델(.mlmodelc) 로더
├── best_plastic.mlpackage      # 학습된 YOLOv11n 커스텀 모델 (Core ML)
├── ByteTrackTracker.swift      # 객체 추적기 (칼만 필터, IoU 매칭, 트랙 생애주기 관리)
├── OCRProcessor.swift          # Vision 기반 OCR 처리
└── Info.plist                  # 카메라 권한 등 앱 설정

data_preprocessing/              # AI Hub 데이터 수집 · 전처리 (Colab)
├── 01_extract_and_merge.py      # zip 압축 해제 + AI Hub 라벨 → COCO 스타일 병합
├── 02_split_dataset.py          # stratified train/val/test 분할 → YOLO 포맷 저장
├── 03_balance_dataset.py        # 클래스 균형 샘플링 (26,000장 최종 데이터셋 구성)
└── README.md

training/                        # 모델 학습 · 평가 · 변환 (Colab)
├── train.py                     # YOLOv11-nano 전이학습
├── evaluate.py                  # test split 성능 평가
├── export_coreml.py             # Core ML(.mlpackage) 변환
├── data.yaml                    # 데이터 설정 예시
└── README.md
```

> `data_preprocessing/`, `training/`은 원본 Colab 노트북(Google Drive)의 최종 실행 코드를 정리한 것으로, 실제로는 Google Drive 마운트 환경에서 실행됩니다. 자세한 내용은 각 폴더의 README를 참고하세요.

<br>

## AI 모델

- **베이스**: YOLOv11-nano (경량화 모델, 엣지 디바이스 최적화)
- **분류 클래스**: 무색(투명) PET / 유색(불투명) PET, 2-class
- **학습 환경**: Google Colab Pro, NVIDIA A100 GPU / epochs 50, batch 32, imgsz 640×640
- **성능**: Precision 0.97+ / Recall 0.98+, mAP50 0.99 / mAP50-95 0.87
- **온디바이스 추론 속도**: iPhone 15 Pro 기준 평균 50~60ms/inference (640×640 입력)
- **변환**: PyTorch(.pt) → Core ML(.mlpackage), 변환 시 NMS 후처리 포함하여 export

### 데이터셋 구축 파이프라인 (`data_preprocessing/`)

AI Hub "재활용품 분류 및 선별 데이터"의 무색단일/유색단일 PET 이미지·라벨을 다음 순서로 가공했습니다.

1. 원천 이미지·라벨(.json) zip 압축 해제 → AI Hub 포맷을 COCO 스타일로 병합
2. 클래스 비율을 유지한 stratified 분할 (train 80% / val 10% / test 10%) → YOLO 포맷(txt) 저장
3. 단일 클래스로만 구성된 이미지를 골라 무색/유색 각각 균등 샘플링 → 최종 26,000장(train 20,000 / val 3,000 / test 3,000) 데이터셋 완성

### 학습·변환 파이프라인 (`training/`)

- `train.py`: YOLOv11-nano를 `data_balanced` 데이터셋으로 전이학습
- `evaluate.py`: test split 기준 mAP·Precision·Recall 평가
- `export_coreml.py`: 학습된 `best.pt`를 `nms=True` 옵션으로 Core ML 변환 (NMS 누락 시 신뢰도 값이 비정상적으로 낮게 나오는 문제 해결)

<br>

## 개발 중 주요 문제 해결

- **동일 객체 중복 안내**: YOLO는 프레임 단위로만 탐지하기 때문에 같은 물체에 다른 ID가 반복 부여되어 TTS가 중복 발화되던 문제 → ByteTrack 기반 추적기를 직접 구현하여 트랙 ID를 유지하고 중복 안내를 방지
- **Core ML 변환 후 신뢰도 값 이상**: 변환된 모델의 objectness/class confidence가 비정상적으로 낮게(0.0002 수준) 출력되던 문제 → export 시 NMS를 명시적으로 포함하여 재변환, 신뢰도 값 정상화
- **OCR 인식 실패**: crop 좌표 계산 오류로 인식 영역이 이미지 밖을 참조하던 문제 → crop 좌표 계산 로직 수정 및 이미지 처리 방식 개선
- **데이터셋 다양성 부족**: 단일 배경 위주 데이터로 학습된 모델이 실제 다양한 배경/조명에서 추론 실패 → 다양한 배경·각도의 이미지를 데이터셋에 추가하여 개선

<br>

## 팀 구성 (캡스톤디자인 2조)

| 이름 | 역할 |
|---|---|
| 송재원 | 프로젝트 관리, 발표 |
| 박채영 | AI 모델 학습, iOS 앱 개발 |
| 염서윤 | 데이터 수집, 데이터 전처리 및 분류 |

<br>

## 향후 계획

- 추가 클래스 학습(이물질 포함 여부 등)을 통한 모델 성능 고도화
- App Store 배포 및 실사용자 피드백 기반 개선
