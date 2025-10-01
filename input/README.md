# 📁 입력 데이터 폴더

여기에 학습할 데이터를 넣어주세요.

## 📂 폴더 구조

```
input/
├── images/           # 이미지 파일들을 여기에 넣으세요
│   ├── img001.jpg
│   ├── img002.jpg
│   ├── img003.png
│   └── ...
└── labels/           # 라벨 파일들을 여기에 넣으세요  
    ├── img001.txt
    ├── img002.txt
    ├── img003.txt
    └── ...
```

## 📝 라벨 파일 형식

각 이미지에 대응하는 `.txt` 파일에 다음 형식으로 작성:

```
RECT,x1,y1,x2,y2,label
RECT,x1,y1,x2,y2,label
...
```

### 예시 (`img001.txt`):
```
RECT,138,167,187,219,fake
RECT,348,249,376,286,real
RECT,100,100,200,200,none
```

- `x1,y1`: 박스 왼쪽 위 좌표
- `x2,y2`: 박스 오른쪽 아래 좌표  
- `label`: `fake`, `real`, `none` (none은 real로 처리됨)

## 🚀 데이터 준비 후 실행

데이터를 넣은 후 다음 명령어로 학습 시작:

```bash
# 1. 데이터 변환 및 준비
python prepare_data.py --image_dir input/images --label_dir input/labels

# 2. 모델 학습
python resume_training.py config --mode both
```