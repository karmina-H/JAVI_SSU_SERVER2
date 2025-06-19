import os
import csv

# 감정 ID 범위 설정
emotion_map = {
    "Happiness": range(1, 51),
    "Surprise": range(51, 101),
    "Neutral": range(101, 151),
    "Fear": range(151, 201),
    "Disgust": range(201, 251),
    "Anger": range(251, 301),
    "Sadness": range(301, 351)
}

def get_emotion_label(index):
    for label, r in emotion_map.items():
        if index in r:
            return label
    return None  # 없는 범위면 None 반환

def generate_csv_from_wav(wav_dir, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "label"])  # 헤더

        for file in sorted(os.listdir(wav_dir)):
            if file.endswith(".wav"):
                try:
                    # 예: 000-245.wav → idx = 245
                    idx = int(file.split("-")[1].split(".")[0])
                    label = get_emotion_label(idx)
                    if label is not None:
                        writer.writerow([os.path.join(wav_dir, file), label])
                except:
                    continue
    print(f"train.csv 생성 완료: {output_csv}")

# 사용 예시
if __name__ == "__main__":
    generate_csv_from_wav(
        # wav_dir="/home/elicer/emotion_project/emotion_wav",  # wav 파일 경로
        wav_dir="/home/elicer/emotion_project/evaluate",  # wav 파일 경로
        # output_csv="/home/elicer/emotion_project/train.csv"
        output_csv="/home/elicer/emotion_project/evaluate.csv"
    )
