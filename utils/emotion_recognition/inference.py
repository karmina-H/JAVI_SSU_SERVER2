import torch
import torchaudio
from transformers import Wav2Vec2Processor
from model import Wav2Vec2ForEmotion  # 사용자 정의 모델 클래스
import os
from pathlib import Path


# --- 1. 초기 설정 (프로그램 시작 시 한 번만 실행) ---

# 라벨 매핑 정의
ID2LABEL = {
    0: "Happiness",
    1: "Surprise",
    2: "Neutral",
    3: "Fear",
    4: "Disgust",
    5: "Anger",
    6: "Sadness"
}

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 모델 및 Processor 전역 변수로 불러오기
try:
    print("모델과 프로세서를 로딩합니다...")
    SCRIPT_DIR = Path(__file__).resolve().parent
    # 모델 파일의 절대 경로를 생성
    MODEL_PATH = SCRIPT_DIR / "emotion_model.pt"
    
    MODEL = Wav2Vec2ForEmotion(num_labels=len(ID2LABEL))
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()  # 추론 모드로 설정

    PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print("로딩 완료.")
except Exception as e:
    print(f"초기화 중 에러 발생: {e}")
    MODEL = None
    PROCESSOR = None

# --- 2. 감정 분석 함수 ---

def predict_emotion(voice_path: str) -> str:
    """
    주어진 음성 파일 경로로부터 감정을 분석하여 문자열로 반환합니다.

    Args:
        voice_path (str): 분석할 .wav 음성 파일의 경로

    Returns:
        str: 분석된 감정 문자열 (예: "Happiness"). 에러 발생 시 None을 반환합니다.
    """
    if not MODEL or not PROCESSOR:
        print("모델 또는 프로세서가 제대로 로딩되지 않았습니다.")
        return None

    if not os.path.exists(voice_path):
        print(f"오류: 파일을 찾을 수 없습니다 - {voice_path}")
        return None

    try:
        # 🔹 오디오 파일 불러오기
        waveform, sr = torchaudio.load(voice_path)

        # 🔹 샘플링 레이트 16kHz로 변환
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        # 🔹 전처리
        inputs = PROCESSOR(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(DEVICE)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)
        
        # 🔹 추론
        with torch.no_grad():
            outputs = MODEL(input_values=input_values, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_id = torch.argmax(logits, dim=-1).item()
        
        # 🔹 결과 반환
        return ID2LABEL[predicted_id]

    except Exception as e:
        print(f"감정 분석 중 에러 발생: {e}")
        return None


# --- 3. 함수 사용 예시 ---

if __name__ == "__main__":
    # 분석할 오디오 파일 경로
    audio_file = "./input.wav"
    
    # 함수 호출
    predicted_emotion = predict_emotion(audio_file)
    
    # 결과 출력
    if predicted_emotion:
        print(f"\n[최종 분석 결과]")
        print(f"파일: {audio_file}")
        print(f"감정: {predicted_emotion}")
    else:
        print("\n감정 분석에 실패했습니다.")