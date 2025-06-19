import torch
import torchaudio
from transformers import Wav2Vec2Processor
from .model import Wav2Vec2ForEmotion  # 사용자 정의 모델 클래스
import os
from pathlib import Path
import torch.nn.functional as F # [추가] 소프트맥스 함수를 사용하기 위해 임포트

# --- 1. 초기 설정 (이전과 동일) ---

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
    MODEL_PATH = SCRIPT_DIR / "emotion_model.pt"
    
    MODEL = Wav2Vec2ForEmotion(num_labels=len(ID2LABEL))
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print("로딩 완료.")
except Exception as e:
    print(f"초기화 중 에러 발생: {e}")
    MODEL = None
    PROCESSOR = None

# --- 2. 감정 분석 함수 ---

# [수정] 반환 값 타입을 (str, float) 튜플로 변경
def predict_emotion(voice_path: str) -> tuple[str, float] | None:
    """
    주어진 음성 파일 경로로부터 감정과 확률을 분석하여 튜플로 반환합니다.

    Args:
        voice_path (str): 분석할 .wav 음성 파일의 경로

    Returns:
        tuple[str, float] | None: (감정 문자열, 확률 값) 튜플. 에러 시 None.
    """
    if not MODEL or not PROCESSOR:
        print("모델 또는 프로세서가 제대로 로딩되지 않았습니다.")
        return None

    if not os.path.exists(voice_path):
        print(f"오류: 파일을 찾을 수 없습니다 - {voice_path}")
        return None

    try:
        # ... (오디오 로딩 및 전처리 부분은 이전과 동일) ...
        waveform, sr = torchaudio.load(voice_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        inputs = PROCESSOR(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(DEVICE)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)
        
        # 🔹 추론
        with torch.no_grad():
            outputs = MODEL(input_values=input_values, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # [추가] 확률 계산
            # Softmax 함수를 적용하여 logits를 확률로 변환
            probabilities = F.softmax(logits, dim=-1)
            
            # [추가] 가장 높은 확률 값과 해당 인덱스를 가져옴
            confidence_score, predicted_id_tensor = torch.max(probabilities, dim=-1)
            
            predicted_id = predicted_id_tensor.item()
            highest_probability = confidence_score.item()
            
        # [수정] 결과 반환 (감정 레이블과 확률 값을 함께 반환)
        predicted_label = ID2LABEL[predicted_id]
        return predicted_label, highest_probability

    except Exception as e:
        print(f"감정 분석 중 에러 발생: {e}")
        return None


# --- 3. 함수 사용 예시 ---

if __name__ == "__main__":
    # 분석할 오디오 파일 경로
    audio_file = "./input.wav"
    
    # [수정] 함수 호출 및 반환 값 처리
    result = predict_emotion(audio_file)
    
    # [수정] 결과 출력
    if result:
        # 튜플에서 감정과 확률 값을 각각 추출
        emotion_label, probability = result
        
        print(f"\n[최종 분석 결과]")
        print(f"파일: {audio_file}")
        print(f"감정: {emotion_label}")
        print(f"확률: {probability * 100:.2f}%") # 소수점 둘째 자리까지 백분율로 표시
    else:
        print("\n감정 분석에 실패했습니다.")