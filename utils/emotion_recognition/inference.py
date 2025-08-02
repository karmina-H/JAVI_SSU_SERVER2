import torch
import torchaudio
from transformers import Wav2Vec2Processor
from model import Wav2Vec2ForEmotion

# 라벨 매핑
label2id = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "neutral": 3
}
id2label = {v: k for k, v in label2id.items()}

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 및 processor 불러오기
model = Wav2Vec2ForEmotion(num_labels=len(id2label))
model.load_state_dict(torch.load("emotion_model.pt", map_location=device))
model = model.to(device)
model.eval()

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# 🔹 오디오 파일 불러오기
filename = "input.wav"
waveform, sr = torchaudio.load(filename)

# 🔹 샘플링 레이트 16kHz로 변환
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)

# 🔹 전처리
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
input_values = inputs["input_values"].to(device)
attention_mask = inputs.get("attention_mask", None)
if attention_mask is not None:
    attention_mask = attention_mask.to(device)

# 🔹 추론
with torch.no_grad():
    outputs = model(input_values=input_values, attention_mask=attention_mask)
    logits = outputs['logits']
    predicted = torch.argmax(logits, dim=-1).item()

print(f"감정 예측 결과: {id2label[predicted]}")
