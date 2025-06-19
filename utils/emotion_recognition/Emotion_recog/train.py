# train.py
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
from torch.optim import AdamW
from model import Wav2Vec2ForEmotion
from dataset import EmotionDataset

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = torch.tensor([item["labels"] for item in batch])
    # pad sequences
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "labels": labels
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 감정 라벨 정의
label2id = {
    "Happiness": 0,
    "Surprise": 1,
    "Neutral": 2,
    "Fear": 3,
    "Disgust": 4,
    "Anger": 5,
    "Sadness": 6
}
id2label = {v: k for k, v in label2id.items()}

# Processor 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Dataset 로드
dataset = EmotionDataset(
    csv_path="train.csv",
    processor=processor,
    label2id=label2id
)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 모델 로드
model = Wav2Vec2ForEmotion(num_labels=len(label2id)).to(device)

# 옵티마이저
optimizer = AdamW(model.parameters(), lr=2e-5)

# 학습 루프
for epoch in range(5):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"📘 Epoch {epoch+1} - 평균 Loss: {avg_loss:.4f}")

# 모델 저장
torch.save(model.state_dict(), "emotion_model.pt")
print("모델 저장 완료: emotion_model.pt")
