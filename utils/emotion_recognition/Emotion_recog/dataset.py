# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class EmotionDataset(Dataset):
    def __init__(self, csv_path, processor, label2id):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row['file_path'])
        inputs = self.processor(waveform.squeeze(), sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(self.label2id[row['label']])
        return inputs
    


