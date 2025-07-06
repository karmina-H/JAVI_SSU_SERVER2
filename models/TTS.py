from typing import List, Dict, Any
import os

import wave
import numpy as np

class TTS():
    """
    A class for generating speech from text using the TTS library.

    This class inherits from the BaseModel class and provides a method for running
    the Text-to-Speech model on text input.
    """

    def __init__(self, **kwargs) -> None:

        # Disable sentence splitting for better output control
        #self.generation_args["split_sentences"] = False

        # XTTS-v2 필수 인자
        self.speaker_wav_path: str = 'example.wav'
        self.language: str = 'ko'

        if not self.speaker_wav_path:
            raise ValueError("XTTS-v2 모델은 'speaker_wav_path' 인자가 반드시 필요합니다.")
        if not self.language:
            raise ValueError("XTTS-v2 모델은 'language' 인자가 반드시 필요합니다.")
        if not os.path.exists(self.speaker_wav_path):
            raise FileNotFoundError(f"지정된 speaker_wav_path '{self.speaker_wav_path}' 파일을 찾을 수 없습니다.")

        from TTS.api import TTS as CoquiTTS
        self.model = CoquiTTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda')

    def forward(self, text: str, output_filepath: str) -> List[int]:
        """
        Generate speech from text using the Text-to-Speech model.

        Args:
            text: The input text for which speech should be generated.

        Returns:
            A list of integers representing the generated audio data.
        """
        print("음성변환 시작")
        audio: List[int] = self.model.tts(
            text=text,
            speaker_wav=self.speaker_wav_path,
            language=self.language,
            split_sentences = False
        )
        print("음성변환 완료")

        # audio_data_np = np.array(audio, dtype=np.int16)

        # try:
        #     with wave.open(output_filepath, 'wb') as wf:
        #         # 채널 수, 샘플링 너비 (바이트 단위), 샘플링 레이트 설정
        #         wf.setnchannels(1) # 보통 모노 1, 스테레오 2
        #         wf.setsampwidth(2) # 16비트 PCM이면 2 바이트
        #         wf.setframerate(16000) # 모델이 생성하는 오디오의 샘플링 레이트

        #         # NumPy 배열을 바이트로 변환하여 파일에 쓰기
        #         wf.writeframes(audio_data_np.tobytes())
            
        #     print(f"오디오가 '{output_filepath}' 파일로 성공적으로 저장되었습니다.")
        # except Exception as e:
        #     print(f"오디오 저장 중 오류 발생: {e}")

        return audio
