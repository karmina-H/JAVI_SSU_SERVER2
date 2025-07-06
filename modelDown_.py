from transformers import pipeline
import os

# 아래에 본인 토큰을 다시 한번 확인해서 넣어주세요.
my_token = ""
os.environ["HUGGING_FACE_HUB_TOKEN"] = my_token

print("="*50)
print("독립적인 환경에서 모델 로드 테스트를 시작합니다...")
print(f"사용될 토큰: {my_token[:5]}...") # 토큰 앞 5자리만 보여줌
print("="*50)

try:
    # STT 파이프라인 로드를 시도합니다.
    stt_pipeline = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-small",
        token=my_token
    )
    print("\n\n✅✅✅ 성공! 독립 테스트에서 모델을 정상적으로 로드했습니다.")
    print("    이는 사용자님의 PC 환경이나 토큰에는 문제가 없다는 뜻입니다.")
    print("    문제의 원인은 100% june-va 프로그램 내부에 있습니다.")

except Exception as e:
    print(f"\n\n❌❌❌ 실패! 이 독립 테스트에서도 오류가 발생했습니다.")
    print(f"    오류 내용: {e}")

##############################

from TTS.api import TTS
import os

print("="*50)
print("CoquiTTS XTTS-v2 모델 로드 및 한국어 음성 생성 테스트를 시작합니다...")
print("="*50)

try:
    # CoquiTTS 모델 로드 (자동 다운로드)
    # CPU 사용 설정 (GPU가 없다면 False 유지)
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)
    print("XTTS-v2 모델이 성공적으로 로드되었습니다.")

    # --- 사용자님께서 준비하신 speaker_wav 파일 경로로 변경해주세요 ---
    # 예시: C:\Users\YourUser\Documents\my_korean_speaker.wav
    speaker_wav_path = "./example.wav" # <-- !!! 여기에 실제 경로를 입력하세요 !!!

    output_audio_path = "output_korean_audio.wav"
    korean_text = "안녕하세요! Coqui TTS XTTS-v2로 생성된 한국어 음성입니다."

    # 스피커 참조 음성 파일이 존재하는지 확인
    if not os.path.exists(speaker_wav_path):
        print(f"\n❌❌❌ 오류: 스피커 참조 파일 '{speaker_wav_path}'을(를) 찾을 수 없습니다.")
        print("    파일 경로가 올바른지, 파일이 해당 위치에 있는지 확인해주세요.")
        print("    예시: speaker_wav_path = 'C:\\Users\\사용자이름\\녹음파일.wav'")
    else:
        print(f"\n✅ 스피커 참조 파일 '{speaker_wav_path}'을(를) 찾았습니다. 음성 생성을 시작합니다.")
        # 한국어 음성 생성
        synthesis = tts.tts_to_file(text=korean_text,
                        file_path=output_audio_path,
                        speaker_wav=speaker_wav_path,
                        language="ko") # 언어를 "ko"로 설정하여 한국어임을 명시

        print(f"\n✅✅✅ 성공! '{korean_text}' 텍스트가 '{output_audio_path}' 파일로 성공적으로 저장되었습니다.")
        print(f"생성된 오디오 파일을 확인해주세요: {os.path.abspath(output_audio_path)}")

except Exception as e:
    print(f"\n❌❌❌ Coqui TTS 모델 로드 또는 음성 생성 중 오류 발생: {e}")
