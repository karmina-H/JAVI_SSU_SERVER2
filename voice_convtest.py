import os
import sys
import pathlib
import soundfile as sf
from utils.voice_conversion.voice_conversion import run_voice_conversion

# 프로젝트 루트 폴더를 Python 경로에 추가하여 모듈을 찾을 수 있도록 설정
# 이 스크립트는 'NeuroSync_Player' 폴더에서 실행하는 것을 기준으로 합니다.
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def load_and_convert(input_wav_path: str, transpose: int = 0):
    """
    WAV 파일을 읽어 음성 변환을 수행하고,
    변환된 오디오 데이터와 샘플레이트를 반환합니다.
    """
    print(f"오디오 파일 로딩 및 변환 중: '{input_wav_path}'...")
    waveform, sr = sf.read(input_wav_path, dtype="float32")
    
    # run_voice_conversion 함수는 (변환된 오디오 배열, 변환된 샘플레이트) 튜플을 반환합니다.
    converted_waveform, converted_sr = run_voice_conversion(
        src_audio=waveform,
        src_sr=sr, 
        transpose=transpose,
        voice_name = "IUNEW"
    )
    
    return converted_waveform, converted_sr

# -------------------------------
# 메인 실행 부분
# -------------------------------
if __name__ == "__main__":
    try:
        # 1. 입력 및 출력 파일 경로 정의
        input_file = "wav_input/korean.wav"
        output_file = "wav_output/korean.wav"
        
        # 출력 폴더가 없으면 자동으로 생성
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        print("음성 변환 프로세스를 시작합니다...")
        
        # 2. 원본 오디오를 로드하고 음성을 변환합니다.
        converted_wave, sr = load_and_convert(input_file, transpose=0)
        
        # 3. 변환된 오디오를 새로운 파일로 저장합니다.
        print(f"변환된 오디오를 다음 경로에 저장합니다: '{output_file}'")
        sf.write(output_file, converted_wave, sr)
        
        print("프로세스가 성공적으로 완료되었습니다!")

    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file}'을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

