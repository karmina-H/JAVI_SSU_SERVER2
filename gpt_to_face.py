from __future__ import annotations
"""
livelink_voice_chat.py — Voice‑to‑Voice GPT Demo (확장판)
-------------------------------------------------------
• 마이크 입력, wav 파일, **텍스트 직접 입력** 3‑way 인터페이스
• 한국어 답변 → 로마자 변환(g2pk) → 영어 TTS → NeuroSync 스트리밍
• OpenAI SDK ≥ 1.30 호환 (audio 네임스페이스)

※ 2025‑06‑14 업데이트
    - voice_name 을 utils/voice_conversion/pretrained/ 아래 폴더 중에서 선택하도록 변경.
    - TTS 결과를 numpy 로 변환 후 RVC 처리, 다시 wav 로 저장하도록 통일.
    - build_voice_reply_* 함수 중복 호출 및 경로/데이터 타입 혼동 수정.
"""

# ────────────────── 외부 라이브러리 및 표준 ───────────────────
import os, sys, queue, tempfile, warnings
from pathlib import Path
from threading import Thread
from contextlib import suppress
from types import GeneratorType
import httpx
import numpy as np
import soundfile as sf
<<<<<<< HEAD
=======
import openai           # OpenAI Python SDK (>=1.30)
>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea
import pygame
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import json

import threading
import socket
<<<<<<< HEAD
import wave
=======
>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea

from g2pk import G2p    # 로마자 변환
from utils.voice_conversion.voice_conversion import run_voice_conversion

# ────────────────── 프로젝트 내부 모듈 ─────────────────────────
from utils.files.file_utils import initialize_directories, ensure_wav_input_folder_exists, list_wav_files
from utils.romanize_ko import romanize_korean
from utils.audio_face_workers import process_wav_file
from livelink.connect.livelink_init import initialize_py_face, create_socket_connection
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.emote_sender.send_emote import EmoteConnect
from utils.emotion_recognition.predict_emotion  import predict_emotion

os.environ["OPENAI_API_KEY"] = ""

from flask import Flask

<<<<<<< HEAD
############# stt,tts,llama init  ####################
from models.LLM import LLM
from models.TTS import TTS
from models.STT import STT

# LLM (Large Language Model) 설정
llm_config = {'disable_chat_history': False,'model': 'llama3.1-8b-instruct-q4_0'}
# STT (Speech-to-Text) 설정
stt_config = {'device': 'cuda','generation_args': {'batch_size': 8},'model': 'openai/whisper-small'}
# TTS (Text-to-Speech) 설정
tts_config = {'device': 'cuda', 'model': 'tts_models/multilingual/multi-dataset/xtts_v2'}



stt_model = STT(**stt_config) if stt_config else None
tts_model = TTS(**tts_config) if tts_config else None
llm_model = LLM(**llm_config)

# if not stt_model.exists():
#     print(f"Invalid stt_model model")
#     exit()
# if not tts_model.exists():
#     print(f"Invalid tts_model model")
#     exit()
if not llm_model.exists():
    print(f"Invalid ollama model")
    exit()

##################

# ────────────────── 설정값 ─────────────────────────────────────
=======
# ────────────────── 설정값 ─────────────────────────────────────
TTS_MODEL = "tts-1"
TRANSCRIBE_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o-mini"
>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea
AUDIO_SAMPLE_RATE = 16_000
OUTPUT_WAV_DIR = Path.cwd() / "wav_cache"
OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
=======
# ────────────────── OpenAI 클라이언트 ─────────────────────────
try:
    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORGANIZATION"),
    )
except openai.OpenAIError as e:
    print(f"Error initializing OpenAI client: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea
# ────────────────── Voice preset discovery ────────────────────
# 이 스크립트가 있는 폴더를 기준으로 상대 경로 설정
BASE_DIR = Path(__file__).resolve().parent
VOICE_PRETRAINED_DIR = BASE_DIR / "utils" / "voice_conversion" / "pretrained"

# (이 부분은 실제 RVC 모델을 사용할 때 필요하므로, 경로가 없어도 일단 실행되도록 수정)
AVAILABLE_VOICES = []
if VOICE_PRETRAINED_DIR.exists() and VOICE_PRETRAINED_DIR.is_dir():
    AVAILABLE_VOICES = [d.name for d in VOICE_PRETRAINED_DIR.iterdir() if d.is_dir()]

if not AVAILABLE_VOICES:
    print("⚠️  Warning: No local RVC voice presets found. Using default OpenAI voices.")
    # 로컬 보이스가 없을 경우, OpenAI 기본 보이스로 대체
    AVAILABLE_VOICES = ["nova", "IU", "KARINA", "ENIME", "Puth"]


# ────────────────── 유니티 연동을 위한 전역 변수 및 Flask 설정 ─────────
current_voice_name = AVAILABLE_VOICES[0]  # 기본값으로 첫 번째 목소리 설정
app = Flask(__name__)

def run_flask_server():
    app.run(host='127.0.0.1', port=5001, debug=False)

@app.route('/set_voice/<voice_name>')
def set_voice(voice_name):
    global current_voice_name
    # 로컬 보이스 목록 또는 OpenAI 기본 보이스 목록에 있는지 확인
    if voice_name in AVAILABLE_VOICES:
        current_voice_name = voice_name
        print(f"✅ Voice changed to: {current_voice_name}")
        return f"Successfully set voice to {current_voice_name}"
    else:
        # 대소문자 무시하고 비교
        for available_voice in AVAILABLE_VOICES:
            if voice_name.lower() == available_voice.lower():
                current_voice_name = available_voice
                print(f"✅ Voice changed to: {current_voice_name}")
                return f"Successfully set voice to {current_voice_name}"
        
        print(f"Input is Basic or  Unknown voice: {voice_name}")
        return f"Set voice to Basic model : {current_voice_name}"

# [추가] TCP 소켓 통신을 위한 설정
TCP_HOST = '127.0.0.1'
TCP_PORT = 9999
unity_conn = None # 유니티와의 TCP 연결을 저장할 전역 변수

# [추가] Unity의 데이터 수신 연결을 처리할 TCP 서버 함수
def run_tcp_server():
    """Unity 클라이언트의 연결을 수락하고 전역 변수에 저장하는 서버"""
    global unity_conn
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((TCP_HOST, TCP_PORT))
    server_socket.listen()
    print(f"✅ TCP Server for Unity is running on {TCP_HOST}:{TCP_PORT}")
    print("Waiting for Unity data client to connect...")
    unity_conn, _ = server_socket.accept()
    print("🔗 Unity data client connected!")

# ────────────────── Emotion&LLM Response To Unity 함수 ─────────────────────────────
def info_sent_to_unity(emotion: str, probability: float, response: str):
    global unity_conn
    """emotion, probability, response 데이터를 JSON으로 묶어 Unity에 전송합니다."""
    if unity_conn is None:
        print("❌ Unity가 TCP 소켓에 연결되지 않았습니다.")
        return

    try:
        # [수정] data_to_send 딕셔너리에 'probability' 항목 추가
        data_to_send = {
            "emotion": emotion,
            "probability": probability,
            "response": response
        }
        
        json_string = json.dumps(data_to_send) + "\n"
        unity_conn.sendall(json_string.encode('utf-8'))
        print(f"전송 완료: {json_string.strip()}")

    except (ConnectionResetError, BrokenPipeError):
        print("❌ Unity TCP 연결이 끊어졌습니다.")
        unity_conn = None
    except Exception as e:
        print(f"전송 중 에러 발생: {e}")



# ────────────────── 오디오 유틸리티 함수 ───────────────────────────

def record_microphone() -> Path:
    """엔터 → 녹음 시작, 엔터 → 종료 후 temp wav 파일 반환"""
    q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n🎙️  Press Enter to start recording, and Enter again to stop.")
    input()  # 첫 번째 Enter 대기
    print("Recording…")
    
    stream = sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="int16", callback=_callback)
    with stream:
        input() # 두 번째 Enter 대기
    
    print("Recording stopped. Processing…")

    if q.empty():
        raise RuntimeError("No audio captured.")
    
    audio_np = np.concatenate([q.get() for _ in range(q.qsize())], axis=0)
    
    # 임시 파일 생성
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
<<<<<<< HEAD
    print(f"녹음된거 path: {path}")
    wav_write(path, AUDIO_SAMPLE_RATE, audio_np)
    return Path(path)


def transcribe_audio(wav_path: Path) -> str:
    """오디오 파일 경로를 받아 텍스트로 변환"""
    audio_data = None
    with wave.open(str(wav_path), 'rb') as wf:
        # 오디오 파일 정보 가져오기
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # 프레임 읽기
        frames = wf.readframes(n_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)
        
    if audio_data is not None:
        # stt_model.forward의 입력은 일반적으로 NumPy 배열 또는 텐서 형태입니다.
        # `openai/whisper-small` 모델의 경우, 일반적으로 16kHz 모노 오디오를 기대합니다.
        # `audio_data` 변수가 이 모델의 입력 형식에 맞아야 합니다.
        transcription = stt_model.forward(audio_data)
        print("변환성공")
        return transcription
    else:
        print("변환실패")
        return None
    

def llama_response(prompt: str, history: list[dict]) -> str:
    """GPT 모델에 프롬프트를 보내고 응답을 받음"""
    answer = llm_model.forward(prompt)
    return answer

def text_to_speech(text: str,speed: float = 1.0) -> Path:
    print("tts함수시작")
    output_file_path = "output.wav"
    synthesis = tts_model.forward(text, output_file_path)
    tts_model.model.synthesizer.save_wav(wav=synthesis, path=output_file_path)
    return Path(output_file_path)
=======
    wav_write(path, AUDIO_SAMPLE_RATE, audio_np)
    return Path(path)

def transcribe_audio(wav_path: Path) -> str:
    """오디오 파일 경로를 받아 텍스트로 변환"""
    with wav_path.open("rb") as f:
        return client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f, response_format="text").strip()

def gpt_response(prompt: str, history: list[dict]) -> str:
    """GPT 모델에 프롬프트를 보내고 응답을 받음"""
    history.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=GPT_MODEL, messages=history, temperature=0.7)
    answer = resp.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": answer})
    return answer

TTS_VOICE   = "nova" 

def text_to_speech(text: str,speed: float = 1.0) -> Path:
    """텍스트를 음성으로 변환하고 임시 파일 경로 반환"""
    resp = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
        speed=speed,
        timeout=httpx.Timeout(20.0, connect=5.0) 
    )
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    resp.stream_to_file(path)
    return Path(path)
>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea

def play_audio(file_path: Path):
    """주어진 경로의 오디오 파일을 재생"""
    try:
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


# ────────────────── 헬퍼: numpy → 지정 경로 wav ────────────────
import time
def numpy_to_wav_in_cache(audio_np: np.ndarray, sr: int) -> Path:
    """wav_cache/rvc_<타임스탬프>.wav 로 저장 후 Path 반환"""
    
    # 고유한 파일 이름을 만들기 위해 현재 시간을 이용합니다.
    timestamp = int(time.time() * 1000)
    file_name = f"rvc_{timestamp}.wav" # 예: rvc_1686835200123.wav
    
    path = OUTPUT_WAV_DIR / file_name
    sf.write(path, audio_np, sr)
    return path


    


# ────────────────── rvc Voice Conversion 함수 ─────────────────────────────

def _rvc_pipeline(tts_path: Path, voice_name: str) -> Path:
    """TTS wav → numpy → RVC → temp wav Path 반환"""
    wav_np, sr = sf.read(tts_path, dtype="float32")
    converted_np, converted_sr = run_voice_conversion(
        src_audio=wav_np,
        src_sr=sr,
        transpose=0,
        voice_name=voice_name,
    )
    return numpy_to_wav_in_cache(converted_np, converted_sr)



# ────────────────── V2V 파이프라인 ─────────────────────────────

def build_voice_reply_audio(wav_path: Path, history: list[dict], voice_name: str) -> Path:
    """음성 입력 → STT → GPT → TTS (RVC는 비활성화)"""
<<<<<<< HEAD
    #voice to text
=======
>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea
    print("1. Transcribing user audio...")
    user_text = transcribe_audio(wav_path)
    print(f"   > User said: {user_text}")


    print("1.5 Predict emotion...")
    emotion_result, emotion_probability  = predict_emotion(wav_path)
    print(f"   > emotion_results: {emotion_result, emotion_probability}")

<<<<<<< HEAD
    #gpt
    print("2. Getting response from GPT...")
    assistant_ko = llama_response(user_text, history)
=======
    print("2. Getting response from GPT...")
    assistant_ko = gpt_response(user_text, history)
>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea
    print(f"   > GPT responds: {assistant_ko}")
    
    print("2.5 Send Emotion and Response To Unity...")
    info_sent_to_unity(emotion_result, emotion_probability, assistant_ko)
    
    
    print(f"3. Converting text to speech with '{voice_name}' voice...")
    # RVC를 사용하지 않으므로, 유니티에서 받은 voice_name을 바로 TTS에 사용
    assistant_en = romanize_korean(assistant_ko)
    tts_path = text_to_speech(assistant_en, speed=0.9)
    
<<<<<<< HEAD
    

=======
>>>>>>> a6732d87576fed2a8e00dedfdf8f7b7a187b1bea
    if voice_name == "Basic":
        return tts_path
    else:
        # _rvc_pipeline 기능 잠시 중지
        return _rvc_pipeline(tts_path, voice_name)
        
        #return tts_path
        


# ────────────────── main() ─────────────────────────────────────
def main():
    # 1. 초기화 (기존 코드의 필수 초기화 로직 포함)
    initialize_directories()  # 필요하다면 이 함수의 주석을 해제하세요.
    wav_dir = Path.cwd() / "wav_input"
    ensure_wav_input_folder_exists(str(wav_dir)) # 필요하다면 이 함수의 주석을 해제하세요.

    ## 두 개의 서버를 각각 다른 스레드에서 실행
    flask_thread = threading.Thread(target=run_flask_server, daemon=True)
    flask_thread.start()
    
    tcp_thread = threading.Thread(target=run_tcp_server, daemon=True)
    tcp_thread.start()
    
     # --- [수정/추가] Unity TCP 클라이언트가 연결될 때까지 메인 스레드 대기 ---
    print("⏳ Waiting for Unity TCP client to connect on port 9999...")
    while unity_conn is None:
        time.sleep(0.5) # 0.5초 간격으로 확인 (파일 상단에 import time 추가 필요)
    print("✅ Unity TCP client is connected!")
    
    # 3D 얼굴 및 통신 초기화 (기존 코드에서 가져옴)
    py_face     = initialize_py_face()
    socket_conn = create_socket_connection()

    # 기본 애니메이션 스레드 시작 (기존 코드에서 가져옴)
    anim_th = Thread(target=default_animation_loop, args=(py_face, stop_default_animation), daemon=True)
    anim_th.start()
    
    
    
    # 2. 서버 실행 안내 메시지 출력
    print("="*50)
    print("🚀 Python Backend Server is running on http://127.0.0.1:5001")
    print(f"Default voice is set to: {current_voice_name}")
    print("Ready to receive voice change commands from Unity.")
    print("Starting microphone listening loop...")
    print("="*50)

    # 3. GPT 대화 기록 초기화
    history = [{
        "role": "system",
        "content": "You are a helpful assistant. 답변은 **한글**로, 최대 100자로 제한해.",
    }]

    # 4. 메인 루프 실행
    while True:
        assistant_audio_path = None
        user_audio_path = None
        
        try:
            # 마이크로 녹음
            user_audio_path = record_microphone()
            
            # 현재 설정된 목소리(current_voice_name)로 음성 처리
            assistant_audio_path = build_voice_reply_audio(user_audio_path, history, current_voice_name)
            
            # <<< 중요! 이 부분이 누락되었습니다.
            # 생성된 음성 파일을 3D 얼굴로 보내서 재생하고 애니메이션 실행
            print(f"✅ Generated response audio. Processing for 3D face...")
            print("asdasdasd", type(assistant_audio_path),type(py_face), type(socket_conn), type(anim_th))
            process_wav_file(assistant_audio_path, py_face, socket_conn, anim_th)
            
        except RuntimeError as e:
            print(f"Audio capture error: {e}")
        except KeyboardInterrupt:
            print("\n⏹️  Exiting microphone loop.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            # 임시 파일 정리 (변수명 수정)
            if user_audio_path and user_audio_path.exists():
                with suppress(PermissionError): user_audio_path.unlink(missing_ok=True)
            if assistant_audio_path and assistant_audio_path.exists():
                with suppress(PermissionError): assistant_audio_path.unlink(missing_ok=True)
            
    # 5. 종료 처리 로직 (기존 코드에서 가져옴)
    print("Shutting down...")
    stop_default_animation.set()
    anim_th.join(timeout=2)
    pygame.quit()
    socket_conn.close()
    
    
if __name__ == "__main__":
    main()
