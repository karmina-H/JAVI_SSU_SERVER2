# -*- coding: utf-8 -*-
"""
livelink_voice_chat.py — Unity 연동 Voice‑to‑Voice GPT Demo
----------------------------------------------------------------
• [최종 수정] 불안정한 TCP 소켓 통신을 제거하고, 안정적인 단일 HTTP 통신 방식으로 변경
• 음성 요청(POST)에 대한 HTTP 응답으로 직접 JSON(감정, GPT 답변)을 반환
• OpenAI SDK ≥ 1.30 호환 (audio 네임스페이스)
"""

# ────────────────── 외부 라이브러리 및 표준 ───────────────────
import os, sys, tempfile, warnings, wave
from pathlib import Path
from threading import Thread, Event
import httpx
import numpy as np
import soundfile as sf
import openai
import pygame
from flask import Flask, request, jsonify
from datetime import datetime
from g2pk import G2p
from utils.voice_conversion.voice_conversion import run_voice_conversion
from utils.romanize_ko import romanize_korean
from utils.audio_face_workers import process_wav_file
from livelink.connect.livelink_init import initialize_py_face, create_socket_connection, create_tcp_connection
from livelink.animations.default_animation import default_animation_loop
from utils.emotion_recognition.predict_emotion import predict_emotion
import socket
import time
import torch
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

if not llm_model.exists():
    print(f"Invalid ollama model")
    exit()
    
# ────────────────── 설정값 ─────────────────────────────────────
TTS_MODEL = "tts-1-hd"
TRANSCRIBE_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o-mini"
AUDIO_SAMPLE_RATE = 16_000
OUTPUT_WAV_DIR = Path.cwd() / "wav_cache"
OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)


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

# ────────────────── Voice preset discovery ────────────────────
BASE_DIR = Path(__file__).resolve().parent
VOICE_PRETRAINED_DIR = BASE_DIR / "utils" / "voice_conversion" / "pretrained"
AVAILABLE_VOICES = ["MaleYoung", "MaleOld", "FemaleYoung", "FemaleOld", "basic"]


# ────────────────── 전역 변수 및 Flask 설정 ─────────
current_voice_name = AVAILABLE_VOICES[0]
app = Flask(__name__)

history = [{"role": "system", "content": "You are a helpful assistant. 답변은 **한글**로, 최대 100자로 제한해."}]

# [삭제] TCP 소켓 관련 전역 변수 제거
# py_face, socket_conn, anim_th 등 3D 얼굴 관련 변수는 유지합니다.
py_face = None
socket_conn = None
anim_th = None

# ────────────────── Unity 연동 API ─────────────────────────────
@app.route('/set_voice/<voice_name>')
def set_voice(voice_name):
    global current_voice_name
    for available_voice in AVAILABLE_VOICES:
        if voice_name.lower() == available_voice.lower():
            current_voice_name = available_voice
            print(f"✅ Voice changed to: {current_voice_name}")
            return f"Successfully set voice to {current_voice_name}"
    print(f"Input is Basic or Unknown voice: {voice_name}")
    current_voice_name = "Basic"
    return f"Set voice to Basic model: {current_voice_name}"


# ────────────────── [삭제] TCP 서버 관련 함수 전체 제거 ───────────────────
# def run_tcp_server(): ...
# def info_sent_to_unity(...): ...



# llama 모델 적용
def llama_response(prompt: str, history: list[dict]) -> str:
    """GPT 모델에 프롬프트를 보내고 응답을 받음"""
    answer = llm_model.forward(prompt)
    return answer



# ────────────────── 오디오 유틸리티 및 V2V 파이프라인 ───────────────────
def transcribe_audio(wav_path: Path) -> str:
    with wav_path.open("rb") as f:
        return client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f, response_format="text").strip()

def gpt_response(prompt: str, history: list[dict]) -> str:
    history.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=GPT_MODEL, messages=history, temperature=0.7)
    answer = resp.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": answer})
    return answer

TTS_VOICE = "nova"
def text_to_speech(text: str, speed: float = 1.0) -> Path:
    resp = client.audio.speech.create(
        model=TTS_MODEL, voice=TTS_VOICE, input=text,
        response_format="wav", speed=speed, timeout=httpx.Timeout(20.0, connect=5.0)
    )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    resp.stream_to_file(path)
    return Path(path)

def text_to_speech_llama(text: str) -> Path:
    if not tts_model:
        raise RuntimeError("TTS 모델이 초기화되지 않았습니다.")

    print(f'🗣️ 로컬 TTS 모델 호출: "{text}"')

    # 1) 합성
    audio = tts_model.forward(text=text, output_filepath="")  # forward가 파일을 쓰지 않는다면 그대로

    # 2) numpy 1D float32 [-1,1]로 정리
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    # 안전 클리핑
    audio = np.clip(audio, -1.0, 1.0)

    # 3) 샘플레이트: 모델에서 얻기
    # Coqui TTS(api) 인스턴스에 보통 아래 중 하나가 있습니다.
    sr = getattr(tts_model.model, "output_sample_rate", None) \
         or getattr(getattr(tts_model.model, "synthesizer", None), "output_sample_rate", None) \
         or 24000  # 최후의 수단

    # 4) 파일로 저장 (PCM16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = Path(f.name)

    sf.write(out_path.as_posix(), audio, sr, subtype="PCM_16")

    print(f"🎵 오디오가 '{out_path}' 파일로 저장되었습니다. (sr={sr}, len={len(audio)/sr:.2f}s)")
    return out_path



def numpy_to_wav_in_cache(audio_np: np.ndarray, sr: int) -> Path:
    timestamp = int(time.time() * 1000)
    file_name = f"rvc_{timestamp}.wav"
    path = OUTPUT_WAV_DIR / file_name
    sf.write(path, audio_np, sr)
    return path

def _rvc_pipeline(tts_path: Path, voice_name: str) -> Path:
    wav_np, sr = sf.read(tts_path, dtype="float32")
    converted_np, converted_sr = run_voice_conversion(
        src_audio=wav_np, src_sr=sr, transpose=0, voice_name=voice_name
    )
    return numpy_to_wav_in_cache(converted_np, converted_sr)

# [수정] build_voice_reply_audio 함수가 (오디오 경로, JSON 데이터) 튜플을 반환하도록 변경
def build_voice_reply_audio(wav_path: Path, history: list[dict], voice_name: str) -> tuple[Path, dict]:
    print("1. Transcribing user audio...")
    user_text = transcribe_audio(wav_path)
    print(f"   > User said: {user_text}")

    print("1.5 Predict emotion...")
    emotion_result, emotion_probability = predict_emotion(wav_path)
    print(f"   > emotion_results: {emotion_result, emotion_probability}")

    #print("2. Getting response from GPT...")
    #assistant_ko = gpt_response(user_text, history)
    
    print("2. Getting response from Llama...")
    assistant_ko = llama_response(user_text, history)
    
    
    assistant_en = romanize_korean(assistant_ko)
    
    print(f"   > GPT responds: {assistant_ko}")
    
    # [삭제] TCP로 데이터를 보내는 함수 호출 제거
    # info_sent_to_unity(...)

    # [추가] Unity 클라이언트로 반환할 JSON 데이터를 생성
    response_data = {
        "emotion": emotion_result,
        "probability": emotion_probability,
        "response": assistant_ko
    }
    
    print(f"3. Converting text to speech with '{voice_name}' voice...")
    #tts_path = text_to_speech(assistant_en, speed=0.9)
    tts_path = text_to_speech_llama(assistant_ko)
    
    
    final_audio_path = tts_path
    #if voice_name != "Basic":
        #final_audio_path = _rvc_pipeline(tts_path, voice_name)
    
    # 생성된 오디오 파일 경로와 함께 response_data 딕셔너리를 반환
    return final_audio_path, response_data

# [수정] voice_input 엔드포인트가 직접 JSON을 반환하도록 변경
@app.route('/voice_input', methods=['POST'])
def handle_voice_input():
    
    print(f"\n{'🎤'*25} VOICE INPUT 요청 처리 시작 {'🎤'*25}")
    
    client_ip = request.remote_addr
    
    if not request.data:
        print("❌ 에러: 요청에 오디오 데이터가 없습니다.")
        return jsonify({"error": "No audio data in request"}), 400

    print(f"📦 데이터 수신 완료: {len(request.data) / 1024:.1f} KB")

    # 1. 수신된 오디오를 임시 파일로 저장
    try:
        timestamp = int(time.time())
        temp_wav_path = OUTPUT_WAV_DIR / f"received_{timestamp}.wav"
        with open(temp_wav_path, 'wb') as f:
            f.write(request.data)
        print(f"💾 수신된 오디오를 임시 파일로 저장: {temp_wav_path}")
    except Exception as e:
        print(f"❌ 임시 파일 저장 실패: {e}")
        return jsonify({"error": f"Failed to save temporary audio file: {e}"}), 500

    # 2. 핵심! 음성 처리 파이프라인 실행
    try:
        global history, current_voice_name
        
        # build_voice_reply_audio 함수는 이제 (오디오 경로, JSON 데이터)를 반환
        final_audio_path, response_json = build_voice_reply_audio(Path(temp_wav_path), history, current_voice_name)
        
        process_wav_file(final_audio_path, py_face, socket_conn, anim_th, client_ip, stop_default_animation_flag)
        
        print(f"✅ 음성 응답 생성 완료: {final_audio_path}")
        print(f"📤 Unity로 응답할 JSON 데이터: {response_json}")
        
        # [핵심 수정] TCP로 보내는 대신, HTTP 응답으로 JSON 데이터를 직접 반환합니다.
        return jsonify(response_json), 200

    except openai.APIError as e:
        print(f"❌ OpenAI API 에러: {e}")
        return jsonify({"error": f"OpenAI API Error: {e}"}), 500
    except Exception as e:
        import traceback
        print(f"❌ 음성 처리 중 알 수 없는 에러 발생: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred during voice processing: {e}"}), 500
    finally:
        print(f"{'🎤'*25} VOICE INPUT 요청 처리 완료 {'🎤'*25}\n")

@app.route('/')
def index():
    return "서비스가 정상적으로 실행 중입니다"

stop_default_animation_flag = Event() 

def manage_unity_connection_and_animation(host='127.0.0.1', port=5002):
    """
    별도의 스레드에서 Unity 클라이언트의 연결을 기다리고,
    연결 성공 시 애니메이션 루프를 시작하는 함수.
    """
    # 이 함수에서 수정할 전역 변수들을 선언합니다.
    global socket_conn, anim_th

    # TCP 서버 소켓 설정
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"🔌 (스레드) Unity 연결 대기 중... (IP: {host}, Port: {port})")

        # 👇 여기가 핵심! 이 accept()는 이제 별도 스레드에서 실행되므로 메인 프로그램을 막지 않습니다.
        client_socket, addr = server_socket.accept()
        
        # 연결 성공 시 전역 변수에 소켓 할당
        socket_conn = client_socket
        print(f"✅ Unity 클라이언트 연결 성공! (from: {addr})")

        # 3D 얼굴 객체(py_face)가 초기화되었다면 애니메이션 스레드 시작
        if py_face:
            anim_th = Thread(target=default_animation_loop, args=(py_face, socket_conn, stop_default_animation_flag), daemon=True)
            anim_th.start()
            print("✅ 3D 얼굴 기본 애니메이션 스레드 시작!")

    except Exception as e:
        print(f"❌ (스레드) Unity 연결 중 에러 발생: {e}")
        print("⚠️ Unity와 연결되지 않아 3D 얼굴 애니메이션을 비활성화합니다.")
        socket_conn = None # 실패 시 None으로 유지
    finally:
        # 연결 대기 소켓은 닫아줍니다.
        server_socket.close()


# ────────────────── main() ─────────────────────────────────────
def main():
    global py_face, socket_conn, anim_th

    # IP 주소 정보 출력
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("="*60)
    print("🌐 네트워크 정보:")
    print(f"   📍 호스트명: {hostname}")
    print(f"   📍 로컬 IP: {local_ip}")
    print(f"   🌐 접속 URL: http://{local_ip}:5001")
    print("="*60)

    # [삭제] TCP 서버 스레드 시작 코드 제거
    # print("🚀 TCP 서버 시작 중...")
    # tcp_thread = ...
    # tcp_thread.start()
    
    # 3D 얼굴 및 통신 초기화 (선택적 기능)
    try:
        '''print("🎭 3D 얼굴 시스템 초기화 중...")
        py_face = initialize_py_face()
        socket_conn = create_socket_connection()
        stop_default_animation_flag =  Event()
        anim_th = Thread(target=default_animation_loop, args=(py_face, stop_default_animation_flag), daemon=True)
        anim_th.start()
        print("✅ 3D 얼굴 시스템 초기화 완료!")'''
        
        print("🎭 3D 얼굴 시스템 초기화 중...")
        py_face = initialize_py_face()

        # 👇 여기를 수정!
        #socket_conn = create_tcp_connection() # UDP 대신 TCP 연결 함수 호출
        connection_thread = Thread(target=manage_unity_connection_and_animation, daemon=True)
        connection_thread.start()



        # 👇 연결 실패 시 처리
        """if socket_conn is None:
            print("⚠️ Unity와 연결되지 않아 3D 얼굴 애니메이션을 비활성화합니다.")
            py_face = None
            # 프로그램이 계속 실행되어야 한다면 아래 코드는 유지
        else:
            stop_default_animation_flag = Event()
            
            anim_th = Thread(target=default_animation_loop, args=(py_face, socket_conn, stop_default_animation_flag), daemon=True)
            anim_th.start()
            print("✅ 3D 얼굴 시스템 초기화 완료!")"""
        
        
        
    except Exception as e:
        print(f"⚠️ 3D 얼굴 초기화 실패 (Flask 서버는 계속 진행): {e}")
        py_face = socket_conn = anivm_th = None
    
    print("="*50)
    print(f"🚀 Flask 서버 시작 중...")
    print(f"📍 로컬 접속: http://localhost:5001")
    print(f"📍 네트워크 접속: http://{local_ip}:5001")
    print(f"📍 API 엔드포인트: http://{local_ip}:5001/voice_input")
    print(f"🎤 기본 음성: {current_voice_name}")
    print("="*50)

    try:
        app.run(
            host='0.0.0.0', 
            port=5001, 
            debug=True,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"❌ Flask 서버 시작 실패: {e}")
    except KeyboardInterrupt:
        print("\n⏹️ 서버 종료 중...")
    finally:
        print("\n⏹️ 서버 종료 중...")
        if 'stop_default_animation' in globals():
            stop_default_animation_flag.set()
        if anim_th and anim_th.is_alive():
            anim_th.join(timeout=2)
        if socket_conn:
            socket_conn.close()
        print("✅ 서버 종료 완료")


if __name__ == "__main__":
    main()