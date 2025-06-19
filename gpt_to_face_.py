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

import numpy as np
import soundfile as sf
import openai           # OpenAI Python SDK (>=1.30)
import pygame
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from g2pk import G2p    # 로마자 변환
from utils.voice_conversion.voice_conversion import run_voice_conversion

# ────────────────── 프로젝트 내부 모듈 ─────────────────────────
from utils.files.file_utils import initialize_directories, ensure_wav_input_folder_exists, list_wav_files
from utils.romanize_ko import romanize_korean
from utils.audio_face_workers import process_wav_file
from livelink.connect.livelink_init import initialize_py_face, create_socket_connection
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.emote_sender.send_emote import EmoteConnect

os.environ["OPENAI_API_KEY"] = ""
# ────────────────── 설정값 ─────────────────────────────────────
ENABLE_EMOTE_CALLS = False
TTS_MODEL        = "tts-1"
VOICE            = "nova"
TRANSCRIBE_MODEL = "whisper-1"
GPT_MODEL        = "gpt-4o-mini"
AUDIO_SAMPLE_RATE = 16_000

# ────────────────── 음성 캐시 디렉터리 ─────────────────────────
OUTPUT_WAV_DIR = Path.cwd() / "wav_cache"
OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────── OpenAI 클라이언트 ─────────────────────────
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION"),
)

warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

# ────────────────── Voice preset discovery ────────────────────
VOICE_PRETRAINED_DIR = Path(__file__).resolve().parent / "utils" / "voice_conversion" / "pretrained"
if not VOICE_PRETRAINED_DIR.exists():
    raise FileNotFoundError(f"Voice preset directory not found: {VOICE_PRETRAINED_DIR}")
AVAILABLE_VOICES = [d.name for d in VOICE_PRETRAINED_DIR.iterdir() if d.is_dir()]
if not AVAILABLE_VOICES:
    raise RuntimeError("No voice presets found in pretrained directory.")

def choose_voice() -> str:
    """사용 가능한 voice preset 중 하나를 선택."""
    print("\n🎤  사용할 보이스를 선택하세요:")
    for idx, name in enumerate(AVAILABLE_VOICES, 1):
        print(f"  {idx}) {name}")
    while True:
        sel = input("> ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(AVAILABLE_VOICES):
            return AVAILABLE_VOICES[int(sel) - 1]
        print("❌ 잘못된 선택입니다. 번호를 다시 입력하세요.")

# ────────────────── 오디오 유틸 ────────────────────────────────

def record_microphone() -> Path:
    """엔터 → 녹음 시작, 엔터 → 종료 후 temp wav 파일 반환"""
    q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("\n🎙️  Enter → 시작, Enter → 종료"); input(); print("Recording…")
    with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="int16", callback=_callback):
        input()
    print("Recording stopped. Processing…")

    if q.empty():
        raise RuntimeError("No audio captured.")
    audio_np = np.concatenate([q.get() for _ in range(q.qsize())], axis=0)
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    wav_write(path, AUDIO_SAMPLE_RATE, audio_np)
    return Path(path)


def transcribe_audio(wav_path: Path) -> str:
    with wav_path.open("rb") as f:
        return client.audio.transcriptions.create(model=TRANSCRIBE_MODEL, file=f, response_format="text").strip()


def gpt_response(prompt: str, history: list[dict]) -> str:
    history.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=GPT_MODEL, messages=history, temperature=0.7)
    answer = resp.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": answer})
    return answer


def text_to_speech_(text: str, speed: float = 1.0) -> Path:
    resp = client.audio.speech.create(model=TTS_MODEL, voice=VOICE, input=text, response_format="wav", speed=speed)
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    resp.stream_to_file(path)
    return Path(path)

# ―― TTS ―――――――――――――――――――――――――――――――――――――――――
TTS_MODEL   = "tts-1"
TTS_VOICE   = "nova"     # OpenAI preset voice (중립적, SNR 높음)
TTS_SPEED   = 0.8         # 0.8 ~ 1.0 권장 범위 — 보통 속도
# -------------------------------------------------------------
def text_to_speech(text: str, speed: float = 1.0) -> Path:
    """텍스트 → wav (nova, speed 0.9)"""
    resp = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
        speed=TTS_SPEED,
    )
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    resp.stream_to_file(path)
    return Path(path)

# ────────────────── 헬퍼: numpy → temp wav ────────────────────

def numpy_to_temp_wav(audio_np: np.ndarray, sr: int) -> Path:
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, audio_np, sr)
    return Path(path)


# ────────────────── 헬퍼: numpy → 지정 경로 wav ────────────────

def numpy_to_wav_in_cache(audio_np: np.ndarray, sr: int) -> Path:
    """wav_cache/voice_name_<timestamp>.wav 로 저장 후 Path 반환"""
    
    file_name = f"rvc.wav"
    path = OUTPUT_WAV_DIR / file_name
    sf.write(path, audio_np, sr)
    return path

# ────────────────── V2V 파이프라인 ─────────────────────────────

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


def build_voice_reply_audio(wav_path: Path, history: list[dict], voice_name: str) -> Path:
    """음성 입력 → STT → GPT → 로마자 → TTS → RVC"""
    user_text    = transcribe_audio(wav_path)
    assistant_ko = gpt_response(user_text, history)
    assistant_en = romanize_korean(assistant_ko)
    tts_path     = text_to_speech(assistant_en, speed=0.9)
    return _rvc_pipeline(tts_path, voice_name)


def build_voice_reply_text(user_text: str, history: list[dict], voice_name: str) -> Path:
    """텍스트 입력 → GPT → 로마자 → TTS → RVC"""
    assistant_ko = gpt_response(user_text, history)
    assistant_en = romanize_korean(assistant_ko)
    tts_path     = text_to_speech(assistant_en, speed=0.9)
    
    return _rvc_pipeline(tts_path, voice_name)

# ────────────────── 입력 소스 제너레이터 ──────────────────────

def iterate_audio_inputs(folder: Path):
    while True:
        files = list_wav_files(str(folder))
        if not files:
            print("Put .wav files in 'wav_input' or use --mic."); break
        for idx, f in enumerate(files, 1):
            print(f"{idx}: {f}")
        sel = input("Select number (q to quit): ").strip()
        if sel.lower() in {"q", "quit", "exit"}: break
        if sel.isdigit() and 1 <= int(sel) <= len(files):
            yield Path(folder / files[int(sel)-1])
        else:
            print("Invalid selection.")

def audio_source_iter(wav_folder: Path):
    while True:
        choice = input("\n▶ 입력 모드\n  1) 🎙️  마이크\n  2) 📂 wav 파일\n  3) ✍️  텍스트\n  q) 종료\n> ").strip().lower()
        if choice == "1":
            try:
                yield ("audio", record_microphone())
            except KeyboardInterrupt:
                print("⏹️  취소")
        elif choice == "2":
            for p in iterate_audio_inputs(wav_folder):
                yield ("audio", p)
        elif choice == "3":
            txt = input("📝  텍스트 입력: \n> ").strip()
            if txt:
                yield ("text", txt)
        elif choice in {"q", "quit", "exit"}: break
        else:
            print("❌ 잘못된 입력")

# ────────────────── main() ─────────────────────────────────────

def main():
    initialize_directories()
    wav_dir = Path.cwd() / "wav_input"
    ensure_wav_input_folder_exists(str(wav_dir))

    py_face     = initialize_py_face()
    socket_conn = create_socket_connection()

    anim_th = Thread(target=default_animation_loop, args=(py_face,), daemon=True)
    anim_th.start()

    voice_name = choose_voice()
    print(f"\n✅ 선택된 보이스: {voice_name}\n")

    history = [{
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "답변은 **한글**로, 최대 100자로 제한해."
        ),
    }]

    for input_type, payload in audio_source_iter(wav_dir):
        if ENABLE_EMOTE_CALLS:
            EmoteConnect.send_emote("startspeaking")
        try:
            if input_type == "audio":
                assistant_audio = build_voice_reply_audio(payload, history, voice_name)
            else:
                assistant_audio = build_voice_reply_text(payload, history, voice_name)
            print("asdasdasd", type(assistant_audio),type(py_face), type(socket_conn), type(anim_th))
            process_wav_file(assistant_audio, py_face, socket_conn, anim_th)
        finally:
            if ENABLE_EMOTE_CALLS:
                EmoteConnect.send_emote("stopspeaking")
            # 임시 파일 정리 (assistant_audio는 항상 Path)
            for p in [assistant_audio] if input_type == "text" else [assistant_audio, payload]:
                if isinstance(p, Path) and p.exists() and p.parent == Path(tempfile.gettempdir()):
                    with suppress(PermissionError):
                        p.unlink(missing_ok=True)

    stop_default_animation.set(); anim_th.join(timeout=2)
    pygame.quit(); socket_conn.close()




if __name__ == "__main__":
    main()
