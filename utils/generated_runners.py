# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.


from threading import Thread, Event, Lock
import numpy as np
import random

from utils.audio.play_audio import play_audio_from_path, play_audio_from_memory
from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal, send_audio_data_to_unreal
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from livelink.connect.livelink_init import initialize_py_face 
from livelink.animations.animation_emotion import determine_highest_emotion,  merge_emotion_data_into_facial_data_wrapper
from livelink.animations.animation_loader import emotion_animations


import struct
import time






queue_lock = Lock()

from threading import Thread, Event
import numpy as np, random

# --- 이미 기존 코드에 있는 import ---
from utils.audio.play_audio import play_audio_from_path, play_audio_from_memory
from livelink.send_to_unreal import (
    pre_encode_facial_data,
    send_pre_encoded_data_to_unreal
)
from livelink.animations.default_animation import default_animation_loop
from livelink.connect.livelink_init import initialize_py_face
from livelink.animations.animation_emotion import (
    determine_highest_emotion,
    merge_emotion_data_into_facial_data_wrapper
)
from livelink.animations.animation_loader import emotion_animations

def run_audio_animation(audio_input,
                        generated_facial_data,
                        py_face,
                        socket_connection,
                        default_animation_thread,
                        stop_default_animation_event):

    #────────────────── 1. 모션 데이터 준비 ──────────────────#
    if (generated_facial_data is not None and
        len(generated_facial_data) > 0 and
        len(generated_facial_data[0]) > 61):

        if isinstance(generated_facial_data, np.ndarray):
            generated_facial_data = generated_facial_data.tolist()

        dominant = determine_highest_emotion(np.array(generated_facial_data))
        if dominant in emotion_animations and emotion_animations[dominant]:
            sel_anim = random.choice(emotion_animations[dominant])
            generated_facial_data = merge_emotion_data_into_facial_data_wrapper(
                                        generated_facial_data, sel_anim)

    encoding_face     = initialize_py_face()
    encoded_face_data = pre_encode_facial_data(generated_facial_data, encoding_face)

    #────────────────── 2. 기존 기본 애니 중단 ───────────────#
    stop_default_animation_event.set()
    if default_animation_thread and default_animation_thread.is_alive():
        default_animation_thread.join(timeout=2.0)

    #────────────────── 3. 동기 이벤트 & 스레드 시작 ─────────#
    start_evt = Event()

    # 3-A. 스피커 재생 스레드
    if isinstance(audio_input, bytes):
        play_thr = Thread(target=play_audio_from_memory,
                          args=(audio_input, start_evt),
                          daemon=True)
        raw_audio = audio_input
    else:
        play_thr = Thread(target=play_audio_from_path,
                          args=(audio_input, start_evt),
                          daemon=True)
        with open(audio_input, "rb") as f:
            raw_audio = f.read()

    # 3-B. 모션 데이터 전송
    face_thr = Thread(target=send_pre_encoded_data_to_unreal,
                      args=(encoded_face_data, start_evt, 60, socket_connection),
                      daemon=True)

    # 3-C. 오디오 PCM 전송
    audio_send_thr = Thread(target=send_audio_data_to_unreal,
                            args=(raw_audio, start_evt, socket_connection),
                            daemon=True)

    #────────────── 스레드 런 & 동기화 ──────────────#
    play_thr.start()
    face_thr.start()
    audio_send_thr.start()
    start_evt.set()                    # 세 스레드 동시에 시작

    play_thr.join()
    face_thr.join()
    audio_send_thr.join()

    #────────────────── 4. 기본 애니 재시작 ─────────────────#
    stop_default_animation_event.clear()
    default_animation_thread = Thread(target=default_animation_loop,
                                      args=(py_face, socket_connection, stop_default_animation_event),
                                      daemon=True)
    default_animation_thread.start()


