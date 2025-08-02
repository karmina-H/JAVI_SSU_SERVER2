# send_to_unreal.py
# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import time
from typing import List
import struct, time, audioop
from threading import Thread, Event, Lock
from livelink.connect.livelink_init import create_socket_connection, FaceBlendShape
from livelink.animations.default_animation import default_animation_data
from livelink.animations.blending_anims import (
    generate_blend_frames,
    combine_frame_streams,
    FAST_BLENDSHAPES,
    default_animation_state,
)


SEND_LOCK = Lock()


def pre_encode_facial_data(facial_data: list, py_face, fps: int = 60, smooth: bool = False) -> list:
    """
    Encodes the full stream:
    1. Blend-IN (idle → capture)
    2. Main captured frames
    3. Blend-OUT (capture → idle **frame 0**)

    Returns
    -------
    encoded_data : list[bytes]
        Ready-to-send UDP packets.
    """
    encoded_data = []
    apply_blink_to_facial_data(facial_data, default_animation_data)

    total_duration = len(facial_data) / fps
    slow_duration  = 0.3 if total_duration < 1.0 else 0.5
    if total_duration < 0.5:
        slow_duration = 0.2

    fast_duration  = 0.1                    # jaw/mouth quick ease
    slow_blend_frames = int(slow_duration * fps)

    fast_blend_in = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        FAST_BLENDSHAPES, mode='in', active_duration_sec=fast_duration
    )

    slow_blend_in = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        set(range(51)) - FAST_BLENDSHAPES, mode='in'
    )

    blend_in_frames = combine_frame_streams(slow_blend_in, fast_blend_in, FAST_BLENDSHAPES)

    for frame in blend_in_frames:
        for i in range(51):
            py_face.set_blendshape(FaceBlendShape(i), frame[i])
        encoded_data.append(py_face.encode())

    main_start = slow_blend_frames
    main_end   = len(facial_data) - slow_blend_frames

    for frame_data in facial_data[main_start:main_end]:
        for i in range(51):
            py_face.set_blendshape(FaceBlendShape(i), frame_data[i])
        encoded_data.append(py_face.encode())

    default_animation_state['current_index'] = 0

    fast_blend_out = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        FAST_BLENDSHAPES, mode='out', active_duration_sec=fast_duration,
        default_start_index=0              
    )

    slow_blend_out = generate_blend_frames(
        facial_data, slow_blend_frames, default_animation_data, fps,
        set(range(51)) - FAST_BLENDSHAPES, mode='out',
        default_start_index=0                                      
    )

    blend_out_frames = combine_frame_streams(slow_blend_out, fast_blend_out, FAST_BLENDSHAPES)

    for frame in blend_out_frames:
        for i in range(51):
            py_face.set_blendshape(FaceBlendShape(i), frame[i])
        encoded_data.append(py_face.encode())

    return encoded_data


def apply_blink_to_facial_data(facial_data: List, default_animation_data: List[List[float]]):
    """
    Updates each frame in facial_data in-place by setting the blink indices (EyeBlinkLeft, EyeBlinkRight)
    to the values from default_animation_data. This ensures that the blink values are present before any blending.
    """
    blink_indices = {FaceBlendShape.EyeBlinkLeft.value, FaceBlendShape.EyeBlinkRight.value}
    default_len = len(default_animation_data)
    for idx, frame in enumerate(facial_data):
        default_idx = idx % default_len
        for blink_idx in blink_indices:
            if blink_idx < len(frame):
                frame[blink_idx] = default_animation_data[default_idx][blink_idx]


def smooth_facial_data(facial_data: list) -> list:
    if len(facial_data) < 2:
        return facial_data.copy()  

    smoothed_data = [facial_data[0]]
    for i in range(1, len(facial_data)):
        previous_frame = facial_data[i - 1]
        current_frame = facial_data[i]
        averaged_frame = [(a + b) / 2 for a, b in zip(previous_frame, current_frame)]
        smoothed_data.append(averaged_frame)
    
    return smoothed_data


'''def send_pre_encoded_data_to_unreal(encoded_facial_data: List[bytes], start_event, fps: int, socket_connection=None):
    try:
        own_socket = False
        if socket_connection is None:
            socket_connection = create_socket_connection()
            own_socket = True

        start_event.wait()  
        frame_duration = 1 / fps  
        start_time = time.time()  

        for frame_index, frame_data in enumerate(encoded_facial_data):
            current_time = time.time()
            elapsed_time = current_time - start_time
            expected_time = frame_index * frame_duration 
            if elapsed_time < expected_time:
                time.sleep(expected_time - elapsed_time)
            elif elapsed_time > expected_time + frame_duration:
                continue

            socket_connection.sendall(frame_data)  

    except KeyboardInterrupt:
        pass
    finally:
        if own_socket:
            socket_connection.close()
'''


import struct
import time
import socket

def send_pre_encoded_data_to_unreal(encoded_facial_data, start_event, fps, socket_connection):
    """
    미리 인코딩된 얼굴 데이터를 TCP 스트림으로 Unity에 전송합니다.
    (기존 코드와 동일 - 이미 TCP에 최적화되어 있음)
    """
    if not socket_connection:
        print("❌ 소켓 연결이 없어 데이터를 전송할 수 없습니다.")
        return

    start_event.wait()
    frame_interval = 1.0 / fps

    try:
        for frame_data in encoded_facial_data:
            start_time = time.time()
            
            # 1. 메시지 프레이밍: 데이터 길이를 4바이트 빅 엔디안 정수로 패킹 (헤더)
            header = struct.pack('>I', len(frame_data))
            
            # 2. 메시지 생성: 헤더 + 실제 데이터
            message = header + frame_data
            
            # 3. 데이터 전송 (sendall은 모든 데이터가 보내질 때까지 블록)
            with SEND_LOCK:
                socket_connection.sendall(message)

            elapsed_time = time.time() - start_time
            sleep_time = frame_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except socket.error as e:
        print(f"❌ 데이터 전송 중 에러 발생: {e}. Unity와 연결이 끊겼을 수 있습니다.")
    except Exception as e:
        print(f"❌ 알 수 없는 에러 발생: {e}")
    finally:
        print("▶️ 오디오 애니메이션 전송 완료.")
   
AUDIO_PACKET_MAGIC = b"PCM_"          # 4 바이트 헤더
AUDIO_CHUNK_MS     = 40               # 40 ms 단위(= 640 샘플 @16 kHz, 모노·int16)     
TARGET_SR          = 16000
TARGET_CH          = 1
SAMPLE_WIDTH       = 2   # int16
       
def _parse_wav_header(wav_bytes: bytes):
    """
    매우 일반적인 44바이트 PCM WAV 헤더를 파싱합니다.
    (fmt/data 추가 청크가 있는 특수 WAV는 별도 파서가 필요)
    """
    if len(wav_bytes) < 44:
        raise ValueError("WAV bytes too short")

    # 채널(2바이트, 오프셋 22), 샘플레이트(4바이트, 오프셋 24), 비트심도(2바이트, 오프셋 34)
    channels    = int.from_bytes(wav_bytes[22:24], "little")
    sample_rate = int.from_bytes(wav_bytes[24:28], "little")
    bits        = int.from_bytes(wav_bytes[34:36], "little")

    # data 청크가 44바이트부터 바로 시작한다고 가정
    data = wav_bytes[44:]
    return channels, sample_rate, bits, data


def send_audio_data_to_unreal(wav_bytes: bytes,
                              start_event,
                              socket_connection,
                              sample_rate: int = TARGET_SR,
                              channels: int = TARGET_CH):
    if not socket_connection:
        print("❌ 소켓 연결이 없어 오디오를 전송할 수 없습니다.")
        return

    in_ch, in_sr, in_bits, pcm = _parse_wav_header(wav_bytes)
    if in_ch == 2:
        pcm = audioop.tomono(pcm, SAMPLE_WIDTH, 0.5, 0.5); in_ch = 1
    if in_sr != TARGET_SR:
        pcm, _ = audioop.ratecv(pcm, SAMPLE_WIDTH, in_ch, in_sr, TARGET_SR, None)

    bytes_per_ms = TARGET_SR * TARGET_CH * SAMPLE_WIDTH / 1000.0
    chunk_size   = int(bytes_per_ms * AUDIO_CHUNK_MS)

    start_event.wait()

    chunk_idx = 0
    for pos in range(0, len(pcm), chunk_size):
        chunk  = pcm[pos: pos + chunk_size]
        header = AUDIO_PACKET_MAGIC + struct.pack("<II", chunk_idx, len(chunk))
        with SEND_LOCK:
            socket_connection.sendall(header + chunk)
        chunk_idx += 1
        # 전송 시간에 맞춰 sleep
        time.sleep(len(chunk) / float(TARGET_SR * TARGET_CH * SAMPLE_WIDTH))

    # ★★★ EOS: 데이터 길이 0으로 종료 신호 전송 ★★★
    eos_header = AUDIO_PACKET_MAGIC + struct.pack("<II", chunk_idx, 0)
    with SEND_LOCK:
        socket_connection.sendall(eos_header)