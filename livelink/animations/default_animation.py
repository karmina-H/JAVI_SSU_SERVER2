# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import time
import socket
import pandas as pd
from threading import Event

from livelink.connect.livelink_init import FaceBlendShape, UDP_IP, UDP_PORT
from livelink.animations.blending_anims import blend_animation_start_end
from livelink.animations.blending_anims import default_animation_state, blend_animation_start_end

def load_animation(csv_path):
    data = pd.read_csv(csv_path)

    data = data.drop(columns=['Timecode', 'BlendshapeCount'])
    # zero'ing eyes so they match the generation position, do some eye control from Unreal or manually.
    cols_to_zero = [1, 2, 3, 4, 8, 9, 10, 11]
    cols_to_zero = [i for i in cols_to_zero if i < data.shape[1]] 
    data.iloc[:, cols_to_zero] = 0.0

    return data.values
# ==================== DEFAULT ANIMATION SETUP ====================

# Path to the default animation CSV file
ground_truth_path = r"livelink/animations/default_anim/default.csv"

# Load the default animation data
default_animation_data = load_animation(ground_truth_path)

# Create the blended default animation data
default_animation_data = blend_animation_start_end(default_animation_data, blend_frames=16)

# Event to signal stopping of the default animation loop
stop_default_animation = Event()

"""def default_animation_loop(py_face):
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((UDP_IP, UDP_PORT))
        while not stop_default_animation.is_set():
            for idx, frame in enumerate(default_animation_data):
                if stop_default_animation.is_set():
                    break
                # update shared state
                default_animation_state['current_index'] = idx

                for i, value in enumerate(frame):
                    py_face.set_blendshape(FaceBlendShape(i), float(value))
                try:
                    s.sendall(py_face.encode())
                except Exception as e:
                    print(f"Error in default animation sending: {e}")

                # maintain 60fps
                total_sleep = 1 / 60
                sleep_interval = 0.005
                while total_sleep > 0 and not stop_default_animation.is_set():
                    time.sleep(min(sleep_interval, total_sleep))
                    total_sleep -= sleep_interval"""


import time
import struct
import socket

# 이 함수가 호출되는 스레드에서 아래 객체들을 접근할 수 있어야 합니다.
# from ... import FaceBlendShape, default_animation_data

def default_animation_loop(py_face, socket_connection, stop_flag):
    """
    [수정됨] 기본 애니메이션을 반복하고, 수립된 TCP 연결을 통해 데이터를 전송합니다.
    """
    # Unity와 TCP 연결이 없으면 함수를 즉시 종료합니다.
    if not socket_connection:
        print("⚠️ 기본 애니메이션 루프: Unity와 연결되지 않아 실행할 수 없습니다.")
        return

    # 60fps를 유지하기 위한 프레임 간격
    frame_interval = 1.0 / 60

    try:
        while not stop_flag.is_set():
            # 기본 애니메이션 데이터의 각 프레임을 순회합니다.
            for frame_values in default_animation_data:
                if stop_flag.is_set():
                    break

                start_time = time.time()

                # 1. py_face 객체에 블렌드셰이프 값 설정
                for i, value in enumerate(frame_values):
                    # FaceBlendShape(i)는 예시이며, 실제 enum/클래스에 맞게 사용해야 합니다.
                    py_face.set_blendshape(FaceBlendShape(i), float(value))

                # 2. 데이터를 전송용으로 인코딩
                encoded_data = py_face.encode()

                # 3. TCP 메시지 프레이밍 (길이 헤더 + 데이터)
                header = struct.pack('>I', len(encoded_data))
                message = header + encoded_data

                # 4. 수립된 TCP 소켓으로 데이터 전송
                socket_connection.sendall(message)

                # 5. 60fps 유지를 위한 딜레이 계산 및 적용
                elapsed_time = time.time() - start_time
                sleep_time = frame_interval - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except (socket.error, BrokenPipeError) as e:
        # 연결이 끊겼을 때 흔히 발생하는 예외들
        print(f"❌ 기본 애니메이션 전송 중 연결 에러 발생: {e}. 루프를 종료합니다.")
    except Exception as e:
        print(f"❌ 기본 애니메이션 루프 중 알 수 없는 에러 발생: {e}")
    finally:
        print("▶️ 기본 애니메이션 루프가 중지되었습니다.")
