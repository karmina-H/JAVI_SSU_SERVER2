# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain a to use this software commercially.

# # livelink_init.py

import socket
from livelink.connect.pylivelinkface import PyLiveLinkFace, FaceBlendShape
import time
import struct

#UDP_IP = "127.0.0.1"
UDP_IP = "0.0.0.0"
UDP_PORT = 11111

def create_socket_connection():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((UDP_IP, UDP_PORT))
    return s

TCP_IP = "0.0.0.0"  # Unity와 같은 PC에서 테스트 시 "127.0.0.1" 사용

TCP_PORT = 5002  # Unity와 통신할 별도의 TCP 포트

# ────────────────── [수정된 핵심 함수] ───────────────────────────────────

def create_tcp_connection(host='0.0.0.0', port=TCP_PORT):
    """
    Unity 클라이언트의 연결을 기다리는 TCP 서버 소켓을 생성하고 반환합니다.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR 옵션을 설정하여 서버 재시작 시 주소 재사용 문제를 방지합니다.
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"🔌 Unity 연결 대기 중... (IP: {host}, Port: {port})")

    try:
        # Unity 클라이언트가 연결될 때까지 여기서 대기합니다.
        client_socket, addr = server_socket.accept()
        print(f"✅ Unity 클라이언트 연결 성공! (from: {addr})")
        return client_socket
    except Exception as e:
        print(f"❌ Unity 연결 대기 중 에러 발생: {e}")
        return None
    finally:
        # 한 클라이언트만 받으므로, 연결 후 서버 리스닝 소켓은 닫아도 무방합니다.
        # 여러 클라이언트를 받으려면 로직 수정이 필요합니다.
        server_socket.close()


def initialize_py_face():
    py_face = PyLiveLinkFace()
    initial_blendshapes = [0.0] * 61
    for i, value in enumerate(initial_blendshapes):
        py_face.set_blendshape(FaceBlendShape(i), float(value))
    return py_face
