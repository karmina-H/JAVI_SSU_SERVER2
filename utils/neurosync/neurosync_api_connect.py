# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import requests
import json
from config import NEUROSYNC_API_KEY, NEUROSYNC_REMOTE_URL, NEUROSYNC_LOCAL_URL

def send_audio_to_neurosync(audio_bytes, remote_ip, use_local=True):
    
    try:
        # Use the local or remote URL depending on the flag
        url = NEUROSYNC_LOCAL_URL if use_local else NEUROSYNC_REMOTE_URL
        headers = {}
        if not use_local:
            headers["API-Key"] = NEUROSYNC_API_KEY

        response = post_audio_bytes(audio_bytes, url, headers, remote_ip)
        response.raise_for_status()  
        json_response = response.json()
        print("send_audio_to_neurosync, json_response : ")
        return parse_blendshapes_from_json(json_response)

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None

def validate_audio_bytes(audio_bytes):
    return audio_bytes is not None and len(audio_bytes) > 0

def post_audio_bytes(audio_bytes, url, headers, remote_ip):
    """오디오 바이트와 함께 원격 IP를 헤더에 추가하여 POST 요청을 보냅니다."""
    headers["Content-Type"] = "application/octet-stream"
    
    # << 추가: 'X-Forwarded-For' 헤더에 원격 IP 주소를 추가합니다.
    if remote_ip:
        headers["X-Forwarded-For"] = remote_ip

    print(f"Sending request to {url} with headers: {headers}") # 확인용 로그
    response = requests.post(url, headers=headers, data=audio_bytes)
    print("response : ", response)
    return response

def parse_blendshapes_from_json(json_response):
    blendshapes = json_response.get("blendshapes", [])
    facial_data = []

    for frame in blendshapes:
        frame_data = [float(value) for value in frame]
        facial_data.append(frame_data)

    return facial_data
