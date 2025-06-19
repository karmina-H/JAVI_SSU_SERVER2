from __future__ import annotations
import sys, pathlib, os, traceback
from typing import Optional, Dict

# --- 1. 안정적인 경로 설정 ---
# 이 스크립트(voice_conversion.py)의 디렉토리
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# 프로젝트 루트 디렉토리 (NeuroSync_Player)
PROJECT_ROOT = SCRIPT_DIR
# 파이썬이 'lib' 패키지를 찾을 수 있도록 프로젝트 루트를 경로에 추가
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- 경로 설정 끝 ---

# --- 2. 자동 진단: 프로젝트 구조 검사 ---
def check_project_structure():
    """RVC 모듈 임포트에 필요한 필수 폴더와 __init__.py 파일을 검사합니다."""
    # 'lib' 폴더 및 하위 __init__.py 파일 검사
    required_inits = [
        PROJECT_ROOT / "lib" / "__init__.py",
        PROJECT_ROOT / "lib" / "infer_pack" / "__init__.py",
    ]
    for init_path in required_inits:
        if not init_path.is_file():
            raise FileNotFoundError(f"치명적 오류: 필수 파일이 없습니다. '{init_path.parent.name}' 폴더 안에 빈 __init__.py 파일을 생성해주세요. 위치: {init_path}")
    print("DEBUG: 프로젝트 구조 검사 통과.")
# 자동 진단 실행
check_project_structure()
# --- 자동 진단 끝 ---


# --- 3. 필수 라이브러리 및 RVC 내부 모듈 임포트 ---
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa
import faiss

try:
    from lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )
    from fairseq import checkpoint_utils
    from rmvpe import RMVPE
    from fairseq.data import Dictionary
    import torch.serialization
    print("내부 RVC 모듈 로딩 성공!")
except ImportError as e:
    print(f"오류: RVC 관련 내부 모듈 임포트 실패! - {e}")
    raise e


# --- 4. 핵심 로직 클래스 ---
class RealTimeVC:
    """
    RVC 추론에 필요한 모든 모델과 설정을 관리하는 클래스.
    """
    def __init__(
        self,
        voice_name: str,
        base_dir_name: str = "pretrained",
        device: Optional[str] = None,
        is_half: bool = True,
    ):
        self.voice_name = voice_name
        self.is_half = is_half

        if device is None:
            if torch.cuda.is_available(): self.device = "cuda:0"
            elif torch.backends.mps.is_available(): self.device = "mps"
            else: self.device = "cpu"
        else:
            self.device = device
        
        if self.is_half and "cuda" not in self.device:
            self.is_half = False

        base_dir = SCRIPT_DIR / base_dir_name
        exp_dir = base_dir / self.voice_name
        self.pth_path = exp_dir / f"{self.voice_name}.pth"
        self.index_path = next(exp_dir.glob("*.index"), None)
        hubert_path = base_dir / "hubert_base.pt"
        rmvpe_path = base_dir / "rmvpe.pt"
        
        if not self.pth_path.exists(): raise FileNotFoundError(f"음성 모델(.pth) 없음: {self.pth_path}")
        if self.index_path is None: raise FileNotFoundError(f"인덱스 파일(.index) 없음: {exp_dir}")
        if not hubert_path.exists(): raise FileNotFoundError(f"Hubert 모델 없음: {hubert_path}")
        if not rmvpe_path.exists(): raise FileNotFoundError(f"RMVPE 모델 없음: {rmvpe_path}")
        
        cpt = torch.load(self.pth_path, map_location="cpu")
        self.tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = cpt.get("f0", 1)
        self.version = cpt.get("version", "v2")

        SynthesizerClass = SynthesizerTrnMs768NSFsid if self.if_f0 else SynthesizerTrnMs768NSFsid_nono
        self.net_g = SynthesizerClass(*cpt["config"], is_half=self.is_half)
        
        del self.net_g.enc_q
        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g.eval().to(self.device)
        if self.is_half: self.net_g = self.net_g.half()

        torch.serialization.add_safe_globals([Dictionary])
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(hubert_path)], suffix="")
        self.hubert_model = models[0].to(self.device)
        self.hubert_model = self.hubert_model.float() 
        if self.is_half:
            self.hubert_model = self.hubert_model.half()
        self.hubert_model.eval()

        self.index = faiss.read_index(str(self.index_path))
        self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        if self.if_f0:
            self.model_rmvpe = RMVPE(str(rmvpe_path), is_half=self.is_half, device=self.device)

    def process(
        self,
        src_audio: np.ndarray,
        src_sr: int,
        transpose: int = 0,
        index_rate: float = 0.75,
) -> np.ndarray:
        """
        src_audio : raw waveform (float32, -1~1)
        src_sr    : sampling rate of src_audio
        Returns   : (converted waveform, self.tgt_sr)
        """

        # ────────────────────────── 0. 입력 전처리 ──────────────────────────
        if src_audio.ndim > 1:
            src_audio = np.mean(src_audio, axis=1)          # mono
        audio_16k = librosa.resample(y=src_audio,
                                    orig_sr=src_sr,
                                    target_sr=16000)
        audio_16k_torch = torch.from_numpy(audio_16k).to(self.device)

        if self.is_half:
            audio_16k_torch = audio_16k_torch.half()
        else:
            audio_16k_torch = audio_16k_torch.float()

        # ─────────────────────── 1. Hubert feature 추출 ─────────────────────
        with torch.no_grad():
            feats = self.hubert_model.extract_features(
                audio_16k_torch.unsqueeze(0), output_layer=12
            )[0]                                           # (1, T, 768)

        # (선택) 인덱스 퓨전 – 원본 코드 그대로 두셔도 무방
        # ------------------------------------------------------------------
        if self.index is not None and index_rate > 0 and self.big_npy.shape[0] > 0:
            # ... (귀하의 기존 인덱스 결합 로직) ...
            pass
        # ------------------------------------------------------------------

        # 1‑b. **프레임 수 확정**  – Hubert 2 배
        feats = feats.repeat_interleave(2, dim=1)          # (1, 2T, 768)

        # ───────────────────── 2. F0 (pitch)  추출/정규화 ───────────────────
        pitch, pitchf = None, None
        if self.if_f0:
            f0 = self.model_rmvpe.infer_from_audio(audio_16k, thred=0.03)
            f0 *= pow(2, transpose / 12.0)

            # mel‑scale → 0 ~ 255 정규화 (귀하의 기존 코드 유지)
            f0_min, f0_max = 50.0, 1100.0
            f0_mel_min = 1127 * np.log(1 + f0_min / 700)
            f0_mel_max = 1127 * np.log(1 + f0_max / 700)

            f0_mel = 1127 * np.log(1 + f0 / 700)
            voiced = f0_mel > 0
            f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
            f0_mel = np.clip(f0_mel, 0, 255)

            pitch  = torch.LongTensor(np.rint(f0_mel)).unsqueeze(0).to(self.device)
            pitchf = torch.FloatTensor(f0).unsqueeze(0).to(self.device)

            # ── 길이 정렬 ──
            seq_len = feats.shape[1]
            if pitch.shape[1] < seq_len:                    # pad 뒤쪽으로
                pad = seq_len - pitch.shape[1]
                pitch  = F.pad(pitch,  (0, pad))
                pitchf = F.pad(pitchf, (0, pad))
            elif pitch.shape[1] > seq_len:                  # 잘라내기
                pitch  = pitch[:,  :seq_len]
                pitchf = pitchf[:, :seq_len]

        # ─────────────────────── 3.  RVC  추론 ─────────────────────────────
        p_len = torch.LongTensor([feats.shape[1]]).to(self.device)
        sid   = torch.LongTensor([0]).to(self.device)

        with torch.no_grad():
            if self.if_f0:
                audio_out = self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
            else:
                audio_out = self.net_g.infer(feats, p_len, sid)[0][0, 0]

        return audio_out.cpu().float().numpy()


# --- 5. 모델 캐시 및 최종 호출 함수 ---
_model_cache: Dict[str, RealTimeVC] = {}

def run_voice_conversion(
    src_audio: np.ndarray,
    src_sr: int,
    voice_name: str = "IU",
    **kwargs,
) -> tuple[np.ndarray, int]:
    global _model_cache
    if voice_name not in _model_cache:
        print(f"'{voice_name}' 모델을 캐시에 로드합니다...")
        try:
            model = RealTimeVC(voice_name=voice_name)
            _model_cache[voice_name] = model
            print("모델 로딩 완료.")
        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
            traceback.print_exc()
            return (src_audio, src_sr)
    
    model = _model_cache[voice_name]
    
    try:
        converted_audio = model.process(src_audio, src_sr, **kwargs)
        return (converted_audio, model.tgt_sr)
    except Exception as e:
        print(f"음성 변환 처리 중 오류 발생: {e}")
        traceback.print_exc()
        return (src_audio, src_sr)
