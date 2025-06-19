from g2pk import G2p
_g2p = G2p()

def romanize_korean(text: str) -> str:
    """
    한글 문장을 g2pk 로마자 문자열로 변환.
    g2pk 0.10.x: group 파라미터 존재
    g2pk 0.9.x 이하: group 없음 → try/except
    """
    try:
        roman = _g2p(text, group=False)         # 신버전
    except TypeError:
        roman = _g2p(text)                      # 구버전 fallback

    # g2pk 는 토큰 리스트 반환 → 공백 join
    if isinstance(roman, list):
        roman = " ".join(roman)
    return roman
