"""
채팅 검열 + 말투 교정 백엔드 서버
- 프론트에서 채팅 텍스트를 받아서
- 로컬 Ollama(exaone3.5:2.4b)에 넘기고
- 비속어 탐지 + 순화된 버전을 받아서 리턴한다.

실행 방법:
    1) 터미널에서 ollama가 켜져 있는지 확인 (ollama run exaone3.5:2.4b 로 실행해둔 상태면 OK)
    2) pip install fastapi uvicorn requests
    3) uvicorn main:app --reload --port 8000
    4) http://localhost:8000/docs 에서 테스트 가능 (Swagger 자동 문서)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json

# -----------------------------
# 기본 설정
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "exaone3.5:2.4b"

app = FastAPI(title="채팅 검열/말투 교정 API")

# 프론트(다른 포트, 예: 3000, 5173 등)에서 호출할 수 있도록 CORS 전체 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# 요청/응답 형태 정의
# -----------------------------
class ChatRequest(BaseModel):
    text: str  # 검사할 채팅 내역 (여러 줄 가능)


class ModerateResult(BaseModel):
    original: str
    filtered: str
    has_profanity: bool
    raw_model_output: str  # 디버깅/데모용 (모델이 실제로 뭐라 답했는지)


# -----------------------------
# Ollama 호출 함수
# -----------------------------
def call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama 서버에 연결할 수 없습니다. 'ollama run exaone3.5:2.4b'가 실행 중인지 확인하세요.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 호출 중 오류: {str(e)}")


# -----------------------------
# 엔드포인트: 채팅 검열 + 말투 교정
# -----------------------------
@app.post("/moderate", response_model=ModerateResult)
def moderate_chat(req: ChatRequest):
    prompt = f"""너는 채팅 내용을 검열하고 순화하는 도우미야.
아래 채팅 내용을 보고 다음 형식의 JSON으로만 답해. 다른 설명은 절대 붙이지 마.

{{
  "has_profanity": true 또는 false,
  "filtered": "비속어나 공격적인 표현을 순화한 전체 문장"
}}

채팅 내용:
{req.text}
"""

    raw_output = call_ollama(prompt)

    # 모델이 JSON 형태로 잘 답했다고 가정하고 파싱 시도
    has_profanity = False
    filtered = raw_output  # 파싱 실패 시 fallback으로 원본 응답 그대로 사용

    try:
        # 모델이 JSON 앞뒤로 잡담을 붙이는 경우가 있어서, { ... } 부분만 잘라내는 안전장치
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        json_str = raw_output[start:end]
        parsed = json.loads(json_str)
        has_profanity = bool(parsed.get("has_profanity", False))
        filtered = parsed.get("filtered", raw_output)
    except Exception:
        # JSON 파싱 실패해도 서버가 죽지 않고 원문 응답을 그대로 보여주도록 처리
        pass

    return ModerateResult(
        original=req.text,
        filtered=filtered,
        has_profanity=has_profanity,
        raw_model_output=raw_output,
    )


# -----------------------------
# 서버 상태 확인용 (프론트/발표 데모에서 "서버 살아있나" 체크용)
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}
