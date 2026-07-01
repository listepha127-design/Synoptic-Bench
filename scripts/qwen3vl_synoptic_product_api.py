import base64
import binascii
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


MODEL_NAME = os.getenv("QWEN3VL_MODEL_NAME", "qwenvl8b-synopticbench")
VLLM_BASE_URL = os.getenv("QWEN3VL_VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_API_KEY = os.getenv("QWEN3VL_VLLM_API_KEY")
PRODUCT_API_KEY = os.getenv("QWEN3VL_PRODUCT_API_KEY")
DEFAULT_PROMPT = (
    "Generate a concise synoptic weather forecast discussion for this image. "
    "Focus on upper-level troughs and ridges, pressure systems, wind patterns, "
    "temperature advection, and important geographic regions."
)

app = FastAPI(title="Qwen3-VL Synoptic Weather Discussion API", version="1.0.0")


class GenerateRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded weather image.")
    prompt: str | None = Field(None, description="Optional generation instruction.")
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    model: str
    discussion: str
    usage: dict[str, Any] | None = None


def _require_config() -> None:
    missing = []
    if not VLLM_API_KEY:
        missing.append("QWEN3VL_VLLM_API_KEY")
    if not PRODUCT_API_KEY:
        missing.append("QWEN3VL_PRODUCT_API_KEY")
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))


def _extract_api_key(authorization: str | None, x_api_key: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            return value[7:].strip()
        return value
    return None


def _validate_api_key(authorization: str | None, x_api_key: str | None) -> None:
    supplied = _extract_api_key(authorization, x_api_key)
    if supplied != PRODUCT_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _normalize_image_data(image_base64: str) -> str:
    raw = image_base64.strip()
    if raw.startswith("data:"):
        header, sep, payload = raw.partition(",")
        if not sep or not header.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Invalid image data URL.")
        raw = payload

    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64.") from exc

    if not decoded:
        raise HTTPException(status_code=400, detail="image_base64 is empty.")

    return "data:image/png;base64," + base64.b64encode(decoded).decode("ascii")


def _call_vllm(payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        VLLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {VLLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"vLLM HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot reach vLLM backend: {exc}") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="vLLM backend timed out.") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/api/v1", response_model=GenerateResponse)
def generate_discussion(
    request: GenerateRequest,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None),
) -> GenerateResponse:
    _validate_api_key(authorization, x_api_key)

    image_url = _normalize_image_data(request.image_base64)
    prompt = request.prompt or DEFAULT_PROMPT

    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": request.max_tokens,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p

    result = _call_vllm(payload)
    choices = result.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="vLLM returned no choices.")

    message = choices[0].get("message") or {}
    return GenerateResponse(
        model=MODEL_NAME,
        discussion=message.get("content") or "",
        usage=result.get("usage"),
    )


if __name__ == "__main__":
    _require_config()
    host = os.getenv("QWEN3VL_PRODUCT_HOST", "127.0.0.1")
    port = int(os.getenv("QWEN3VL_PRODUCT_PORT", "7600"))
    uvicorn.run(app, host=host, port=port)
