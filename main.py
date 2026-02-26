import os
import time
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PROXY_SHARED_SECRET = os.environ.get("PROXY_SHARED_SECRET", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")

def guard(x_proxy_secret: Optional[str]):
    if PROXY_SHARED_SECRET and x_proxy_secret != PROXY_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

def sanitize(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.pop("elevenlabs_extra_body", None)
    payload.pop("user_id", None)

    # NEW: strip ElevenLabs-only / unsupported OpenAI Chat Completions args
    payload.pop("reasoning_effort", None)
    payload.pop("max_output_tokens", None)  # sometimes sent by UIs; chat.completions uses max_tokens
    payload.pop("temperature", None)  # optional: keep if you want, but safe to allow
    payload.pop("top_p", None)        # optional

    # Ensure correct token field for chat.completions
    if "max_output_tokens" in payload and "max_tokens" not in payload:
        payload["max_tokens"] = payload.pop("max_output_tokens")

    payload["stream"] = True
    return payload

async def stream_openai(payload: Dict[str, Any]):
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise HTTPException(status_code=502, detail=f"OpenAI error {r.status_code}: {body.decode('utf-8', errors='ignore')}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                yield (line + "\n\n").encode("utf-8")

@app.post("/chat/completions")
async def chat_completions(request: Request, x_proxy_secret: Optional[str] = Header(default=None)):
    guard(x_proxy_secret)
    payload = sanitize(await request.json())
    return StreamingResponse(stream_openai(payload), media_type="text/event-stream")

@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "ts": int(time.time())})
