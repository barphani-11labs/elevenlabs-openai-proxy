python
import os
import time
import asyncio
import re
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PROXY_SHARED_SECRET = os.environ.get("PROXY_SHARED_SECRET", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")


def guard(x_proxy_secret: Optional[str]):
    """
    Simple shared-secret auth so only ElevenLabs (with your header) can call this proxy.
    """
    if PROXY_SHARED_SECRET and x_proxy_secret != PROXY_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def sanitize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    ElevenLabs sends extra fields that OpenAI Chat Completions rejects.
    Strip/normalize them here.
    """
    # ElevenLabs extras
    payload.pop("elevenlabs_extra_body", None)
    payload.pop("user_id", None)

    # Unsupported by OpenAI Chat Completions
    payload.pop("reasoning_effort", None)

    # Token field normalization (ElevenLabs may send max_output_tokens)
    if "max_output_tokens" in payload and "max_tokens" not in payload:
        payload["max_tokens"] = payload.pop("max_output_tokens")
    else:
        payload.pop("max_output_tokens", None)

    # You can keep temperature/top_p if you want; leaving them as-is is fine.
    # If you prefer stricter control, uncomment the next two lines:
    # payload.pop("temperature", None)
    # payload.pop("top_p", None)

    # Ensure streaming (ElevenLabs expects SSE)
    payload["stream"] = True
    return payload


async def stream_openai(payload: Dict[str, Any]):
    """
    Streams OpenAI SSE bytes through to ElevenLabs.
    Adds a small retry/backoff for 429 token-per-minute bursts so calls don't die mid-stream.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0)

    # Retry a few times on rate-limit bursts (429)
    for attempt in range(3):
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as r:
                if r.status_code == 429:
                    body = await r.aread()
                    text = body.decode("utf-8", errors="ignore")

                    # Try to parse "Please try again in Xs" from OpenAI message
                    m = re.search(r"try again in ([0-9.]+)s", text, flags=re.IGNORECASE)
                    wait_s = float(m.group(1)) if m else 2.5

                    # Add tiny buffer and cap wait to keep calls responsive
                    await asyncio.sleep(min(wait_s + 0.3, 6.0))
                    continue

                if r.status_code != 200:
                    body = await r.aread()
                    raise HTTPException(
                        status_code=502,
                        detail=f"OpenAI error {r.status_code}: {body.decode('utf-8', errors='ignore')}",
                    )

                # IMPORTANT: stream raw bytes to preserve SSE formatting
                async for chunk in r.aiter_bytes():
                    if chunk:
                        yield chunk
                return

    raise HTTPException(
        status_code=502,
        detail="OpenAI rate-limited repeatedly (429). Please retry shortly.",
    )


@app.post("/chat/completions")
async def chat_completions(request: Request, x_proxy_secret: Optional[str] = Header(default=None)):
    guard(x_proxy_secret)
    payload = sanitize(await request.json())
    return StreamingResponse(stream_openai(payload), media_type="text/event-stream")
@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "ts": int(time.time())})
```
