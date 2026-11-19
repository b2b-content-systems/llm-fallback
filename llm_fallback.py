"""llm-fallback: ask Claude, and ask Gemini when Claude cannot answer.

One function, `ask()`. It sends the prompt to Claude. If Claude cannot answer
(no key, expired key, rate limit, server error, network down), it sends the
same prompt to Gemini and tells you which model answered and why.

A bad request is not an outage. If Claude rejects the request itself (HTTP
400), `ask()` raises instead of hiding the bug behind a second model.

It also knows which settings each model accepts, so a fallback never fails
on a parameter the other model rejects:

- Claude 5 models and Opus 4.7+ reject `temperature`. It is dropped for them.
- Gemini 2.x takes `thinkingBudget` and `temperature`.
- Gemini 3.x and later take `thinkingLevel` and reject `temperature`,
  `topP` and `topK`. Thinking tokens count against the output limit, so
  3.x calls get extra headroom.

Standard library only. Keys come from ANTHROPIC_API_KEY and GEMINI_API_KEY.
"""
from __future__ import annotations

import json
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

CLAUDE_MODEL = "claude-sonnet-5-5"
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI3_HEADROOM = 500
TIMEOUT = 60

# (status, body) <- transport(url, headers, payload)
Transport = Callable[[str, dict, dict], tuple[int, dict]]


class Unavailable(Exception):
    """The provider cannot answer right now. Try the other one."""


class BadRequest(Exception):
    """The request itself is wrong. Do not fall back; fix the call."""


@dataclass
class Answer:
    text: str
    model: str
    provider: str
    fallback_reason: str | None = None
    tried: list[str] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return self.fallback_reason is not None


def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without CA certificates. Use certifi's
    # bundle when it is installed; otherwise the system default.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def http_transport(url: str, headers: dict, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_ssl_context()) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read() or b"{}")
        except ValueError:
            body = {}
        return e.code, body
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
        raise Unavailable(f"network: {e}") from e


def _classify(provider: str, status: int, body: dict) -> None:
    if 200 <= status < 300:
        return
    msg = json.dumps(body)[:300]
    if status in (401, 403):
        raise Unavailable(f"{provider} auth failed ({status})")
    if status in (408, 429) or status >= 500:
        raise Unavailable(f"{provider} unavailable ({status})")
    raise BadRequest(f"{provider} rejected the request ({status}): {msg}")


# ---- per-model settings -------------------------------------------------

def claude_accepts_temperature(model: str) -> bool:
    m = re.match(r"claude-(?:opus|sonnet|haiku|fable)-(\d+)(?:-(\d+))?", model)
    if not m:
        return True
    major, minor = int(m.group(1)), int(m.group(2) or 0)
    if major >= 5:
        return False
    return not (model.startswith("claude-opus") and (major, minor) >= (4, 7))


def gemini_generation(model: str, max_tokens: int, temperature: float | None) -> dict:
    m = re.match(r"gemini-(\d+)", model)
    major = int(m.group(1)) if m else 3
    if major >= 3:
        return {"maxOutputTokens": max_tokens + GEMINI3_HEADROOM,
                "thinkingConfig": {"thinkingLevel": "low"}}
    cfg: dict = {"maxOutputTokens": max_tokens, "thinkingConfig": {"thinkingBudget": 0}}
    if temperature is not None:
        cfg["temperature"] = temperature
    return cfg


# ---- providers -----------------------------------------------------------

def ask_claude(prompt: str, *, system: str | None = None, model: str = CLAUDE_MODEL,
               max_tokens: int = 1024, temperature: float | None = None,
               transport: Transport = http_transport, api_key: str | None = None) -> Answer:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise Unavailable("Claude: no ANTHROPIC_API_KEY")
    payload: dict = {"model": model, "max_tokens": max_tokens,
                     "messages": [{"role": "user", "content": prompt}]}
    if system:
        payload["system"] = system
    if temperature is not None and claude_accepts_temperature(model):
        payload["temperature"] = temperature
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    status, body = transport("https://api.anthropic.com/v1/messages", headers, payload)
    _classify("Claude", status, body)
    text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
    return Answer(text=text, model=body.get("model", model), provider="claude")


def ask_gemini(prompt: str, *, system: str | None = None, model: str = GEMINI_MODEL,
               max_tokens: int = 1024, temperature: float | None = None,
               transport: Transport = http_transport, api_key: str | None = None) -> Answer:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise Unavailable("Gemini: no GEMINI_API_KEY")
    payload: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                     "generationConfig": gemini_generation(model, max_tokens, temperature)}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": key, "content-type": "application/json"}
    status, body = transport(url, headers, payload)
    _classify("Gemini", status, body)
    parts = (body.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return Answer(text=text, model=body.get("modelVersion", model), provider="gemini")


def ask(prompt: str, *, system: str | None = None, claude_model: str = CLAUDE_MODEL,
        gemini_model: str = GEMINI_MODEL, max_tokens: int = 1024,
        temperature: float | None = None, transport: Transport = http_transport) -> Answer:
    """Claude first. Gemini only when Claude cannot answer. Raises BadRequest
    on a malformed call, and Unavailable when both are down."""
    common = dict(system=system, max_tokens=max_tokens, temperature=temperature, transport=transport)
    try:
        ans = ask_claude(prompt, model=claude_model, **common)
        ans.tried = ["claude"]
        return ans
    except Unavailable as why:
        reason = str(why)
    try:
        ans = ask_gemini(prompt, model=gemini_model, **common)
    except Unavailable as why:
        raise Unavailable(f"both down. {reason}; {why}") from None
    ans.fallback_reason = reason
    ans.tried = ["claude", "gemini"]
    return ans


if __name__ == "__main__":
    import sys
    a = ask(" ".join(sys.argv[1:]) or "Say hello in five words.")
    note = f" (fallback: {a.fallback_reason})" if a.fell_back else ""
    print(f"[{a.model}{note}]\n{a.text}")
