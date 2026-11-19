# llm-fallback

Ask Claude. If Claude cannot answer, ask Gemini, and say so.

I run marketing tools that call an AI model all day: film briefs, document pipelines, a church's weekly bulletin. When the model is down, rate-limited or out of credit, the tool should keep working and tell me which model answered. When my own request is wrong, it should fail loudly instead of hiding the bug behind a second model. This file does both.

```python
from llm_fallback import ask

a = ask("Write a two-line product update for a product launch.", system="Plain English.")
print(a.text)
print(a.model, a.fallback_reason)   # e.g. "gemini-3.6-flash", "Claude unavailable (529)"
```

## What it does

| Claude says | What happens |
|---|---|
| answers | you get Claude's answer |
| no key, 401, 403 | Gemini answers; `fallback_reason` says why |
| 408, 429, 5xx, network down | Gemini answers; `fallback_reason` says why |
| 400 (your request is wrong) | raises `BadRequest`; no fallback |
| both down | raises `Unavailable` with both reasons |

## Settings each model accepts

A fallback is useless if the second model rejects the call. Each provider gets only the settings it takes:

- **Claude 5 and Opus 4.7+:** `temperature` is dropped.
- **Gemini 2.x:** `thinkingBudget` and `temperature`.
- **Gemini 3.x and later:** `thinkingLevel: "low"`, no `temperature`, `topP` or `topK`. Thinking counts against the output limit, so the call gets 500 tokens of extra room.

## Install

Python 3.10 or later, standard library only. If `certifi` is installed it uses its certificates (python.org builds on macOS ship without any). Copy `llm_fallback.py` into your project and set the keys:

```
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
python llm_fallback.py "Say hello in five words."
```

Change the defaults with `claude_model=` and `gemini_model=`. Pass `transport=` to test without the network (see the tests).

MIT licence.
