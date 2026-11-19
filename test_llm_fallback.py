import pytest

import llm_fallback as lf

CLAUDE_OK = (200, {"model": "claude-sonnet-5-5", "content": [{"type": "text", "text": "hi from claude"}]})
GEMINI_OK = (200, {"modelVersion": "gemini-3.6-flash", "candidates": [{"content": {"parts": [
    {"text": "thinking...", "thought": True}, {"text": "hi from gemini"}]}}]})


def fake(claude, gemini, calls=None):
    def transport(url, headers, payload):
        if calls is not None:
            calls.append((url, payload))
        return claude if "anthropic" in url else gemini
    return transport


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")


def test_claude_answers_first():
    a = lf.ask("hi", transport=fake(CLAUDE_OK, GEMINI_OK))
    assert (a.provider, a.text, a.fell_back) == ("claude", "hi from claude", False)


@pytest.mark.parametrize("status", [401, 429, 500, 529])
def test_falls_back_when_claude_cannot_answer(status):
    a = lf.ask("hi", transport=fake((status, {}), GEMINI_OK))
    assert a.provider == "gemini" and a.text == "hi from gemini"
    assert str(status) in a.fallback_reason


def test_falls_back_without_a_claude_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    a = lf.ask("hi", transport=fake(CLAUDE_OK, GEMINI_OK))
    assert a.provider == "gemini" and "no ANTHROPIC_API_KEY" in a.fallback_reason


def test_bad_request_fails_as_itself():
    with pytest.raises(lf.BadRequest):
        lf.ask("hi", transport=fake((400, {"error": "bad"}), GEMINI_OK))


def test_both_down():
    with pytest.raises(lf.Unavailable, match="both down"):
        lf.ask("hi", transport=fake((503, {}), (503, {})))


def test_temperature_dropped_for_claude_5():
    calls = []
    lf.ask("hi", temperature=0.7, transport=fake(CLAUDE_OK, GEMINI_OK, calls))
    assert "temperature" not in calls[0][1]
    assert lf.claude_accepts_temperature("claude-sonnet-4-5")
    assert not lf.claude_accepts_temperature("claude-opus-4-7")
    assert not lf.claude_accepts_temperature("claude-haiku-5-5")


def test_gemini_settings_follow_the_model():
    g3 = lf.gemini_generation("gemini-3.6-flash", 1000, 0.7)
    assert g3["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "temperature" not in g3 and g3["maxOutputTokens"] == 1500
    g2 = lf.gemini_generation("gemini-2.5-flash", 1000, 0.7)
    assert "thinkingBudget" in g2["thinkingConfig"] and g2["temperature"] == 0.7
