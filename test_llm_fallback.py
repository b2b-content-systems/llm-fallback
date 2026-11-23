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



# ---- images, schemas, finish, model gone ---------------------------------------------

def test_images_go_to_both_providers_as_inline_data():
    calls = []
    img = lf.Image(b"\x89PNG fake", "image/png")
    lf.ask_claude("read it", images=[img], transport=fake(CLAUDE_OK, GEMINI_OK, calls))
    lf.ask_gemini("read it", images=[img], transport=fake(CLAUDE_OK, GEMINI_OK, calls))
    claude, gemini = calls[0][1], calls[1][1]
    block = claude["messages"][0]["content"][0]
    assert block["type"] == "image" and block["source"]["media_type"] == "image/png"
    assert gemini["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/png"
    assert gemini["contents"][0]["parts"][-1] == {"text": "read it"}


def test_a_schema_is_native_for_gemini_and_in_words_for_claude():
    calls = []
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    lf.ask_gemini("x", json_schema=schema, model="gemini-3.6-flash", transport=fake(CLAUDE_OK, GEMINI_OK, calls))
    lf.ask_claude("x", json_schema=schema, transport=fake(CLAUDE_OK, GEMINI_OK, calls))
    g = calls[0][1]["generationConfig"]
    assert g["responseMimeType"] == "application/json" and g["responseJsonSchema"] == schema
    assert g["thinkingConfig"] == {"thinkingLevel": "low"} and "temperature" not in g
    assert "JSON Schema" in calls[1][1]["messages"][0]["content"]


def test_finish_says_when_a_reply_was_cut_off():
    cut_g = (200, {"candidates": [{"content": {"parts": [{"text": "{"}]}, "finishReason": "MAX_TOKENS"}]})
    cut_c = (200, {"content": [{"type": "text", "text": "{"}], "stop_reason": "max_tokens"})
    assert lf.ask_gemini("x", transport=fake(CLAUDE_OK, cut_g)).cut_off
    assert lf.ask_claude("x", transport=fake(cut_c, GEMINI_OK)).cut_off
    assert not lf.ask_gemini("x", transport=fake(CLAUDE_OK, GEMINI_OK)).cut_off


def test_a_missing_model_is_model_gone_not_an_outage():
    with pytest.raises(lf.ModelGone):
        lf.ask_gemini("x", transport=fake(CLAUDE_OK, (404, {"error": {"message": "not found"}})))
    assert issubclass(lf.ModelGone, lf.BadRequest)



# ---- image output -------------------------------------------------------------------

def test_an_image_model_returns_pictures_and_takes_no_thinking_settings():
    import base64
    calls = []
    png = b"\x89PNG fake"
    ok = (200, {"modelVersion": "gemini-3.1-flash-image", "candidates": [{"content": {"parts": [
        {"text": "here"}, {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}}]},
        "finishReason": "STOP"}]})
    a = lf.ask_gemini("draw", model="gemini-3.1-flash-image", output="image", aspect_ratio="3:4",
                      images=[lf.Image(b"ref")], transport=fake(CLAUDE_OK, ok, calls))
    assert a.images == [lf.Image(png, "image/png")] and a.finish == "stop"
    gen = calls[0][1]["generationConfig"]
    assert gen["responseModalities"] == ["IMAGE"] and gen["imageConfig"] == {"aspectRatio": "3:4"}
    assert "thinkingConfig" not in gen and "temperature" not in gen


@pytest.mark.parametrize("body", [
    {"candidates": [{"content": {"parts": []}, "finishReason": "IMAGE_SAFETY"}]},
    {"candidates": [{"content": {"parts": []}, "finishReason": "PROHIBITED_CONTENT"}]},
    {"promptFeedback": {"blockReason": "SAFETY"}},
])
def test_a_refusal_is_an_answer_not_an_error(body):
    a = lf.ask_gemini("draw", model="gemini-3.1-flash-image", output="image", transport=fake(CLAUDE_OK, (200, body)))
    assert a.finish == "refused" and a.images == []
