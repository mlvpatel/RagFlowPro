"""Unit tests for the frontend API client (no server, no streamlit)."""

from frontend import api_utils


class _FakeResp:
    def __init__(self, lines=None, payload=None):
        self._lines = lines or []
        self._payload = payload

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line

    def json(self):
        return self._payload


def test_headers_include_api_key():
    headers = api_utils._headers()
    assert headers["X-API-Key"] == api_utils.API_KEY
    assert headers["Content-Type"] == "application/json"


def test_stream_chat_parses_sse_tokens_then_done(monkeypatch):
    lines = [
        'data: {"token": "Hello"}',
        "",
        'data: {"token": " world"}',
        'data: {"done": true, "session_id": "abc"}',
    ]
    monkeypatch.setattr(
        api_utils.requests, "post", lambda *a, **k: _FakeResp(lines=lines)
    )

    events = list(api_utils.stream_chat("hi", None, "gpt-4o-mini"))

    assert [e["token"] for e in events if "token" in e] == ["Hello", " world"]
    done = [e for e in events if e.get("done")]
    assert done and done[0]["session_id"] == "abc"


def test_upload_document_posts_file_to_v1(monkeypatch):
    captured = {}

    def fake_post(url, files=None, headers=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        return _FakeResp(payload={"task_id": "t1", "status": "processing"})

    monkeypatch.setattr(api_utils.requests, "post", fake_post)

    out = api_utils.upload_document(b"data", "notes.txt")

    assert out["task_id"] == "t1"
    assert captured["url"].endswith("/v1/upload-doc")
    assert captured["files"]["file"][0] == "notes.txt"
