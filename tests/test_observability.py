import pytest

from app import observability


def test_langfuse_v2_handler_is_importable_with_langchain_1x_thanks_to_the_compat_layer():
    """Regression: without the shim, Langfuse v2 failed with ModuleNotFoundError on langchain 1.x."""
    observability.install_langchain_compat()
    from langfuse.callback import CallbackHandler

    assert CallbackHandler is not None


@pytest.mark.parametrize("public,secret", [("", ""), ("pk-lf-...", "sk-lf-..."), ("pk-lf-x", "")])
def test_tracing_is_disabled_for_missing_or_placeholder_keys(monkeypatch, public, secret):
    settings = type("S", (), {"langfuse_public_key": public, "langfuse_secret_key": secret})()
    monkeypatch.setattr(observability, "get_settings", lambda: settings)
    assert not observability.tracing_enabled()
    assert observability.make_callbacks("session-1", "question") == []


def test_real_looking_keys_enable_tracing(monkeypatch):
    settings = type("S", (), {"langfuse_public_key": "pk-lf-local-dev", "langfuse_secret_key": "sk-lf-local-dev"})()
    monkeypatch.setattr(observability, "get_settings", lambda: settings)
    assert observability.tracing_enabled()


def test_broken_tracing_never_breaks_the_request(monkeypatch):
    settings = type(
        "S",
        (),
        {"langfuse_public_key": "pk-x", "langfuse_secret_key": "sk-x", "langfuse_host": "http://localhost:3000"},
    )()
    monkeypatch.setattr(observability, "get_settings", lambda: settings)

    def boom():
        raise RuntimeError("langfuse is not installed")

    monkeypatch.setattr(observability, "install_langchain_compat", boom)
    assert observability.make_callbacks("s", "q") == []  # the error is swallowed so the user's answer is unaffected
