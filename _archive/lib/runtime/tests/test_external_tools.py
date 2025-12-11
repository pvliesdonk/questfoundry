import os
import sys
import types
from pathlib import Path

import numpy as np
from questfoundry.runtime.tools.creative_tools import StableDiffusion
from questfoundry.runtime.tools.export_tools import PandocConvert
from questfoundry.runtime.tools.research_tools import LoreIndex, WebSearch


def test_web_search_mock_without_env(monkeypatch):
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    tool = WebSearch()
    result = tool.run("test query")
    assert result["status"] == "mock"
    assert result["results"] == []


def test_web_search_success(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "https://searx.example")

    class FakeResponse:
        def __init__(self):
            self._json = {
                "results": [
                    {"title": "t1", "url": "u1", "content": "c1", "engine": "g"},
                    {"title": "t2", "url": "u2", "content": "c2", "engine": "g"},
                ]
            }

        def raise_for_status(self):
            return None

        def json(self):
            return self._json

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr("questfoundry.runtime.tools.research_tools.httpx.Client", FakeClient)
    result = WebSearch().run("hello")
    assert result["status"] == "success"
    assert len(result["results"]) == 2


def test_stable_diffusion_mock(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("IMAGE_PROVIDER", raising=False)
    tool = StableDiffusion()
    out = tool.run("a castle")
    assert out["status"] in {"mock", "error"}


def test_stable_diffusion_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("IMAGE_PROVIDER", "dalle")

    class FakeImages:
        def __init__(self):
            self.data = [types.SimpleNamespace(url="https://example/image.png")]

    class FakeOpenAI:
        def __init__(self):
            self.images = self

        def generate(self, **kwargs):  # compatibility with client.images.generate
            return FakeImages()

    fake_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    res = StableDiffusion().run("sunset")
    assert res["status"] == "success"
    assert res["url"].startswith("https://example")


def test_pandoc_mock_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("questfoundry.runtime.tools.export_tools._find_binary", lambda name: None)
    infile = tmp_path / "a.md"
    infile.write_text("# Test")
    res = PandocConvert().run(str(infile), output_format="pdf")
    assert res["status"] == "mock"


def test_pandoc_invocation(monkeypatch, tmp_path):
    infile = tmp_path / "a.md"
    infile.write_text("# Title")
    calls = {}

    def fake_find(name):
        return "/usr/bin/pandoc"

    def fake_run(args, check, stdout, stderr):
        calls["args"] = args
        class R:
            pass

        return R()

    monkeypatch.setattr("questfoundry.runtime.tools.export_tools._find_binary", fake_find)
    monkeypatch.setattr("subprocess.run", fake_run)
    res = PandocConvert().run(str(infile), output_format="pdf")
    assert res["status"] == "success"
    assert "/usr/bin/pandoc" in calls["args"][0]


def test_lore_index_keyword_fallback(monkeypatch):
    tool = LoreIndex()
    # force keyword path
    monkeypatch.setattr(tool, "_load_vectorizer", lambda: None)
    monkeypatch.setattr(tool, "_embed_with_ollama", lambda texts: None)
    docs = ["Alpha beta", "Gamma delta", "Beta gamma"]
    res = tool.run("beta", documents=docs)
    assert res["status"] == "keyword"
    assert res["results"]


def test_lore_index_semantic(monkeypatch):
    class StubVectorizer:
        def fit_transform(self, docs):
            # two docs -> identity matrix
            return np.array([[1.0, 0.0], [0.0, 1.0]])

        def transform(self, queries):
            return np.array([[1.0, 0.0]])

    tool = LoreIndex()
    monkeypatch.setattr(tool, "_load_vectorizer", lambda: StubVectorizer())
    monkeypatch.setattr(tool, "_embed_with_ollama", lambda texts: None)
    docs = ["Doc similar", "Doc different"]
    res = tool.run("query", documents=docs)
    assert res["status"] == "semantic"
    assert res["results"][0]["text"] == "Doc similar"


def test_lore_index_ollama(monkeypatch):
    tool = LoreIndex()

    def fake_embed(texts):
        # first is query, next two docs
        return [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]

    monkeypatch.setattr(tool, "_embed_with_ollama", fake_embed)
    monkeypatch.setenv("OLLAMA_MODEL", "nomic-embed-text:latest")
    res = tool.run("query", documents=["Doc similar", "Doc different"])
    assert res["status"] == "semantic"
    assert res["model"] == "nomic-embed-text:latest"
    assert res["results"][0]["text"] == "Doc similar"
