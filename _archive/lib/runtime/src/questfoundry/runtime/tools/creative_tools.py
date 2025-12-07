from __future__ import annotations

"""Creative tools: image generation (multi-provider).

Providers (auto-selected by env IMAGE_PROVIDER unless explicitly passed):
- "dalle"   : OpenAI Images API (OPENAI_API_KEY)
- "gemini"  : Google AI Studio image generation (GOOGLE_API_KEY)
- "a1111"   : Automatic1111 / Stable Diffusion WebUI (A1111_URL)

If no provider is configured, the tool returns a mock response.
"""

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class StableDiffusion(BaseTool):
    name: str = "stable_diffusion"
    description: str = "Generate an image from a prompt using configured provider"

    def _run(
        self,
        prompt: str,
        provider: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:  # type: ignore[override]
        selected = (provider or os.getenv("IMAGE_PROVIDER") or "").lower().strip()
        if not selected:
            # Try inferring from available envs
            if os.getenv("A1111_URL"):
                selected = "a1111"
            elif os.getenv("GOOGLE_API_KEY"):
                selected = "gemini"
            elif os.getenv("OPENAI_API_KEY"):
                selected = "dalle"

        try:
            if selected == "dalle":
                return self._run_openai(prompt)
            if selected == "gemini":
                return self._run_gemini(prompt)
            if selected == "a1111":
                return self._run_a1111(prompt, **kwargs)
        except Exception as exc:  # pragma: no cover - surfaced via tests with monkeypatch
            logger.warning("image generation failed for provider %s: %s", selected, exc)
            return {
                "status": "error",
                "provider": selected or "none",
                "message": str(exc),
            }

        return {
            "status": "mock",
            "provider": selected or "none",
            "message": "No image provider configured; returning mock",
        }

    def _run_openai(self, prompt: str) -> dict[str, Any]:
        from openai import OpenAI  # type: ignore

        client = OpenAI()
        resp = client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024")
        url = resp.data[0].url if resp.data else None
        return {"status": "success", "provider": "dalle", "url": url, "prompt": prompt}

    def _run_gemini(self, prompt: str) -> dict[str, Any]:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("imagen-3.0-generate-002")
        result = model.generate_image(prompt=prompt)
        # google SDK returns a Binary object with ._repr_html_ or .base64_data
        b64 = getattr(result, "base64_data", None)
        if b64:
            file_path = self._write_temp_image(b64)
        else:
            file_path = None
        return {
            "status": "success",
            "provider": "gemini",
            "path": file_path,
            "prompt": prompt,
        }

    def _run_a1111(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        base_url = os.getenv("A1111_URL")
        if not base_url:
            raise RuntimeError("A1111_URL not set")

        payload = {
            "prompt": prompt,
            "steps": kwargs.get("steps", 20),
            "cfg_scale": kwargs.get("cfg_scale", 7.0),
            "width": kwargs.get("width", 768),
            "height": kwargs.get("height", 768),
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(base_url.rstrip("/") + "/sdapi/v1/txt2img", json=payload)
        resp.raise_for_status()
        data = resp.json()
        images = data.get("images") or []
        if not images:
            raise RuntimeError("no images returned from A1111")

        file_path = self._write_temp_image(images[0])
        return {
            "status": "success",
            "provider": "a1111",
            "path": file_path,
            "prompt": prompt,
        }

    @staticmethod
    def _write_temp_image(b64_data: str) -> str:
        raw = base64.b64decode(b64_data)
        fd, path = tempfile.mkstemp(suffix=".png", prefix="qf_img_")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return str(Path(path))
