"""Utilities to download released QuestFoundry specs from GitHub."""

from __future__ import annotations

import json
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Final

DEFAULT_CACHE_DIR: Final = Path.home() / ".cache" / "questfoundry" / "spec"
GITHUB_REPO: Final = "pvliesdonk/questfoundry-spec"
API_BASE: Final = f"https://api.github.com/repos/{GITHUB_REPO}"
USER_AGENT: Final = "questfoundry-prompt-generator"


class SpecFetchError(RuntimeError):
    """Raised when fetching a released spec fails."""


def _request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 400:  # pragma: no cover
                raise SpecFetchError(f"Request failed with status {response.status}")
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except urllib.error.URLError as exc:  # pragma: no cover
        raise SpecFetchError(f"Unable to reach GitHub: {exc}") from exc


def _download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            destination.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    except urllib.error.URLError as exc:
        raise SpecFetchError(f"Failed to download release archive: {exc}") from exc


def _extract_zip(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        extract_root = Path(tempfile.mkdtemp(prefix="qf-spec-extract-"))
        try:
            archive.extractall(extract_root)
            try:
                unpacked_root = next(extract_root.iterdir())
            except StopIteration as exc:  # pragma: no cover
                raise SpecFetchError("Downloaded archive was empty") from exc
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(unpacked_root, target_dir)
        finally:
            shutil.rmtree(extract_root, ignore_errors=True)


def _is_valid_spec_root(spec_root: Path) -> bool:
    return (spec_root / "05-behavior").is_dir()


def _fetch_release_info(tag: str | None = None) -> dict[str, Any]:
    if tag:
        url = f"{API_BASE}/releases/tags/{tag}"
    else:
        url = f"{API_BASE}/releases/latest"
    return _request_json(url)


def download_latest_release_spec(
    cache_dir: Path | None = None, tag: str | None = None
) -> Path:
    """Download the latest QuestFoundry spec release if needed."""

    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)

    release_info = _fetch_release_info(tag)
    tag_name = release_info["tag_name"]
    spec_dir = cache_root / tag_name
    if _is_valid_spec_root(spec_dir):
        return spec_dir

    archive_url = release_info.get("zipball_url")
    if not archive_url:
        raise SpecFetchError("Release response missing archive URL")

    with tempfile.TemporaryDirectory(prefix="qf-spec-download-") as tmp:
        archive_path = Path(tmp) / "spec-release.zip"
        _download_file(archive_url, archive_path)
        _extract_zip(archive_path, spec_dir)

    if not _is_valid_spec_root(spec_dir):
        raise SpecFetchError("Downloaded spec archive is missing 05-behavior/")

    metadata = {"tag": tag_name, "source": archive_url}
    (spec_dir / ".questfoundry-spec.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return spec_dir
