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
GITHUB_REPO: Final = "pvliesdonk/questfoundry"
API_BASE: Final = f"https://api.github.com/repos/{GITHUB_REPO}"
USER_AGENT: Final = "questfoundry-compiler"


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
    """Extract a zip archive and locate the spec root directory."""
    with zipfile.ZipFile(archive_path) as archive:
        extract_root = Path(tempfile.mkdtemp(prefix="qf-spec-extract-"))
        try:
            archive.extractall(extract_root)

            # Locate the directory that contains 05-behavior
            # Some release zips contain a top-level 'spec' directory (spec-all.zip),
            # while GitHub repo zipballs contain the repo root with a nested 'spec/'.
            candidate_root: Path | None = None

            # Check if 05-behavior is directly in extract root
            if (extract_root / "05-behavior").is_dir():
                candidate_root = extract_root
            else:
                # Search for 05-behavior in subdirectories
                for subdir in extract_root.rglob("05-behavior"):
                    if subdir.is_dir():
                        # Found 05-behavior, its parent is the spec root
                        candidate_root = subdir.parent
                        break

            if candidate_root is None:
                raise SpecFetchError(
                    "Downloaded archive is missing a spec/05-behavior/ directory"
                )

            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(candidate_root, target_dir)
        finally:
            shutil.rmtree(extract_root, ignore_errors=True)


def _is_valid_spec_root(spec_root: Path) -> bool:
    return (spec_root / "05-behavior").is_dir()


def _fetch_release_info(tag: str | None = None) -> dict[str, Any]:
    if tag:
        url = f"{API_BASE}/releases/tags/{tag}"
        return _request_json(url)

    # No explicit tag: prefer a release whose tag_name starts with 'spec-v'
    releases = _request_json(f"{API_BASE}/releases")
    if isinstance(releases, list):
        for rel in releases:
            if not isinstance(rel, dict):
                continue
            tag_name = rel.get("tag_name")
            if isinstance(tag_name, str) and tag_name.startswith("spec-v"):
                return rel

    # Fallback to the GitHub 'latest' release if no spec-tagged release found
    return _request_json(f"{API_BASE}/releases/latest")


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

    # Prefer an attached asset named 'spec-all.zip' (browser_download_url)
    archive_url = None
    assets = release_info.get("assets") or []
    if isinstance(assets, list):
        for asset in assets:
            name = asset.get("name") if isinstance(asset, dict) else None
            if isinstance(name, str) and name.lower() == "spec-all.zip":
                archive_url = asset.get("browser_download_url")
                break

    # Fallback to the release zipball if no spec-all asset found
    if not archive_url:
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
