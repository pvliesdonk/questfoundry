"""Utilities to download released QuestFoundry specs from GitHub.

This module enables the runtime to download and cache spec releases from GitHub,
allowing users to update to the latest spec without reinstalling the package.

Based on: lib/compiler/src/questfoundry_compiler/spec_fetcher.py

Environment Variables:
    QF_SPEC_SOURCE: Control spec source selection (auto/monorepo/bundled/download)
        - auto: Try monorepo → bundled → download (default)
        - monorepo: Only use monorepo spec (fail if not found)
        - bundled: Only use bundled resources (fail if not found)
        - download: Always download latest spec from GitHub
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR: Final = Path.home() / ".cache" / "questfoundry" / "spec"
GITHUB_REPO: Final = "pvliesdonk/questfoundry"
API_BASE: Final = f"https://api.github.com/repos/{GITHUB_REPO}"
USER_AGENT: Final = "questfoundry-runtime"


def get_spec_source_preference() -> str:
    """
    Get spec source preference from environment variable.

    Returns:
        One of: 'auto', 'monorepo', 'bundled', 'download'
    """
    source = os.getenv("QF_SPEC_SOURCE", "auto").lower()
    valid_sources = {"auto", "monorepo", "bundled", "download"}

    if source not in valid_sources:
        logger.warning(
            f"Invalid QF_SPEC_SOURCE value: '{source}'. "
            f"Valid values: {', '.join(valid_sources)}. Using 'auto'."
        )
        return "auto"

    return source


class SpecFetchError(RuntimeError):
    """Raised when fetching a released spec fails."""


def _request_json(url: str) -> dict[str, Any]:
    """Make a JSON request to GitHub API."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 400:
                raise SpecFetchError(f"Request failed with status {response.status}")
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except urllib.error.URLError as exc:
        raise SpecFetchError(f"Unable to reach GitHub: {exc}") from exc


def _download_file(url: str, destination: Path) -> None:
    """Download a file from URL to destination path."""
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

            # Locate the directory that contains 05-definitions
            # Some release zips contain a top-level 'spec' directory (spec-all.zip),
            # while GitHub repo zipballs contain the repo root with a nested 'spec/'.
            candidate_root: Path | None = None

            # Check if 05-definitions is directly in extract root
            if (extract_root / "05-definitions").is_dir():
                candidate_root = extract_root
            else:
                # Search for 05-definitions in subdirectories
                for subdir in extract_root.rglob("05-definitions"):
                    if subdir.is_dir():
                        # Found 05-definitions, its parent is the spec root
                        candidate_root = subdir.parent
                        break

            if candidate_root is None:
                raise SpecFetchError(
                    "Downloaded archive is missing a spec/05-definitions/ directory"
                )

            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(candidate_root, target_dir)
        finally:
            shutil.rmtree(extract_root, ignore_errors=True)


def _is_valid_spec_root(spec_root: Path) -> bool:
    """Check if a directory is a valid spec root (contains 05-definitions)."""
    return (spec_root / "05-definitions").is_dir()


def _fetch_release_info(tag: str | None = None) -> dict[str, Any]:
    """Fetch release information from GitHub API."""
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


def get_cached_spec_path(tag: str | None = None, cache_dir: Path | None = None) -> Path | None:
    """
    Get path to cached spec if it exists.

    Args:
        tag: Spec version tag (e.g., 'spec-v1.0.0'). If None, uses latest.
        cache_dir: Custom cache directory. If None, uses default.

    Returns:
        Path to cached spec root, or None if not cached.
    """
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    if not cache_root.exists():
        return None

    if tag:
        spec_dir = cache_root / tag
        if _is_valid_spec_root(spec_dir):
            return spec_dir
        return None

    # Find latest cached spec (lexicographically last spec-v* directory)
    spec_dirs = sorted(cache_root.glob("spec-v*"), reverse=True)
    for spec_dir in spec_dirs:
        if _is_valid_spec_root(spec_dir):
            return spec_dir

    return None


def download_latest_release_spec(
    cache_dir: Path | None = None, tag: str | None = None, force: bool = False
) -> Path:
    """
    Download the latest QuestFoundry spec release from GitHub.

    This downloads spec releases to a local cache directory, allowing the runtime
    to use newer specs without reinstalling the package.

    Args:
        cache_dir: Custom cache directory. If None, uses ~/.cache/questfoundry/spec/
        tag: Specific release tag to download. If None, downloads latest spec release.
        force: Force re-download even if already cached.

    Returns:
        Path to the downloaded spec root directory.

    Raises:
        SpecFetchError: If download fails or spec is invalid.

    Example:
        >>> spec_path = download_latest_release_spec()
        >>> roles_dir = spec_path / "05-definitions" / "roles"
    """
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)

    release_info = _fetch_release_info(tag)
    tag_name = release_info["tag_name"]
    spec_dir = cache_root / tag_name

    # Check if already cached
    if not force and _is_valid_spec_root(spec_dir):
        logger.info(f"Using cached spec: {tag_name}")
        return spec_dir

    logger.info(f"Downloading spec release: {tag_name}")

    # Prefer an attached asset named 'spec-all.zip' (browser_download_url)
    archive_url = None
    assets = release_info.get("assets") or []
    if isinstance(assets, list):
        for asset in assets:
            name = asset.get("name") if isinstance(asset, dict) else None
            if isinstance(name, str) and name.lower() == "spec-all.zip":
                archive_url = asset.get("browser_download_url")
                logger.info("Found spec-all.zip asset")
                break

    # Fallback to the release zipball if no spec-all asset found
    if not archive_url:
        archive_url = release_info.get("zipball_url")
        logger.info("Using release zipball (no spec-all.zip found)")

    if not archive_url:
        raise SpecFetchError("Release response missing archive URL")

    with tempfile.TemporaryDirectory(prefix="qf-spec-download-") as tmp:
        archive_path = Path(tmp) / "spec-release.zip"
        logger.info(f"Downloading from: {archive_url}")
        _download_file(archive_url, archive_path)
        logger.info(f"Extracting to: {spec_dir}")
        _extract_zip(archive_path, spec_dir)

    if not _is_valid_spec_root(spec_dir):
        raise SpecFetchError("Downloaded spec archive is missing 05-definitions/")

    # Write metadata file
    metadata = {"tag": tag_name, "source": archive_url}
    (spec_dir / ".questfoundry-spec.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info(f"Spec downloaded successfully: {tag_name}")
    return spec_dir
