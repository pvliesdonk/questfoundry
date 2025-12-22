"""
Corpus file parser.

Extracts YAML frontmatter and markdown sections from corpus files.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Regex patterns
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


@dataclass
class CorpusSection:
    """A section extracted from a corpus file."""

    heading: str
    level: int  # 1=H1, 2=H2, 3=H3
    line_start: int
    content: str

    def __post_init__(self) -> None:
        # Normalize content
        self.content = self.content.strip()


@dataclass
class CorpusFrontmatter:
    """Parsed YAML frontmatter from a corpus file."""

    title: str
    summary: str
    topics: list[str]
    cluster: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusFrontmatter:
        """Create from parsed YAML dict."""
        return cls(
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            topics=data.get("topics", []),
            cluster=data.get("cluster", ""),
        )

    def validate(self) -> list[str]:
        """Validate frontmatter, return list of errors."""
        errors = []

        if not self.title:
            errors.append("Missing required field: title")
        elif len(self.title) < 5:
            errors.append(f"Title too short (min 5 chars): {len(self.title)}")

        if not self.summary:
            errors.append("Missing required field: summary")
        elif len(self.summary) < 20:
            errors.append(f"Summary too short (min 20 chars): {len(self.summary)}")
        elif len(self.summary) > 300:
            errors.append(f"Summary too long (max 300 chars): {len(self.summary)}")

        if not self.topics:
            errors.append("Missing required field: topics")
        elif len(self.topics) < 3:
            errors.append(f"Too few topics (min 3): {len(self.topics)}")

        valid_clusters = {
            "narrative-structure",
            "prose-and-language",
            "genre-conventions",
            "audience-and-access",
            "world-and-setting",
            "emotional-design",
            "scope-and-planning",
        }
        if not self.cluster:
            errors.append("Missing required field: cluster")
        elif self.cluster not in valid_clusters:
            errors.append(f"Invalid cluster '{self.cluster}', must be one of: {valid_clusters}")

        return errors


@dataclass
class CorpusFile:
    """A parsed corpus file with frontmatter and sections."""

    path: Path
    frontmatter: CorpusFrontmatter
    sections: list[CorpusSection] = field(default_factory=list)
    content_hash: str = ""
    raw_content: str = ""

    def __post_init__(self) -> None:
        if self.raw_content and not self.content_hash:
            self.content_hash = hashlib.sha256(self.raw_content.encode()).hexdigest()[:16]


def parse_corpus_file(file_path: Path) -> CorpusFile | None:
    """
    Parse a corpus file, extracting frontmatter and sections.

    Args:
        file_path: Path to the corpus markdown file

    Returns:
        CorpusFile with parsed data, or None if parsing fails
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read corpus file {file_path}: {e}")
        return None

    # Extract frontmatter
    frontmatter_match = FRONTMATTER_PATTERN.match(content)
    if not frontmatter_match:
        logger.warning(f"No YAML frontmatter found in {file_path}")
        return None

    try:
        frontmatter_yaml = yaml.safe_load(frontmatter_match.group(1))
        if not isinstance(frontmatter_yaml, dict):
            logger.warning(f"Frontmatter is not a dict in {file_path}")
            return None
        frontmatter = CorpusFrontmatter.from_dict(frontmatter_yaml)
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML frontmatter in {file_path}: {e}")
        return None

    # Validate frontmatter
    errors = frontmatter.validate()
    if errors:
        logger.warning(f"Frontmatter validation errors in {file_path}: {errors}")
        # Continue anyway - we can still index the file

    # Extract sections from content after frontmatter
    body = content[frontmatter_match.end() :]
    sections = _extract_sections(body)

    return CorpusFile(
        path=file_path,
        frontmatter=frontmatter,
        sections=sections,
        raw_content=content,
    )


def _extract_sections(content: str) -> list[CorpusSection]:
    """
    Extract sections from markdown content.

    Sections are defined by H1-H3 headings. Each section includes
    content up to the next heading of equal or higher level.
    """
    sections: list[CorpusSection] = []
    lines = content.split("\n")

    current_heading: str | None = None
    current_level: int = 0
    current_start: int = 0
    current_content: list[str] = []

    for i, line in enumerate(lines):
        heading_match = HEADING_PATTERN.match(line)

        if heading_match:
            # Save previous section if exists
            if current_heading is not None:
                section_content = "\n".join(current_content).strip()
                if section_content:  # Only add non-empty sections
                    sections.append(
                        CorpusSection(
                            heading=current_heading,
                            level=current_level,
                            line_start=current_start,
                            content=section_content,
                        )
                    )

            # Start new section
            hashes = heading_match.group(1)
            current_level = len(hashes)
            current_heading = heading_match.group(2).strip()
            current_start = i + 1  # 1-indexed line numbers
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_heading is not None:
        section_content = "\n".join(current_content).strip()
        if section_content:
            sections.append(
                CorpusSection(
                    heading=current_heading,
                    level=current_level,
                    line_start=current_start,
                    content=section_content,
                )
            )

    return sections


def parse_corpus_directory(corpus_dir: Path) -> list[CorpusFile]:
    """
    Parse all corpus files in a directory.

    Args:
        corpus_dir: Path to the corpus directory

    Returns:
        List of successfully parsed CorpusFile objects
    """
    if not corpus_dir.exists():
        logger.warning(f"Corpus directory does not exist: {corpus_dir}")
        return []

    files: list[CorpusFile] = []

    for file_path in corpus_dir.glob("*.md"):
        parsed = parse_corpus_file(file_path)
        if parsed:
            files.append(parsed)
        else:
            logger.warning(f"Failed to parse corpus file: {file_path}")

    logger.info(f"Parsed {len(files)} corpus files from {corpus_dir}")
    return files
