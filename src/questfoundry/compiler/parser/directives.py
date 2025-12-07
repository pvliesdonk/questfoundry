"""MyST directive parser for extracting structured data from domain files.

This module provides the core parsing functionality for QuestFoundry's domain
specification files. It extracts custom directives from MyST (Markedly Structured
Text) markdown files, which serve as the single source of truth for roles,
workflows, artifacts, and protocol definitions.

Directive Syntax
----------------
Directives use the MyST fence syntax with YAML content::

    :::{directive-name}
    key: value
    nested:
      - item1
      - item2
    :::

The parser extracts all directives from a file and returns them as structured
:class:`Directive` objects with validated YAML content.

Supported Directive Types
-------------------------
The parser recognizes the following directive categories:

**Role Directives** (define agent personas):
    - ``role-meta``: Core role identity (id, archetype, agency, mandate)
    - ``role-tools``: Tools available to the role
    - ``role-constraints``: Behavioral constraints and guidelines
    - ``role-prompt``: Jinja2 system prompt template

**Loop Directives** (define workflow graphs):
    - ``loop-meta``: Loop identity and entry configuration
    - ``graph-node``: Individual workflow nodes
    - ``graph-edge``: Transitions between nodes with conditions
    - ``quality-gate``: Validation checkpoints

**Ontology Directives** (define data structures):
    - ``artifact-type``: Artifact definitions (like story hooks, scenes)
    - ``artifact-field``: Individual fields within artifacts
    - ``enum-type``: Enumeration types with values
    - ``enum-value``: Individual enum values (alternative syntax)

**Protocol Directives** (define runtime behavior):
    - ``intent-type``: Intent message types for inter-role communication
    - ``routing-rule``: Conditional routing logic
    - ``quality-bar``: Quality threshold definitions

Example Usage
-------------
Parse a single file::

    from pathlib import Path
    from questfoundry.compiler.parser import parse_myst_file

    result = parse_myst_file(Path("domain/roles/showrunner.md"))

    if result.has_errors:
        for error in result.errors:
            print(f"Error: {error}")
    else:
        role_meta = result.get_directive(DirectiveType.ROLE_META)
        print(f"Parsed role: {role_meta['id']}")

Parse a string directly::

    from questfoundry.compiler.parser import parse_myst_string, DirectiveType

    content = '''
    # My Role

    :::{role-meta}
    id: my_role
    archetype: Assistant
    agency: medium
    mandate: Help users
    :::
    '''

    result = parse_myst_string(content)
    meta = result.get_directive(DirectiveType.ROLE_META)

Parse an entire domain directory::

    from questfoundry.compiler.parser import parse_domain_directory

    results = parse_domain_directory(Path("src/questfoundry/domain"))
    for role_result in results["roles"]:
        print(f"Parsed: {role_result.source_path}")

See Also
--------
:mod:`questfoundry.compiler.models.ir` : IR models that consume parsed directives
:mod:`questfoundry.domain.ARCHITECTURE` : Full specification of directive schemas
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DirectiveType(Enum):
    """Enumeration of all supported MyST directive types.

    Each directive type corresponds to a specific kind of domain specification
    element. The enum value is the exact string used in MyST files (e.g.,
    ``:::{role-meta}`` uses ``ROLE_META``).

    Directive types are grouped into four categories:

    Role Directives
        Define agent personas and their behaviors. Each role file typically
        contains one ``role-meta`` plus supporting directives.

    Loop Directives
        Define workflow graphs as state machines. Loops connect roles via
        nodes and edges with conditional transitions.

    Ontology Directives
        Define the data model - artifacts (like HookCard, Scene) and their
        fields, plus enumeration types for constrained values.

    Protocol Directives
        Define runtime behavior including intent types for communication,
        routing rules, and quality thresholds.

    Examples
    --------
    Convert from string::

        directive_type = DirectiveType.from_string("role-meta")
        assert directive_type == DirectiveType.ROLE_META

    Check directive category::

        if directive_type in {DirectiveType.ROLE_META, DirectiveType.ROLE_TOOLS}:
            print("This is a role directive")
    """

    # Role directives - define agent personas
    ROLE_META = "role-meta"
    """Core role identity: id, abbr, archetype, agency level, mandate."""

    ROLE_TOOLS = "role-tools"
    """List of tools/functions available to this role."""

    ROLE_CONSTRAINTS = "role-constraints"
    """Behavioral constraints and guidelines for the role."""

    ROLE_PROMPT = "role-prompt"
    """Jinja2 template for the role's system prompt."""

    # Loop directives - define workflow graphs
    LOOP_META = "loop-meta"
    """Loop identity: id, name, trigger type, entry point node."""

    GRAPH_NODE = "graph-node"
    """A node in the workflow graph, mapping to a role."""

    GRAPH_EDGE = "graph-edge"
    """A transition between nodes with optional condition."""

    QUALITY_GATE = "quality-gate"
    """Validation checkpoint requiring quality bars to pass."""

    # Ontology directives - define data structures
    ARTIFACT_TYPE = "artifact-type"
    """Artifact definition: id, name, fields, storage policy."""

    ARTIFACT_FIELD = "artifact-field"
    """Individual field within an artifact (alternative syntax)."""

    ENUM_TYPE = "enum-type"
    """Enumeration type with constrained values."""

    ENUM_VALUE = "enum-value"
    """Individual enum value (alternative syntax)."""

    # Protocol directives - define runtime behavior
    INTENT_TYPE = "intent-type"
    """Intent message type for inter-role communication."""

    ROUTING_RULE = "routing-rule"
    """Conditional routing logic for workflow transitions."""

    QUALITY_BAR = "quality-bar"
    """Quality threshold definition for validation gates."""

    @classmethod
    def from_string(cls, name: str) -> DirectiveType | None:
        """Convert a directive name string to DirectiveType enum.

        Parameters
        ----------
        name : str
            The directive name as it appears in MyST files (e.g., "role-meta").

        Returns
        -------
        DirectiveType | None
            The corresponding DirectiveType, or None if not recognized.

        Examples
        --------
        >>> DirectiveType.from_string("role-meta")
        <DirectiveType.ROLE_META: 'role-meta'>

        >>> DirectiveType.from_string("unknown")
        None
        """
        for directive_type in cls:
            if directive_type.value == name:
                return directive_type
        return None


@dataclass
class Directive:
    """A parsed MyST directive with its content.

    Represents a single directive block extracted from a MyST file. The directive
    contains both the parsed YAML content (as a dictionary) and metadata about
    its location in the source file.

    Attributes
    ----------
    type : DirectiveType
        The type of directive (e.g., ROLE_META, GRAPH_NODE).
    content : dict[str, Any]
        The parsed YAML content as a dictionary. For ``role-prompt`` directives,
        this contains ``{"template": "..."}`` with the raw Jinja2 template.
    line_number : int
        The 1-based line number where the directive starts in the source file.
    raw_content : str
        The original unparsed content between the directive fences.

    Examples
    --------
    Access content using dict-like syntax::

        directive = result.get_directive(DirectiveType.ROLE_META)
        role_id = directive["id"]
        agency = directive.get("agency", "medium")

    Check available fields::

        print(directive.keys())  # ['id', 'abbr', 'archetype', 'agency', 'mandate']
    """

    type: DirectiveType
    """The directive type enum value."""

    content: dict[str, Any]
    """Parsed YAML content as a dictionary."""

    line_number: int
    """1-based line number of directive start in source file."""

    raw_content: str = ""
    """Original unparsed content between directive fences."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the directive content with optional default.

        Parameters
        ----------
        key : str
            The key to look up in the content dictionary.
        default : Any, optional
            Value to return if key is not found. Defaults to None.

        Returns
        -------
        Any
            The value for the key, or the default if not found.
        """
        return self.content.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get a value from the directive content.

        Parameters
        ----------
        key : str
            The key to look up in the content dictionary.

        Returns
        -------
        Any
            The value for the key.

        Raises
        ------
        KeyError
            If the key is not found in the content.
        """
        return self.content[key]

    def keys(self) -> list[str]:
        """Get all keys in the directive content.

        Returns
        -------
        list[str]
            List of all keys present in the content dictionary.
        """
        return list(self.content.keys())


@dataclass
class ParseResult:
    """Result of parsing a MyST file.

    Contains all directives extracted from a file, any prose sections (markdown
    headers and their content), and any errors encountered during parsing.

    The result provides convenient methods for accessing directives by type,
    which is the primary way to extract specific configuration elements.

    Attributes
    ----------
    directives : list[Directive]
        All successfully parsed directives from the file.
    prose_sections : dict[str, str]
        Mapping of markdown header text to the prose content under that header.
        Useful for extracting documentation alongside structured data.
    errors : list[str]
        List of error messages for any parsing failures. Check ``has_errors``
        before using the result.
    source_path : Path | None
        Path to the source file, if parsed from a file (None if parsed from string).

    Examples
    --------
    Check for errors before processing::

        result = parse_myst_file(path)
        if result.has_errors:
            for error in result.errors:
                logger.error(error)
            raise ValueError("Parse errors encountered")

    Get a singular directive (e.g., role-meta)::

        meta = result.get_directive(DirectiveType.ROLE_META)
        if meta:
            print(f"Role: {meta['id']}")

    Get multiple directives (e.g., all graph nodes)::

        nodes = result.get_directives(DirectiveType.GRAPH_NODE)
        for node in nodes:
            print(f"Node: {node['id']} -> Role: {node['role']}")

    Access prose documentation::

        if "Overview" in result.prose_sections:
            print(result.prose_sections["Overview"])
    """

    directives: list[Directive] = field(default_factory=list)
    """All successfully parsed directives."""

    prose_sections: dict[str, str] = field(default_factory=dict)
    """Mapping of header text to prose content."""

    errors: list[str] = field(default_factory=list)
    """List of error messages from parsing failures."""

    source_path: Path | None = None
    """Path to source file, or None if parsed from string."""

    def get_directives(self, directive_type: DirectiveType) -> list[Directive]:
        """Get all directives of a specific type.

        Use this for directive types that can appear multiple times in a file,
        such as ``graph-node`` or ``graph-edge``.

        Parameters
        ----------
        directive_type : DirectiveType
            The type of directives to retrieve.

        Returns
        -------
        list[Directive]
            List of matching directives (may be empty).

        Examples
        --------
        >>> nodes = result.get_directives(DirectiveType.GRAPH_NODE)
        >>> len(nodes)
        3
        """
        return [d for d in self.directives if d.type == directive_type]

    def get_directive(self, directive_type: DirectiveType) -> Directive | None:
        """Get the first directive of a specific type.

        Use this for directive types that should appear at most once in a file,
        such as ``role-meta`` or ``loop-meta``. If multiple exist, returns
        the first one found.

        Parameters
        ----------
        directive_type : DirectiveType
            The type of directive to retrieve.

        Returns
        -------
        Directive | None
            The first matching directive, or None if not found.

        Examples
        --------
        >>> meta = result.get_directive(DirectiveType.ROLE_META)
        >>> meta["id"]
        'showrunner'
        """
        directives = self.get_directives(directive_type)
        return directives[0] if directives else None

    @property
    def has_errors(self) -> bool:
        """Check if parsing produced any errors.

        Returns
        -------
        bool
            True if there were any parsing errors, False otherwise.
        """
        return len(self.errors) > 0


# Regex pattern for MyST directive blocks
# Matches: :::{directive-name}\n...content...\n:::
DIRECTIVE_PATTERN = re.compile(
    r"^:::\{([a-z-]+)\}\s*$\n"  # Opening fence with directive name
    r"(.*?)"  # Content (non-greedy)
    r"^:::\s*$",  # Closing fence
    re.MULTILINE | re.DOTALL,
)

# Pattern for markdown headers
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def parse_myst_string(content: str, source_path: Path | None = None) -> ParseResult:
    """Parse MyST content string and extract all directives.

    This is the core parsing function that processes MyST markdown content and
    extracts all QuestFoundry directives. It handles YAML parsing, error
    collection, and prose section extraction.

    The parser recognizes directives in the MyST fence syntax::

        :::{directive-name}
        yaml: content
        here: values
        :::

    Special handling is applied for ``role-prompt`` directives, which contain
    Jinja2 templates that should not be parsed as YAML.

    Parameters
    ----------
    content : str
        The MyST markdown content to parse. May contain multiple directives.
    source_path : Path | None, optional
        Path to the source file, used for error messages. Defaults to None.

    Returns
    -------
    ParseResult
        Result object containing:
        - ``directives``: List of successfully parsed Directive objects
        - ``prose_sections``: Dict mapping headers to their content
        - ``errors``: List of error messages for any failures
        - ``source_path``: The provided source path

    Notes
    -----
    - Unknown directive types are reported as errors but don't stop parsing
    - Invalid YAML is reported as an error for that directive only
    - List-only YAML content is wrapped as ``{"items": [...]}``
    - Scalar-only YAML content is wrapped as ``{"value": ...}``

    Examples
    --------
    Parse a simple role definition::

        content = '''
        # Showrunner

        :::{role-meta}
        id: showrunner
        archetype: Product Owner
        agency: high
        mandate: Manage by Exception
        :::
        '''
        result = parse_myst_string(content)
        meta = result.get_directive(DirectiveType.ROLE_META)
        assert meta["id"] == "showrunner"

    Handle parse errors::

        result = parse_myst_string(":::{unknown}\\nfoo: bar\\n:::")
        assert result.has_errors
        assert "unknown" in result.errors[0].lower()
    """
    result = ParseResult(source_path=source_path)

    # Find all directive blocks
    for match in DIRECTIVE_PATTERN.finditer(content):
        directive_name = match.group(1)
        directive_content = match.group(2).strip()

        # Calculate line number
        line_number = content[: match.start()].count("\n") + 1

        # Try to parse as DirectiveType
        directive_type = DirectiveType.from_string(directive_name)

        if directive_type is None:
            result.errors.append(f"Unknown directive '{directive_name}' at line {line_number}")
            continue

        # Parse YAML content
        try:
            # Handle special case for role-prompt which may contain Jinja2 templates
            if directive_type == DirectiveType.ROLE_PROMPT:
                # Store as raw string under 'template' key
                parsed_content = {"template": directive_content}
            else:
                parsed_content = yaml.safe_load(directive_content) or {}

            # Ensure we have a dict
            if not isinstance(parsed_content, dict):
                # If it's a list (like role-tools), wrap it
                if isinstance(parsed_content, list):
                    parsed_content = {"items": parsed_content}
                else:
                    parsed_content = {"value": parsed_content}

        except yaml.YAMLError as e:
            result.errors.append(f"Invalid YAML in '{directive_name}' at line {line_number}: {e}")
            continue

        directive = Directive(
            type=directive_type,
            content=parsed_content,
            line_number=line_number,
            raw_content=directive_content,
        )
        result.directives.append(directive)

    # Extract prose sections (headers and their content)
    # This is useful for documentation extraction
    current_header: str | None = None
    current_content: list[str] = []

    # Remove directive blocks from content for prose extraction
    prose_content = DIRECTIVE_PATTERN.sub("", content)

    for line in prose_content.split("\n"):
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            # Save previous section
            if current_header is not None:
                result.prose_sections[current_header] = "\n".join(current_content).strip()

            current_header = header_match.group(2).strip()
            current_content = []
        elif current_header is not None:
            current_content.append(line)

    # Save last section
    if current_header is not None:
        result.prose_sections[current_header] = "\n".join(current_content).strip()

    return result


def parse_myst_file(path: Path) -> ParseResult:
    """Parse a MyST file and extract all directives.

    Convenience wrapper around :func:`parse_myst_string` that handles file
    reading and error handling.

    Parameters
    ----------
    path : Path
        Path to the MyST markdown file to parse.

    Returns
    -------
    ParseResult
        Result object containing directives, prose sections, and any errors.
        If the file cannot be read, the result will contain an error message
        but no directives.

    Examples
    --------
    Parse a role definition file::

        from pathlib import Path
        result = parse_myst_file(Path("domain/roles/showrunner.md"))

        if result.has_errors:
            print(f"Errors in {result.source_path}:")
            for error in result.errors:
                print(f"  - {error}")
        else:
            meta = result.get_directive(DirectiveType.ROLE_META)
            print(f"Successfully parsed role: {meta['id']}")

    See Also
    --------
    parse_myst_string : Parse content from a string
    parse_domain_directory : Parse all files in a domain directory
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        result = ParseResult(source_path=path)
        result.errors.append(f"Failed to read file: {e}")
        return result

    return parse_myst_string(content, source_path=path)


def parse_domain_directory(domain_path: Path) -> dict[str, list[ParseResult]]:
    """Parse all MyST files in the domain directory structure.

    Scans the standard domain directory layout and parses all ``.md`` files
    in each subdirectory. Files starting with ``_`` or having uppercase names
    (like ``ARCHITECTURE.md``) are skipped.

    Expected directory structure::

        domain/
        ├── roles/           # Role definitions (showrunner.md, plotwright.md, ...)
        ├── loops/           # Workflow definitions (story_spark.md, ...)
        ├── ontology/        # Data models (artifacts.md, enums.md, ...)
        └── protocol/        # Runtime config (intents.md, routing.md, ...)

    Parameters
    ----------
    domain_path : Path
        Path to the root domain directory.

    Returns
    -------
    dict[str, list[ParseResult]]
        Dictionary with keys ``"roles"``, ``"loops"``, ``"ontology"``, ``"protocol"``,
        each mapping to a list of ParseResult objects for files in that subdirectory.
        Missing subdirectories result in empty lists.

    Examples
    --------
    Parse entire domain and check for errors::

        results = parse_domain_directory(Path("src/questfoundry/domain"))

        all_errors = []
        for category, parse_results in results.items():
            for result in parse_results:
                if result.has_errors:
                    all_errors.extend(
                        f"{result.source_path}: {err}"
                        for err in result.errors
                    )

        if all_errors:
            print("Parse errors found:")
            for error in all_errors:
                print(f"  - {error}")

    Count directives by type across all files::

        from collections import Counter

        results = parse_domain_directory(Path("src/questfoundry/domain"))
        type_counts = Counter()

        for category, parse_results in results.items():
            for result in parse_results:
                for directive in result.directives:
                    type_counts[directive.type] += 1

        print(type_counts)

    See Also
    --------
    parse_myst_file : Parse a single file
    """
    results: dict[str, list[ParseResult]] = {
        "roles": [],
        "loops": [],
        "ontology": [],
        "protocol": [],
    }

    for subdir in results:
        subdir_path = domain_path / subdir
        if not subdir_path.exists():
            continue

        for md_file in subdir_path.glob("*.md"):
            # Skip non-domain files like ARCHITECTURE.md
            if md_file.name.startswith("_") or md_file.name.isupper():
                continue

            result = parse_myst_file(md_file)
            results[subdir].append(result)

    return results
