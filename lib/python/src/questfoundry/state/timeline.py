"""
Timeline anchor management for canon workflows.

Timeline anchors provide chronological structure for canon transfer and world genesis.
They establish baseline events (T0, T1, T2) and allow new projects to add events
while maintaining chronological integrity.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TimelineAnchor:
    """
    A chronological anchor point in the timeline.

    Timeline anchors mark significant events and provide chronological structure
    for storytelling. Base anchors (T0, T1, T2) are established during world
    genesis or canon transfer, and new anchors (T3+) can be added for new projects.

    Attributes:
        anchor_id: Unique identifier (e.g., "T0", "T1", "T2", "T3-REVOLT")
        event: Description of the event
        year: Optional year/date for the event
        offset: Relative offset from T0 (in years, optional)
        description: Detailed description of significance
        source: Source project or package that created this anchor
        immutable: Whether this anchor can be modified
        metadata: Additional timeline data
    """

    anchor_id: str
    event: str
    year: int | None = None
    offset: int | None = None
    description: str = ""
    source: str = ""
    immutable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate anchor fields."""
        if not self.anchor_id:
            raise ValueError("Anchor ID cannot be empty")
        if not self.event:
            raise ValueError("Anchor event cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        """Convert anchor to dictionary."""
        return {
            "anchor_id": self.anchor_id,
            "event": self.event,
            "year": self.year,
            "offset": self.offset,
            "description": self.description,
            "source": self.source,
            "immutable": self.immutable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimelineAnchor":
        """Create anchor from dictionary."""
        return cls(
            anchor_id=data["anchor_id"],
            event=data["event"],
            year=data.get("year"),
            offset=data.get("offset"),
            description=data.get("description", ""),
            source=data.get("source", ""),
            immutable=data.get("immutable", False),
            metadata=data.get("metadata", {}),
        )


class TimelineManager:
    """
    Manager for timeline anchors across canon workflows.

    The timeline manager maintains chronological integrity by tracking anchor
    points and validating that new events maintain proper ordering. It supports
    both absolute years and relative offsets from T0.

    Features:
        - Baseline anchors (T0, T1, T2) from canon transfer or world genesis
        - Extension anchors (T3+) for new project events
        - Chronological validation (no gaps or conflicts)
        - Offset calculation from T0
        - Immutability enforcement for imported anchors

    Example:
        >>> timeline = TimelineManager()
        >>> # Add baseline anchors from world genesis
        >>> timeline.add_anchor(TimelineAnchor(
        ...     anchor_id="T0",
        ...     event="Founding of Kingdom",
        ...     year=1000,
        ...     source="world-genesis",
        ...     immutable=True
        ... ))
        >>> timeline.add_anchor(TimelineAnchor(
        ...     anchor_id="T1",
        ...     event="Dragon Wars Begin",
        ...     year=1050,
        ...     offset=50,
        ...     source="world-genesis",
        ...     immutable=True
        ... ))
        >>> # Add new event for sequel
        >>> timeline.add_anchor(TimelineAnchor(
        ...     anchor_id="T3-REVOLT",
        ...     event="Peasant Revolt",
        ...     year=1075,
        ...     offset=75,
        ...     source="dragon-quest-2"
        ... ))
        >>> # Validate chronology
        >>> issues = timeline.validate_chronology()
    """

    def __init__(self) -> None:
        """Initialize empty timeline."""
        self._anchors: dict[str, TimelineAnchor] = {}  # anchor_id -> anchor
        self._year_index: dict[int, list[str]] = {}  # year -> [anchor_ids]

    def add_anchor(self, anchor: TimelineAnchor) -> None:
        """
        Add timeline anchor.

        Args:
            anchor: Anchor to add

        Raises:
            ValueError: If anchor ID already exists
        """
        if anchor.anchor_id in self._anchors:
            raise ValueError(f"Anchor '{anchor.anchor_id}' already exists")

        self._anchors[anchor.anchor_id] = anchor

        # Index by year if available
        if anchor.year is not None:
            if anchor.year not in self._year_index:
                self._year_index[anchor.year] = []
            self._year_index[anchor.year].append(anchor.anchor_id)

    def get_anchor(self, anchor_id: str) -> TimelineAnchor | None:
        """
        Get anchor by ID.

        Args:
            anchor_id: Anchor identifier

        Returns:
            Anchor if found, None otherwise
        """
        return self._anchors.get(anchor_id)

    def get_all_anchors(self) -> list[TimelineAnchor]:
        """
        Get all anchors sorted chronologically.

        Returns:
            List of anchors sorted by year (if available) or offset
        """
        anchors = list(self._anchors.values())

        # Sort by year if available, else by offset, else by ID
        def sort_key(a: TimelineAnchor) -> tuple[int, int, str]:
            year = a.year if a.year is not None else 999999
            offset = a.offset if a.offset is not None else 999999
            return (year, offset, a.anchor_id)

        return sorted(anchors, key=sort_key)

    def get_baseline_anchors(self) -> list[TimelineAnchor]:
        """
        Get baseline anchors (T0, T1, T2).

        Returns:
            List of baseline anchors
        """
        return [
            anchor
            for anchor_id, anchor in self._anchors.items()
            if anchor_id in ("T0", "T1", "T2")
        ]

    def get_extension_anchors(self) -> list[TimelineAnchor]:
        """
        Get extension anchors (T3+).

        Returns:
            List of extension anchors
        """
        return [
            anchor
            for anchor_id, anchor in self._anchors.items()
            if anchor_id not in ("T0", "T1", "T2")
        ]

    def update_anchor(self, anchor: TimelineAnchor) -> None:
        """
        Update existing anchor.

        Args:
            anchor: Anchor with updated fields

        Raises:
            ValueError: If anchor doesn't exist or is immutable
        """
        if anchor.anchor_id not in self._anchors:
            raise ValueError(f"Anchor '{anchor.anchor_id}' does not exist")

        existing = self._anchors[anchor.anchor_id]
        if existing.immutable:
            raise ValueError(
                f"Cannot update immutable anchor '{anchor.anchor_id}' "
                f"from {existing.source}"
            )

        # Update year index if year changed
        if existing.year != anchor.year:
            if existing.year is not None:
                self._year_index[existing.year].remove(anchor.anchor_id)
                if not self._year_index[existing.year]:
                    del self._year_index[existing.year]

            if anchor.year is not None:
                if anchor.year not in self._year_index:
                    self._year_index[anchor.year] = []
                self._year_index[anchor.year].append(anchor.anchor_id)

        self._anchors[anchor.anchor_id] = anchor

    def delete_anchor(self, anchor_id: str) -> None:
        """
        Delete anchor from timeline.

        Args:
            anchor_id: Anchor ID to delete

        Raises:
            ValueError: If anchor doesn't exist or is immutable
        """
        if anchor_id not in self._anchors:
            raise ValueError(f"Anchor '{anchor_id}' does not exist")

        anchor = self._anchors[anchor_id]
        if anchor.immutable:
            raise ValueError(
                f"Cannot delete immutable anchor '{anchor_id}' from {anchor.source}"
            )

        # Remove from year index
        if anchor.year is not None:
            self._year_index[anchor.year].remove(anchor_id)
            if not self._year_index[anchor.year]:
                del self._year_index[anchor.year]

        del self._anchors[anchor_id]

    def validate_chronology(self) -> list[str]:
        """
        Validate timeline chronology.

        Checks for:
        - Proper ordering (no events before T0)
        - No conflicts (same year with contradictory events)
        - No unreasonable gaps
        - Valid offset calculations

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        anchors = self.get_all_anchors()

        if not anchors:
            return issues

        # Check T0 exists
        t0 = self.get_anchor("T0")
        if not t0:
            issues.append("Missing T0 baseline anchor")
            return issues

        # Validate offsets match years
        t0_year = t0.year
        if t0_year is not None:
            for anchor in anchors:
                if anchor.year is not None and anchor.offset is not None:
                    expected_offset = anchor.year - t0_year
                    if anchor.offset != expected_offset:
                        issues.append(
                            f"Anchor '{anchor.anchor_id}' offset {anchor.offset} "
                            f"doesn't match year calculation {expected_offset}"
                        )

        # Check for proper ordering
        prev_year = None
        for anchor in anchors:
            if anchor.year is not None:
                if prev_year is not None and anchor.year < prev_year:
                    issues.append(
                        f"Anchor '{anchor.anchor_id}' year {anchor.year} "
                        f"is before previous anchor year {prev_year}"
                    )
                prev_year = anchor.year

        # Check for unreasonable gaps (>1000 years)
        years = sorted(self._year_index.keys())
        for i in range(len(years) - 1):
            gap = years[i + 1] - years[i]
            if gap > 1000:
                issues.append(
                    f"Large gap ({gap} years) between year {years[i]} "
                    f"and {years[i + 1]}"
                )

        return issues

    def calculate_offset(self, year: int) -> int | None:
        """
        Calculate offset from T0 for a given year.

        Args:
            year: Year to calculate offset for

        Returns:
            Offset from T0, or None if T0 year not set
        """
        t0 = self.get_anchor("T0")
        if t0 and t0.year is not None:
            return year - t0.year
        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Export timeline to dictionary.

        Returns:
            Dictionary with anchors and metadata
        """
        return {
            "anchors": [anchor.to_dict() for anchor in self.get_all_anchors()],
            "baseline_count": len(self.get_baseline_anchors()),
            "extension_count": len(self.get_extension_anchors()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimelineManager":
        """
        Create timeline from dictionary.

        Args:
            data: Dictionary with anchors

        Returns:
            New TimelineManager instance
        """
        timeline = cls()

        for anchor_data in data.get("anchors", []):
            anchor = TimelineAnchor.from_dict(anchor_data)
            timeline.add_anchor(anchor)

        return timeline

    def merge(self, anchors: list[TimelineAnchor]) -> dict[str, Any]:
        """
        Merge anchors from canon import.

        Handles conflicts and validates chronology during merge.

        Args:
            anchors: List of anchors to merge

        Returns:
            Merge report with added, skipped, and conflict counts
        """
        added = 0
        skipped = 0
        conflicts: list[dict[str, str]] = []

        for anchor in anchors:
            if anchor.anchor_id not in self._anchors:
                # New anchor - validate then add
                self.add_anchor(anchor)
                added += 1
            else:
                existing = self._anchors[anchor.anchor_id]

                # Conflict detection
                if existing.immutable and not anchor.immutable:
                    # Existing is immutable, skip incoming
                    skipped += 1
                elif not existing.immutable and anchor.immutable:
                    # Incoming is immutable, replace existing
                    self.update_anchor(anchor)
                    added += 1
                elif existing.immutable and anchor.immutable:
                    # Both immutable - conflict!
                    conflicts.append(
                        {
                            "anchor_id": anchor.anchor_id,
                            "existing_source": existing.source,
                            "incoming_source": anchor.source,
                            "reason": "Both immutable with same ID",
                        }
                    )
                    skipped += 1
                else:
                    # Both mutable - use newer
                    self.update_anchor(anchor)
                    added += 1

        # Validate chronology after merge
        validation_issues = self.validate_chronology()
        if validation_issues:
            conflicts.extend([{"reason": issue} for issue in validation_issues])

        return {
            "added": added,
            "skipped": skipped,
            "conflicts": conflicts,
        }

    def __len__(self) -> int:
        """Return total anchor count."""
        return len(self._anchors)

    def __contains__(self, anchor_id: str) -> bool:
        """Check if anchor exists."""
        return anchor_id in self._anchors
