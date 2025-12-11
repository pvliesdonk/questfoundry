"""QuestFoundry Stores - Hot and Cold storage for creative artifacts.

Hot Store: Ephemeral, in-memory workspace for active creative work.
Cold Store: Persistent SQLite + files for player-safe canon.

Rule: Hot discovers and argues. Cold agrees and ships.
"""

from questfoundry.runtime.stores.cold_store import (
    AssetProvenance,
    AssetType,
    BookMetadata,
    ColdAsset,
    ColdSection,
    ColdSnapshot,
    ColdStore,
    get_cold_store,
)
from questfoundry.runtime.stores.hot_store import HotStore

__all__ = [
    # Hot Store
    "HotStore",
    # Cold Store
    "ColdStore",
    "BookMetadata",
    "ColdSection",
    "ColdAsset",
    "ColdSnapshot",
    "AssetType",
    "AssetProvenance",
    "get_cold_store",
]
