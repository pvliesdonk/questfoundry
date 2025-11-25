from __future__ import annotations

import json
from pathlib import Path

import pytest

from questfoundry.runtime.core.state_manager import StateManager
from questfoundry.runtime.tools import ReadColdSOT, ReadHotSOT, WriteColdSOT, WriteHotSOT
from questfoundry.runtime.core.cold_store import ColdStore


@pytest.fixture()
def state_manager() -> StateManager:
    return StateManager()


@pytest.fixture()
def base_state(state_manager: StateManager):
    return state_manager.initialize_state(loop_id="story_spark", context={"scene_text": "x"})


def test_read_hot_returns_nested(base_state):
    base_state["hot_sot"]["section_briefs"] = [{"id": "A"}]
    tool = ReadHotSOT()

    result = tool._run(key="section_briefs", state=base_state)

    assert result == [{"id": "A"}]


def test_write_hot_appends_list(base_state):
    base_state["hot_sot"]["hooks"] = ["h1"]
    tool = WriteHotSOT()

    update = tool._run(key="hooks", value="h2", state=base_state)

    assert update["hot_sot"]["hooks"] == ["h1", "h2"]


def test_read_cold_from_store(tmp_path: Path):
    cold_store = ColdStore(base_dir=tmp_path)
    cold_store.save_cold("proj", {"canon": {"entries": 1}})
    tool = ReadColdSOT(cold_store=cold_store)

    result = tool._run(key="canon.entries", state=None, project_id="proj")

    assert result == 1


def test_write_cold_persists_and_returns(tmp_path: Path, base_state):
    cold_store = ColdStore(base_dir=tmp_path)
    tool = WriteColdSOT(cold_store=cold_store)

    update = tool._run(
        key="canon.section1", value={"title": "t"}, state=base_state, project_id="proj"
    )

    persisted = cold_store.load_cold("proj")
    assert update["cold_sot"]["canon"]["section1"]["title"] == "t"
    assert persisted["canon"]["section1"]["title"] == "t"
