import pytest

from emporio_agent.tools import StoreTools


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    # Tuesday, two days after the last order in the dataset
    monkeypatch.setenv("EMPORIO_NOW", "2026-03-24T10:30")


@pytest.fixture(scope="session")
def tools():
    return StoreTools()
