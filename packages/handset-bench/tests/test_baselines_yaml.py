"""Tests for the reproducibility contract in baselines.yaml."""

from pathlib import Path

import pytest
import yaml
from handset_bench.adapters.registry import resolve
from handset_bench.conditions import CONDITIONS
from handset_bench.textset.loader import TEXTSET_NAME

BASELINES = Path(__file__).resolve().parents[3] / "baselines.yaml"

#: Adapters whose vendor package is not installed in the local test environment.
VENDOR_ONLY = {"handset_bench.adapters.zipvoice:ZipVoiceAdapter"}


@pytest.fixture(scope="module")
def contract() -> dict:
    if not BASELINES.exists():
        pytest.skip("baselines.yaml not present")
    return yaml.safe_load(BASELINES.read_text())


def test_contract_exists():
    assert BASELINES.exists(), "the reproducibility contract must be committed"


def test_textset_matches_the_frozen_set(contract):
    assert contract["textset"] == TEXTSET_NAME


def test_conditions_match_the_harness(contract):
    assert set(contract["conditions"]) == set(CONDITIONS)


def test_every_system_has_a_pinned_version(contract):
    """A run that cannot be attributed to a version is discarded, not published."""
    for entry in contract["systems"]:
        version = entry.get("version", "")
        assert version, f"{entry['name']} has no version"
        assert version.lower() not in {"latest", "main", "head", "unknown"}


def test_every_system_has_an_image_and_settings(contract):
    for entry in contract["systems"]:
        assert entry.get("image"), f"{entry['name']} has no image"
        assert "settings" in entry, f"{entry['name']} has no settings block"


def test_every_adapter_path_is_well_formed(contract):
    for entry in contract["systems"]:
        assert ":" in entry["adapter"], entry["adapter"]


def test_locally_installable_adapters_resolve(contract):
    for entry in contract["systems"]:
        if entry["adapter"] in VENDOR_ONLY:
            continue
        assert resolve(entry["adapter"]) is not None


def test_exactly_one_headline_listener(contract):
    roles = [listener["role"] for listener in contract["listeners"]]
    assert roles.count("headline") == 1


def test_headline_listener_is_parakeet(contract):
    headline = next(
        listener for listener in contract["listeners"] if listener["role"] == "headline"
    )
    assert "parakeet" in headline["name"]


def test_every_listener_decodes_greedily(contract):
    """Any sampling would break the reproducibility guarantee."""
    for listener in contract["listeners"]:
        assert listener["decode"] == "greedy"


def test_prompt_policy_uses_heldout_speakers(contract):
    policy = contract["prompt_policy"]
    assert "heldout" in policy["source"]
    assert policy["n_speakers"] == 3


def test_zipvoice_variants_are_separate_entries(contract):
    """A faster mode and a better mode are two rows, not one measured twice."""
    names = {entry["name"] for entry in contract["systems"]}
    assert {"zipvoice", "zipvoice_distill"} <= names


def test_zipvoice_distill_records_the_upstream_step_count(contract):
    """Upstream ships 8 steps, not the 4 the brief assumed. Measured as shipped."""
    entry = next(e for e in contract["systems"] if e["name"] == "zipvoice_distill")
    assert entry["settings"]["num_step"] == 8
    assert entry["settings"]["guidance_scale"] == 3.0


def test_pending_systems_state_why_they_are_absent(contract):
    """An unrun system is a stated fact, not a silent omission."""
    for entry in contract.get("pending", []):
        assert entry.get("reason"), f"{entry['name']} gives no reason"


def test_no_system_name_is_duplicated(contract):
    names = [entry["name"] for entry in contract["systems"]]
    assert len(set(names)) == len(names)
