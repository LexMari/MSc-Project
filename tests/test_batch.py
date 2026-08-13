"""Tests for sim_core/batch.py"""
import csv
import os
import tempfile
import yaml

from sim_core.batch import run_batch

def test_batch_runs_every_combination_in_the_cross_product():
    """A sweep with two fields of sizes 2 and 3 should produce 6 runs,
    each showing up as its own run_id."""
    batch_config = {
        "name": "cross product test",
        "base_scenario": "scenarios/radar_spoof_masking.yaml",
        "output_csv": os.path.join(tempfile.gettempdir(), "_batch_cross_product_test.csv"),
        "sweep": {
            "random_seed": {"count": 2},
            "solo.fusion_policy": ["camera_priority", "majority_vote", "confidence_weighted"],
        },
    }
    path = os.path.join(tempfile.gettempdir(), "_batch_cross_product_config.yaml")
    with open(path, "w") as f:
        yaml.dump(batch_config, f)
    rows = run_batch(path, quiet=True)
    run_ids = {row["run_id"] for row in rows}
    assert len(run_ids) == 6

def test_batch_reproduces_the_known_per_policy_pattern():
    """confidence_weighted should be defeated by radar_spoof_masking
    every time, the other three policies never."""
    batch_config = {
        "name": "policy pattern test",
        "base_scenario": "scenarios/radar_spoof_masking.yaml",
        "output_csv": os.path.join(tempfile.gettempdir(), "_batch_policy_pattern_test.csv"),
        "sweep": {
            "random_seed": {"count": 5},
            "solo.fusion_policy": ["camera_priority", "majority_vote", "confidence_weighted", "plausibility_filtered"],
        },
    }
    path = os.path.join(tempfile.gettempdir(), "_batch_policy_pattern_config.yaml")
    with open(path, "w") as f:
        yaml.dump(batch_config, f)
    rows = run_batch(path, quiet=True)

    for row in rows:
        if row["vehicle_id"] != "solo":
            continue
        if row["solo.fusion_policy"] == "confidence_weighted":
            assert row["collided"] is True
        else:
            assert row["collided"] is False

def test_batch_writes_a_csv_with_a_row_per_vehicle():
    """A scenario with more than one vehicle should produce one row per
    vehicle per run, not one row per run."""
    output_path = os.path.join(tempfile.gettempdir(), "_batch_multi_vehicle_test.csv")
    batch_config = {
        "name": "multi vehicle row test",
        "base_scenario": "scenarios/background_traffic.yaml",
        "output_csv": output_path,
        "sweep": {"random_seed": {"count": 1}},
    }
    path = os.path.join(tempfile.gettempdir(), "_batch_multi_vehicle_config.yaml")
    with open(path, "w") as f:
        yaml.dump(batch_config, f)
    rows = run_batch(path, quiet=True)
    assert len(rows) > 1, "expected more than one vehicle's worth of rows given background traffic spawns"

    with open(output_path) as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)

def test_batch_seed_count_shorthand_produces_a_range():
    """{count: N} should sweep seeds 0 through N-1, not N copies of the
    same seed."""
    batch_config = {
        "name": "seed shorthand test",
        "base_scenario": "scenarios/radar_spoof_masking.yaml",
        "output_csv": os.path.join(tempfile.gettempdir(), "_batch_seed_shorthand_test.csv"),
        "sweep": {"random_seed": {"count": 4}},
    }
    path = os.path.join(tempfile.gettempdir(), "_batch_seed_shorthand_config.yaml")
    with open(path, "w") as f:
        yaml.dump(batch_config, f)
    rows = run_batch(path, quiet=True)
    seeds_used = {row["random_seed"] for row in rows}
    assert seeds_used == {0, 1, 2, 3}
