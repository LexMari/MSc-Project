"""Tests for sim_core/batch.py"""
import csv
import os
import tempfile
import yaml

from sim_core.batch import run_batch

def test_batch_runs_every_combination():
    """A sweep with two fields of sizes 2 and 3 should produce 6 runs"""
    batch_config = {
        "name": "combination test",
        "base_scenario": "scenarios/radar_spoof_masking.yaml",
        "output_csv": os.path.join(tempfile.gettempdir(), "_batch_combination_test.csv"),
        "sweep": {
            "random_seed": {"count": 2},
            "solo.fusion_policy": ["camera_priority", "majority_vote", "confidence_weighted"],
        },
    }
    path = os.path.join(tempfile.gettempdir(), "_batch_combination_test.yaml")
    with open(path, "w") as f:
        yaml.dump(batch_config, f)
    rows = run_batch(path, quiet=True)
    run_ids = {row["run_id"] for row in rows}
    assert len(run_ids) == 6

def test_batch_reproduces_known_policy_pattern():
    """confidence_weighted should be defeated by radar_spoof_masking every time"""
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

def test_batch_writes_csv_with_row_per_vehicle():
    """A scenario with more than one vehicle should produce one row per vehicle per run"""
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
    assert len(rows) > 1, "expected more than one vehicle's worth of rows"

    with open(output_path) as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(rows)

def test_batch_seed_count_produces_correct_range():
    """{count: N} should sweep seeds 0 through N-1"""
    batch_config = {
        "name": "seed test",
        "base_scenario": "scenarios/radar_spoof_masking.yaml",
        "output_csv": os.path.join(tempfile.gettempdir(), "_batch_seed_test.csv"),
        "sweep": {"random_seed": {"count": 4}},
    }
    path = os.path.join(tempfile.gettempdir(), "_batch_seed_config.yaml")
    with open(path, "w") as f:
        yaml.dump(batch_config, f)
    rows = run_batch(path, quiet=True)
    seeds_used = {row["random_seed"] for row in rows}
    assert seeds_used == {0, 1, 2, 3}
