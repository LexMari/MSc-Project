"""Generates graphs from a batch experiment's CSV output

Usage: python main_graph.py results/some_batch.csv [swept_field]

Produces two PNGs in results/graphs/:
  - <name>_collision_rate.png   bar chart of collision rate per group
  - <name>_min_speed.png        box plot of min_speed_mph per group
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTCOME_COLUMNS = {"run_id", "vehicle_id", "collided", "severity", "collision_time", "min_speed_mph", "final_speed_mph"}

FIELD_UNITS = {
    "start_distance": "m",
    "start_speed": "m/s",
    "duration": "s",
    "timestep": "s",
    "visibility": "fraction, 0-1",
}

def _axis_label(swept_field: str) -> str:
    bare_field = swept_field.split(".")[-1]
    unit = FIELD_UNITS.get(bare_field)
    return f"{swept_field} ({unit})" if unit else swept_field

def load_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))

def guess_swept_field(rows: list[dict]) -> str:
    candidate_fields = [f for f in rows[0].keys() if f not in OUTCOME_COLUMNS]
    scored = []
    for field in candidate_fields:
        n_distinct = len({r[field] for r in rows})
        if n_distinct > 1:
            scored.append((n_distinct, field))
    if scored:
        scored.sort()
        return scored[0][1]
    if candidate_fields:
        return candidate_fields[0]
    raise ValueError("no swept field found in this CSV - pass one as a second argument")

def _sort_groups(group_keys):
    try:
        return sorted(group_keys, key=float)
    except ValueError:
        return sorted(group_keys)

def plot_collision_rate(rows: list[dict], swept_field: str, out_path: str, title: str) -> None:
    totals = defaultdict(int)
    collisions = defaultdict(int)
    for row in rows:
        group = row[swept_field]
        totals[group] += 1
        if row["collided"] == "True":
            collisions[group] += 1

    groups = _sort_groups(totals.keys())
    rates = [100.0 * collisions[g] / totals[g] for g in groups]

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 5))
    bars = ax.bar(groups, rates, color="#c0392b")
    ax.set_ylabel("Collision rate (%)")
    ax.set_xlabel(_axis_label(swept_field))
    ax.set_title(title)
    ax.set_ylim(0, 105)
    for bar, group in zip(bars, groups):
        n = totals[group]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{collisions[group]}/{n}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_min_speed(rows: list[dict], swept_field: str, out_path: str, title: str) -> None:
    by_group = defaultdict(list)   # group -> list of (min_speed_mph, collided) tuples
    for row in rows:
        if row["min_speed_mph"]:
            by_group[row[swept_field]].append((float(row["min_speed_mph"]), row["collided"] == "True"))

    groups = _sort_groups(by_group.keys())
    data = [[v for v, _ in by_group[g]] for g in groups]
    multi_run = any(len(d) > 1 for d in data)

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 5))

    if multi_run:
        ax.boxplot(data, tick_labels=groups, showfliers=False, zorder=1,
                   boxprops={"color": "#888888"}, whiskerprops={"color": "#888888"}, capprops={"color": "#888888"})
    else:
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups)
        ax.set_xlim(0.5, len(groups) + 0.5)

    for i, g in enumerate(groups, start=1):
        n = len(by_group[g])
        for j, (speed, collided) in enumerate(by_group[g]):
            jitter = 0.0 if n == 1 else (j / max(n - 1, 1) - 0.5) * 0.5
            colour = "#c0392b" if collided else "#2980b9"
            ax.scatter(i + jitter, speed, color=colour, s=28, zorder=3, edgecolor="white", linewidth=0.6)

    ax.scatter([], [], color="#c0392b", label="collided")
    ax.scatter([], [], color="#2980b9", label="no collision")
    ax.legend(loc="upper right", framealpha=0.9)

    ax.set_ylabel("Lowest speed reached (mph)")
    ax.set_xlabel(_axis_label(swept_field))
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def main(csv_path: str, swept_field: str | None = None) -> None:
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    if swept_field is None:
        swept_field = guess_swept_field(rows)
        print(f"(no field given - using {swept_field!r}")

    os.makedirs("results/graphs", exist_ok=True)
    stem = os.path.splitext(os.path.basename(csv_path))[0]

    collision_path = f"results/graphs/{stem}_collision_rate.png"
    plot_collision_rate(rows, swept_field, collision_path, f"Collision rate by {_axis_label(swept_field)}\n({stem})")
    print(f"saved {collision_path}")

    speed_path = f"results/graphs/{stem}_min_speed.png"
    plot_min_speed(rows, swept_field, speed_path, f"Lowest speed reached by {_axis_label(swept_field)}\n({stem})")
    print(f"saved {speed_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main_graph.py results/some_batch.csv [swept_field]")
        sys.exit(1)
    path = sys.argv[1]
    field = sys.argv[2] if len(sys.argv) > 2 else None
    main(path, field)
