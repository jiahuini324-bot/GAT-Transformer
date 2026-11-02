import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def draw(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = json.loads((directory / "graph.json").read_text())
    case = json.loads((directory / "case.json").read_text())
    points = np.asarray(graph["positions_m"])
    with (directory / "edges.csv").open(newline="") as stream:
        edges = list(csv.DictReader(stream))
    with (directory / "results/displacement.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    displacement = np.asarray(
        [[float(row[key]) for key in ("ux_m", "uy_m", "uz_m")] for row in rows]
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    for result in (False, True):
        figure = plt.figure(figsize=(9.2, 6.8))
        axes = figure.add_subplot(111, projection="3d")
        positions = points + displacement if result else points
        for kind, color, width, alpha in (
            (2, "#d5a047", 0.7, 0.5),
            (1, "#53677a", 1.5, 0.85),
            (0, "#263b50", 1.7, 0.9),
        ):
            segments = [
                [positions[int(row["source"])], positions[int(row["target"])]]
                for row in edges
                if int(row["type"]) == kind and float(row["initial_damage"]) == 0
            ]
            axes.add_collection3d(
                Line3DCollection(segments, colors=color, linewidths=width, alpha=alpha)
            )
        if result:
            values = displacement[:, 2] * 1000
            points_plot = axes.scatter(
                *positions.T,
                c=values,
                cmap="viridis",
                s=33,
                depthshade=False,
                vmin=values.min(),
                vmax=0,
            )
            bar = figure.colorbar(points_plot, ax=axes, shrink=0.65, pad=0.05)
            bar.set_label("Terminal Z displacement (mm)")
            axes.set_title(
                f"Example 01 | FE graph-node displacement\nt = {case['result_time_s']:.7f} s, deformation scale = 1",
                pad=18,
            )
        else:
            axes.scatter(*positions.T, color="#263b50", s=14, depthshade=False)
            removed = [
                [points[int(row["source"])], points[int(row["target"])]]
                for row in edges
                if float(row["initial_damage"]) > 0
            ]
            axes.add_collection3d(
                Line3DCollection(
                    removed, colors="#cf4144", linewidths=3, linestyles="dashed"
                )
            )
            axes.set_title(
                "Example 01 | Structural graph\n3 storeys, 3 x 3 bays, corner-column removal",
                pad=18,
            )
            axes.legend(
                handles=[
                    Line2D([0], [0], color="#263b50", label="Beam / column"),
                    Line2D([0], [0], color="#d5a047", label="Slab diagonal"),
                    Line2D(
                        [0],
                        [0],
                        color="#cf4144",
                        linestyle="--",
                        label="Removed column",
                    ),
                ],
                loc="upper left",
                bbox_to_anchor=(0.02, 0.96),
                frameon=False,
            )
        axes.set_xlabel("X (m)", labelpad=10)
        axes.set_ylabel("Y (m)", labelpad=10)
        axes.set_zlabel("")
        axes.text2D(
            -0.06,
            0.5,
            "Z (m)",
            transform=axes.transAxes,
            rotation=90,
            ha="center",
            va="center",
        )
        axes.set_xlim(-1, points[:, 0].max() + 1)
        axes.set_ylim(-1, points[:, 1].max() + 1)
        axes.set_zlim(-0.5, points[:, 2].max() + 0.5)
        axes.set_box_aspect(np.ptp(points, axis=0))
        axes.view_init(elev=23, azim=42)
        figure.subplots_adjust(left=0.02, right=0.94, bottom=0.04, top=0.90)
        figure.savefig(
            output / ("terminal_displacement.png" if result else "graph.png"),
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path(__file__).parent / "example")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    draw(arguments.case, arguments.output or arguments.case / "figures")
