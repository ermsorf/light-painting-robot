#!/usr/bin/env python3
"""Plot Cartesian paths stored in CSV files.

Example:
    python3 plot_path.py deer.csv
    python3 plot_path.py deer.csv --output deer.png --no-show
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_COLUMNS = (
    "Contour Index",
    "Point Index",
    "X (m)",
    "Y (m)",
    "Z (m)",
)


def read_path(csv_path: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    """Read the required Cartesian coordinates from a path CSV."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"{csv_path} is missing required columns: {missing}")

        contours: Dict[str, List[Tuple[float, float, float]]] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                contour = row["Contour Index"]
                point = (
                    float(row["X (m)"]),
                    float(row["Y (m)"]),
                    float(row["Z (m)"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid coordinate at line {line_number}") from error

            contours.setdefault(contour, []).append(point)

    if not contours:
        raise ValueError(f"{csv_path} does not contain any path points")
    return contours


def plot_path(contours: Dict[str, List[Tuple[float, float, float]]], title: str) -> None:
    """Render each contour as a separate line in a 3D matplotlib plot."""
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(111, projection="3d")
    for contour, points in contours.items():
        x_values, y_values, z_values = zip(*points)
        axes.plot(x_values, y_values, z_values, marker=".", label=f"Contour {contour}")

    axes.set_title(title)
    axes.set_xlabel("X (m)")
    axes.set_ylabel("Y (m)")
    axes.set_zlabel("Z (m)")
    if len(contours) > 1:
        axes.legend()
    figure.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="CSV file containing the Cartesian path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="save the plot to this image file",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open a graphical window (useful with --output)",
    )
    args = parser.parse_args()

    contours = read_path(args.csv_file)
    plot_path(contours, args.csv_file.stem)

    import matplotlib.pyplot as plt

    if args.output:
        plt.savefig(args.output, dpi=150)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()