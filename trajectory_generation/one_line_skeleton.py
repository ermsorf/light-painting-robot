"""Skeletonize a one-line drawing and extract its main contour."""

from pathlib import Path
from typing import Sequence

import cv2 as cv
import numpy as np
from pdf2image import convert_from_path


THRESHOLD_VALUE = 127
OPEN_CONTOUR_ENABLED = True


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from the specified file path."""
    image = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return image


def load_pdf(pdf_path: str | Path) -> np.ndarray:
    """Load the first page of a PDF as a grayscale image."""
    pages = convert_from_path(pdf_path, dpi=300)
    if not pages:
        raise FileNotFoundError(f"No pages found in PDF: {pdf_path}")
    return cv.cvtColor(np.array(pages[0]), cv.COLOR_RGB2GRAY)


def threshold_image(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image into a binary foreground image."""
    _, threshold = cv.threshold(
        image,
        THRESHOLD_VALUE,
        255,
        cv.THRESH_BINARY_INV,
    )
    return threshold


def skeletonize(binary_image: np.ndarray) -> np.ndarray:
    """Reduce a binary foreground image to one-pixel-wide centerlines."""
    if hasattr(cv, "ximgproc") and hasattr(cv.ximgproc, "thinning"):
        return cv.ximgproc.thinning(
            binary_image,
            cv.ximgproc.THINNING_ZHANGSUEN,
        )

    skeleton = binary_image > 0

    while True:
        neighbours = (
            skeleton[:-2, 1:-1],
            skeleton[:-2, 2:],
            skeleton[1:-1, 2:],
            skeleton[2:, 2:],
            skeleton[2:, 1:-1],
            skeleton[2:, :-2],
            skeleton[1:-1, :-2],
            skeleton[:-2, :-2],
        )
        neighbour_count = sum(neighbours)
        transitions = sum(
            current & ~following
            for current, following in zip(
                neighbours,
                neighbours[1:] + neighbours[:1],
            )
        )
        centre = skeleton[1:-1, 1:-1]
        remove = (
            centre
            & (neighbour_count >= 2)
            & (neighbour_count <= 6)
            & (transitions == 1)
            & ~(neighbours[0] & neighbours[2] & neighbours[4])
            & ~(neighbours[2] & neighbours[4] & neighbours[6])
        )
        first_changed = bool(remove.any())
        if first_changed:
            centre[remove] = False

        neighbours = (
            skeleton[:-2, 1:-1],
            skeleton[:-2, 2:],
            skeleton[1:-1, 2:],
            skeleton[2:, 2:],
            skeleton[2:, 1:-1],
            skeleton[2:, :-2],
            skeleton[1:-1, :-2],
            skeleton[:-2, :-2],
        )
        neighbour_count = sum(neighbours)
        transitions = sum(
            current & ~following
            for current, following in zip(
                neighbours,
                neighbours[1:] + neighbours[:1],
            )
        )
        centre = skeleton[1:-1, 1:-1]
        remove = (
            centre
            & (neighbour_count >= 2)
            & (neighbour_count <= 6)
            & (transitions == 1)
            & ~(neighbours[0] & neighbours[2] & neighbours[6])
            & ~(neighbours[0] & neighbours[4] & neighbours[6])
        )
        second_changed = bool(remove.any())
        if second_changed:
            centre[remove] = False

        if not first_changed and not second_changed:
            break

    return skeleton.astype(np.uint8) * 255


def simplify_contour(
    contour: np.ndarray,
    target_points: int | None,
    epsilon: float | None,
) -> np.ndarray:
    """Reduce a contour to a target number of points or a fixed tolerance."""
    if target_points is not None and target_points < 2:
        raise ValueError("target_points must be at least 2 or None.")
    if epsilon is not None and epsilon < 0:
        raise ValueError("epsilon must be non-negative or None.")

    if epsilon is not None:
        return cv.approxPolyDP(contour, epsilon, False)
    if target_points is None or len(contour) <= target_points:
        return contour

    low = 0.0
    high = cv.arcLength(contour, False)
    approximation = contour

    for _ in range(50):
        middle = (low + high) / 2
        candidate = cv.approxPolyDP(contour, middle, False)
        if len(candidate) > target_points:
            low = middle
        else:
            high = middle
            approximation = candidate

    return approximation


def open_contour_at_largest_gap(contour: np.ndarray) -> np.ndarray:
    """Reorder a closed contour to start after its largest internal gap."""
    if len(contour) < 3:
        return contour

    points = contour[:, 0, :].astype(float)
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not len(distances):
        return contour

    median_distance = float(np.median(distances))
    endpoint_distance = float(np.linalg.norm(points[-1] - points[0]))
    if median_distance > 0 and endpoint_distance > 3.0 * median_distance:
        print("Contour already appears open; skipping contour opening.")
        return contour

    split_idx = int(np.argmax(distances))
    largest_gap = float(distances[split_idx])
    print(f"Largest contour gap: {largest_gap:.2f} pixels")
    print(f"Split index: {split_idx}")

    return np.concatenate([
        contour[split_idx + 1:],
        contour[:split_idx + 1],
    ])


def extract_skeleton_contour(
    threshold: np.ndarray,
    target_points: int | None = None,
    epsilon: float | None = 2.0,
) -> list[np.ndarray]:
    """Extract and simplify the largest contour from a skeletonized image."""
    skeleton = skeletonize(threshold)
    contours, _ = cv.findContours(
        skeleton,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("No centerline points were found.")

    contour = max(contours, key=len)
    simplified = simplify_contour(contour, target_points, epsilon)
    if len(simplified) < 2:
        raise ValueError("Skeleton approximation produced too few points.")
    if OPEN_CONTOUR_ENABLED:
        simplified = open_contour_at_largest_gap(simplified)
    return [simplified]


def scale_contours(
    contours: Sequence[np.ndarray],
    drawing_width: float,
    drawing_x: float,
    center_y: float,
    center_z: float,
) -> list[list[float | int]]:
    """Convert image contours into robot coordinates in metres."""
    valid_contours = [contour for contour in contours if len(contour) > 0]
    if not valid_contours:
        raise ValueError("No valid contours were found.")

    all_points = np.vstack([contour[:, 0, :] for contour in valid_contours])
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    image_span = max(max_x - min_x, max_y - min_y)
    if image_span == 0:
        raise ValueError("Contours have no measurable size.")

    scale = drawing_width / image_span
    trajectory: list[list[float | int]] = []

    for contour_id, contour in enumerate(valid_contours):
        for point_index, (x_pixel, y_pixel) in enumerate(contour[:, 0, :]):
            y = (x_pixel - (min_x + max_x) / 2) * scale
            z = -(y_pixel - (min_y + max_y) / 2) * scale
            trajectory.append([
                contour_id,
                point_index,
                drawing_x,
                center_y + y,
                center_z + z,
            ])

    return trajectory
