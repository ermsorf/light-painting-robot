"""Detect and prepare drawable outlines from thresholded images."""

import cv2 as cv
import numpy as np
from pdf2image import convert_from_path


THRESHOLD_VALUE = 127
TRAJECTORY_MODE = "logo"
CONTOUR_EPSILON: float | None = 3.0
CLOSE_LOGO_CONTOURS = True
MAX_LOGO_SEGMENT_LENGTH_PIXELS: float | None = 100.0
MIN_CONTOUR_POINTS = 20
MIN_CONTOUR_LENGTH = 50.0


def load_pdf(pdf_path: str) -> np.ndarray:
    """Load the first PDF page as a grayscale image."""
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
            skeleton[:-2, 1:-1], skeleton[:-2, 2:],
            skeleton[1:-1, 2:], skeleton[2:, 2:],
            skeleton[2:, 1:-1], skeleton[2:, :-2],
            skeleton[1:-1, :-2], skeleton[:-2, :-2],
        )
        count = sum(neighbours)
        transitions = sum(
            current & ~following
            for current, following in zip(
                neighbours,
                neighbours[1:] + neighbours[:1],
            )
        )
        centre = skeleton[1:-1, 1:-1]
        remove = (
            centre & (count >= 2) & (count <= 6)
            & (transitions == 1)
            & ~(neighbours[0] & neighbours[2] & neighbours[4])
            & ~(neighbours[2] & neighbours[4] & neighbours[6])
        )
        first_changed = bool(remove.any())
        centre[remove] = False

        neighbours = (
            skeleton[:-2, 1:-1], skeleton[:-2, 2:],
            skeleton[1:-1, 2:], skeleton[2:, 2:],
            skeleton[2:, 1:-1], skeleton[2:, :-2],
            skeleton[1:-1, :-2], skeleton[:-2, :-2],
        )
        count = sum(neighbours)
        transitions = sum(
            current & ~following
            for current, following in zip(
                neighbours,
                neighbours[1:] + neighbours[:1],
            )
        )
        centre = skeleton[1:-1, 1:-1]
        remove = (
            centre & (count >= 2) & (count <= 6)
            & (transitions == 1)
            & ~(neighbours[0] & neighbours[2] & neighbours[6])
            & ~(neighbours[0] & neighbours[4] & neighbours[6])
        )
        second_changed = bool(remove.any())
        centre[remove] = False
        if not first_changed and not second_changed:
            break
    return skeleton.astype(np.uint8) * 255


def _skeleton_graph(
    skeleton: np.ndarray,
) -> tuple[
    set[tuple[int, int]],
    dict[tuple[int, int], list[tuple[int, int]]],
]:
    """Build an 8-connected graph from white skeleton pixels."""
    pixels = {
        (int(y), int(x))
        for y, x in np.argwhere(skeleton > 0)
    }
    neighbours: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in pixels:
        adjacent = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in pixels:
                    adjacent.append((y + dy, x + dx))
        neighbours[(y, x)] = adjacent
    return pixels, neighbours


def _skeleton_paths(
    skeleton: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[tuple[int, int]]]:
    """Trace graph edges between endpoints and junctions."""
    pixels, neighbours = _skeleton_graph(skeleton)

    endpoints = [
        pixel for pixel, adjacent in neighbours.items()
        if len(adjacent) == 1
    ]
    junctions = [
        pixel for pixel, adjacent in neighbours.items()
        if len(adjacent) >= 3
    ]
    nodes = set(endpoints + junctions)
    print(f"Number of endpoints: {len(endpoints)}")
    print(f"Number of junctions: {len(junctions)}")
    visited_edges: set[
        tuple[tuple[int, int], tuple[int, int]]
    ] = set()

    def edge_key(
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        first, second = sorted((first, second))
        return first, second

    paths: list[np.ndarray] = []
    for start in nodes:
        for next_pixel in neighbours[start]:
            key = edge_key(start, next_pixel)
            if key in visited_edges:
                continue

            path = [start]
            previous = start
            current = next_pixel
            visited_edges.add(key)
            while True:
                path.append(current)
                if current in nodes:
                    break
                candidates = [
                    candidate for candidate in neighbours[current]
                    if candidate != previous
                ]
                if not candidates:
                    break
                previous, current = current, candidates[0]
                visited_edges.add(edge_key(previous, current))

            paths.append(np.asarray(
                [[x, y] for y, x in path],
                dtype=np.int32,
            ).reshape(-1, 1, 2))

    # A closed skeleton loop has no endpoint or junction.
    for start in pixels - nodes:
        unvisited = [
            neighbour for neighbour in neighbours[start]
            if edge_key(start, neighbour) not in visited_edges
        ]
        if not unvisited:
            continue

        path = [start]
        previous = start
        current = unvisited[0]
        visited_edges.add(edge_key(start, current))
        while current != start:
            path.append(current)
            candidates = [
                candidate for candidate in neighbours[current]
                if candidate != previous
                and edge_key(current, candidate) not in visited_edges
            ]
            if not candidates:
                break
            previous, current = current, candidates[0]
            visited_edges.add(edge_key(previous, current))
        paths.append(np.asarray(
            [[x, y] for y, x in path],
            dtype=np.int32,
        ).reshape(-1, 1, 2))

    return paths, endpoints, junctions


def _extract_graph_contours(threshold: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract and simplify robot segments by tracing the skeleton graph."""
    skeleton = skeletonize(threshold)
    segments, _, _ = _skeleton_paths(skeleton)
    print(f"Number of segments: {len(segments)}")

    simplified_segments: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        segment_length = cv.arcLength(segment, False)
        print(
            f"Segment {index}: "
            f"{len(segment)} points, length {segment_length:.2f}"
        )
        if not valid_contour(segment):
            print(f"Segment {index}: discarded as noise")
            continue

        simplified = simplify_contour(segment, closed=False)
        if len(simplified) < 2:
            print(f"Segment {index}: discarded after simplification")
            continue
        print(
            f"Segment {index}: "
            f"{len(simplified)} points after simplification"
        )
        simplified_segments.append(simplified)

    if not simplified_segments:
        raise ValueError("No valid contours remained after filtering.")
    return simplified_segments, skeleton


def simplify_contour(
    contour: np.ndarray,
    epsilon: float | None = CONTOUR_EPSILON,
    closed: bool = False,
) -> np.ndarray:
    """Simplify one open or closed contour."""
    if epsilon is None:
        return contour
    if epsilon < 0:
        raise ValueError("CONTOUR_EPSILON must be non-negative or None.")
    return cv.approxPolyDP(contour, epsilon, closed=closed)


def close_contour(contour: np.ndarray) -> np.ndarray:
    """Append the first point so a logo outline is explicitly drawable."""
    if len(contour) < 2:
        return contour
    if np.array_equal(contour[0], contour[-1]):
        return contour
    return np.concatenate([contour, contour[:1]])


def densify_contour(
    contour: np.ndarray,
    max_segment_length: float | None,
    closed: bool = False,
) -> np.ndarray:
    """Insert points so no contour segment exceeds a pixel-length limit."""
    if max_segment_length is None:
        return contour
    if max_segment_length <= 0:
        raise ValueError("max_segment_length must be positive or None.")
    if len(contour) < 2:
        return contour

    points = contour[:, 0, :].astype(float)
    densified: list[np.ndarray] = []
    segment_count = len(points) if closed else len(points) - 1
    for index in range(segment_count):
        start = points[index]
        end = points[(index + 1) % len(points)]
        segment = end - start
        length = float(np.linalg.norm(segment))
        subdivisions = max(1, int(np.ceil(length / max_segment_length)))
        for step in range(subdivisions):
            fraction = step / subdivisions
            densified.append(start + fraction * segment)

    if not closed:
        densified.append(points[-1])

    return np.rint(np.asarray(densified)).astype(np.int32).reshape(-1, 1, 2)


def valid_contour(contour: np.ndarray) -> bool:
    """Filter contours using point count or geometric length."""
    return (
        len(contour) > MIN_CONTOUR_POINTS
        or cv.arcLength(contour, False) > MIN_CONTOUR_LENGTH
    )


def contour_statistics(
    contour: np.ndarray,
    simplified: np.ndarray,
) -> None:
    """Print geometry and simplification statistics for one contour."""
    print(
        f"Contour: area={cv.contourArea(contour):.2f}, "
        f"perimeter={cv.arcLength(contour, True):.2f}, "
        f"points={len(contour)}, simplified_points={len(simplified)}"
    )


def extract_contours(
    threshold: np.ndarray,
    mode: str = TRAJECTORY_MODE,
) -> tuple[list[np.ndarray], np.ndarray | None, np.ndarray | None]:
    """Extract independent contours for logo, skeleton, or graph mode."""
    if mode == "graph":
        contours, skeleton = _extract_graph_contours(threshold)
        return contours, skeleton, None

    if mode == "logo":
        source = threshold
        contours, hierarchy = cv.findContours(
            source,
            cv.RETR_TREE,
            cv.CHAIN_APPROX_NONE,
        )
    elif mode == "skeleton":
        source = skeletonize(threshold)
        contours, hierarchy = cv.findContours(
            source,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_NONE,
        )
    else:
        raise ValueError("TRAJECTORY_MODE must be 'logo', 'skeleton', or 'graph'.")

    print(f"Number of contours found: {len(contours)}")
    simplified_contours: list[np.ndarray] = []
    for index, contour in enumerate(contours):
        if not valid_contour(contour):
            print(f"Contour {index}: discarded as noise")
            continue
        simplified = simplify_contour(contour, closed=mode == "logo")
        if len(simplified) < 2:
            print(f"Contour {index}: discarded after simplification")
            continue
        if mode == "logo":
            simplified = densify_contour(
                simplified,
                MAX_LOGO_SEGMENT_LENGTH_PIXELS,
                closed=True,
            )
        if mode == "logo" and CLOSE_LOGO_CONTOURS:
            simplified = close_contour(simplified)
        print(f"Contour {index}:")
        contour_statistics(contour, simplified)
        simplified_contours.append(simplified)

    if not simplified_contours:
        raise ValueError("No valid contours remained after filtering.")
    return simplified_contours, source, hierarchy
