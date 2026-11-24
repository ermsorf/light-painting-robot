import numpy as np


def project_2d_to_sphere(points_2d, sphere_radius=0.8, camera_distance=1.0, image_distance=0.1):
    """
    Project 2D (x, z) points onto a sphere.

    Coordinate system:
        X = horizontal
        Y = camera/depth axis
        Z = vertical

    Sphere center: (0, 0, 0)
    Camera:        (0, camera_distance, 0)
    """

    points_2d = np.asarray(points_2d, dtype=float)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("points_2d must be an array of (x, z) points")
    if not np.all(np.isfinite(points_2d)):
        raise ValueError("points_2d must contain only finite coordinates")
    if sphere_radius <= 0 or camera_distance <= sphere_radius:
        raise ValueError("camera_distance must be greater than sphere_radius")

    # Camera position
    camera = np.array([0.0, camera_distance, 0.0])

    # Image plane at the front surface of the sphere
    image_y = image_distance  # y-coordinate of the image plane

    points_3d = []

    for x, z in points_2d:

        # 2D (x, z) -> 3D point on image plane
        image_point = np.array([
            x,
            image_y,
            z
        ])

        # Ray from camera through image point
        direction = image_point - camera

        # Normalize direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            raise ValueError("A projected point cannot be at the camera position")
        direction /= direction_norm

        # Ray:
        # P(t) = camera + t * direction
        #
        # Sphere:
        # x² + y² + z² = R²

        b = 2 * np.dot(camera, direction)
        c = np.dot(camera, camera) - sphere_radius**2

        discriminant = b**2 - 4 * c

        if discriminant < 0:
            points_3d.append([np.nan, np.nan, np.nan])
            continue

        # Two intersections with the sphere
        t1 = (-b - np.sqrt(discriminant)) / 2
        t2 = (-b + np.sqrt(discriminant)) / 2

        # Keep intersections in front of the camera
        positive_t = [t for t in (t1, t2) if t >= 0]

        if not positive_t:
            points_3d.append([np.nan, np.nan, np.nan])
            continue

        # Closest intersection
        t = min(positive_t)

        # Calculate 3D point
        point_3d = camera + t * direction

        points_3d.append(point_3d)

    return np.array(points_3d)


def adjust_3d_to_robot_frame(points_3d):
    # z_offset = 0.215
    z_offset = 0.215  
    y_offset = -0.1 # Adjust this value based on your robot's configuration
    """adjust the center of the sphere to be matching the center of the robots 
    moving range. The center of the sphere is at (0, 0, 0) and the robot's moving range is at"""
    points_3d[:, 2] += z_offset
    points_3d[:, 1] += y_offset  # Invert the y-axis to match the robot's coordinate system
    return points_3d
    