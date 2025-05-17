import cv2
import math
import numpy as np

def mark_point_on_mask(mask, xc, yc, color=(255, 0, 0), radius=5, thickness=-1):
    """
    Marks a point on the given droplet mask image at the specified coordinates.

    Parameters:
        mask (numpy.ndarray): The droplet mask image.
        xc (int): X-coordinate of the point.
        yc (int): Y-coordinate of the point.
        color (tuple): Color of the point (default is blue in BGR format).
        radius (int): Radius of the point (default is 5).
        thickness (int): Thickness of the point. Use -1 for a filled circle (default is -1).

    Returns:
        numpy.ndarray: The mask image with the point marked.
    """
    # Ensure the coordinates are within the image bounds
    if 0 <= xc < mask.shape[1] and 0 <= yc < mask.shape[0]:
        # Draw the point on the mask
        cv2.circle(mask, (xc, yc), radius, color, thickness)
    else:
        print("Coordinates are out of bounds.")
    return mask

def mark_bounding_box_on_mask(mask, BBox_X, BBox_Y, BBox_W, BBox_H, color=(0, 255, 0), thickness=2):
    """
    Marks a bounding box on the given droplet mask image.

    Parameters:
        mask (numpy.ndarray): The droplet mask image.
        BBox_X (int): X-coordinate of the top-left corner of the bounding box.
        BBox_Y (int): Y-coordinate of the top-left corner of the bounding box.
        BBox_W (int): Width of the bounding box.
        BBox_H (int): Height of the bounding box.
        color (tuple): Color of the bounding box (default is green in BGR format).
        thickness (int): Thickness of the bounding box lines (default is 2).

    Returns:
        numpy.ndarray: The mask image with the bounding box marked.
    """
    # Ensure the bounding box coordinates are within the image bounds
    x1, y1 = BBox_X, BBox_Y
    x2, y2 = BBox_X + BBox_W, BBox_Y + BBox_H
    if 0 <= x1 < mask.shape[1] and 0 <= y1 < mask.shape[0] and 0 <= x2 <= mask.shape[1] and 0 <= y2 <= mask.shape[0]:
        # Draw the rectangle on the mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), color, thickness)
    else:
        print("Bounding box coordinates are out of bounds.")
    return mask

def mark_ellipse_on_mask(mask, Ellipse_XC, Ellipse_YC, Ellipse_A, Ellipse_B, Ellipse_Theta, color=(0, 0, 255), thickness=2):
    """
    Marks an ellipse on the given droplet mask image.
    """
    # Convert values to native Python int types
    xc = int(Ellipse_XC)
    yc = int(Ellipse_YC)
    axes = (int(Ellipse_A), int(Ellipse_B))
    angle = float(Ellipse_Theta)

    # Ensure the ellipse center is within image bounds
    if 0 <= xc < mask.shape[1] and 0 <= yc < mask.shape[0]:
        # Draw the ellipse
        cv2.ellipse(mask, (xc, yc), axes, angle, 0, 360, color, thickness)
    else:
        print("Ellipse center coordinates are out of bounds.")
    return mask

def compute_tangent_slope(x_pt, y_pt, xc, yc, a, b):
    numerator = -(x_pt - xc) * (b**2)
    denominator = (y_pt - yc) * (a**2)
    if abs(denominator) < 1e-10:
        return None  # vertical tangent
    return numerator / denominator

def compute_angle_from_slope(slope):
    if slope is None:
        return 90.0  # vertical
    return abs(math.degrees(math.atan(slope)))

def draw_tangent(img, x_pt, y_pt, slope, length=300):
    if slope is None:
        x1_ = x2_ = x_pt
        y1_ = y_pt - length
        y2_ = y_pt + length
    else:
        inv_len = length / math.sqrt(1 + slope**2)
        x1_ = int(x_pt + inv_len)
        y1_ = int(y_pt + slope * (x1_ - x_pt))

        x2_ = int(x_pt - inv_len)
        y2_ = int(y_pt + slope * (x2_ - x_pt))

    cv2.line(img, (x1_, y1_), (x2_, y2_), (255, 0, 255), 2)

def compute_intersection_points(mask, frame_num, BBox_Y,BBox_H):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in frame {frame_num:03d}.\n")

    contour = max(contours, key=cv2.contourArea)
    bottom_y = BBox_Y + BBox_H

    # BGR image for bounding box + tangents
    right_img = cv2.cvtColor(gray_mask, cv2.COLOR_GRAY2BGR)

    # Intersection points near bottom line
    intersection_points = []
    for pt in contour:
        px, py = pt[0]
        if abs(py - bottom_y) <= 1:
            intersection_points.append((px, py))

    if len(intersection_points) < 2:
        print(f"Could not find two intersection points for frame {frame_num:03d}.\n")

    intersection_points.sort(key=lambda p: p[0])
    (x1, y1), (x2, y2) = intersection_points[0], intersection_points[-1]

    # Mark them red
    cv2.circle(right_img, (x1, y1), 5, (0, 0, 255), -1)
    cv2.circle(right_img, (x2, y2), 5, (0, 0, 255), -1)

    # Fit ellipse
    if len(contour) < 5:
        print(f"Not enough contour points to fit ellipse for frame {frame_num:03d}.\n")
    return intersection_points, right_img

def mark_tangent_and_angle_on_mask(mask, frame_num, xc, yc, a, b, BBox_Y,BBox_H):
    intersection_points, right_img = compute_intersection_points(mask, frame_num, BBox_Y,BBox_H)
    intersection_points.sort(key=lambda p: p[0])
    (x1, y1), (x2, y2) = intersection_points[0], intersection_points[-1]
    slope1 = compute_tangent_slope(x1, y1, xc, yc, a, b)
    slope2 = compute_tangent_slope(x2, y2, xc, yc, a, b)
    angle1 = compute_angle_from_slope(slope1)
    angle2 = compute_angle_from_slope(slope2)

    draw_tangent(right_img, x1, y1, slope1)
    draw_tangent(right_img, x2, y2, slope2)

    # Put angles
    cv2.putText(right_img, f"{angle1:.2f} DEG", (x1 + 5, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(right_img, f"{angle2:.2f} DEG", (x2 + 5, y2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return right_img

def mark_tangent_and_angle_given_angle(mask, xc, yc, a, b, theta_deg, tangent_angles_deg, bbox_y=None, bbox_h=None):
    """
    Draw tangents at the points where a horizontal line intersects the ellipse.
    Each tangent uses a specified angle for its slope.

    Parameters:
        mask (np.ndarray): Image to draw on.
        xc, yc (float): Center of the ellipse.
        a, b (float): Semi-major and semi-minor axes.
        theta_deg (float): Rotation angle of the ellipse.
        tangent_angles_deg (list of float): List of tangent angles (in degrees).
        bbox_y, bbox_h (float, optional): Defines the horizontal line as bbox_y + bbox_h.

    Returns:
        np.ndarray: Image with marked tangents and angles.
    """
    output = mask.copy()
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Base line y = y_line
    if bbox_y is None or bbox_h is None:
        y_line = yc
    else:
        y_line = bbox_y + bbox_h

    # Sample ellipse points
    intersections = []
    for t in np.linspace(0, 2 * np.pi, 1000):
        x_ell = a * math.cos(t)
        y_ell = b * math.sin(t)

        x_rot = x_ell * cos_t - y_ell * sin_t + xc
        y_rot = x_ell * sin_t + y_ell * cos_t + yc

        if abs(y_rot - y_line) < 1.0:  # tolerance
            intersections.append((x_rot, y_rot))

    if len(intersections) > 2:
        intersections = sorted(intersections, key=lambda pt: pt[0])
        intersections = [intersections[0], intersections[-1]]

    if len(intersections) < 2:
        print(f"Warning: Found only {len(intersections)} intersections")
        return output

    for i, (x0, y0) in enumerate(intersections):
        if i >= len(tangent_angles_deg):
            break

        angle_deg = -tangent_angles_deg[i] if i == 0 else tangent_angles_deg[i]  # <-- NEGATE first (left side)
        angle_rad = math.radians(angle_deg)

        slope = None if abs(math.cos(angle_rad)) < 1e-6 else math.tan(angle_rad)

        draw_tangent(output, int(x0), int(y0), slope)
        cv2.putText(output, f"{angle_deg:.1f} DEG", (int(x0) + 5, int(y0) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.circle(output, (int(x0), int(y0)), 4, (0, 0, 255), -1)

    return output
