import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc
import math
from skimage.measure import EllipseModel
import os
import csv

# Extracts the largest connected droplet region from a binary mask.
def extract_droplet_pixels(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(filled == 255)
    return np.column_stack((xs, ys)), largest

# Extracts the upper portion of the droplet contour for stable ellipse fitting.
def extract_upper_arc_points(mask, ratio=0.9):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    contour_pts = largest[:, 0, :]
    y_thresh = np.min(contour_pts[:, 1]) + ratio * (np.max(contour_pts[:, 1]) - np.min(contour_pts[:, 1]))
    upper_arc_pts = contour_pts[contour_pts[:, 1] < y_thresh]
    return upper_arc_pts

# Fits an ellipse to the upper arc of the droplet using a conic least-squares fit.
def fit_ellipse_to_upper_arc(mask, ratio=0.9):
    upper_arc_pts = extract_upper_arc_points(mask, ratio)
    if upper_arc_pts is None or len(upper_arc_pts) < 5:
        return None
    model = EllipseModel()
    if model.estimate(upper_arc_pts):
        xc, yc, a, b, theta = model.params
        return (xc, yc), a, b, np.degrees(theta), model
    return None

# Computes the angle between a line and the horizontal, in degrees.
def compute_contact_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = abs(math.degrees(angle_rad))
    return angle_deg

# Draws an arc at the vertex between two lines, to visualize the contact angle.
def draw_angle_arc(ax, vertex, p1, p2, radius=20, color='red'):
    def angle_between(p_from, p_to):
        return np.degrees(np.arctan2(p_to[1] - p_from[1], p_to[0] - p_from[0]))
    angle1 = angle_between(vertex, p1)
    angle2 = angle_between(vertex, p2)
    theta1, theta2 = sorted([angle1, angle2])
    if theta2 - theta1 > 180:
        theta1, theta2 = theta2, theta1 + 360
    arc = Arc(vertex, 2*radius, 2*radius, angle=0, theta1=theta1, theta2=theta2, color=color, lw=2)
    ax.add_patch(arc)

def analyze_droplet_frame(mask_path, image_path, frame_id=None, save_step=100, save_path=None):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    pixels, largest_contour = extract_droplet_pixels(binary)
    volume = len(pixels)
    ellipse_result = fit_ellipse_to_upper_arc(binary)
    if ellipse_result is None:
        raise RuntimeError("Failed to fit ellipse")
    (xc, yc), a, b, angle_deg, model = ellipse_result

    theta_rad = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    t = np.linspace(0, 2*np.pi, 720)
    xt = a * np.cos(t)
    yt = b * np.sin(t)
    xe = xc + xt * cos_t - yt * sin_t
    ye = yc + xt * sin_t + yt * cos_t

    x, y, w, h = cv2.boundingRect(largest_contour)
    x -= 1
    w += 1
    bottom_y = y + h

    points_on_surface = [(x_, y_) for x_, y_ in zip(xe, ye) if abs(y_ - bottom_y) <= 2]
    if len(points_on_surface) < 2:
        points_on_surface = [(x, bottom_y), (x + w, bottom_y)]

    points_on_surface.sort(key=lambda p: p[0])
    left_surface = points_on_surface[0]
    right_surface = points_on_surface[-1]

    points_on_ellipse = list(zip(xe, ye))
    points_on_ellipse_in_bbox = [(x_, y_) for x_, y_ in points_on_ellipse if y <= y_ <= y + h]
    epsilon = 5
    left_candidates = [p for p in points_on_ellipse_in_bbox if abs(p[0] - x) <= epsilon]
    right_candidates = [p for p in points_on_ellipse_in_bbox if abs(p[0] - (x + w)) <= epsilon]
    leftmost_touch = max(left_candidates, key=lambda p: p[1]) if left_candidates else left_surface
    rightmost_touch = max(right_candidates, key=lambda p: p[1]) if right_candidates else right_surface
    leftmost_touch = (x, leftmost_touch[1])
    rightmost_touch = (x + w, rightmost_touch[1])

    left_angle = compute_contact_angle(left_surface, leftmost_touch)
    right_angle = compute_contact_angle(rightmost_touch, right_surface)

    # Optional save or show
    if frame_id is not None and frame_id % save_step == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.scatter(pixels[:, 0], pixels[:, 1], s=1, color='skyblue', alpha=0.05)
        draw_ellipse(ax, xc, yc, a, b, angle_deg)
        draw_bounding_box(ax, largest_contour)

        ax.plot([x, x + w], [bottom_y, bottom_y], color='black', linestyle='--')
        for pt in [left_surface, right_surface, leftmost_touch, rightmost_touch]:
            ax.plot(pt[0], pt[1], 'ro')
        ax.plot([left_surface[0], leftmost_touch[0]], [left_surface[1], leftmost_touch[1]], color='blue')
        ax.plot([right_surface[0], rightmost_touch[0]], [right_surface[1], rightmost_touch[1]], color='blue')
        draw_angle_arc(ax, left_surface, leftmost_touch, (left_surface[0] + 30, left_surface[1]), radius=25)
        draw_angle_arc(ax, right_surface, rightmost_touch, (right_surface[0] - 30, right_surface[1]), radius=25)
        ax.set_title(f"Frame {frame_id:03d} — Volume: {volume} pixels\nContact Angles: Left: {left_angle:.2f}°, Right: {right_angle:.2f}°", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    return left_angle, right_angle, volume

# Draws the ellipse using matplotlib patches.
def draw_ellipse(ax, xc, yc, a, b, angle_deg):
    ellipse_patch = patches.Ellipse((xc, yc), 2*a, 2*b, angle=angle_deg, edgecolor='green', facecolor='none', lw=2)
    ax.add_patch(ellipse_patch)

# Draws bounding box around droplet and returns its parameters.
def draw_bounding_box(ax, contour):
    x, y, w, h = cv2.boundingRect(contour)
    x -= 1
    w += 1
    bottom_y = y + h
    ax.plot([x, x + w], [bottom_y, bottom_y], color='black', linestyle='--')
    rect_patch = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='orange', facecolor='none')
    ax.add_patch(rect_patch)
    return x, y, w, h, bottom_y

# Generates sample points along the fitted ellipse.
def sample_ellipse_points(xc, yc, a, b, angle_deg):
    theta_rad = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    t = np.linspace(0, 2*np.pi, 360)
    xt = a * np.cos(t)
    yt = b * np.sin(t)
    xe = xc + xt * cos_t - yt * sin_t
    ye = yc + xt * sin_t + yt * cos_t
    return xe, ye

# Finds contact points where the ellipse intersects the bottom edge of the bounding box.
def find_surface_contacts(xe, ye, bottom_y):
    points_on_surface = [(x_, y_) for x_, y_ in zip(xe, ye) if abs(y_ - bottom_y) <= 5] # 5 for tolerance
    points_on_surface.sort(key=lambda p: p[0])
    return points_on_surface[0], points_on_surface[-1]

# Finds the lowest points on the vertical sides of the bounding box that touch the ellipse.
def find_bbox_wall_contacts(x, y, w, h, xe, ye, left_surface, right_surface):
    points_on_ellipse = list(zip(xe, ye))
    points_on_ellipse_in_bbox = [(x_, y_) for x_, y_ in points_on_ellipse if y <= y_ <= y + h]
    epsilon = 5
    left_candidates = [p for p in points_on_ellipse_in_bbox if abs(p[0] - x) <= epsilon]
    right_candidates = [p for p in points_on_ellipse_in_bbox if abs(p[0] - (x + w)) <= epsilon]
    leftmost_touch = max(left_candidates, key=lambda p: p[1]) if left_candidates else left_surface
    rightmost_touch = max(right_candidates, key=lambda p: p[1]) if right_candidates else right_surface
    leftmost_touch = (x, leftmost_touch[1])
    rightmost_touch = (x + w, rightmost_touch[1])
    return leftmost_touch, rightmost_touch

# Draws all visual markers and lines between contact points.
def draw_all_lines(ax, left_surface, right_surface, leftmost_touch, rightmost_touch):
    for pt in [left_surface, right_surface, leftmost_touch, rightmost_touch]:
        ax.plot(pt[0], pt[1], 'ro')
    ax.plot([left_surface[0], leftmost_touch[0]], [left_surface[1], leftmost_touch[1]], color='blue')
    ax.plot([right_surface[0], rightmost_touch[0]], [right_surface[1], rightmost_touch[1]], color='blue')

if __name__ == "__main__":
    os.makedirs("5. Robust Estimation and Evaluation Methods/droplet_analysis/analysis_plots", exist_ok=True)

    summary_csv = "5. Robust Estimation and Evaluation Methods/droplet_analysis/droplet_analysis.csv"
    previous_volume = None

    with open(summary_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Left_Angle", "Right_Angle", "Volume", "Volume_Loss"])

        for i in range(0, 5000, 100):
            frame_id = i
            frame_str = f"{i:03d}"
            mask_path = f"5. Robust Estimation and Evaluation Methods/droplet_masks/frame_{frame_str}_mask.png"
            image_path = f"3. Segmentation and Detection Models/processed_data/data/frame_{frame_str}.png"
            output_path = f"5. Robust Estimation and Evaluation Methods/droplet_analysis/analysis_plots/analysis_{frame_str}.png"

            try:
                left_angle, right_angle, volume = analyze_droplet_frame(
                    mask_path, image_path, frame_id=frame_id, save_step=1, save_path=output_path
                )
                volume_loss = None if previous_volume is None else previous_volume - volume
                previous_volume = volume
                writer.writerow([frame_str, left_angle, right_angle, volume, volume_loss])
                print(f"Saved and logged frame {frame_str}")
            except Exception as e:
                print(f"Skipped frame {frame_str}: {e}")