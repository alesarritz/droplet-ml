import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import os
from skimage.measure import EllipseModel

mask_dir = "5. Robust Estimation and Evaluation Methods/droplet_masks"
image_dir = "3. Segmentation and Detection Models/processed_data/data"
output_dir = "5. Robust Estimation and Evaluation Methods/ellipse_fitting/ellipse_fitting_plots"
os.makedirs(output_dir, exist_ok=True)

frame_step = 100
max_frame = 5000

def extract_droplet_pixels(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(filled == 255)
    return np.column_stack((xs, ys)), largest

def extract_upper_arc_points(mask, ratio=0.9):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    contour_pts = largest[:, 0, :]
    y_thresh = np.min(contour_pts[:, 1]) + ratio * (np.max(contour_pts[:, 1]) - np.min(contour_pts[:, 1]))
    upper_arc_pts = contour_pts[contour_pts[:, 1] < y_thresh]
    return upper_arc_pts

def fit_covariance_ellipse(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return mean, eigvals, eigvecs

def fit_ellipse_to_upper_arc(mask, ratio=0.9):
    upper_arc_pts = extract_upper_arc_points(mask, ratio)
    if upper_arc_pts is None or len(upper_arc_pts) < 5:
        return None

    model = EllipseModel()
    if model.estimate(upper_arc_pts):
        xc, yc, a, b, theta = model.params
        return (xc, yc), a, b, np.degrees(theta)
    return None

for i in range(0, max_frame, frame_step):
    mask_path = os.path.join(mask_dir, f"frame_{i:03d}_mask.png")
    real_image_path = os.path.join(image_dir, f"frame_{i:03d}.png")

    if not os.path.exists(mask_path) or not os.path.exists(real_image_path):
        print(f"Skipping frame {mask_path} due to missing files.")
        continue

    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    points, largest_contour = extract_droplet_pixels(binary)
    if points is None or len(points) < 5:
        continue

    real_img = cv2.imread(real_image_path)
    if real_img is None:
        continue

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Covariance-Based Ellipse Fit', 'Fitzgibbon Ellipse Fit (Upper Arc)', 'OpenCV Ellipse Fit']
    colors = ['red', 'green', 'blue']

    for ax in axs:
        ax.imshow(cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB))
        ax.scatter(points[:, 0], points[:, 1], s=1, color='skyblue', alpha=0.05)
        ax.set_aspect('equal')
        ax.axis('off')

    # Covariance ellipse (standard)
    mean_c, eigvals_c, eigvecs_c = fit_covariance_ellipse(points)
    width_c = 2 * 2 * np.sqrt(eigvals_c[0])
    height_c = 2 * 2 * np.sqrt(eigvals_c[1])
    angle_c_deg = np.degrees(np.arctan2(eigvecs_c[1, 0], eigvecs_c[0, 0]))
    axs[0].add_patch(patches.Ellipse(mean_c, width_c, height_c, angle=angle_c_deg,
                                     edgecolor=colors[0], facecolor='none', lw=2))

    # Fitzgibbon on Upper Arc
    result_upper = fit_ellipse_to_upper_arc(binary, ratio=0.9)
    if result_upper:
        (xc, yc), a, b, angle_deg = result_upper
        axs[1].add_patch(patches.Ellipse((xc, yc), 2*a, 2*b, angle=angle_deg,
                                         edgecolor=colors[1], facecolor='none', lw=2))

    # OpenCV ellipse fit
    if len(largest_contour) >= 5:
        ellipse_cv = cv2.fitEllipse(largest_contour)
        (xcv, ycv), (maj, minr), angle_cv = ellipse_cv
        axs[2].add_patch(patches.Ellipse((xcv, ycv), maj, minr, angle=angle_cv,
                                         edgecolor=colors[2], facecolor='none', lw=2))

    for ax, title in zip(axs, titles):
        ax.set_title(f"{title}")

    fig.suptitle(f"Frame frame_{i:03d}.png: Ellipse Fitting Methods", fontsize=14)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"ellipse_fit_{i:03d}.png")
    plt.savefig(output_path)
    plt.close()
