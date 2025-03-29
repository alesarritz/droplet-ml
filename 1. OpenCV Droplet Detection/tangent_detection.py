import cv2
import numpy as np
import os
import glob
import math

# Input and output folder
input_folder = "output_images"
output_folder = "output_tangents"
os.makedirs(output_folder, exist_ok=True)

def detect_tangents(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_path}")
        return None

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color 
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a binary mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No contours found in {image_path}")
        return None

    # Get the largest contour (assuming it's the blue ellipse)
    contour = max(contours, key=cv2.contourArea)

    # Find the two intersection points with the bottom line
    height, width = image.shape[:2]
    y_bottom = 847 - 1  # Bottom-most y-coordinate

    intersection_points = []
    for point in contour:
        x, y = point[0]
        if y >= y_bottom - 2: 
            intersection_points.append((x, y))

    # Ensure we have exactly two points
    if len(intersection_points) < 2:
        print(f"Could not detect two intersection points in {image_path}")
        return None

    # Sort by x-coordinate
    intersection_points = sorted(intersection_points, key=lambda p: p[0])
    (x1, y1), (x2, y2) = intersection_points[0], intersection_points[-1]
    
    # Fit an ellipse to the contour
    if len(contour) >= 5:  # Minimum required points
        ellipse = cv2.fitEllipse(contour)
        (xc, yc), (major_axis, minor_axis), angle = ellipse

        # Convert ellipse parameters
        a = major_axis / 2  # Semi-major axis
        b = minor_axis / 2  # Semi-minor axis

        # Compute exact tangent slopes at intersection points
        def compute_tangent_slope(x, y, xc, yc, a, b):
            """Computes the correct slope of the tangent to an ellipse at (x, y)."""
            numerator = -(x - xc) * (b ** 2)
            denominator = (y - yc) * (a ** 2)
            if denominator == 0:
                return None  # Vertical tangent case
            return numerator / denominator

        slope1 = compute_tangent_slope(x1, y1, xc, yc, a, b)
        slope2 = compute_tangent_slope(x2, y2, xc, yc, a, b)

        def draw_tangent(x, y, slope, img):
            """Draws a correctly computed tangent line extending outward from (x, y)."""
            if slope is None:
                # Vertical tangent case
                x_tangent1 = x
                y_tangent1 = y - 150  # Extend upwards
                x_tangent2 = x
                y_tangent2 = y + 150  # Extend downwards
            else:
                # Compute line endpoints
                length = 250  # Extend tangent
                x_tangent1 = int(x + length / math.sqrt(1 + slope ** 2))
                y_tangent1 = int(y + slope * (x_tangent1 - x))

                x_tangent2 = int(x - length / math.sqrt(1 + slope ** 2))
                y_tangent2 = int(y + slope * (x_tangent2 - x))

            # Draw the correctly computed tangent in magenta
            cv2.line(img, (x_tangent1, y_tangent1), (x_tangent2, y_tangent2), (255, 0, 255), 1)

        # Draw correct tangent lines
        draw_tangent(x1, y1, slope1, image)
        draw_tangent(x2, y2, slope2, image)

        # Compute angles with the horizontal line (y = constant)
        def compute_angle(slope):
            if slope is None:
                return 90  # Vertical line
            angle = math.atan(abs(slope)) * (180 / math.pi)  # Convert to degrees
            return round(angle, 2)

        angle1 = compute_angle(slope1)
        angle2 = compute_angle(slope2)

        # Save the processed image
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        print(f"Final image shape before saving: {image.shape}")
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path} - Angle1: {180 - angle1}°, Angle2: {180 - angle2}°")

        return (angle1, angle2)

    print(f"Could not fit an ellipse in {image_path}")
    return None

image_paths = glob.glob(os.path.join(input_folder, "*.png"))
for image_path in image_paths:
    detect_tangents(image_path)
