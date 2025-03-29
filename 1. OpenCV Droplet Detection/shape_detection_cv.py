import cv2
import numpy as np
import os

# Input and output directories
input_folder = "../dataset_120"
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

def detect_surface(image_path, color_image, binary):
    height, width = binary.shape
    min_surface_y = height  # Start with max height
    surface_x1, surface_x2 = 0, width
    min_segment_length = width // 10  # Minimum valid segment length (1/10 of image width)
    
    temp_x1, temp_x2 = None, None  # Temporary segment bounds
    temp_y = height  # Temporary surface height tracking
    
    for x in range(width):
        column = binary[:, x]
        black_pixels = np.where(column == 0)[0]  # Indices of black pixels
        if len(black_pixels) > 0:
            transition_y = black_pixels[-1]  # Last black pixel before transition
            
            if transition_y < temp_y:  # Found a new highest transition
                temp_y = transition_y
                temp_x1, temp_x2 = x, x
            elif transition_y == temp_y:
                temp_x2 = x  # Extend current segment
            else:  # Transition dropped, finalize previous segment if valid
                if temp_x2 - temp_x1 >= min_segment_length:
                    min_surface_y, surface_x1, surface_x2 = temp_y, temp_x1, temp_x2
                temp_y = height  # Reset tracking
                temp_x1, temp_x2 = None, None
    
    # Final check for last segment
    if temp_x2 and temp_x2 - temp_x1 >= min_segment_length:
        min_surface_y, surface_x1, surface_x2 = temp_y, temp_x1, temp_x2
    
    # Extend the detected surface line across the entire width of the image
    surface_x1 = 0
    surface_x2 = width - 1
    
    # Draw the detected surface line in red
    cv2.line(color_image, (surface_x1, min_surface_y), (surface_x2, min_surface_y), (0, 0, 255), 3)
    
    # Return the line equation y = mx + q (m=0, q=min_surface_y since it's horizontal)
    return min_surface_y, 0  # q, m

def detect_ellipse(image_path, output_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Add black padding at the bottom 
    image = cv2.copyMakeBorder(image, 0, 300, 0, 0, cv2.BORDER_CONSTANT, value=0)

    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Thresholding to create a binary image
    _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Detect the surface and get its Y position and equation parameters (q, m)
    surface_y, surface_m = detect_surface(image_path, color_image, binary)
    
    # Find contours again
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Identify the droplet contour as the largest contour above the surface
    droplet_contour = None
    for contour in contours:
        if cv2.boundingRect(contour)[1] < surface_y:  # Contour is above the surface
            if droplet_contour is None or cv2.contourArea(contour) > cv2.contourArea(droplet_contour):
                droplet_contour = contour
    
    if droplet_contour is None:
        print(f"No droplet detected in {image_path}")
        return surface_y, surface_m, None
    
    # Filter contour points to keep only those above the surface
    filtered_points = np.array([point for point in droplet_contour[:, 0, :] if point[1] < surface_y])

    ellipse = None
    if len(filtered_points) >= 5:
        ellipse = cv2.fitEllipse(filtered_points)
        cv2.ellipse(color_image, ellipse, (255, 0, 0), 3)  # Draw ellipse in blue

    # Save output image
    cv2.imwrite(output_path, color_image)


def process_images():
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            detect_ellipse(image_path, output_path)
            print(f"Processed: {filename}")

if __name__ == "__main__":
    process_images()
    print("Processing complete. Output images saved in 'output_images'.")