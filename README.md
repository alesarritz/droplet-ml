# Droplet ML  

## Introduction  
Droplet ML is a project focused on detecting the contact angle of a liquid droplet on a surface. The project utilizes computer vision techniques to identify the droplet's shape, fit an elliptical model, and compute the contact angle by determining the tangents at the intersection points between the droplet and the surface.  

## Section 1: Droplet Detection with OpenCV  
In this section, OpenCV is used for:  

- **Shape Detection (Elliptical Fitting)**: The algorithm processes grayscale images to detect the droplet. It first identifies the surface on which the droplet rests and then fits an ellipse to the detected contour of the droplet.  
- **Tangent Calculation & Angle Measurement**: Once the ellipse is detected, the algorithm identifies the intersection points between the ellipse and the surface. The tangent lines at these points are computed, and their angles with respect to the surface are determined.  

**Output**: OpenCV's elliptical fitting in certain conditions fails to detect the correct droplet shape, while the surface's detection is always successful. The angle measurement doesn't provide precise results due to the incorrect elliptical fitting.

## Section 2: Droplet Segmentation with Meta's SAM model