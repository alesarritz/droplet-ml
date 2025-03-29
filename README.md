# Droplet ML  

## Introduction  
Droplet ML is a project focused on detecting the contact angle of a liquid droplet on a surface. The project utilizes computer vision techniques to identify the droplet's shape, fit an elliptical model, and compute the contact angle by determining the tangents at the intersection points between the droplet and the surface.  

## Section 1: Droplet Detection with OpenCV  
In this section, OpenCV is used for Droplet Detection through Elliptical Fitting. 

#### Methodology

- **Shape Detection (Elliptical Fitting)**: The algorithm processes grayscale images to detect the droplet. It first identifies the surface on which the droplet rests and then fits an ellipse to the detected contour of the droplet.  
- **Tangent Calculation & Angle Measurement**: Once the ellipse is detected, the algorithm identifies the intersection points between the ellipse and the surface. The tangent lines at these points are computed, and their angles with respect to the surface are determined.  

#### Observations

- **Strengths**: The OpenCV-based method consistently detects the surface on which the droplet rests, providing a reliable reference for angle measurement. In clean and well-lit scenarios, the ellipse fitting algorithm can accurately outline the droplet shape.
- **Limitations**: The elliptical fitting often fails in challenging conditions, such as when the droplet has low contrast with the background or is partially occluded. As a result, the contact angle computation becomes imprecise due to inaccurate tangent estimation at the droplet's edge.


## Section 2: Droplet Segmentation with Meta's SAM model

In this section, we explored the use of [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) to automatically segment droplets in real-world images. The goal was to overcome the limitations of traditional OpenCV-based methods by leveraging SAM's advanced transformer-based segmentation capabilities.

#### Methodology

- **Model & Inference**: We used the `vit_h` variant of SAM and the official pre-trained checkpoint. For each image, we applied the `SamAutomaticMaskGenerator`, which produces multiple candidate masks based on visual features.
- **Visualization & Saving**: Each mask was randomly colored and overlaid onto the original image to visualize segmentation. The masks were saved as RGB images with clearly separated segments using OpenCV.

#### Observations

- **Strengths**: SAM demonstrated strong segmentation capabilities in complex visual contexts, even with variable lighting and textured backgrounds. In many cases, it successfully isolated the droplet region with high precision.
- **Limitations**: The model tends to over-segment — producing numerous masks for irrelevant regions (e.g., background elements, shadows, reflections). This behavior introduces noise and requires post-processing or mask filtering to isolate the actual droplet.