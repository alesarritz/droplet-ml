import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, num_frames=120):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Select 120 evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    count = 0
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(output_folder, f'frame_{count:03d}.png')
            cv2.imwrite(frame_path, frame)
            count += 1
        else:
            print(f"Warning: Could not read frame at index {i}")
    
    # Release video capture
    cap.release()
    print(f"Extracted {count} frames to {output_folder}")


video_path = "20211021_HR13_1_test_converted.mp4"  
output_folder = "dataset_120_real"  
extract_frames(video_path, output_folder)

