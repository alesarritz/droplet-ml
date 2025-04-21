import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

reference_path = '5. Robust Estimation and Evaluation Methods/droplet_masks'
prediction_path = '5. Robust Estimation and Evaluation Methods/droplet_unet'
output_csv = '5. Robust Estimation and Evaluation Methods/segmentation_quality'
loss_threshold = 5.0  # % of pixels lost in the reference mask

def get_frame_list(path):
    return sorted(glob.glob(os.path.join(path, '*.png')))

def extract_index_ref(filename):
    basename = os.path.basename(filename)
    return int(basename.replace("frame_", "").replace("_mask.png", ""))

def extract_index_pred(filename):
    basename = os.path.basename(filename)
    return int(basename.replace("frame_", "").replace(".png", ""))

def load_binary_mask(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load {filepath}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def match_frame_indices(ref_path, pred_path):
    ref_files = get_frame_list(ref_path)
    pred_files = get_frame_list(pred_path)

    ref_indices = {extract_index_ref(f): f for f in ref_files}
    pred_indices = {extract_index_pred(f): f for f in pred_files}

    common_indices = sorted(set(ref_indices.keys()) & set(pred_indices.keys()))

    print(f"Reference masks found: {len(ref_indices)}")
    print(f"Prediction masks found: {len(pred_indices)}")
    print(f"Common frame indices: {len(common_indices)}")

    return common_indices, ref_indices, pred_indices

def compare_masks(idx, ref_file, pred_file):
    ref_mask = load_binary_mask(ref_file)
    pred_mask = load_binary_mask(pred_file)

    ref_area = np.count_nonzero(ref_mask)
    if ref_area == 0:
        print(f"Frame {idx:03d}: Reference mask is empty. Skipping.")
        return None

    missed_mask = cv2.bitwise_and(ref_mask, cv2.bitwise_not(pred_mask))
    missed_area = np.count_nonzero(missed_mask)

    loss_rate = missed_area / ref_area * 100

    intersection = np.logical_and(ref_mask, pred_mask).sum()
    union = np.logical_or(ref_mask, pred_mask).sum()
    iou = intersection / union if union != 0 else 0

    dice = (2 * intersection) / (ref_mask.sum() + pred_mask.sum()) if (ref_mask.sum() + pred_mask.sum()) != 0 else 0

    is_valid = loss_rate <= loss_threshold

    print(f"Frame {idx:03d} | Loss: {loss_rate:.2f}% | IoU: {iou:.3f} | Dice: {dice:.3f} | Valid: {is_valid}")

    return {
        'frame': idx,
        'loss_rate': loss_rate,
        'iou': iou,
        'dice': dice,
        'valid': is_valid
    }

def generate_summary(results):
    if not results:
        print("No valid frames were processed.")
        return

    valid_count = sum(r['valid'] for r in results)
    total_count = len(results)

    print(f"\nOverall Valid Frames: {valid_count}/{total_count} "
          f"({(valid_count / total_count * 100):.2f}%)")

    frames = [r['frame'] for r in results]
    loss_rates = [r['loss_rate'] for r in results]

    plt.figure(figsize=(10, 5))
    plt.plot(frames, loss_rates, marker='o', label='Pixel Loss Rate (%)')
    plt.axhline(y=loss_threshold, color='r', linestyle='--', label=f'Threshold = {loss_threshold}%')
    plt.title('Segmentation Quality Over Time')
    plt.xlabel('Frame Index')
    plt.ylabel('Loss Rate (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot to PNG 
    plot_path = os.path.join(os.path.dirname(output_csv), 'segmentation_quality_plot.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    plt.show()

    # Save CSV report
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved report to {output_csv}")


if __name__ == '__main__':
    common_indices, ref_indices, pred_indices = match_frame_indices(reference_path, prediction_path)
    if not common_indices:
        print("No matching frame indices found. Please check file naming or folder paths.")
    else:
        results = []
        for idx in common_indices:
            result = compare_masks(idx, ref_indices[idx], pred_indices[idx])
            if result:
                results.append(result)

        generate_summary(results)
