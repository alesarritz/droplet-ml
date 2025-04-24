import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
csv_path = "5. Robust Estimation and Evaluation Methods/analysis/droplet_analysis.csv"
output_dir = "5. Robust Estimation and Evaluation Methods/analysis/droplet_analysis_summary"
os.makedirs(output_dir, exist_ok=True)

# Load and clean data
df = pd.read_csv(csv_path)

# Convert types
numeric_columns = [
    "Left_Angle", "Right_Angle", "Volume", "Volume_Loss",
    "Ellipse_XC", "Ellipse_YC", "Ellipse_A", "Ellipse_B", "Ellipse_Theta",
    "BBox_X", "BBox_Y", "BBox_W", "BBox_H"
]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
df["Valid"] = df["Valid"].astype(bool)

# Filter only valid rows
df_valid = df[df["Valid"]].dropna()

# Derived features
df_valid["Ellipse_Aspect_Ratio"] = df_valid["Ellipse_A"] / df_valid["Ellipse_B"]
df_valid["BBox_Aspect_Ratio"] = df_valid["BBox_W"] / df_valid["BBox_H"]

# Basic summary statistics
summary_stats = df_valid.describe().loc[["mean", "std", "min", "max"]]
summary_stats.to_csv(os.path.join(output_dir, "summary_statistics_valid.csv"))

# Full grouped statistics (valid vs invalid)
extended_stats = df.groupby("Valid").describe().transpose()
extended_stats.to_csv(os.path.join(output_dir, "full_statistics_by_validity.csv"))

# Utility for plotting
def save_line_plot(data, cols, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    for col in cols:
        plt.plot(data["Frame"], data[col], label=col)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Generate and save plots
save_line_plot(df_valid, ["Left_Angle", "Right_Angle"], "Contact Angles Over Time", "Angle (°)", "contact_angles.png")
save_line_plot(df_valid, ["Volume"], "Droplet Volume Over Time", "Volume (pixels)", "volume.png")
save_line_plot(df_valid, ["Volume_Loss"], "Volume Loss Over Time", "Loss (pixels)", "volume_loss.png")
save_line_plot(df_valid, ["Ellipse_A", "Ellipse_B"], "Ellipse Axes Over Time", "Length (pixels)", "ellipse_axes.png")
save_line_plot(df_valid, ["BBox_W", "BBox_H"], "Bounding Box Dimensions Over Time", "Pixels", "bbox_dimensions.png")
save_line_plot(df_valid, ["Ellipse_Aspect_Ratio"], "Ellipse Aspect Ratio Over Time", "a / b", "ellipse_aspect_ratio.png")
save_line_plot(df_valid, ["BBox_Aspect_Ratio"], "Bounding Box Aspect Ratio Over Time", "W / H", "bbox_aspect_ratio.png")

print("Summary statistics and plots saved in:", output_dir)
