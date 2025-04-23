import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
csv_path = "5. Robust Estimation and Evaluation Methods/droplet_analysis/droplet_analysis_full.csv"
output_dir = "5. Robust Estimation and Evaluation Methods/droplet_analysis/droplet_analysis_summary"
os.makedirs(output_dir, exist_ok=True)

# Load and clean data
df = pd.read_csv(csv_path)
df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
df["Left_Angle"] = pd.to_numeric(df["Left_Angle"], errors="coerce")
df["Right_Angle"] = pd.to_numeric(df["Right_Angle"], errors="coerce")
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
df["Volume_Loss"] = pd.to_numeric(df["Volume_Loss"], errors="coerce")

df.dropna(inplace=True)

# Save summary statistics
summary_stats = df.describe().loc[["mean", "std", "min", "max"]]
summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"))

print("Summary Statistics:")
print(summary_stats)

# Contact angle plot
plt.figure(figsize=(10, 5))
plt.plot(df["Frame"], df["Left_Angle"], label="Left Contact Angle", marker='o', markersize=4)
plt.plot(df["Frame"], df["Right_Angle"], label="Right Contact Angle", marker='x', markersize=4)
plt.title("Contact Angles Over Time")
plt.xlabel("Frame")
plt.ylabel("Angle (°)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "contact_angles_over_time.png"))
plt.close()

# Volume plot
plt.figure(figsize=(10, 5))
plt.plot(df["Frame"], df["Volume"], color="green")
plt.title("Droplet Volume Over Time")
plt.xlabel("Frame")
plt.ylabel("Volume (pixels)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "volume_over_time.png"))
plt.close()

# Volume loss plot
plt.figure(figsize=(10, 5))
plt.plot(df["Frame"], df["Volume_Loss"], color="red")
plt.title("Volume Loss Per Frame")
plt.xlabel("Frame")
plt.ylabel("Loss (pixels)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "volume_loss_over_time.png"))
plt.close()

print("Analysis complete. Outputs saved in:", output_dir)
