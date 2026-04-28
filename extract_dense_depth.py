# extract_dense_depth.py
import numpy as np
import struct
import os
import cv2

def read_depth_map(path):
    """Read COLMAP binary depth map"""
    with open(path, "rb") as f:
        # Read header
        header = ""
        while True:
            c = f.read(1).decode("utf-8")
            if c == "\n":
                break
            header += c
        # Parse width, height, channels
        width, height, channels = map(int, header.split("&"))
        # Read data
        depth = np.frombuffer(
            f.read(width * height * channels * 4),
            dtype=np.float32
        ).reshape(height, width, channels)
    return depth[:, :, 0]

workspace  = "/home/arua/projects/3dgs/datasets/prepared/Eiffel_Tower/subset_50/undistorted"
depth_src  = os.path.join(workspace, "stereo/depth_maps")
depth_dst  = "/home/arua/projects/3dgs/datasets/prepared/Eiffel_Tower/subset_50_1776x1182/depth_dense"
os.makedirs(depth_dst, exist_ok=True)

for fname in sorted(os.listdir(depth_src)):
    if not fname.endswith(".geometric.bin"):
        continue

    depth = read_depth_map(os.path.join(depth_src, fname))

    # Save raw depth
    stem = fname.replace(".geometric.bin", "").rsplit('.', 1)[0]
    np.save(os.path.join(depth_dst, stem + ".npy"), depth)

    # Save colorized visualization
    valid       = depth > 0
    depth_norm  = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        depth_norm[valid] = ((depth[valid] - d_min) /
                             (d_max - d_min) * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(depth_dst, stem + "_viz.png"), depth_color)

    print(f"{fname}")
    print(f"  Shape: {depth.shape}")
    print(f"  Valid pixels: {valid.sum()} / {depth.size} "
          f"({100*valid.sum()/depth.size:.1f}%)")
    if valid.any():
        print(f"  Depth range: {depth[valid].min():.2f} to "
              f"{depth[valid].max():.2f} m")

print(f"\nDone. Dense depth maps saved to {depth_dst}/")