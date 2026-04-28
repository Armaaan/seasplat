import numpy as np
import struct
import os
import cv2

def read_depth_map(path):
    """
    Read COLMAP binary depth map.
    Format: width&height&channels\n followed by float32 data
    """
    with open(path, "rb") as f:
        # Read header as bytes until newline
        header = b""
        while True:
            byte = f.read(1)
            if byte == b"\n" or byte == b"":
                break
            header += byte

        # Parse width, height, channels from header
        # Header format: "width&height&channels"
        header_str = header.decode("ascii")
        parts = header_str.split("&")
        width    = int(parts[0])
        height   = int(parts[1])
        channels = int(parts[2])

        # Read float32 data
        num_elements = width * height * channels
        data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
        depth = data.reshape(height, width, channels)

    return depth[:, :, 0]  # return first channel


workspace  = "/home/arua/projects/3dgs/datasets/prepared/Eiffel_Tower/subset_50/undistorted"
depth_src  = os.path.join(workspace, "stereo/depth_maps")
depth_dst  = "/home/arua/projects/3dgs/datasets/prepared/Eiffel_Tower/subset_50_1776x1182/depth_dense"
os.makedirs(depth_dst, exist_ok=True)

# Process only geometric depth maps
files = [f for f in sorted(os.listdir(depth_src))
         if f.endswith(".geometric.bin")]

print(f"Found {len(files)} depth maps")

for fname in files:
    try:
        depth = read_depth_map(os.path.join(depth_src, fname))

        # Save raw depth
        stem = fname.replace(".geometric.bin", "").rsplit('.', 1)[0]
        np.save(os.path.join(depth_dst, stem + ".npy"), depth)

        # Save colorized visualization
        valid      = depth > 0
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            depth_norm[valid] = ((depth[valid] - d_min) /
                                 (d_max - d_min) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        depth_color[~valid] = [0, 0, 0]
        cv2.imwrite(os.path.join(depth_dst, stem + "_viz.png"), depth_color)

        print(f"{fname}")
        print(f"  Shape: {depth.shape}")
        print(f"  Valid: {valid.sum()}/{depth.size} "
              f"({100*valid.sum()/depth.size:.1f}%)")
        if valid.any():
            print(f"  Range: {depth[valid].min():.2f} to "
                  f"{depth[valid].max():.2f} m")

    except Exception as e:
        print(f"ERROR on {fname}: {e}")

print(f"\nDone. Saved to {depth_dst}/")