#!/usr/bin/env python3
import os
import glob
import argparse

import h5py
import imageio
import numpy as np
import matplotlib.cm as cm

def find_global_range(files, dataset):
    gmin, gmax = np.inf, -np.inf
    for fp in files:
        with h5py.File(fp, 'r') as h5:
            if dataset in h5:
                arr = h5[dataset][:]
                gmin = min(gmin, float(np.min(arr)))
                gmax = max(gmax, float(np.max(arr)))
    if not np.isfinite(gmin) or not np.isfinite(gmax):
        raise RuntimeError(f"No valid data for '{dataset}' in any file")
    if gmin == gmax:
        gmin, gmax = gmin - 1, gmax + 1
    return gmin, gmax

def normalize(frame, vmin, vmax):
    f = np.clip(frame, vmin, vmax)
    return (f - vmin) / (vmax - vmin)

def apply_colormap(normed, cmap_name='viridis'):
    """normed: float array in [0,1], returns uint8 RGB array."""
    cmap = cm.get_cmap(cmap_name)
    colored = cmap(normed)[:, :, :3]   # drop alpha
    return (colored * 255).astype(np.uint8)

def h5_to_video(input_dir, dataset, output, fps, vmin, vmax, cmap,
                crf, preset):
    files = sorted(glob.glob(os.path.join(input_dir, '*.h5')))
    if not files:
        raise FileNotFoundError(f"No .h5 files in {input_dir!r}")

    # determine scaling range
    if vmin is None or vmax is None:
        print("→ Computing global data range …")
        vmin, vmax = find_global_range(files, dataset)
    print(f"→ Value range: [{vmin:.3f},{vmax:.3f}] → [0,1], cmap='{cmap}'")

    writer = None
    for fp in files:
        with h5py.File(fp, 'r') as h5:
            if dataset not in h5:
                print(f"  skip {os.path.basename(fp)} (no '{dataset}')")
                continue
            frame = np.array(h5[dataset][...], dtype=float)
            normed = normalize(frame, vmin, vmax)
            rgb = apply_colormap(normed, cmap)

            if writer is None:
                writer = imageio.get_writer(
                    output,
                    format='FFMPEG',
                    mode='I',
                    fps=fps,
                    codec='rawvideo',
                    quality=None,
                    output_params=[
                        '-pix_fmt', 'rgb24'
                    ]
                )
            writer.append_data(rgb)

    if writer is None:
        raise RuntimeError("No frames written—check your dataset name.")
    writer.close()
    print("→ Video saved to", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Convert HDF5 series to a COLOR video with fixed data range"
    )
    p.add_argument("input_dir", help="Directory of .h5 files")
    p.add_argument("dataset",   help="Dataset name inside each .h5")
    p.add_argument("output",    help="Output video file (e.g. out.mp4)")
    p.add_argument("--fps",   type=int,   default=10,    help="Frames per second")
    p.add_argument("--vmin",  type=float, default=None,  help="Min data value (auto)")
    p.add_argument("--vmax",  type=float, default=None,  help="Max data value (auto)")
    p.add_argument("--cmap",  default="viridis",         help="Matplotlib colormap")
    p.add_argument("--crf",   type=int,   default=18,    help="FFmpeg CRF (lower=better quality)")
    p.add_argument("--preset",default="slow",            help="FFmpeg preset (slower=better compression)")

    args = p.parse_args()
    h5_to_video(
        input_dir=args.input_dir,
        dataset=args.dataset,
        output=args.output,
        fps=args.fps,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        crf=args.crf,
        preset=args.preset
    )
