#!/usr/bin/env python3
import os
import glob
import argparse

import h5py
import imageio
import numpy as np


def h5_series_to_video(input_dir, dataset_name, output_path, fps=10):
    """
    Convert a series of .h5 files in input_dir into a video.
    Reads the specified dataset from each file as a 2D or 3D array,
    then appends it as a frame in the output video.
    """
    # Find and sort all .h5 files
    files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    writer = None
    for filepath in files:
        with h5py.File(filepath, "r") as h5:
            if dataset_name not in h5:
                print(
                    f"Warning: '{dataset_name}' not in {os.path.basename(filepath)}; skipping"
                )
                continue

            data = h5[dataset_name][:]
            frame = np.asarray(data)

            # On first valid frame, initialize video writer
            if writer is None:
                # Determine format arguments
                codec_args = {"format": "FFMPEG", "mode": "I", "fps": fps}
                # Grayscale frames must be expanded to H×W×1
                if frame.ndim == 2:
                    frame = frame[:, :, np.newaxis]
                writer = imageio.get_writer(output_path, **codec_args)

            writer.append_data(frame)

    if writer is None:
        raise RuntimeError(f"No valid frames found for dataset '{dataset_name}'")
    writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert a sequence of .h5 files to a video by extracting one dataset as frames"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing .h5 files (will process all files ending in .h5)",
    )
    parser.add_argument(
        "dataset_name", help="Name of the dataset inside each .h5 to use as a frame"
    )
    parser.add_argument(
        "output_video", help="Path to the output video (e.g., movie.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output video (default: 10)",
    )
    args = parser.parse_args()

    h5_series_to_video(
        input_dir=args.input_dir,
        dataset_name=args.dataset_name,
        output_path=args.output_video,
        fps=args.fps,
    )
    print(f"Video saved to {args.output_video}")


if __name__ == "__main__":
    main()
