import sys
import os
import h5py
import numpy as np


def explore_h5_file(file_path):
    """
    Explore and display the contents of an HDF5 file.

    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file
    """

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return

    try:
        with h5py.File(file_path, "r") as f:
            print(f"\n{'='*50}")
            print(f"Contents of {file_path}:")
            print(f"{'='*50}")

            # Define a function to recursively explore groups and datasets
            def print_item_info(name, obj):
                indent = "  " * name.count("/")

                if isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}/")
                    # Print attributes of the group
                    if len(obj.attrs) > 0:
                        print(f"{indent}  Attributes:")
                        for key, value in obj.attrs.items():
                            print(f"{indent}    {key}: {value}")

                elif isinstance(obj, h5py.Dataset):
                    shape_str = str(obj.shape)
                    dtype_str = str(obj.dtype)
                    print(
                        f"{indent}Dataset: {name} (Shape: {shape_str}, Type: {dtype_str})"
                    )

                    # Print a sample of the data for small datasets
                    if len(obj.shape) == 0 or (
                        np.prod(obj.shape) < 10 and len(obj.shape) <= 2
                    ):
                        try:
                            print(f"{indent}  Data: {obj[...]}")
                        except:
                            print(f"{indent}  Data: <Unable to display>")

                    # Print attributes of the dataset
                    if len(obj.attrs) > 0:
                        print(f"{indent}  Attributes:")
                        for key, value in obj.attrs.items():
                            print(f"{indent}    {key}: {value}")

            # Visit all items in the file
            f.visititems(print_item_info)

            print(f"{'='*50}")

    except Exception as e:
        print(f"Error opening file: {e}")


# Example usage
if __name__ == "__main__":
    # Replace this with your HDF5 file path
    if len(sys.argv) != 2:
        print("Usage: python h5_dataset.py <path_to_h5_file>")
        sys.exit(1)
    h5_file_path = sys.argv[1]

    explore_h5_file(h5_file_path)
