import os
import h5py
import numpy as np
import glob

def convert_1d_to_2d(data, shape):
    if len(data) != np.prod(shape):
        raise ValueError(f"Cannot reshape array of size {len(data)} into shape {shape}")
    
    return data.reshape(shape)

def divide_1d_array(data, num_parts):
    if num_parts <= 0:
        raise ValueError("Number of parts must be greater than zero.")
    
    part_size = len(data) // num_parts
    return [data[i * part_size:(i + 1) * part_size] for i in range(num_parts)]

def reshape_h5(h5_file):
    with h5py.File(os.path.join("data", h5_file), "r") as f:
        reshaped_data = {}
        reshaped_data.update(reshape_convection(f))
        reshaped_data.update(reshape_diffusion(f))
        reshaped_data.update(reshape_bc1(f))
        reshaped_data.update(reshape_rhs1(f))
        reshaped_data.update({
            "extra/dP": f["extra/dP"][:],
            "extra/rhs2": f["extra/rhs2"][:],
            "force": f["force"][:],
            "p": f["p"][:],
            "v": f["v"][:],
            "u": f["u"][:],
        })

        os.makedirs("reshape", exist_ok=True)
        with h5py.File(os.path.join("reshape", h5_file), "w") as out_f:
            for key, value in reshaped_data.items():
                out_f.create_dataset(key, data=value)

def reshape_convection(f):
    data = f["convection/0"][:]
    divided_data = divide_1d_array(data, 2)
    reshaped_data = {
        "convection/0/0": convert_1d_to_2d(divided_data[0], (512, 511)),
        "convection/0/1": convert_1d_to_2d(divided_data[1], (511, 512))
    }
    data = f["convection/1"][:]
    divided_data = divide_1d_array(data, 2)
    reshaped_data.update({
        "convection/1/0": convert_1d_to_2d(divided_data[0], (512, 511)),
        "convection/1/1": convert_1d_to_2d(divided_data[1], (511, 512))
    })

    return reshaped_data

def reshape_diffusion(f):
    data = f["diffusion/0"][:]
    divided_data = divide_1d_array(data, 2)
    reshaped_data = {
        "diffusion/0/0": convert_1d_to_2d(divided_data[0], (512, 511)),
        "diffusion/0/1": convert_1d_to_2d(divided_data[1], (511, 512))
    }

    return reshaped_data

def reshape_bc1(f):
    data = f["extra/bc1"][:]
    divided_data = divide_1d_array(data, 2)
    reshaped_data = {
        "extra/bc1/0": convert_1d_to_2d(divided_data[0], (512, 511)),
        "extra/bc1/1": convert_1d_to_2d(divided_data[1], (511, 512))
    }

    return reshaped_data

def reshape_rhs1(f):
    data = f["extra/rhs1"][:]
    divided_data = divide_1d_array(data, 2)
    reshaped_data = {
        "extra/rhs1/0": convert_1d_to_2d(divided_data[0], (512, 511)),
        "extra/rhs1/1": convert_1d_to_2d(divided_data[1], (511, 512))
    }

    return reshaped_data

def __main__():
    h5_files = [os.path.basename(file) for file in glob.glob("data/*00.h5")]
    for h5_file in h5_files:
        print(f"Processing {h5_file}...")
        reshape_h5(h5_file)

if __name__ == "__main__":
    __main__()