#!/usr/bin/env python

import argparse
import os
import h5py
import pandas as pd


def extract_data(h5_file):
    data = {}

    def visit_item(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Split the path into parts
            parts = name.split("/")
            current = data

            # Navigate/create the nested structure
            for part in parts[:-1]:  # All parts except the last (dataset name)
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Add the dataset to the final location
            current[parts[-1]] = obj[()]

    with h5py.File(h5_file, "r") as f:
        f.visititems(visit_item)

    return data


def get_group_list(data):
    groups = []
    groups = list(data.keys())
    return groups


def get_dataset_list(data):
    datasets = set()

    for _, contents in data.items():
        datasets.update(contents.keys())

    return list(datasets)


def convert_data(data):
    converted = {}
    for group, contents in data.items():
        for dataset, value in contents.items():
            if dataset not in converted:
                converted[dataset] = [{} for _ in range(len(value))]

            for idx, val in enumerate(value):
                converted[dataset][idx][group] = val

    return converted


def export_to_md(data, level, output):
    result = ""

    for dataset, values in data.items():
        result += f"{"#"*level} {dataset}\n\n"
        df = pd.DataFrame(values)
        result += df.to_markdown(index=False) + "\n\n"

    if output:
        output_file = output + ".md"
        with open(output_file, "w") as f:
            f.write(result)
        print(f"Exported to {output_file}")
    else:
        print(result)


def export_to_csv(data, output):
    for dataset, values in data.items():
        df = pd.DataFrame(values)
        output_file = output + f"_{dataset}.csv"
        df.to_csv(output_file, index=False)
        print(f"Exported {dataset} to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HDF5 datasets.")
    parser.add_argument("input_file", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("--output", "-o", type=str, help="Path to the output files.")
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "md"],
        default="md",
        help="Output format: 'csv' or 'md'. Default is 'csv'.",
    )
    parser.add_argument(
        "--md_level",
        "-ml",
        type=int,
        default=4,
        help="Markdown header level for datasets. Default is 4.",
    )
    args = parser.parse_args()

    if args.format == "csv" and args.output is None:
        print("Output directory is required for CSV format.")
        exit(1)

    basename = os.path.basename(args.input_file)
    output_dir = None
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_dir = os.path.join(args.output, f"{os.path.splitext(basename)}")

    data = extract_data(args.input_file)
    converted_data = convert_data(data)

    if args.format == "md":
        export_to_md(converted_data, args.md_level, output_dir)
    elif args.format == "csv":
        export_to_csv(converted_data, output_dir)
    else:
        print("Unsupported format. Use 'csv' or 'md'.")
        exit(1)
