"""utils module.

General utility functions used across finetuning_embeddings.
"""

import os

import torch


def check_gpu() -> None:
    """Print information if a CUDA GPU is available."""
    if torch.cuda.is_available():
        print("You have a GPU available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory Allocated: "
            f"{torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB"
        )
        print(
            f"Memory Cached: "
            f"{torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB"
        )
        print(
            f"Total Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("You do not have a GPU available.")


def validate_file_path(file_path: str, is_output: bool = False) -> bool:
    """Validate if a file path exists (input) or its directory is writable.

    Supports Excel (.xls, .xlsx), CSV (.csv), and JSON (.json) files.
    Prints which type of file was validated.

    Parameters:
        file_path (str): Path to validate.
        is_output (bool): True if file is for output, False if input.

    Returns:
        bool: True if valid for intended use, otherwise False.

    Raises:
        ValueError: If the extension is unsupported.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return False

    file_path = os.path.abspath(file_path)
    lower = file_path.lower()

    if lower.endswith((".xls", ".xlsx")):
        file_type = "Excel"
    elif lower.endswith(".csv"):
        file_type = "CSV"
    elif lower.endswith(".json"):
        file_type = "JSON"
    else:
        raise ValueError(
            "Unsupported file type. Must be .xls, .xlsx, .csv or .json"
        )

    if is_output:
        parent_dir = os.path.dirname(file_path) or "."
        if os.path.exists(parent_dir) and os.access(parent_dir, os.W_OK):
            print(f"{file_type} file path is valid for output.")
            return True
        return False
    else:
        if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            print(f"{file_type} file validated for input.")
            return True
        return False
