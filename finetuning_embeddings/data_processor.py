"""data_processor module.

Contains data processing utilities for embedding fine-tuning.
"""
import os

import pandas as pd
from utils import validate_file_path


class DataProcessor:
    """A class for processing data files for embedding fine-tuning.

    Attributes:
        input_file (str): Path to the input data file.
            Supported formats: .xls, .xlsx, .csv, .json.
        output_file (str): Path to output file.
            Parent directory must be writable.
    """

    def __init__(self, input_file: str, output_file: str):
        """Initialize DataProcessor with input and output file paths.

        Validates that the input file exists and is readable, and that
        the output directory exists and is writable. Stores absolute
        paths for consistent downstream operations.
        """
        # Validate the input file:
        # --> must exist, be a regular file, and match supported formats
        if not validate_file_path(input_file, is_output=False):
            raise ValueError(f"Invalid input file: {input_file}")

        # Validate the output path: parent directory must exist and be writable
        if not validate_file_path(output_file, is_output=True):
            raise ValueError(
                f"Invalid output path or directory not writable: {output_file}"
            )

        # Convert provided paths to absolute paths to avoid issues with relative paths
        # and ensure consistency in logging, error messages, and file operations
        self.input_file = os.path.abspath(input_file)
        self.output_file = os.path.abspath(output_file)

    def load_data(self) -> pd.DataFrame:
        """Load data from the input file into a pandas DataFrame.

        Supports Excel, CSV, and JSON. JSON support handles
        both JSON-lines and standard JSON.

        Excel inputs:
          - Multiple sheets: load 'Sheet1', error if missing.
          - Single sheet: load the sole sheet.

        Raises:
            ValueError: Unsupported file type, missing sheet,
                or fewer than two columns.
        """
        _, ext = os.path.splitext(self.input_file.lower())
        if ext in (".xls", ".xlsx"):
            excel_file = pd.ExcelFile(self.input_file, engine="openpyxl")
            sheets = excel_file.sheet_names
            if len(sheets) > 1:
                if "Sheet1" not in sheets:
                    raise ValueError(
                        "Excel file has multiple sheets but no 'Sheet1'."
                    )
                df = excel_file.parse("Sheet1")
            else:
                df = excel_file.parse(sheets[0])
        elif ext == ".csv":
            df = pd.read_csv(self.input_file)
        elif ext == ".json":
            try:
                df = pd.read_json(self.input_file, lines=True)
            except ValueError:
                df = pd.read_json(self.input_file)
        else:
            raise ValueError(
                "Unsupported file type: must be .xls, .xlsx, .csv or .json."
            )

        if df.shape[1] < 2:
            raise ValueError(
                "Dataset not suitable: requires at least two columns."
            )

        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply processing steps to the loaded DataFrame.

        To be implemented in the next commit.
        """
        raise NotImplementedError

    def save_data(self, df: pd.DataFrame) -> None:
        """Save processed DataFrame to the output file.

        Excel: .xls/.xlsx via openpyxl engine.
        CSV: standard CSV.
        JSON: standard JSON array (orient='records').
        """
        _, ext = os.path.splitext(self.output_file.lower())
        if ext in (".xls", ".xlsx"):
            df.to_excel(
                self.output_file,
                index=False,
                engine="openpyxl"
            )
        elif ext == ".csv":
            df.to_csv(self.output_file, index=False)
        elif ext == ".json":
            df.to_json(self.output_file, orient="records", lines=True)
        else:
            raise ValueError(
                "Unsupported file type: must be .xls, .xlsx, .csv or .json."
            )
