"""Process the input Excel file and generate new data using OpenAI API."""

import os
import re
import time

import pandas as pd
from openai import OpenAI

from finetuning_embeddings.prompter import (
    generate_system_question_prompt,
    generate_user_question_prompt,
)
from finetuning_embeddings.utils import validate_file_path


class DataProcessor:
    """A class to process input Excel data and generate new data using OpenAI API."""

    def __init__(
        self,
        input_file: str,
        output_file: str,
        api_key: str,
        openai_api_model: str,
    ):
        """Initialize the DataProcessor with paths and API key.

        Args:
            input_file (str): Path to the input Excel file.
            output_file (str): Path to the output Excel file.
            api_key (str): Key for accessing OpenAI models for generation.
            openai_api_model (str): OpenAI model selected for generation.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.openai_client = OpenAI(api_key=api_key)
        self.openai_model = openai_api_model

    # Helper to decide which columns are required for minimal or full info
    def _decide_required_columns(self, present_columns: set[str]) -> list[str]:
        """Returns the list of columns required.

        based on the present columns in the DataFrame in order to
        decide the columns we are going to provide in the prompt for generation.
        """
        minimal_required = ["Instruction", "Answer"]
        full_required = [
            "Instruction",
            "Answer",
            "Context",
            "Topic",
            "Notes",
            "Document Name"
        ]
        # Check if all full_required columns are in present_columns
        if all(col in present_columns for col in full_required):
            return full_required
        else:
            return minimal_required

    # Helper to decide whether specified colum is missing or contains empty/NaN cells
    def _ensure_non_empty_column(self, df: pd.DataFrame, column_name: str) -> None:
        """Raise ValueError if the column is missing or contains empty/NaN cells."""
        if column_name not in df.columns:
            raise ValueError(f"Missing required column: '{column_name}'")
        if (
            df[column_name].isnull().any()
            or (df[column_name].astype(str).str.strip() == "").any()
        ):
            raise ValueError(f"Some rows have an empty or NaN '{column_name}' value.")

    # Helper to validate presence of critical columns
    def _validate_critical_columns(
        self, df: pd.DataFrame, critical_columns: list[str]
    ) -> None:
        """Ensures that all required_columns exist in df."""
        for col in critical_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")

    # Helper to generate LLM responses via using OpenAI API
    def _generate_llm_response(
        self, current_row: pd.Series, developer_prompt: str, option: str
    ) -> str:
        """Generate a response (new question) by querying the OpenAI chat API.

        This function sends a chat completion request
        to the OpenAI API using the specified model,
        combining a developer prompt and a user prompt
        into a conversation. It attempts to retrieve a
        non-empty response from the API. The function
        will retry up to three times (with a 10-second wait
        between attempts) if the API returns an empty response.
        If, after three attempts, no valid response
        is obtained, the function raises a RuntimeError.
        If an exception occurs during the API call, the
        exception is printed and then re-raised.

        Args:
            current_row (pd.Series): The DataFrame row for user prompt.
            developer_prompt (str): The system prompt.
            option (str): The identifier for generation type (question, answer).

        Returns:
            str: A non-empty response string from the OpenAI API.

        Raises:
            Exception: If an exception occurs during the API call.
            RuntimeError: If a valid response is not received after 3 attempts.
        """
        # Generate user_prompt based on the provided option.
        option_lower = option.lower()
        if option_lower == "question":
            user_prompt = generate_user_question_prompt(
                instruction=current_row["Instruction"],
                answer=current_row["Answer"],
                context=current_row["Context"],
                topic=current_row.get(
                    "Topic", "Not provided at the moment."
                ),
                notes=current_row.get(
                    "Notes", "Not provided at the moment."
                ),
            )
        else:
            raise ValueError(
                f"Invalid option provided: {option}. Expected 'question'."
            )

        max_retries = 3
        retries = 0

        while retries < max_retries:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "developer", "content": developer_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                openai_response = response.choices[0].message.content
                if openai_response:
                    return str(openai_response)
                retries += 1
                print(
                    f"Attempt {retries}: API returned an empty string."
                )
                time.sleep(10)
            except Exception as e:
                print(f"API call failed: {e}")
                raise e

        raise RuntimeError(
            f"Failed to get a valid response after {max_retries} attempts."
        )

    def _clean_generation_text(self, text: str) -> str:
        """Removes the prefix "Output:" from the provided text if it exists.

        and strips any leading or trailing whitespace.

        Args:
            text (str): The input string which may begin with "Output:".

        Returns:
            str: The cleaned text without the prefix and with trimmed whitespace.
        """
        cleaned_text = re.sub(r"^output:\s*", "", text, flags=re.IGNORECASE)
        return cleaned_text.strip()

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

    def process_data(self) -> bool:
        """Process the input Excel file and generate new data using OpenAI API.

        The Data Generator SHALL accept two types of input:
           1) Minimal Information Set: includes only "Instruction" and "Answer".
           2) Full Information Set: includes "Instruction", "Answer",
              "Context", "Topic", and Notes".

        Returns:
            bool: True if processing is successful.
        """
        # 1. Validate file paths
        # Validate input file
        try:
            if not validate_file_path(self.input_file):
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
        except ValueError as ve:
            # Handles cases when the file path is not an Excel file
            raise ValueError(f"Invalid input file: {self.input_file}. {ve}") from ve

        # Note: we set is_output=True to perform output-specific checks.
        try:
            if not validate_file_path(self.output_file, is_output=True):
                raise ValueError(
                    f"Invalid output path: {self.output_file}"
                )
        except ValueError as ve:
            # Handles cases when the file path is not an Excel file
            raise ValueError(f"Invalid output file: {self.output_file}. {ve}") from ve

        # 2. Read the Excel file
        df = self.load_data()

        # 3. Validate the presence of critical columns
        self._validate_critical_columns(df, ["Instruction", "Answer"])

        # 4. Always enforce that "Instruction" isn't empty/NaN (both minimal and full)
        self._ensure_non_empty_column(df, "Instruction")

        # 5. Always enforce that "Answer" isn't empty/NaN (both minimal and full)
        self._ensure_non_empty_column(df, "Answer")

        # 6. Decide which columns are required depending on minimal or full input
        required_columns = self._decide_required_columns(set(df.columns))

        # 7. If in full input mode, also ensure 'Requirement Section' is non-empty
        if "Context" in required_columns:
            self._ensure_non_empty_column(df, "Context")

        # 8. Call developer (system) prompt for question generation
        developer_prompt_question = generate_system_question_prompt()

        # 9. Generate the positive chunk for each row
        df["new_question"] = df.apply(
            lambda row: self._generate_llm_response(
                row, developer_prompt_question, "question"
            ),
            axis=1,
        )

        # 10. Apply the cleaning function to all values in the "new_question" column
        df["new_question"] = df["new_question"].apply(
            self._clean_generation_text
        )

        # 11. Have the new generation as separate rows
        extras_nq = df.copy()
        extras_nq["Instruction"] = extras_nq["new_question"]

        # 12. Combine all sets of entries
        final_df = pd.concat([df, extras_nq], ignore_index=True)

        # 13. Remove the temporary 'new_question' column if present
        if "new_question" in final_df.columns:
            final_df = final_df.drop(columns=["new_question"])

        # 14. Save the updated data to the output file
        self.save_data(final_df)

        return True
