"""DataGenerator module."""

import os

from finetuning_embeddings.processor import DataProcessor


class DataGenerator:
    """DataGenerator is a class designed to generate data using the OpenAI API.

    It manages the API key and API model required for the process, reads an input file
    to determine the necessary data generation approach (minimal or full information),
    and delegates the data generation task to a DataProcessor, which handles
    the processing and saving of data to an Excel file.

    Attributes:
        api_key (Optional[str]): The OpenAI API key, provided during initialization or
                                 obtained from the env variable 'OPENAI_API_KEY'.
        model (Optional[str]): The OpenAI API model, provided during initialization.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        """Initializes the DataGenerator instance.

        Args:
            api_key (Optional[str]): API key for OpenAI. If not provided, retrieved
                                     from the 'OPENAI_API_KEY' env variable.
            model (Optional[str]): The API model for OpenAI.
        """
        # Retrieve the API key from argument or environment variable.
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required to initialize DataGenerator."
            )
        self.api_key = api_key

        # Retrieve the API model from argument.
        self.model = model

    def get_api_key(self) -> str:
        """Retrieves the API key.

        Returns:
            (str): The API key stored in the instance.
        """
        return self.api_key

    def get_model(self) -> str:
        """Retrieves the API model.

        Returns:
            (str): The API model stored in the instance.
        """
        return self.model

    def generate(self, input_file: str, output_file: str) -> bool:
        """Generates data using the OpenAI API and saves the output to an Excel file.

        The process includes:
        1. Initializing a DataProcessor instance.
        2. Delegating the data generation process to the DataProcessor.

        Args:
            input_file (str): Path to the input file that contains necessary columns to
                              determine the type of data generation.
            output_file (str): Path where the generated Excel file will be saved.

        Returns:
            bool: True if the data processing is successful, False otherwise.

        Raises:
            ValueError: If no API key is provided.
        """
        # Initialize the DataProcessor with input file, output file, and API key.
        data_processor = DataProcessor(
            input_file=input_file,
            output_file=output_file,
            # Retrieve the final API key (from instance or environment)
            api_key=self.get_api_key(),
            # Retrieve the final API model (from instance or environment)
            openai_api_model=self.get_model(),
        )

        # Delegate the generation process to the DataProcessor and return its result.
        return data_processor.process_data()
