"""Collect and run embedding experiments.

This script:
1. Loads hyperparameter settings and script names from a YAML config.
2. Generates all combinations of training epochs, learning rates, and model names.
3. Executes each experiment inside a Podman container.
4. Stops the pod once all experiments complete.
"""

import os
import itertools
import yaml
import argparse

def load_parameters(method, yaml_file="experiments/configs/embedding_configs.yaml"):
    """
    Load the experiment parameters and target script for a given method.

    :param method:    Identifier for the embedding method (e.g., 'triplet').
    :param yaml_file: Path to the YAML file defining model-parameters and script.
    :return:          A tuple (parameters_list, python_script_filename).
    :raises ValueError: If the method key is not found in the YAML.
    """
    with open(yaml_file) as file:
        data = yaml.safe_load(file)

    if method in data:
        # 'model-parameters' is expected to be a list of parameter groups
        parameters = data[method]["model-parameters"]
        # 'script' is the Python file to run for this method
        python_script = data[method]["script"]
    else:
        raise ValueError(f"Unknown method: {method}")

    return parameters, python_script

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run embedding experiments with different methods."
    )
    parser.add_argument(
        "--method", type=str, required=True,
        help="Embedding method to use (e.g., 'triplet', 'contrastive', 'matryoshka')."
    )
    parser.add_argument(
        "--pod-name", type=str, default=os.environ.get("POD_NAME"),
        help="Name of the Podman pod (env var POD_NAME if not provided)."
    )
    parser.add_argument(
        "--yaml-file", type=str,
        default="experiments/embedding_configs.yaml",
        help="Path to the YAML configuration file defining experiments."
    )
    args = parser.parse_args()

    method    = args.method
    pod_name  = args.pod_name
    yaml_file = args.yaml_file

    if not pod_name:
        raise ValueError(
            "Pod name must be provided via --pod-name or the POD_NAME environment variable."
        )

    print(f"Experiment method: {method}")
    print(f"Pod name:           {pod_name}")

    # Load hyperparameters list and script name from the YAML config
    parameters, python_script = load_parameters(method, yaml_file)

    # Unpack hyperparameter options
    # parameters is expected to be a list of dicts, in a fixed order:
    # 0: num_train_epochs, 1: experiment_name, 2: run_name template,
    # 3: output_dir template, 4: learning_rate, 5: model_name
    num_train_epochs_options = parameters[0]["num_train_epochs"]
    experiment_name         = parameters[1]["experiment_name"]
    run_name                = parameters[2]["run_name"]
    base_output_dir         = parameters[3]["output_dir"]
    learning_rate_options   = parameters[4]["learning_rate"]
    model_name_options      = parameters[5]["model_name"]

    # Generate all combinations of (epochs, learning_rate, model_name)
    parameter_combinations = list(itertools.product(
        num_train_epochs_options,
        learning_rate_options,
        model_name_options
    ))

    # Execute each experiment combination inside the Podman container
    for num_epochs, lr, model_name in parameter_combinations:
        # Substitute the model name into the output directory path
        output_dir = base_output_dir.replace("mpnet-base-all-nli", model_name)

        # Build the command-line arguments for the training script
        element_args = (
            f"--num_train_epochs {num_epochs} "
            f"--learning_rate {lr} "
            f"--model_name {model_name} "
            f"--output_dir {output_dir} "
            f"--experiment_name {experiment_name}"
            f"--run_name {run_name}"
        )

        print(
            f"Running {method} experiment with "
            f"epochs={num_epochs}, lr={lr}, model={model_name}"
        )
        print(f"Command args: {element_args}")

        # Invoke the training script inside the specified pod
        os.system(
            f"podman exec {pod_name}_dev "
            f"python experiments/scripts/{python_script} {element_args}"
        )

    print("All experiments finished. Stopping the pod.")
    # Stop the Podman pod using a Makefile target or custom script
    os.system("make stop")

if __name__ == "__main__":
    main()
