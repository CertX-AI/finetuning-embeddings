"""
exp_finetuning_embeddings_mnrl_triplet.py

This script fine-tunes a SentenceTransformer model using a custom dataset and logs
the process and results to MLflow. The script supports argument parsing for various
training parameters, allowing for flexibility and easy experimentation.

The script performs the following steps:
1. Loads and prepares the dataset for fine-tuning.
2. Initializes the SentenceTransformer model and defines the loss function.
3. Configures training arguments based on input parameters.
4. Logs the training arguments to MLflow for tracking.
5. Creates evaluators for the development and test datasets.
6. Trains the model using the specified configurations.
7. Evaluates the model on both the development and test datasets.
8. Logs evaluation metrics to MLflow.
9. Saves the final trained model to the specified output directory.

Arguments:
    --num_train_epochs: Number of training epochs (default: 1).
    --experiment_name: Name of the MLflow experiment (required).
    --run_name: Name of the MLflow run (required).
    --output_dir: Directory to save the trained model (required).
    --learning_rate: Initial learning rate for the optimizer (default: 2e-5).
    --model_name: Pre-trained model name to fine-tune (required).

Usage Example:
    python exp_finetune_embeddings_mnrl_triplet.py --num_train_epochs 1 \
        --experiment_name "triplet_pair_finetuning" --run_name "fine_tune_run_1" \
        --output_dir "docs/models/mpnet-base-all-nli-triplet/final" --learning_rate 3e-5 \
        --model_name "all-mpnet-base-v2"
"""

import argparse
import mlflow
from sentence_transformers import SentenceTransformerTrainer
from finetuning_embeddings.embedding_utils import (
    prepare_finetune_dataset,
    initialize_model,
    get_MNRL_or_matryoshka_loss,
    create_training_arguments,
    create_tripletloss_evaluator,
)

def main(num_train_epochs, experiment_name, run_name, output_dir, learning_rate, model_name):
    """
    Main function to run the fine-tuning process for a SentenceTransformer model.

    :param num_train_epochs: Number of training epochs.
    :param run_name: The name of the MLflow run.
    :param output_dir: Directory where model checkpoints will be saved.
    :param learning_rate: The initial learning rate for the optimizer.
    :param model_name: The name of the pre-trained model to use for fine-tuning.
    """

    # Initialize MLflow
    mlflow.set_tracking_uri("http://host.containers.internal:5000")  # Set to your local MLflow server
    mlflow.set_experiment(experiment_name)  # Set your experiment name

    with mlflow.start_run():
        # 1. Load the dataset to finetune on
        print("Loading dataset...")
        finetune_dataset = prepare_finetune_dataset(flag=True)
        train_dataset = finetune_dataset["train"].select(range(10_000))  # Subset for testing
        eval_dataset = finetune_dataset["dev"]
        test_dataset = finetune_dataset["test"]

        # 2. Load the model to finetune and define the MNRL loss function
        print(f"Loading model '{model_name}'...")
        training_model =  initialize_model(model_name, use_prompts=False)

        # 3. Define the MNRL loss function
        print(f"Creating an Instance for Multiple Negatives Ranking Loss...")
        loss = get_MNRL_or_matryoshka_loss(
            model= training_model,
            use_matryoshka=False
        )

        # 4. Specify training arguments
        print("Configuring training arguments...")
        training_args = create_training_arguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            run_name=run_name
        )

        # 4. Log training arguments to MLflow
        mlflow.log_params({
            "output_dir": training_args.output_dir,
            "num_train_epochs": training_args.num_train_epochs,
            "train_batch_size": training_args.per_device_train_batch_size,
            "eval_batch_size": training_args.per_device_eval_batch_size,
            "learning_rate": training_args.learning_rate,
            "warmup_ratio": training_args.warmup_ratio,
            "fp16": training_args.fp16,
            "bf16": training_args.bf16,
            "eval_strategy": training_args.eval_strategy,
            "eval_steps": training_args.eval_steps,
            "save_strategy": training_args.save_strategy,
            "save_steps": training_args.save_steps,
            "save_total_limit": training_args.save_total_limit,
            "logging_steps": training_args.logging_steps,
            "run_name": training_args.run_name,
        })

        # 5. Create the evaluator for the dev dataset
        print("Creating dev evaluator...")
        dev_evaluator = create_tripletloss_evaluator(eval_dataset, name="all-nli-dev")

        # 6. Create a trainer and start training
        print("Starting training...")
        trainer = SentenceTransformerTrainer(
            model=training_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )
        trainer.train()

        # 7. Evaluate the trained model on the dev set after training
        print("Evaluating on dev dataset...")
        dev_score = dev_evaluator(training_model)
        mlflow.log_metrics(dev_score)

        # 8. Create the evaluator for the test dataset
        print("Creating test evaluator...")
        test_evaluator = create_tripletloss_evaluator(test_dataset, name="all-nli-test")

        # 9. Evaluate the trained model on the test set after training
        print("Evaluating on test dataset...")
        test_score = test_evaluator(training_model)
        mlflow.log_metrics(test_score)

        # 10. Save the trained model locally
        print(f"Saving the model to {output_dir}...")
        training_model.save_pretrained(output_dir)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Fine-tune a SentenceTransformer model"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the MLflow experiment"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the MLflow run"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the model"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pre-trained model name"
    )

    args = parser.parse_args()
    print("Arguments are successfully parsed!!!")

    # Call the main function with parsed arguments
    main(
        num_train_epochs=args.num_train_epochs,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
    )

    print(f"Embedding model {args.model_name} is successfully finetuned!!!")
    print(f"Multiple Negatives Ranking Loss (MNRL) technique has been used !!!")
    print(f"Embedding model can be found in {args.output_dir} folder!!!")
