"""embedding_utils module.

Utility functions for computing and managing embeddings.
"""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from datasets import Dataset, concatenate_datasets
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    NanoBEIREvaluator,
    SequentialEvaluator,
    TripletEvaluator,
)
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    CachedMultipleNegativesSymmetricRankingLoss,
    ContrastiveLoss,
    Matryoshka2dLoss,
    MatryoshkaLoss,
    MegaBatchMarginLoss,
    MultipleNegativesRankingLoss,
    MultipleNegativesSymmetricRankingLoss,
    OnlineContrastiveLoss,
    TripletLoss,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import cos_sim


##########################  MODEL ##########################
def initialize_model(
    model_name: str,
    prompts: dict[str, str] | None = None,
    default_prompt_name: str | None = None,
    use_prompts: bool = True
) -> SentenceTransformer:
    """Initialize a SentenceTransformer model with optional prompt templates.

    :param model_name: The name of the pre-trained model
                       (e.g., "xlm-roberta-base").
    :param prompts: Optional dict mapping prompt names to prompt texts.
                    Example mapping:
                      "classification": "Classify the following text: ",
                      "retrieval":      "Retrieve semantically similar text: ",
                      "clustering":     "Identify the topic or theme: ",
    :param default_prompt_name: Optional key from `prompts` to use by default.
                                If None, no default prompt is applied.
    :param use_prompts: If False, ignore prompts and default_prompt_name.
    :return: A configured SentenceTransformer instance.
    """
    # If prompts are disabled, load plain model
    if not use_prompts:
        return SentenceTransformer(model_name)

    # Validate default prompt name
    if default_prompt_name is not None:
        if prompts is None:
            raise ValueError(
                f"Cannot set default_prompt_name='{default_prompt_name}' "
                "without providing any prompts."
            )
        if default_prompt_name not in prompts:
            raise ValueError(
                f"default_prompt_name='{default_prompt_name}' "
                f"is not in prompts keys: {list(prompts.keys())}"
            )

    # Load without any prompts
    if prompts is None and default_prompt_name is None:
        return SentenceTransformer(model_name)

    # Load with prompts but no default
    if prompts is not None and default_prompt_name is None:
        return SentenceTransformer(
            model_name,
            prompts=prompts
        )

    # Load with both prompts and a default prompt
    return SentenceTransformer(
        model_name,
        prompts=prompts,
        default_prompt_name=default_prompt_name
    )


##########################  MODEL ##########################

################### LIST LOSS FUNCTIONS ####################

# Mapping from input keys to available SBERT loss class names
_LOSSES_BY_INPUT: dict[str, list[str]] = {
    "single_sentences": [
        "BatchAllTripletLoss",
        "BatchHardSoftMarginTripletLoss",
        "BatchHardTripletLoss",
        "BatchSemiHardTripletLoss",
        "ContrastiveTensionLoss",
        "DenoisingAutoEncoderLoss",
    ],
    "anchor_anchor_pairs": [
        "ContrastiveTensionLossInBatchNegatives",
    ],
    "damaged_original_pairs": [
        "DenoisingAutoEncoderLoss",
    ],
    "sentence_a_sentence_b_pairs": [
        "SoftmaxLoss",
        "CoSENTLoss",
        "AnglELoss",
        "CosineSimilarityLoss",
    ],
    "anchor_positive_pairs": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "MultipleNegativesSymmetricRankingLoss",
        "CachedMultipleNegativesSymmetricRankingLoss",
        "MegaBatchMarginLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
    ],
    "anchor_pos_neg_pairs": [
        "ContrastiveLoss",
        "OnlineContrastiveLoss",
    ],
    "anchor_positive_negative_triplets": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "TripletLoss",
        "GISTEmbedLoss",
        "CachedGISTEmbedLoss",
    ],
    "anchor_positive_multiple_negatives": [
        "MultipleNegativesRankingLoss",
        "CachedMultipleNegativesRankingLoss",
        "CachedGISTEmbedLoss",
    ],
}

def print_available_losses(input_type: str) -> None:
    """Print the available loss function names for a given normalized input_type key.

    :param input_type: One of the keys in _LOSSES_BY_INPUT,
                       e.g. "anchor_positive_pairs" or "single_sentences".
    """
    if input_type not in _LOSSES_BY_INPUT:
        print(f"Unknown input_type '{input_type}'. Valid types are:")
        for key in _LOSSES_BY_INPUT:
            print(f" - {key}")
        return

    print(f"Available losses for '{input_type}':")
    for loss_name in _LOSSES_BY_INPUT[input_type]:
        print(f" - {loss_name}")

################### LIST LOSS FUNCTIONS ####################

###################### LOSS FUNCTIONS ######################

############### ANCHOR-POSITIVE PAIRS LOSSES ###############

def get_MNRL_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None
) -> MultipleNegativesRankingLoss | MatryoshkaLoss:
    """Return either a MultipleNegativesRankingLoss or a MatryoshkaLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   Whether to wrap the base loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims if using Matryoshka (defaults).
    :return:                 MultipleNegativesRankingLoss or MatryoshkaLoss.
    """
    # Provide default nested dimensions if none were supplied
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base retrieval loss: MultipleNegativesRankingLoss uses
    # in-batch negatives to train sentence embeddings for retrieval tasks.
    base_loss = MultipleNegativesRankingLoss(model)

    # If the user requested Matryoshka, wrap the base loss:
    # - MatryoshkaLoss will train several projection heads in nested order,
    #   from the largest dimension down to the smallest, enforcing each head
    #   to mimic the next-larger head.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise, simply return the standard MultipleNegativesRankingLoss.
    return base_loss


def get_cached_MNRL_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None,
    batch_size: int = 32
) -> CachedMultipleNegativesRankingLoss | MatryoshkaLoss:
    """Return either a CachedMultipleNegativesRankingLoss or a MatryoshkaLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   Whether to wrap the base loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims if using Matryoshka (defaults).
    :param batch_size:       Mini-batch size for the forward pass.
    :return:                 CachedMultipleNegativesRankingLoss or MatryoshkaLoss.
    """
    # Provide default nested dimensions if none were supplied
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base retrieval loss:
    # CachedMultipleNegativesRankingLoss caches embeddings for negatives
    # to speed up training when in-batch negatives are reused.
    base_loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=batch_size
    )

    # If Matryoshka is requested, wrap the cached loss:
    # - Trains nested projection heads from largest to smallest dimension
    #   enforcing each head to mimic the next larger one.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise, return the cached multiple negatives ranking loss.
    return base_loss

def get_symm_MNRL_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None
) -> MultipleNegativesSymmetricRankingLoss | MatryoshkaLoss:
    """Return either a MultipleNegativesSymmetricRankingLoss or a MatryoshkaLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   Whether to wrap the base loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims if using Matryoshka (defaults).
    :return:                 MultipleNegativesSymmetricRankingLoss or MatryoshkaLoss.
    """
    # Provide default nested dimensions if none were supplied
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base symmetric retrieval loss:
    # MultipleNegativesSymmetricRankingLoss uses both directions
    # of in-batch negatives for symmetric training.
    base_loss = MultipleNegativesSymmetricRankingLoss(model)

    # If Matryoshka is requested, wrap the symmetric loss:
    # - Trains nested projection heads from largest to smallest dimension,
    #   each mimicking the next larger head.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise, return the symmetric ranking loss directly.
    return base_loss

def get_cached_symm_MNRL_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None,
    batch_size: int = 32
) -> CachedMultipleNegativesSymmetricRankingLoss | MatryoshkaLoss:
    """Return either a CachedMultipleNegativesSymmetricRankingLoss or a MatryoshkaLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   Whether to wrap the base loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims if using Matryoshka (defaults).
    :param batch_size:       Mini-batch size for the forward pass.
    :return:                 CachedMultipleNegativesSymmRankingLoss or MatryoshkaLoss.
    """
    # Provide default nested dimensions if none were supplied
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base cached symmetric retrieval loss:
    # CachedMultipleNegativesSymmetricRankingLoss caches both query
    # and passage embeddings for symmetric in-batch negatives, speeding up
    # training when examples are reused.
    base_loss = CachedMultipleNegativesSymmetricRankingLoss(
        model,
        mini_batch_size=batch_size
    )

    # If Matryoshka is requested, wrap the cached symmetric loss:
    # - Trains nested projection heads from largest to smallest dimension,
    #   enforcing each smaller head to mimic the next larger head.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise, return the cached symmetric ranking loss directly.
    return base_loss

def get_megabatch_margin_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None,
    train_batch_size: int = 50
) -> MegaBatchMarginLoss | MatryoshkaLoss:
    """Return either a MegaBatchMarginLoss or a MatryoshkaLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   Whether to wrap the base loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims if using Matryoshka (defaults).
    :param train_batch_size: Size for the mini-batches.
    :return:                 MegaBatchMarginLoss or MatryoshkaLoss.
    """
    # Default nested dimensions if none provided
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base margin loss:
    # MegaBatchMarginLoss uses larger "mega-batches" of negatives
    # to compute a margin-based loss for retrieval tasks.
    base_loss = MegaBatchMarginLoss(
        model,
        mini_batch_size=train_batch_size
    )

    # Wrap in MatryoshkaLoss if requested:
    # - Trains nested projection heads from largest to smallest dim,
    #   enforcing each smaller head to mimic the next larger head.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise return the standard MegaBatchMarginLoss
    return base_loss

def get_MNRL_or_matryoshka2d_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None
) -> MultipleNegativesRankingLoss | Matryoshka2dLoss:
    """Return either a MultipleNegativesRankingLoss or a Matryoshka2dLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   Whether to wrap the base loss in Matryoshka2dLoss.
    :param matryoshka_dims:  Nested embedding dims if using Matryoshka2d (defaults).
    :return:                 MultipleNegativesRankingLoss or Matryoshka2dLoss.
    """
    # Provide default nested dimensions if none were supplied
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base retrieval loss: MultipleNegativesRankingLoss uses
    # in-batch negatives to train sentence embeddings for retrieval tasks.
    base_loss = MultipleNegativesRankingLoss(model)

    # If the user requested Matryoshka2dLoss, wrap the base loss:
    # - Matryoshka2dLoss will train several projection heads in nested order,
    #   from the largest dimension down to the smallest, enforcing each head
    #   to mimic the next-larger head.
    if use_matryoshka:
        return Matryoshka2dLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise, simply return the standard MultipleNegativesRankingLoss.
    return base_loss

############### ANCHOR-POSITIVE PAIRS LOSSES ###############

######### (ANCHOR, POSITIVE/NAGATIVE) PAIRS LOSSES #########

def get_contrastive_loss_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None
) -> ContrastiveLoss | MatryoshkaLoss:
    """Return either a ContrastiveLoss or a MatryoshkaLoss-wrapped ContrastiveLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   If True, wrap the base contrastive loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims for Matryoshka (defaults to
                             [768, 512, 256, 128, 64] if None).
    :return:                 A ContrastiveLoss or MatryoshkaLoss instance.
    """
    # Default nested dimensions if none provided
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base contrastive loss:
    # ContrastiveLoss trains on (anchor, positive/negative) pairs with a margin
    base_loss = ContrastiveLoss(model)

    # Wrap in MatryoshkaLoss if requested:
    # - Trains a series of projection heads from largest to smallest dim,
    #   enforcing each smaller head to mimic its predecessor.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise return the standard contrastive loss
    return base_loss

def get_online_contrastive_loss_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None
) -> OnlineContrastiveLoss | MatryoshkaLoss:
    """Return OnlineContrastiveLoss or a MatryoshkaLoss-wrapped OnlineContrastiveLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   If True, wrap the base loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims for Matryoshka (defaults to
                             [768, 512, 256, 128, 64] if None).
    :return:                 An OnlineContrastiveLoss or MatryoshkaLoss instance.
    """
    # Provide default nested dimensions if none were supplied
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base online contrastive loss:
    # OnlineContrastiveLoss mines hard positives and negatives within each batch
    base_loss = OnlineContrastiveLoss(model)

    # If Matryoshka is requested, wrap the base loss to train nested heads:
    # - Each projection head from largest to smallest dim is trained to mimic
    #   the next-larger head, fostering multi-scale representations.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise return the standard OnlineContrastiveLoss
    return base_loss

######### (ANCHOR, POSITIVE/NAGATIVE) PAIRS LOSSES #########

####### (ANCHOR, POSITIVE, NAGATIVE) TRIPLETS LOSSES #######

def get_triplet_loss_or_matryoshka_loss(
    model: SentenceTransformer,
    use_matryoshka: bool = False,
    matryoshka_dims: list[int] | None = None
) -> TripletLoss | MatryoshkaLoss:
    """Return either a TripletLoss or a MatryoshkaLoss-wrapped TripletLoss.

    :param model:            A SentenceTransformer instance.
    :param use_matryoshka:   If True, wrap the base triplet loss in MatryoshkaLoss.
    :param matryoshka_dims:  Nested embedding dims for Matryoshka (defaults to
                             [768, 512, 256, 128, 64] if None).
    :return:                 A TripletLoss or MatryoshkaLoss instance.
    """
    # Default nested dimensions if none provided
    if matryoshka_dims is None:
        matryoshka_dims = [768, 512, 256, 128, 64]

    # Create the base triplet loss:
    # TripletLoss trains on (anchor, positive, negative) triplets enforcing
    # anchor-positive similarity > anchor-negative similarity by a margin.
    base_loss = TripletLoss(model)

    # Wrap in MatryoshkaLoss if requested:
    # - Trains a series of projection heads from largest to smallest dim,
    #   each smaller head mimicking the next larger head.
    if use_matryoshka:
        return MatryoshkaLoss(
            model=model,
            loss=base_loss,
            matryoshka_dims=matryoshka_dims
        )

    # Otherwise return the standard TripletLoss
    return base_loss

####### (ANCHOR, POSITIVE, NAGATIVE) TRIPLETS LOSSES #######

###################### LOSS FUNCTIONS ######################

#################### TRAINING ARGUMENTS ####################

def create_training_arguments(
    output_dir: str = "experiments/results/mpnet-base-all-nli-triplet",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    fp16: bool = True,
    bf16: bool = False,
    eval_strategy: str = "steps",
    eval_steps: int = 200,
    save_strategy: str = "steps",
    save_steps: int = 200,
    save_total_limit: int = 2,
    logging_steps: int = 200,
    run_name: str = "mpnet-base-all-nli-triplet"
) -> SentenceTransformerTrainingArguments:
    """Create TrainingArguments with specified or default parameters.

    :param output_dir: Directory where model checkpoints will be saved.
    :param num_train_epochs: Number of training epochs.
    :param per_device_train_batch_size: Batch size per device during training.
    :param per_device_eval_batch_size: Batch size per device during evaluation.
    :param learning_rate: The initial learning rate for the optimizer.
    :param warmup_ratio: The warmup ratio for the learning rate scheduler.
    :param fp16: Whether to use FP16 precision during training.
    :param bf16: Whether to use BF16 precision during training (if GPU supports it).
    :param eval_strategy: The evaluation strategy to use during training.
    :param eval_steps: Number of update steps between two evaluations.
    :param save_strategy: The checkpoint save strategy.
    :param save_steps: Number of update steps between two checkpoint saves.
    :param save_total_limit: Maximum number of checkpoints to keep.
    :param logging_steps: Number of update steps between two logs.
    :param run_name: Name of the run for logging and tracking.
    :return: A configured SentenceTransformerTrainingArguments object.
    """
    # 1) Compute timestamp in Europe/Zurich
    now_zurich = datetime.now(ZoneInfo("Europe/Zurich"))
    timestamp = now_zurich.strftime("%Y%m%d_%H%M%S")
    # 2) Build run name
    curr_run_name = f"{run_name}_{timestamp}"

    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Correct usage
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        run_name=curr_run_name,
        report_to="mlflow",
    )

def create_training_arguments_matryoshka(
    output_dir: str = "experiments/results/matryoshka/bge-base-en-v1.5",
    num_train_epochs: int = 4,
    per_device_train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    per_device_eval_batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    tf32: bool = True,
    bf16: bool = True,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_dim_128_cosine_ndcg@10",
    run_name: str = "bge-base-matryoshka"
) -> SentenceTransformerTrainingArguments:
    """Create TrainingArguments with specified or default parameters.

    :param output_dir: Directory where model checkpoints will be saved.
    :param num_train_epochs: Number of training epochs.
    :param per_device_train_batch_size: Batch size per device during training.
    :param gradient_accumulation_steps: Number of steps to accumulate gradients before
                                        performing a backward/update pass.
    :param per_device_eval_batch_size: Batch size per device during evaluation.
    :param learning_rate: The initial learning rate for the optimizer.
    :param warmup_ratio: The warmup ratio for the learning rate scheduler.
    :param tf32: Whether to use TF32 precision during training (for NVIDIA Ampere GPUs).
    :param bf16: Whether to use BF16 precision during training (if GPU supports it).
    :param eval_strategy: The evaluation strategy to use during training.
    :param save_strategy: The checkpoint save strategy.
    :param save_total_limit: Maximum number of checkpoints to keep.
    :param load_best_model_at_end: Decision for loading the best model at the end.
    :param metric_for_best_model: Metric to determine the best model.
    :param run_name: Name of the run for logging and tracking.
    :return: A configured SentenceTransformerTrainingArguments object.
    """
    # 1) Compute timestamp in Europe/Zurich
    now_zurich = datetime.now(ZoneInfo("Europe/Zurich"))
    timestamp = now_zurich.strftime("%Y%m%d_%H%M%S")
    # 2) Build run name
    curr_run_name = f"{run_name}_{timestamp}"

    # 3) TrainingArguments with MLflow reporting and timestamped run name
    return SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        tf32=tf32,
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        run_name=curr_run_name,
        report_to="mlflow"
    )

#################### TRAINING ARGUMENTS ####################

######################## EVALUATORS ########################

# Input pairs: (anchor, positive, negative) triplets
def create_tripletloss_evaluator(
    dataset: Dataset,
    name: str
) -> TripletEvaluator:
    """Create a TripletEvaluator for a given dataset and return the evaluator.

    :param dataset: The dataset containing (anchor, positive, negative) triplets.
    :param name: The name of the evaluator.
    :return: The TripletEvaluator for the specified dataset.
    """
    # Print the size of each segment
    print(f"Size of anchor segment: {len(dataset['anchor'])}")
    print(f"Size of positive segment: {len(dataset['positive'])}")
    print(f"Size of negative segment: {len(dataset['negative'])}")

    # Create and return the evaluator
    evaluator = TripletEvaluator(
        anchors=dataset["anchor"],
        positives=dataset["positive"],
        negatives=dataset["negative"],
        name=name
    )
    return evaluator

# Input pairs: (anchor, positive) pairs
def create_NanoBEIR_evaluator(
    query_prompts: dict[str, str],
    datasets: list[
        Literal[
            "climatefever",
            "dbpedia",
            "fever",
            "fiqa2018",
            "hotpotqa",
            "msmarco",
            "nfcorpus",
            "nq",
            "quoraretrieval",
            "scidocs",
            "arguana",
            "scifact",
            "touche2020",
        ]
    ] | None = None,
) -> NanoBEIREvaluator:
    """Create a NanoBEIREvaluator for a given dataset list and return the evaluator.

    It can be used for quickly evaluating the retrieval performance of a model
    before commiting to a full evaluation.
    :param datasets: A set of datasets based on the BEIR collections.
    :return: The NanoBEIREvaluator for the specified dataset.
    """
    # Create and return the evaluator
    evaluator = NanoBEIREvaluator(
        dataset_names=datasets,
        query_prompts=query_prompts,
    )
    return evaluator

# Input pairs option: (anchor, positive) pairs
def create_informationretrieval_evaluator(
    train_dataset: Dataset, test_dataset: Dataset, name: str
) -> InformationRetrievalEvaluator:
    """Create an InformationRetrievalEvaluator for a given dataset.

    :param train_dataset: The dataset containing 'anchor' and 'positive' segments.
    :param test_dataset: The dataset containing 'anchor' and 'positive' segments.
    :param name: The name of the evaluator.
    :return: The InformationRetrievalEvaluator for the specified dataset.
    """
    # Concatenate train and test datasets to create a corpus
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Convert the datasets to dictionaries
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"], strict=False))
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"], strict=False))

    # Create a mapping of relevant documents (1 in our case) for each query
    # Each query is only relevant to its own passage
    relevant_docs: dict[str, set[str]] = {
        qid: {qid} for qid in queries
    }

    # Initialize the Information Retrieval Evaluator
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        score_functions={"cosine": cos_sim},
    )

    return ir_evaluator

# Input pairs option: (anchor, positive) pairs
def create_matryoshka_info_ret_evaluator(
    train_dataset: Dataset,
    test_dataset: Dataset
) -> SequentialEvaluator:
    """Create a SequentialEvaluator with matryoshka and return the evaluator.

    :param train_dataset: The dataset containing 'anchor' and 'positive' segments.
    :param test_dataset: The dataset containing 'anchor' and 'positive' segments.
    :param name: The name of the evaluator.
    :return: The SequentialEvaluator for the specified dataset.
    """
    # Define Matryoshka dimensions
    matryoshka_dimensions = [768, 512, 256, 128, 64]  # Large to small dimensions

    # Concatenate train and test datasets to create a corpus
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

    # Convert the datasets to dictionaries
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"], strict=False))
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"], strict=False))

    # Create a mapping of relevant documents (1 in our case) for each query
    # Each query is only relevant to its own passage
    relevant_docs: dict[str, set[str]] = {
        qid: {qid} for qid in queries
    }

    matryoshka_evaluators = []

    # Iterate over the different dimensions
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # Truncate the embeddings to a certain dimension
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)

    # Create and return a sequential evaluator
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    return evaluator

######################## EVALUATORS ########################
