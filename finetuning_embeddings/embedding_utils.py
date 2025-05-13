"""embedding_utils module.

Utility functions for computing and managing embeddings.
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    CachedMultipleNegativesSymmetricRankingLoss,
    ContrastiveLoss,
    MatryoshkaLoss,
    MegaBatchMarginLoss,
    MultipleNegativesRankingLoss,
    MultipleNegativesSymmetricRankingLoss,
    OnlineContrastiveLoss,
    TripletLoss,
)


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
