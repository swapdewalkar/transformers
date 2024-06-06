from ...utils import logging
from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class DeepSeekV2Tokenizer(RobertaTokenizerFast):
    """
    Construct a DeepSeekV2 tokenizer. Based on Byte-Pair-Encoding.

    This tokenizer inherits from [`RobertaTokenizerFast`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            File containing the vocabulary.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"<sep>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two
    """