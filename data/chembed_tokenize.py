from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

UNKNOWN_TOKEN = '[UNK]'
END_OF_MOLECULE_TOKEN = '[EOM]'

CURRENT_DEFAULT_TOKENIZER = "./tokenizers/tokenizer-chembldb-16-06-2025.json"

def train_bpe_tokenizer(filepaths: list[str], vocab_size: int = 1024, special_tokens: list[str] =[UNKNOWN_TOKEN, END_OF_MOLECULE_TOKEN]):
    tokenizer = Tokenizer(BPE(unk_token=UNKNOWN_TOKEN))
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)

    # Do we need more special tokens?
    # special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    # Include pre-tokenizer for punctuation
    tokenizer.pre_tokenizer = Punctuation()

    tokenizer.train(
        filepaths,
        trainer,
    )

    return tokenizer

def load_chembed_tokenizer(filepath: str = CURRENT_DEFAULT_TOKENIZER):
    """
    We'll use this as a hopefully temporary converter because we're training with the 
    "Tokenizers" library and loading with the transformers library for fast use
    """
    return PreTrainedTokenizerFast(
        tokenizer_file = filepath,
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<cls>",
        sep_token="<sep>",
        bos_token="<s>",
        eos_token="<pad>",
    )
