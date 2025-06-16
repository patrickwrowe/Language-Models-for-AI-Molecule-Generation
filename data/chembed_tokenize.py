from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation

UNKNOWN_TOKEN = '[UNK]'
END_OF_MOLECULE_TOKEN = '[EOM]'

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

