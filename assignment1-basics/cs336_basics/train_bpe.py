import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenization(text: str, special_tokens: list[str]):
    """
    Implement a regex-based pre-tokenizer and apply it on the corpus.
    """
    freq_table: dict[tuple[bytes, ...], int] = {}

    # avoid storing pre_tokenized words
    for match in re.finditer(PAT, text):
        pre_token = match.group() # split is a match object, not string
        # iterating over encoded string gives integers
        pre_token_bytes = tuple(bytes([s]) for s in pre_token.encode("utf-8"))
        freq_table[pre_token_bytes] = freq_table.get(pre_token_bytes, 0) + 1 

    return freq_table

def compute_pairs_in_words(freq_table: dict[tuple[bytes, ...], int]):
    """
    Count adjacent pairs in pre-tokenized words.
    """
    pass

def merge(freq_table: dict[tuple[bytes, ...], int]):
    """
    Compute the BPE merges (i.e., train the BPE tokenizer).
    """
    pass

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Training a (byte-level) BPE tokenizer.
    """
    # vocab init: 256 byte values + special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    end_id = 256
    for special_token in special_tokens:
        vocab[end_id] = special_token.encode("utf-8")
        end_id += 1

    # pre-tokenization
    freq_table = pre_tokenization(text, special_tokens)

    pass