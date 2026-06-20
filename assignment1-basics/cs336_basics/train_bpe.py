import regex as re
from pathlib import Path

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenization(text: str, special_tokens: list[str]):
    """
    Implement a regex-based pre-tokenizer and apply it on the corpus.
    """
    freq_table: dict[tuple[bytes, ...], int] = {}
    # split text on special tokens, 
    # use escape to avoid splitting text on partial special tokens
    delimiter = "|".join(re.escape(token) for token in special_tokens)
    text_split = re.split(delimiter, text)

    # avoid storing pre_tokenized words
    for match in re.finditer(PAT, text_split):
        pre_token = match.group() # split is a match object, not string
        # iterating over encoded string gives integers
        pre_token_bytes = tuple(bytes([s]) for s in pre_token.encode("utf-8"))
        freq_table[pre_token_bytes] = freq_table.get(pre_token_bytes, 0) + 1 

    return freq_table

def count_pairs_in_words(freq_table: dict[tuple[bytes, ...], int]):
    """
    Count adjacent pairs in pre-tokenized words.
    """
    pairs_table: dict[tuple[bytes, bytes], int] = {}
    for word, count in freq_table.items():
        pre = word[0]
        for cur in word[1:]:
            pair = (pre, cur)
            pairs_table[pair] = pairs_table.get(pair, 0) + count
            pre = cur

    return pairs_table

def merge_once(freq_table: dict[tuple[bytes, ...], int], target_merge: tuple[bytes, bytes]):
    """
    Apply a merge of tokens on frequency table.
    """
    first, second = target_merge[0], target_merge[1]
    new_token = first + second
    new_freq_table = {}

    for word, count in freq_table.items():
        new_word = word
        word_lenth = len(word)
        pre = 0
        for cur in range(1, word_lenth):
            if word[pre] == first and word[cur] == second:
                # replace with new word after merging
                new_word = new_word[:pre] + (new_token,) + new_word[cur + 1:]
            pre = cur
        new_freq_table[new_word] = count

    return new_freq_table

def merge(freq_table: dict[tuple[bytes, ...], int], vocabulary: dict[int, bytes], end_id: int, vocab_size: int):
    """
    Compute the BPE merges (i.e., train the BPE tokenizer).
    """
    merges: list[tuple[bytes, bytes]] = []
    new_freq_table = freq_table
    merge_times = vocab_size - len(vocabulary)

    for _ in range(merge_times):
        pairs_table = count_pairs_in_words(new_freq_table)
        # first find the most frequent fair; choose the lexicographically greatest pair if tied
        most_common_pair = max(pairs_table, key=lambda pair: (pairs_table[pair], pair))
        new_freq_table = merge_once(new_freq_table, most_common_pair)
        merges.append(most_common_pair)
        vocabulary[end_id] = most_common_pair[0] + most_common_pair[1]
        end_id += 1

    return vocabulary, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Training a (byte-level) BPE tokenizer.
    """
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    # vocab init: 256 byte values + special tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    end_id = 256
    for special_token in special_tokens:
        vocab[end_id] = special_token.encode("utf-8")
        end_id += 1

    # pre-tokenization
    freq_table = pre_tokenization(text, special_tokens)

    # Compute BPE merges (i.e., Train BPE tokenizer)
    new_vocab, merges = merge(freq_table, vocab, end_id, vocab_size)

    return new_vocab, merges