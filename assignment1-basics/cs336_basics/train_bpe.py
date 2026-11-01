import regex as re
import multiprocessing as mp
import os
import time
import threading
import psutil
import pickle
from pathlib import Path
from collections import Counter
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
peak_memory = 0
stop_monitor = threading.Event()

def monitor_memory():
    global peak_memory

    parent = psutil.Process(os.getpid())

    while not stop_monitor.is_set():
        total_memory = 0

        processes = [parent] + parent.children(recursive=True)

        for proc in processes:
            try:
                total_memory += proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        peak_memory = max(peak_memory, total_memory)

        stop_monitor.wait(0.05)

def count_word_freq(text: str, special_tokens: list[str]):
    """
    Perform pre-tokenization on text chunk:
        1) Split on special tokens (special tokens do not participate in training stats)
        2) Apply GPT-2 regex pre-tokenization
        3) Encode each piece to UTF-8 bytes and split into single-byte tokens
        4) Count token-sequence frequencies
    """
    freq_table: dict[tuple[bytes, ...], int] = {}
    # use escape to avoid splitting text on partial special tokens
    delimiter = "|".join(re.escape(token) for token in special_tokens)
    text_split = re.split(delimiter, text)
    for text_chunk in text_split:
        # avoid storing pre_tokenized words
        for match in re.finditer(PAT, text_chunk):
            pre_token = match.group() # split is a match object, not string
            # iterating over encoded string gives integers
            pre_token_bytes = tuple(bytes([s]) for s in pre_token.encode("utf-8"))
            freq_table[pre_token_bytes] = freq_table.get(pre_token_bytes, 0) + 1 
    
    return freq_table

def process_chunk(input_path: str, start: int, end: int, special_tokens: list[str]):
    """
    Worker entry point for processing a single file chunk.
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
    text = chunk.decode("utf-8")
    return count_word_freq(text, special_tokens)

def pre_tokenization(input_path: str, special_tokens: list[str], n_process: int, n_chunks: int):
    """
    Implement a regex-based pre-tokenizer and apply it on the corpus.
    """
    freq_table = Counter()
    split_special_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_chunks, split_special_token)

    text_segmented = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(processes=n_process) as pool:
        for partial_freq in pool.starmap(process_chunk, text_segmented):
            freq_table.update(partial_freq)

    return dict(freq_table)

def count_pair_in_word(word: tuple[bytes, ...]):
    """
    Count occurrences of adjacent pairs in a word.
    """
    count: dict[tuple[bytes, bytes], int] = {}
    pre = word[0]
    for cur in word[1:]:
        pair = (pre, cur)
        count[pair] = count.get(pair, 0) + 1
        pre = cur

    return count

def build_pair_stats(freq_table: dict[tuple[bytes, ...], int]):
    """
    Build:
      - pair_counts: global weighted counts for each adjacent pair
      - pair_to_words: inverted index (pair -> set of words containing that pair)
    """
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for word, word_freq in freq_table.items():
        pair_in_word_count = count_pair_in_word(word)
        for pair, occurrence in pair_in_word_count.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + occurrence * word_freq
            pair_to_words.setdefault(pair, set()).add(word)

    return pair_counts, pair_to_words

def merge_once(word: tuple[bytes, ...], target_merge: tuple[bytes, bytes]):
    """
    Apply a merge of tokens on a word.
    """
    new_word = []
    word_len = len(word)
    fir, sec = target_merge[0], target_merge[1]
    i = 0
    while i < word_len:
        if i < word_len - 1 and word[i] == fir and word[i + 1] == sec:
            new_word.append(fir + sec)
            i += 2
        else:
            new_word.append(word[i])
            i += 1

    return tuple(new_word)

def delete_stats(pair_counts: dict[tuple[bytes, bytes], int], 
                 pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
                 word: tuple[bytes, ...],
                 word_freq: int):
    """
    Remove stats of adjacent pairs in a word from pair_counts and pair_to_words.
    """
    pair_stats = count_pair_in_word(word)
    for pair, pair_freq in pair_stats.items():
        new_occurrences = pair_counts[pair] - pair_freq * word_freq
        if not new_occurrences:
            del pair_counts[pair]
        else:
            pair_counts[pair] = new_occurrences

        words_contained_pair = pair_to_words[pair]
        words_contained_pair.remove(word)
        if not words_contained_pair:
            del pair_to_words[pair]

def add_stats(pair_counts: dict[tuple[bytes, bytes], int], 
              pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
              word: tuple[bytes, ...],
              word_freq: int):
    """
    Add stats of adjacent pairs in a word to pair_counts and pair_to_words.
    """
    pair_stats = count_pair_in_word(word)
    for pair, pair_freq in pair_stats.items():
        pair_counts[pair] = pair_counts.get(pair, 0) + pair_freq * word_freq
        pair_to_words.setdefault(pair, set()).add(word)


def apply_complete_merge(freq_table: dict[tuple[bytes, ...], int], vocabulary: dict[int, bytes], end_id: int, vocab_size: int):
    """
    Compute the BPE merges.
    """
    merges: list[tuple[bytes, bytes]] = []
    pair_counts, pair_to_words = build_pair_stats(freq_table)
    merge_times = vocab_size - len(vocabulary)

    for _ in range(merge_times):
        # first find the most frequent fair; choose the lexicographically greatest pair if tied
        most_common_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(most_common_pair)
        vocabulary[end_id] = most_common_pair[0] + most_common_pair[1]
        end_id += 1

        affected_words = pair_to_words[most_common_pair].copy()
        for affected in affected_words:
            word_freq = freq_table[affected]
            delete_stats(pair_counts, pair_to_words, affected, word_freq)
            del freq_table[affected]
            new_word = merge_once(affected, most_common_pair)
            freq_table[new_word] = word_freq
            add_stats(pair_counts, pair_to_words, new_word, word_freq)

    return vocabulary, merges

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], n_process: int = 4):
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
    freq_table = pre_tokenization(input_path, special_tokens, n_process, n_chunks=n_process * 32)

    # Compute BPE merges
    new_vocab, merges = apply_complete_merge(freq_table, vocab, end_id, vocab_size)

    return new_vocab, merges

if __name__ == "__main__":
    stop_monitor.clear()
    monitor = threading.Thread(target=monitor_memory)
    start_time = time.perf_counter()
    monitor.start()

    try:
        vocab, merges = train_bpe(
            input_path="data/TinyStoriesV2-GPT4-train.txt",
            vocab_size=10000,
            special_tokens=['<|endoftext|>']
        )
    finally:
        stop_monitor.set()
        monitor.join()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(list(vocab.values()), f)
    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    print(f"Execution Time: {elapsed_time:.4f} seconds")
    print(f"Peak Memory: {peak_memory / 1024**2:.2f} MiB")
    longest_token = max(vocab.values(), key=lambda x: len(x))
    print(longest_token)