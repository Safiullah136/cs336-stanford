import regex as re
from collections import defaultdict
from tqdm.contrib.concurrent import process_map
from save_bpe import *

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def word2bytes(word: str):
    return tuple( bytes([b]) for b in list(word.encode("utf-8")) )

def merge_dicts(dicts):
    merged = defaultdict(int)
    for dict in dicts:
        for k, v in dict.items():
            merged[k] += v
    return merged

def update_neighbours(old, new, stats, freq, pair_to_words, word_idx):
    stats[old] -= freq
    stats[new] += freq

    pair_to_words[new].add(word_idx)


def merge_word(word, pair, stats, freq, pair_to_words, idx):
    new_word = []
    i = 0
    a, b = pair
    merged = a + b

    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:   
            if i > 0:
                old_left = (word[i-1], a)
                new_left = (new_word[-1], merged)
                
                update_neighbours(old_left, new_left, stats, freq, pair_to_words, idx)                 

            if i < len(word) - 2 and not (i < len(word) - 3 and word[i+2] == a and word[i+3] == b):
                old_right = (b, word[i+2])
                new_right = (merged, word[i+2])
                
                update_neighbours(old_right, new_right, stats, freq, pair_to_words, idx)

            new_word.append(merged)
            i+=2
        else:
            new_word.append(word[i])
            i+=1

    return new_word

def read_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def get_max_pair(stats):
    return max(stats.keys(), key=lambda p: (stats[p], p))

def get_basic_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    for idx, tok in enumerate(special_tokens):
        vocab[idx + 256] = tok.encode("utf-8")
    return vocab

def split_on_special(text, special_tokens):
    if not special_tokens:
        return [text]
    
    delimiter =  "|".join(re.escape(tok) for tok in special_tokens)
    chunks = re.split(delimiter, text)
    return [c for c in chunks if c]

def collect_words_from_chunk(chunk):
    words = defaultdict(int)
    for word in PAT.finditer(string=chunk):
        word_bytes = word2bytes(word.group(0))
        words[word_bytes] += 1

    return words

def get_stats(words):
    stats = defaultdict(int)
    pair_to_words = defaultdict(set)
    for index, (word, freq) in enumerate(words):
        for pair in zip(word, word[1:]):
            stats[pair] += freq
            pair_to_words[pair].add(index)

    return stats, pair_to_words

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    text = read_file(input_path)
    chunks = split_on_special(text, special_tokens)

    if len(chunks) < 4: word_dicts = list(map(collect_words_from_chunk, chunks))
    else: word_dicts = process_map(collect_words_from_chunk, chunks, chunksize=1)

    words = merge_dicts(word_dicts)
    words = [(list(w), f) for w, f in words.items()]

    stats, pair_to_words = get_stats(words)

    merges = []
    vocab = get_basic_vocab(special_tokens)
    num_merges = vocab_size - 256 - len(special_tokens)
    for merge_no in range(num_merges):
        if not stats: break
        top_pair = get_max_pair(stats)

        for i in pair_to_words[top_pair]:
            old_word, freq = words[i]
            new_word = merge_word(old_word, top_pair, stats, freq, pair_to_words, i)
            words[i] = (new_word, freq)

        del stats[top_pair]
        del pair_to_words[top_pair]

        vocab[merge_no + 256 + len(special_tokens)] = top_pair[0] + top_pair[1]
        merges.append( top_pair )

    return (vocab, merges)

if __name__ == "__main__":
    vocab, merges = train_bpe("./data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
    # save_tokenizer_yaml('tokenizer_tinystories_valid.yaml', vocab, merges)
    # loaded_vocab, loaded_merges = load_tokenizer_yaml('tokenizer_tinystories_valid.yaml')
    # assert set(vocab.keys()) == set(loaded_vocab.keys())
    # assert set(vocab.values()) == set(loaded_vocab.values())
    # assert merges == loaded_merges