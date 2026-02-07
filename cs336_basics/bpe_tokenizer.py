from typing import Iterator, Iterable
import json
from .common import word2bytes, PAT, split_on_special

def apply_merges(word_bytes, merges, vocab_to_id):
    word_bytes = list(word_bytes)

    while True:
        min_token_id = float("inf")
        best_pair_idx = -1

        for i in range (len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined

        if best_pair_idx == -1:
            break

        word_bytes = (word_bytes[:best_pair_idx] + [merged] + word_bytes[best_pair_idx+2:])

    return tuple(word_bytes)


def merge_then_encode(text, merges, vocab_to_id):
    words = PAT.findall(text)
    tokens = []
    for word in words:
        word_bytes = word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes, merges, vocab_to_id)
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
    return tokens


class BPETokenizer:
    def __init__(self, merges: list[tuple[bytes, bytes]], vocab: dict[int, bytes], special_tokens: list[str] | None =None):
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens if special_tokens else []
        special_token_bytes = [i.encode("utf-8") for i in self.special_tokens]
        self.vocab_to_id = {v: k for k, v in vocab.items()}

        for tok_bytes in special_token_bytes:
            if tok_bytes not in self.vocab_to_id:
                new_id = len(self.vocab)
                self.vocab_to_id[tok_bytes] = new_id
                self.vocab[tok_bytes] = new_id


    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v) 
                     for k, v in vocab_data.items() }
            
        with open(merges_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]

            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]
            
        return cls(merges, vocab, special_tokens)
    

    def encode(self, text: str) -> list[int]:
        chunks = split_on_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append( self.vocab_to_id[ chunk.encode("utf-8") ] )
            else:
                tokens.extend( merge_then_encode(chunk, self.merges, self.vocab_to_id) )
        
        return tokens


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[i] for i in ids]).decode("utf-8", errors="replace")
    