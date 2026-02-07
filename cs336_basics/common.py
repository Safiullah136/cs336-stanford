import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def word2bytes(word: str):
    return tuple( bytes([b]) for b in list(word.encode("utf-8")) )

def split_on_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    
    delimiter =  "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: delimiter = f"({delimiter})"
    
    chunks = re.split(delimiter, text)
    return [c for c in chunks if c]