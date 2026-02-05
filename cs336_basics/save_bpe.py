import yaml

def save_tokenizer_yaml(fname, vocab, merges):
    serialized_vocab = {
        k: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
        for k, v in vocab.items()
    }

    serialized_merges = [
        (a.decode("utf-8", errors="replace"), b.decode("utf-8", errors="replace"))
        for a, b in merges
    ]

    with open(fname, "w", encoding="utf-8") as f:
        yaml.dump({ "vocab": serialized_vocab, "merges": serialized_merges },
                  f,
                  allow_unicode=True,
                  sort_keys=False)
        
def load_tokenizer_yaml(fname):
    with open(fname, "r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    vocab = {
        int(k): v.encode("utf-8") if isinstance(v, str) else v
        for k, v in data["vocab"].items()
    }

    merges = {
        (a.encode("utf-8"), b.encode("utf-8"))
        for (a, b) in data["merges"]
    }

    return vocab, merges