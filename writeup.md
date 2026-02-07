(a) What Unicode character does chr(0) return?

'\x00'

(b) How does this character’s string representation (__repr__()) differ from its printed representation?

string representation: '\x00'

printed representation: 

(c) What happens when this character occurs in text? 

it is included as '\x00' but not printed in its string representation.

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

UTF-16 and UTF-32 contain a lot of zeros at the start of most common character representations which will make combination of zeros with other bytes in tokenizer byte-pairs, wasting vocablory space.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

bytestring: [128], 0b1000 0000 (invalid byte format in utf-8).
This function is incorrect because encode('utf-8') expects bytes to be in a specific format and user might provide a valid bytestring but not according to format of utf-8 bytes. 

(c) Give a two byte sequence that does not decode to any Unicode character(s).

[128, 128]


(transformer_accounting):
a) 50,257*1600 (embeddings) + 48 * (4*1600*1600 + 3*6,400*1600 + 2*1600) (transformers) + (1*1600 + 50,257*1600) final_layers
total parameters: 2127057600
memory: 68065843200 bytes = 63.4 GB

b) Matmuls:
    Attention: 48 * 4 * 1600 * 1600 * 1024 = 503316480000 FLOPs = 504 GFLOPs
    SwiGLU: 48 * 3 * 1600 * 6400 * 1024 = 1,509,949,440,000 FLOPs = 1.6 TFLOPs
    Final Linear: 2 * 50,257 * 1600 * 1024 = 164682137600 = 165 GFLOPs

c) FFN

d) GPT-2 small (12 layers, 768 d_model, 12 heads):
    Attention: 12 * 4 * 768 * 768 * 1024 = 29 GFLOPs
    SwiGLU: 12 * 3 * 768 * 3072 * 1024 = 87 GFLOPs
    Final Linear: 2 * 50,257 * 768 * 1024 =  = 79.1 GFLOPs

    GPT-2 medium (24 layers, 1024 d_model, 16 heads):
    Attention: 24 * 4 * 1024 * 1024 * 1024 = 103.1 GFLOPs
    SwiGLU: 24 * 3 * 1024 * 4096 * 1024 = 310 GFLOPs
    Final Linear: 2 * 50,257 * 1024 * 1024 = 106 GFLOPs

    GPT-2 large (36 layers, 1280 d_model, 20 heads):
    Attention: 36 * 4 * 1280 * 1280 * 1024 = 242 GFLOPs
    SwiGLU: 36 * 3 * 1280 * 5120 * 1024 = 725 GFLOPs
    Final Linear: 2 * 50,257 * 1280 * 1024 = 132 GFLOPs

FFN takes more FLOPs as model size increases.

d) Matmuls:
    Attention: 48 * 4 * 1600 * 1600 * 16,384 = 8.06 TFLOPs
    SwiGLU: 48 * 3 * 1600 * 6400 * 16,384 = 24.16 TFLOPs
    Final Linear: 2 * 50,257 * 1600 * 16,384 = 2.7 TFLOPs

Total FLOPs get (16,384/1024)x. Contribution of model components remain same. 