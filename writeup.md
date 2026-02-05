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

