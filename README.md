# Huffman Compression

A Python implementation of Huffman coding for file compression and decompression. This project builds Huffman trees based on byte frequency and uses them to compress data with variable-length binary codes.

## Features

- **Huffman Tree Construction** — Builds optimal binary trees from byte frequency data using merge sort
- **Compression** — Compresses files using generated Huffman codes with padding for byte alignment
- **Decompression** — Reconstructs original files using stored tree structure and symbol codes
- **Tree Optimization** — Improves tree layout without changing its structure by swapping nodes based on frequency
- **Multiple Tree Formats** — Supports postorder and general tree serialization/deserialization

## How It Works

### Compression

1. **Frequency Analysis** — Scans input file and counts occurrence of each byte
2. **Tree Building** — Constructs Huffman tree via merge sort of frequency pairs
3. **Code Generation** — Assigns variable-length binary codes (0/1 paths) to each symbol
4. **Encoding** — Replaces each byte with its code, pads to byte boundary
5. **File Writing** — Stores tree structure + metadata + compressed data

### Decompression

1. **Tree Reconstruction** — Reads stored tree nodes and rebuilds the Huffman tree
2. **Code Reversal** — Inverts code dictionary (binary string → byte)
3. **Decoding** — Traverses tree following compressed bits until symbol found, repeats for full data

## Usage

Run interactively:

```bash
python compress.py
```

Choose an option:
- `c` — Compress a file (outputs `filename.huf`)
- `d` — Decompress a `.huf` file (outputs `filename.orig`)
- Any other key — Exit

### Example

```bash
python compress.py
Press c to compress, d to decompress, or other key to exit: c
File to compress: document.txt
Bits per symbol: 4.8
Compressed document.txt in 0.12 seconds.
```

## Core Functions

### Compression

| Function | Purpose |
|----------|---------|
| `build_frequency_dict(text)` | Count byte occurrences in data |
| `build_huffman_tree(freq_dict)` | Construct Huffman tree from frequencies |
| `get_codes(tree)` | Generate symbol → binary code mapping |
| `compress_bytes(text, codes)` | Encode data using codes |
| `compress_file(in_file, out_file)` | Full compression pipeline |

### Decompression

| Function | Purpose |
|----------|---------|
| `generate_tree_general(node_lst, root)` | Rebuild tree from node list (any order) |
| `generate_tree_postorder(node_lst, root)` | Rebuild tree from postorder node list |
| `decompress_bytes(tree, text, size)` | Decode compressed data using tree |
| `decompress_file(in_file, out_file)` | Full decompression pipeline |

### Utilities

| Function | Purpose |
|----------|---------|
| `avg_length(tree, freq_dict)` | Calculate average bits per symbol |
| `number_nodes(tree)` | Label internal nodes in postorder |
| `improve_tree(tree, freq_dict)` | Optimize tree by swapping leaf symbols |

## Project Structure

```
├── compress.py      # Main compression/decompression functions
├── huffman.py       # HuffmanTree class definition
├── utils.py         # Helper functions (node serialization, bit manipulation)
├── tests/           # Unit tests (doctests included in functions)
└── README.md        # This file
```

## Data Format

Compressed file structure (binary):

```
[1 byte: num_nodes] [num_nodes * 4 bytes: tree data] [4 bytes: original size] [compressed data]
```

Each tree node (4 bytes):
- Byte 0: Left type (0=leaf, 1=internal)
- Byte 1: Left data (symbol or node index)
- Byte 2: Right type
- Byte 3: Right data

## Performance

Compression ratio depends on byte frequency distribution. Text files with repeated patterns typically compress 20-40%. Files with high entropy (random data) may expand due to overhead.

Example (from doctest):
```
Symbols: {3: 2 occurrences, 2: 7 occurrences, 9: 1 occurrence}
Bits per symbol: 1.9
Original: 10 bytes (80 bits)
Compressed: ~19 bits
Ratio: ~24%
```

## Testing

Run doctests:

```bash
python -m doctest compress.py -v
python -m doctest huffman.py -v
```

All functions include example usage in docstrings.

## Requirements

- Python 3.9+
- `utils.py` (custom module with helper functions)
- `huffman.py` (HuffmanTree class)

## License

Educational purposes.
