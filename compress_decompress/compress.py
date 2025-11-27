"""
Assignment 2 starter code
CSC148, Winter 2025
Instructors: Bogdan Simion, Rutwa Engineer, Marc De Benedetti, Romina Piunno

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for i in text:
        if isinstance(i, int):
            if i in d:
                d[i] += 1
            else:
                d[i] = 1
    return d


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    lst = []
    for key, value in freq_dict.items():
        lst.append([value, key])

    if len(lst) > 1:
        lst = _huffman_sort(lst)
        return _build_huffman_tree_helper(lst)

    value = lst[0][1]
    second = (value + 1) % 256
    return HuffmanTree(None, HuffmanTree(value), HuffmanTree(second))


def _build_huffman_tree_helper(tree_list: list) -> HuffmanTree:
    """
    Helper for build_huffman_tree
    """
    smallest = tree_list.pop(0)
    second_smallest = tree_list.pop(0)

    if isinstance(smallest[1], int):
        s1_tree = HuffmanTree(smallest[1])
    else:
        s1_tree = smallest[1]

    if isinstance(second_smallest[1], int):
        s2_tree = HuffmanTree(second_smallest[1])
    else:
        s2_tree = second_smallest[1]

    total = smallest[0] + second_smallest[0]
    huffman_tree = HuffmanTree(None, s1_tree, s2_tree)
    tree_list.append([total, huffman_tree])

    tree_list = _huffman_sort(tree_list)
    main_tree = tree_list[0][1]

    if len(tree_list) > 1:
        return _build_huffman_tree_helper(tree_list)
    return main_tree


def _huffman_sort(lst: list) -> list:
    """
    Helper for sorting a list
    """
    if len(lst) < 2:
        return lst[:]
    else:
        mid = len(lst) // 2
        left_sorted = _huffman_sort(lst[:mid])
        right_sorted = _huffman_sort(lst[mid:])
        return _merge(left_sorted, right_sorted)


def _merge(lst1: list, lst2: list) -> list:
    """
    Helper for _huffman_sort
    """
    i = 0
    j = 0
    merged = []

    while i < len(lst1) and j < len(lst2):
        if lst1[i][0] <= lst2[j][0]:
            merged.append(lst1[i])
            i += 1
        else:
            merged.append(lst2[j])
            j += 1

    return merged + lst1[i:] + lst2[j:]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.
    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    d = {}
    binary_string = ""
    return _get_codes_helper(tree, d, binary_string)


def _get_codes_helper(tree: HuffmanTree, d: dict, b: str) -> dict[int, str]:
    """
    Helper for get_codes
    """
    if tree.is_leaf():
        if tree.symbol is not None:
            d[tree.symbol] = b
            return d
    if tree.left:
        if tree.symbol is not None:
            d[tree.symbol] = b
        _get_codes_helper(tree.left, d, b + "0")
    if tree.right:
        if tree.symbol is not None:
            d[tree.symbol] = b
        _get_codes_helper(tree.right, d, b + "1")
    return d


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    lst = []
    _number_nodes_helper(tree, lst)
    num = 0
    for i_n in lst:
        if i_n.symbol is None:
            i_n.number = num
            num += 1


def _number_nodes_helper(tree: HuffmanTree, lst: list) -> None:
    """
    Helper for number_nodes
    """
    if tree is None:
        return
    _number_nodes_helper(tree.left, lst)
    _number_nodes_helper(tree.right, lst)
    lst.append(tree)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    no_of_bits = 0
    total = 0
    codes_got = get_codes(tree)
    for key, value in freq_dict.items():
        total += value
    for key, value in codes_got.items():
        code_length = len(value)
        no_of_bits += (code_length * freq_dict[key])
    return no_of_bits / total


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    compressed_b = []
    for byte in text:
        compressed_b.append(codes[byte])
    # Learnt about string join() here:
    # https://www.w3schools.com/python/ref_string_join.asp
    compressed_string = "".join(compressed_b)
    length = len(compressed_string)
    if length % 8 != 0:
        zeros = 8 - (length % 8)
        compressed_string += '0' * zeros

    a = []
    compress_bytes_helper(a, compressed_string)
    return bytes(a)


def compress_bytes_helper(a: list, string: str) -> bytes:
    """
    Helper for compress_bytes
    """
    i = 0
    while i < len(string):
        j = i + 8
        chunk = string[i:j]
        # Learnt about int() here:
        # https://www.geeksforgeeks.org/python-int-function/
        b = int(chunk, 2)
        a.append(b)
        i += 8
    return bytes(a)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    b = []
    if tree.is_leaf():
        return bytes(b)
    if tree.left:
        b.extend(tree_to_bytes(tree.left))
    if tree.right:
        b.extend(tree_to_bytes(tree.right))
    x = tree.left.is_leaf()
    r = tree.right.is_leaf()
    _tree_to_bytes_helper(tree, b, x, r)
    return bytes(b)


def _tree_to_bytes_helper(tree: HuffmanTree, b: list, x: bool, r: bool) -> None:
    """
    Helper for tree_to_bytes
    """
    if x:
        b.append(0)
        b.append(tree.left.symbol)
    else:
        b.append(1)
        b.append(tree.left.number)
    if r:
        b.append(0)
        b.append(tree.right.symbol)
    else:
        b.append(1)
        b.append(tree.right.number)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    if root_index < 0 or root_index >= len(node_lst):
        raise IndexError
    else:
        root = node_lst[root_index]
        return _g_t_g_helper(node_lst, root)


def _g_t_g_helper(nodes: list[ReadNode], root: ReadNode) -> HuffmanTree:
    """
    Helper for generate_tree_general
    """
    is_leaf_left = _g_t_p_helper2(root.l_type)
    is_leaf_right = _g_t_p_helper2(root.r_type)

    if not is_leaf_left:
        left = _g_t_g_helper(nodes, nodes[root.l_data])
    else:
        left = _g_t_g_p_helper2(root, 1)

    if not is_leaf_right:
        right = _g_t_g_helper(nodes, nodes[root.r_data])
    else:
        right = _g_t_g_p_helper2(root, 2)

    return HuffmanTree(None, left, right)


def _g_t_g_p_helper2(root: ReadNode, count: int) -> HuffmanTree:
    """
    Helper for _g_t_g_helper and _g_t_p_helper
    """
    if count == 1:
        return HuffmanTree(root.l_data)
    else:
        return HuffmanTree(root.r_data)


def _g_t_p_helper2(node_type: int) -> bool:
    """
    Helper for _g_t_g_helper and _g_t_p_helper
    """
    if node_type == 0:
        return True
    return False


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    if root_index < 0 or root_index >= len(node_lst):
        raise IndexError
    else:
        index = [root_index]
        return _g_t_p_helper(node_lst, index)


def _g_t_p_helper(nodes: list[ReadNode], index: list[int]) -> HuffmanTree:
    """
    Helper for generate_tree_postorder
    """
    node = nodes[index[0]]
    is_leaf_left = _g_t_p_helper2(node.l_type)
    is_leaf_right = _g_t_p_helper2(node.r_type)

    index[0] -= 1

    if is_leaf_right:
        right = _g_t_g_p_helper2(node, 2)
    else:
        right = _g_t_p_helper(nodes, index)

    if is_leaf_left:
        left = _g_t_g_p_helper2(node, 1)
    else:
        left = _g_t_p_helper(nodes, index)

    return HuffmanTree(None, left, right)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    got_codes = {}
    code_got = get_codes(tree)
    for key, value in code_got.items():
        got_codes[value] = key

    return bytes(_d_b_helper(got_codes, size, text))


def _d_b_helper(d: dict, size: int, text: bytes) -> list:
    """
    Helper for decompress_bytes
    """
    count = 0
    making_str = _d_b_helper2(text)
    total = []
    so_far = []
    for _ in range(size):
        # Got the idea to use a nested loop from here:
        # https://www.geeksforgeeks.org/python-nested-loops/
        while True:
            so_far.append(making_str[count])
            count += 1
            # Learnt about string join() here:
            # https://www.w3schools.com/python/ref_string_join.asp
            make_str = ''.join(so_far)
            if make_str in d:
                total.append(d[make_str])
                so_far.clear()
                # Found exactly how to use the break statement from here:
                # https://www.geeksforgeeks.org/loops-in-python/
                break
    return total


def _d_b_helper2(text: bytes) -> str:
    """
    Helper for _d_b_helper
    """
    i = []
    for byte in text:
        # Learnt to use format syntax from:
        # https://www.theunterminatedstring.com/python-bits-and-bytes/
        j = f"{byte:08b}"
        i.append(j)
    # Learnt about string join() here:
    # https://www.w3schools.com/python/ref_string_join.asp
    return "".join(i)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_postorder(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    lst = []
    lst2 = []
    lst3 = []
    for key, value in freq_dict.items():
        lst.append([value, key])
    lst = _huffman_sort(lst)
    for i in lst:
        lst3.append(i[1])
    _i_t_helper(tree, lst2)
    i = 0
    while i < len(lst2):
        if freq_dict[lst2[i].symbol] != freq_dict[lst3[i]]:
            lst2[i].symbol = lst3[i]
        i += 1

    print(lst2)


def _i_t_helper(tree: HuffmanTree, lst2: list) -> None:
    """ Helper for improve_tree """
    if tree.is_leaf():
        lst2.insert(0, tree)
    else:
        if tree.left:
            _i_t_helper(tree.left, lst2)
        if tree.right:
            _i_t_helper(tree.right, lst2)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
