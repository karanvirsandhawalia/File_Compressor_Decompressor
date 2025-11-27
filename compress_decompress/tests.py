from __future__ import annotations

from random import shuffle

import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import binary, integers, dictionaries, text

from compress import *

def test_simple_tree():
    """Test a tree with two leaves."""
    original_tree = HuffmanTree(None,
                                HuffmanTree(5, None, None),
                                HuffmanTree(7, None, None))
    number_nodes(original_tree)  # Assign numbers to internal nodes if needed

    # Serialize to bytes
    byte_repr = tree_to_bytes(original_tree)
    # Convert bytes to ReadNodes
    nodes = bytes_to_nodes(byte_repr)
    # Reconstruct the tree
    reconstructed_tree = generate_tree_postorder(nodes, len(nodes) - 1)

    # Verify the reconstructed tree matches the original
    assert reconstructed_tree == original_tree, "Reconstruction failed for simple tree"

    # Print results (for debugging)
    print("Original:", original_tree)
    print("Reconstructed:", reconstructed_tree)
    print("Bytes:", byte_repr)
    print("Nodes:", nodes)

def test_larger_tree():
    """Test a more complex Huffman tree."""
    left = HuffmanTree(None,
                       HuffmanTree(1, None, None),
                       HuffmanTree(2, None, None))
    right = HuffmanTree(None,
                        HuffmanTree(3, None, None),
                        HuffmanTree(4, None, None))
    original_tree = HuffmanTree(None, left, right)
    number_nodes(original_tree)

    byte_repr = tree_to_bytes(original_tree)
    nodes = bytes_to_nodes(byte_repr)
    reconstructed_tree = generate_tree_postorder(nodes, len(nodes) - 1)

    assert reconstructed_tree == original_tree, "Reconstruction failed for larger tree"
    print("Original:", original_tree)
    print("Reconstructed:", reconstructed_tree)
    print("Bytes:", byte_repr)
    print("Nodes:", nodes)



def test_nested_tree():
    """Test a tree with nested internal nodes."""
    left_subtree = HuffmanTree(None,
                               HuffmanTree(10, None, None),
                               HuffmanTree(12, None, None))
    right_subtree = HuffmanTree(15, None, None)
    original_tree = HuffmanTree(None, left_subtree, right_subtree)
    number_nodes(original_tree)  # Assign numbers to internal nodes

    # Serialize and reconstruct
    byte_repr = tree_to_bytes(original_tree)
    nodes = bytes_to_nodes(byte_repr)
    reconstructed_tree = generate_tree_postorder(nodes, len(nodes) - 1)

    print("Original:", original_tree)
    print("Reconstructed:", reconstructed_tree)
    print("Bytes:", byte_repr)
    print("Nodes:", nodes)
    # Verify
    assert reconstructed_tree == original_tree, "Reconstruction failed for nested tree"
