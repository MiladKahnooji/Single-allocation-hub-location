"""Validated input loading for the single-allocation hub location project."""

from .data import MatrixPair, MatrixValidationError, load_matrix, load_matrix_pair

__all__ = ["MatrixPair", "MatrixValidationError", "load_matrix", "load_matrix_pair"]
