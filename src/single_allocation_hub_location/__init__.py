"""Validated input loading for the single-allocation hub location project."""

from .data import MatrixPair, MatrixValidationError, load_matrix, load_matrix_pair
from .scenarios import FlowScenarios, generate_flow_scenarios

__all__ = [
    "FlowScenarios",
    "MatrixPair",
    "MatrixValidationError",
    "generate_flow_scenarios",
    "load_matrix",
    "load_matrix_pair",
]
