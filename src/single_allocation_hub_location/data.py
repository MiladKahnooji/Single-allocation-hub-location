"""Load finite, nonnegative square matrices from CSV or XLSX files."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from openpyxl import load_workbook

Matrix: TypeAlias = tuple[tuple[float, ...], ...]


class MatrixValidationError(ValueError):
    """Raised when a matrix input is missing, unsupported, or malformed."""


@dataclass(frozen=True)
class MatrixPair:
    """Matching flow and distance matrices loaded from validated input files."""

    flow: Matrix
    distance: Matrix

    @property
    def size(self) -> int:
        """Return the shared number of origins and destinations."""
        return len(self.flow)


def load_matrix(path: str | Path) -> Matrix:
    """Load one square matrix from a CSV or XLSX file.

    Values must be numeric, finite, and nonnegative. Blank cells and headers are
    intentionally rejected so input errors are surfaced at the boundary.
    """
    input_path = Path(path)
    if not input_path.is_file():
        raise MatrixValidationError(f"Matrix file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv(input_path)
    elif suffix == ".xlsx":
        rows = _read_xlsx(input_path)
    else:
        raise MatrixValidationError(
            f"Unsupported matrix format for {input_path}: expected .csv or .xlsx"
        )
    return _validate_rows(_remove_column_index_header(rows), input_path)


def load_matrix_pair(flow_path: str | Path, distance_path: str | Path) -> MatrixPair:
    """Load matching flow and distance matrices through the same validation path."""
    flow = load_matrix(flow_path)
    distance = load_matrix(distance_path)
    if len(flow) != len(distance):
        raise MatrixValidationError(
            "Flow and distance matrices must have matching dimensions: "
            f"{len(flow)}x{len(flow)} != {len(distance)}x{len(distance)}"
        )
    return MatrixPair(flow=flow, distance=distance)


def _read_csv(path: Path) -> list[list[object]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as matrix_file:
            return [row for row in csv.reader(matrix_file)]
    except UnicodeDecodeError as error:
        raise MatrixValidationError(f"Could not decode CSV matrix {path}: {error}") from error


def _read_xlsx(path: Path) -> list[list[object]]:
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as error:
        raise MatrixValidationError(f"Could not read XLSX matrix {path}: {error}") from error
    try:
        worksheet = workbook.active
        return [list(row) for row in worksheet.iter_rows(values_only=True)]
    finally:
        workbook.close()


def _validate_rows(rows: list[list[object]], path: Path) -> Matrix:
    if not rows:
        raise MatrixValidationError(f"Matrix {path} is empty")
    width = len(rows[0])
    if width == 0:
        raise MatrixValidationError(f"Matrix {path} is empty")
    if len(rows) != width:
        raise MatrixValidationError(
            f"Matrix {path} must be square; found {len(rows)} rows and {width} columns"
        )

    validated_rows: list[tuple[float, ...]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != width:
            raise MatrixValidationError(
                f"Matrix {path} has inconsistent row length at row {row_number}: "
                f"expected {width}, found {len(row)}"
            )
        validated_rows.append(
            tuple(_validate_value(value, path, row_number, column_number) for column_number, value in enumerate(row, start=1))
        )
    return tuple(validated_rows)


def _remove_column_index_header(rows: list[list[object]]) -> list[list[object]]:
    """Remove the simple 0..n-1 XLSX-style column header used by supplied data."""
    if not rows or len(rows) != len(rows[0]) + 1:
        return rows
    try:
        header = tuple(int(value) for value in rows[0])
    except (TypeError, ValueError):
        return rows
    if header == tuple(range(len(rows[0]))):
        return rows[1:]
    return rows


def _validate_value(value: object, path: Path, row: int, column: int) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise MatrixValidationError(f"Matrix {path} has a blank value at row {row}, column {column}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise MatrixValidationError(
            f"Matrix {path} has a nonnumeric value at row {row}, column {column}: {value!r}"
        ) from error
    if not math.isfinite(number):
        raise MatrixValidationError(
            f"Matrix {path} has a non-finite value at row {row}, column {column}"
        )
    if number < 0:
        raise MatrixValidationError(
            f"Matrix {path} has a negative value at row {row}, column {column}"
        )
    return number
