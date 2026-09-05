from pathlib import Path

import pytest
from openpyxl import Workbook

from single_allocation_hub_location import (
    MatrixValidationError,
    load_matrix,
    load_matrix_pair,
)


FIXTURES = Path(__file__).parent / "fixtures"


def write_xlsx(path: Path, rows: list[list[object]]) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    for row in rows:
        worksheet.append(row)
    workbook.save(path)


def test_load_matrix_pair_reads_csv_and_xlsx(tmp_path: Path) -> None:
    distance_path = tmp_path / "distance.xlsx"
    write_xlsx(distance_path, [[0, 4], [4, 0]])

    matrices = load_matrix_pair(FIXTURES / "flow.csv", distance_path)

    assert matrices.size == 2
    assert matrices.flow == ((0.0, 2.0), (3.0, 0.0))
    assert matrices.distance == ((0.0, 4.0), (4.0, 0.0))


def test_load_matrix_removes_a_simple_xlsx_column_index_header(tmp_path: Path) -> None:
    path = tmp_path / "flow.xlsx"
    write_xlsx(path, [[0, 1], [0, 2], [3, 0]])

    assert load_matrix(path) == ((0.0, 2.0), (3.0, 0.0))


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[0, 1, 2], [1, 0, 3]], "must be square"),
        ([[0, -1], [1, 0]], "negative value"),
        ([[0, "not-a-number"], [1, 0]], "nonnumeric value"),
    ],
)
def test_load_matrix_reports_invalid_xlsx_values(
    tmp_path: Path, rows: list[list[object]], message: str
) -> None:
    path = tmp_path / "invalid.xlsx"
    write_xlsx(path, rows)

    with pytest.raises(MatrixValidationError, match=message):
        load_matrix(path)


def test_load_matrix_reports_non_finite_csv_value(tmp_path: Path) -> None:
    path = tmp_path / "non-finite.csv"
    path.write_text("0,inf\n1,0\n", encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="non-finite value"):
        load_matrix(path)


def test_load_matrix_reports_inconsistent_csv_rows(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent.csv"
    path.write_text("0,1\n1\n", encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="inconsistent row length"):
        load_matrix(path)


def test_load_matrix_rejects_rectangular_csv_that_looks_like_an_index_header(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rectangular.csv"
    path.write_text("0,1\n2,3\n4,5\n", encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="must be square"):
        load_matrix(path)


def test_load_matrix_pair_reports_mismatched_dimensions(tmp_path: Path) -> None:
    distance_path = tmp_path / "distance.csv"
    distance_path.write_text("0,1,2\n1,0,3\n2,3,0\n", encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="matching dimensions"):
        load_matrix_pair(FIXTURES / "flow.csv", distance_path)


def test_load_matrix_reports_missing_and_unsupported_files(tmp_path: Path) -> None:
    with pytest.raises(MatrixValidationError, match="does not exist"):
        load_matrix(tmp_path / "missing.csv")

    path = tmp_path / "matrix.txt"
    path.write_text("0,1\n1,0\n", encoding="utf-8")
    with pytest.raises(MatrixValidationError, match="Unsupported matrix format"):
        load_matrix(path)
