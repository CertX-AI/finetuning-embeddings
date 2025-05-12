import os
import pytest
from unittest.mock import patch, MagicMock

from finetuning_embeddings.utils import check_gpu, validate_file_path


# ----------------------------
# Tests for check_gpu()
# ----------------------------

@patch('torch.cuda.is_available', return_value=True)
@patch('torch.cuda.get_device_name', return_value='Test GPU')
@patch('torch.cuda.memory_allocated', return_value=1024**3 * 1.23)   # 1.23 GB
@patch('torch.cuda.memory_reserved', return_value=1024**3 * 0.45)    # 0.45 GB
@patch('torch.cuda.get_device_properties')
@patch('torch.cuda.get_device_capability', return_value=(8, 0))
def test_check_gpu_available(
    mock_capability,
    mock_properties,
    mock_reserved,
    mock_allocated,
    mock_name,
    mock_available,
    capsys
):
    # Simulate a total of 10 GB on the device
    props = MagicMock()
    props.total_memory = 1024**3 * 10
    mock_properties.return_value = props

    check_gpu()
    out = capsys.readouterr().out.splitlines()

    assert out[0] == "You have a GPU available!"
    assert out[1] == "Device Name: Test GPU"
    assert out[2] == "Memory Allocated: 1.23 GB"
    assert out[3] == "Memory Cached: 0.45 GB"
    assert out[4] == "Total Memory: 10.00 GB"
    assert out[5] == "CUDA Capability: (8, 0)"


@patch('torch.cuda.is_available', return_value=False)
def test_check_gpu_not_available(mock_available, capsys):
    check_gpu()
    assert capsys.readouterr().out.strip() == "You do not have a GPU available."


# ----------------------------
# Tests for validate_file_path()
# ----------------------------

@pytest.mark.parametrize("ext,file_type,message", [
    ("csv",   "CSV",   "CSV file validated for input."),
    ("xls",   "Excel", "Excel file validated for input."),
    ("xlsx",  "Excel", "Excel file validated for input."),
    ("json",  "JSON",  "JSON file validated for input."),
])
def test_validate_input_existing(tmp_path, ext, file_type, message, capsys):
    # create an actual file
    f = tmp_path / f"test.{ext}"
    f.write_text("hello")
    result = validate_file_path(str(f), is_output=False)
    out = capsys.readouterr().out.strip()

    assert result is True
    assert out == message


@pytest.mark.parametrize("ext,file_type,message", [
    ("csv",   "CSV",   "CSV file path is valid for output."),
    ("xls",   "Excel", "Excel file path is valid for output."),
    ("xlsx",  "Excel", "Excel file path is valid for output."),
    ("json",  "JSON",  "JSON file path is valid for output."),
])
def test_validate_output_dir_exists(tmp_path, ext, file_type, message, capsys):
    # directory exists (tmp_path), file may not exist yet
    f = tmp_path / f"out.{ext}"
    result = validate_file_path(str(f), is_output=True)
    out = capsys.readouterr().out.strip()

    assert result is True
    assert out == message


def test_validate_input_nonexistent(tmp_path, capsys):
    f = tmp_path / "nofile.csv"
    result = validate_file_path(str(f), is_output=False)
    out = capsys.readouterr().out

    assert result is False
    assert out == ""


def test_validate_output_nonexistent_dir(tmp_path, capsys, monkeypatch):
    # simulate parent dir does not exist or not writable
    f = tmp_path / "no_dir" / "out.csv"
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    monkeypatch.setattr(os, "access", lambda p, mode: False)

    result = validate_file_path(str(f), is_output=True)
    out = capsys.readouterr().out

    assert result is False
    assert out == ""


def test_validate_invalid_extension(tmp_path):
    f = tmp_path / "bad.txt"
    with pytest.raises(ValueError, match="Unsupported file type. Must be .xls, .xlsx, .csv or .json"):
        validate_file_path(str(f))


@pytest.mark.parametrize("path", ["", "   ", None, 123])
def test_validate_invalid_path_inputs(path):
    assert validate_file_path(path) is False


def test_validate_case_insensitivity(tmp_path, capsys):
    f = tmp_path / "MyFile.CsV"
    f.write_text("data")
    result = validate_file_path(str(f), is_output=False)
    out = capsys.readouterr().out.strip()

    assert result is True
    assert out == "CSV file validated for input."
