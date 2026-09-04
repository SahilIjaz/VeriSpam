"""Dataset loading utilities: turns an uploaded file into a DataFrame + file metadata."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


@dataclass
class LoadResult:
    dataframe: pd.DataFrame | None
    filename: str
    size_bytes: int
    extension: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.dataframe is not None

    @property
    def size_kb(self) -> float:
        return self.size_bytes / 1024

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def load_uploaded_file(uploaded_file) -> LoadResult:
    """Read a Streamlit UploadedFile (CSV or Excel) into a pandas DataFrame.

    Never raises: read/parse failures are captured on LoadResult.error so the
    caller can render them as validation feedback instead of crashing the app.
    """
    return _load_bytes(uploaded_file.name, uploaded_file.getvalue())


def load_local_file(path: Path) -> LoadResult:
    """Read a file already on disk (e.g. the bundled sample dataset) the same
    way an upload would be read, so both paths share one parsing code path."""
    path = Path(path)
    return _load_bytes(path.name, path.read_bytes())


def _load_bytes(filename: str, raw: bytes) -> LoadResult:
    size_bytes = len(raw)
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        return LoadResult(
            dataframe=None,
            filename=filename,
            size_bytes=size_bytes,
            extension=extension,
            error=f"Unsupported file format '{extension or 'unknown'}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    if size_bytes == 0:
        return LoadResult(
            dataframe=None,
            filename=filename,
            size_bytes=size_bytes,
            extension=extension,
            error="The uploaded file is empty (0 bytes).",
        )

    try:
        if extension == ".csv":
            df = pd.read_csv(BytesIO(raw), encoding_errors="replace")
        else:
            df = pd.read_excel(BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - surface any parser failure as validation feedback
        return LoadResult(
            dataframe=None,
            filename=filename,
            size_bytes=size_bytes,
            extension=extension,
            error=f"Could not parse file as {extension}: {exc}",
        )

    return LoadResult(
        dataframe=df,
        filename=filename,
        size_bytes=size_bytes,
        extension=extension,
    )
