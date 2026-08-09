"""Безопасное хранение и выдача исходных файлов ваучеров."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

MAX_NAME_LENGTH = 80
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

# Расширения, которые оператор может просмотреть в карточке ваучера.
ALLOWED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@dataclass(frozen=True)
class StoredUpload:
    """Результат сохранения загруженного файла."""

    path: Path
    stored_name: str
    original_filename: str
    content_type: str | None
    sha256: str


def safe_basename(original_filename: str | None) -> str:
    """Возвращает безопасное имя файла без путей и спецсимволов."""
    raw = (original_filename or "").replace("\\", "/")
    raw = raw.rsplit("/", 1)[-1].strip()
    suffix = Path(raw).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        suffix = ".bin"
    stem = _UNSAFE_CHARS.sub("_", Path(raw).stem).strip("._-")
    if not stem:
        stem = "voucher"
    return f"{stem[:MAX_NAME_LENGTH]}{suffix}"


def store_upload(
    stream: BinaryIO,
    original_filename: str | None,
    content_type: str | None,
    folder: Path,
) -> StoredUpload:
    """Сохраняет поток в folder под безопасным именем с префиксом sha256."""
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    tmp_fd, tmp_name = tempfile.mkstemp(dir=folder, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        sha256 = digest.hexdigest()
        stored_name = f"{sha256[:16]}_{safe_basename(original_filename)}"
        dest = folder / stored_name
        shutil.move(str(tmp_path), dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return StoredUpload(
        path=dest,
        stored_name=stored_name,
        original_filename=safe_basename(original_filename),
        content_type=content_type or None,
        sha256=sha256,
    )


def resolve_stored_file(name: str, folder: Path) -> Path:
    """Возвращает существующий файл внутри folder или бросает ValueError.

    Защищает от выхода за пределы каталога (`..`, абсолютные пути, симлинки).
    """
    if not name:
        raise ValueError("Пустое имя файла")
    base = folder.resolve()
    candidate = (base / name).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"Путь вне каталога хранения: {name}")
    if not candidate.is_file():
        raise ValueError(f"Файл не найден: {name}")
    return candidate


def media_type_for(path: Path, content_type: str | None = None) -> str:
    """Медиа-тип по расширению файла (content_type из загрузки как подсказка)."""
    suffix = path.suffix.lower()
    if suffix in MEDIA_TYPES:
        return MEDIA_TYPES[suffix]
    if content_type is not None and content_type in MEDIA_TYPES.values():
        return content_type
    return "application/octet-stream"


def preview_kind(path: Path) -> str:
    """`pdf`, `image` или `none` — как показывать файл в карточке."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in ALLOWED_SUFFIXES:
        return "image"
    return "none"
