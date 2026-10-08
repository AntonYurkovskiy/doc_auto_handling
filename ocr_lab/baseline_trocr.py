"""T14: бейзлайн — нынешний TrOCR (kazars24/trocr-base-handwritten-ru) на новых кропах.

Та же модель, что в приложении (:mod:`app.services.voucher_trocr`), но:

- с beam search (``num_return_sequences=5``, псевдовероятности — softmax по top-5 оценкам
  последовательностей), а не top-1, как ``trocr_image``;
- на кропах подполей T10 (:mod:`ocr_lab.cut_crops`), а не на строке целиком;
- в двух вариантах подготовки кропа (см. :func:`prepare_image`): ``plain`` — как сейчас
  в приложении (процессор ресайзит кроп в квадрат 384×384, пропорции не сохраняются),
  ``padded`` — кроп дополняется белым до квадрата перед ресайзом (пропорции сохранены).

Код приложения (``app/services/voucher_trocr.py``) не меняется и не импортируется: здесь
своя (временная) загрузка модели ради доступа к ``generate(num_beams=...)``.

CLI::

    python -m ocr_lab.baseline_trocr infer --split val --variant plain
    python -m ocr_lab.baseline_trocr infer --split test --variant padded --max-minutes 9
    python -m ocr_lab.baseline_trocr timing --n 50 --variant plain

``infer`` дописывает JSONL (:mod:`ocr_lab.predictions`) в
``data/ocr/models/trocr_baseline/{split}_{variant}.jsonl``: уже обработанные сканы (по
``scan_id`` в существующем файле) пропускаются, поэтому прогон можно продолжать после
остановки по ``--max-minutes`` или сбоя. ``timing`` меряет время на ``n`` кропах — для
оценки длительности прогона заранее (п. 4 промпта).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from app.config import settings
from ocr_lab.cut_crops import read_gray
from ocr_lab.paths import MANIFEST, MODELS_DIR, WORK_DIR, configure_model_caches, ensure_dir
from ocr_lab.predictions import (
    SUBFIELDS,
    Candidate,
    ScanPrediction,
    read_predictions,
)

if TYPE_CHECKING:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

#: Индекс кропов T10 (``scan_id, variant, subfield, path, ...``, пути относительно ``WORK_DIR``).
INDEX_CSV = WORK_DIR / "crops_index.csv"
BASELINE_DIR = MODELS_DIR / "trocr_baseline"

#: Варианты подготовки кропа (п. 2 промпта).
VARIANTS: tuple[str, ...] = ("plain", "padded")
SPLITS: tuple[str, ...] = ("train", "val", "test", "all")

NUM_RETURN_SEQUENCES = 5
NUM_BEAMS = 5

#: Похожие по начертанию символы → цифра (минимальный набор из промпта T14).
_DIGIT_FIXUPS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "s": "5"})
_DIGIT_RE = re.compile(r"\d")

_processor: TrOCRProcessor | None = None
_model: VisionEncoderDecoderModel | None = None
_device: str = ""


def text_to_int(text: str) -> int | None:
    """Текст распознавания → целое число.

    Сначала минимальные замены похожих символов (``O``/``o``→``0``, ``l``/``I``→``1``,
    ``S``/``s``→``5``), затем остаются только цифры. Пустой результат — ``None``
    (отсутствие предсказания, а не 0).
    """
    digits = "".join(_DIGIT_RE.findall(text.translate(_DIGIT_FIXUPS)))
    return int(digits) if digits else None


def load_model(
    device: str | None = None,
) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel, str]:
    """Загрузить процессор и модель TrOCR (один раз на процесс, кэш на уровне модуля).

    Модель та же, что в приложении (``settings.trocr_model``), но своя загрузка — нужен
    прямой доступ к ``generate(num_beams=..., num_return_sequences=...)`` для beam search,
    которого нет в ``app.services.voucher_trocr.trocr_image`` (она отдаёт только top-1).
    """
    global _processor, _model, _device
    if _processor is not None and _model is not None and (device is None or device == _device):
        return _processor, _model, _device
    configure_model_caches()
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(settings.trocr_model)
    model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model)
    dev = device or settings.trocr_device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    model.eval()
    _processor, _model, _device = processor, model, dev
    return processor, model, dev


def load_crop_image(path: Path) -> Image.Image:
    """Серый кроп PNG (путь может содержать кириллицу) → PIL RGB."""
    return Image.fromarray(read_gray(path)).convert("RGB")


def pad_to_square(image: Image.Image) -> Image.Image:
    """Дополнить изображение белым до квадрата по центру, сохранив пропорции."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def prepare_image(image: Image.Image, variant: str) -> Image.Image:
    """Подготовка кропа к варианту (п. 2 промпта).

    - ``plain`` — как есть: процессор TrOCR сам ресайзит прямоугольный кроп в квадрат
      384×384 (как сейчас в приложении), пропорции не сохраняются.
    - ``padded`` — кроп дополняется белым до квадрата перед процессором, пропорции целы.
    """
    if variant == "padded":
        return pad_to_square(image)
    if variant == "plain":
        return image
    raise ValueError(f"неизвестный вариант подготовки кропа: {variant}")


def beam_predict(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: str,
    *,
    num_return_sequences: int = NUM_RETURN_SEQUENCES,
    num_beams: int = NUM_BEAMS,
    max_new_tokens: int | None = None,
) -> list[tuple[str, float]]:
    """Beam search на одном кропе: до ``num_return_sequences`` текстов с псевдовероятностями.

    Псевдовероятность — softmax по оценкам возвращённых последовательностей
    (``sequences_scores``, сумма лог-вероятностей токенов с учётом длины), а не по всему
    пространству возможных строк.
    """
    import torch

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens or settings.trocr_max_new_tokens,
            num_beams=max(num_beams, num_return_sequences),
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
        )
    texts = [t.strip() for t in processor.batch_decode(output.sequences, skip_special_tokens=True)]
    scores = output.sequences_scores
    if scores is None:
        probs = [1.0 / len(texts)] * len(texts) if texts else []
    else:
        probs = [float(p) for p in torch.softmax(scores, dim=0)]
    return list(zip(texts, probs, strict=True))


def to_candidates(
    beam: list[tuple[str, float]], *, top_k: int = NUM_RETURN_SEQUENCES
) -> list[Candidate]:
    """Результаты beam search → кандидаты подполя: текст → число.

    Совпадающие после ``text_to_int`` значения складываются (несколько написаний одного
    числа среди top-5), затем сортировка по убыванию вероятности и обрезка до ``top_k``.
    Тексты без цифр (``text_to_int`` вернул ``None``) не дают кандидата. Сумма зажимается
    в ``[0, 1]`` — softmax по float32-тензору иногда даёт 1.0000001 (погрешность округления),
    а строгая проверка формата T13 (``ocr_lab.predictions``) такого не допускает.
    """
    agg: dict[int, float] = {}
    for text, p in beam:
        value = text_to_int(text)
        if value is not None:
            agg[value] = agg.get(value, 0.0) + p
    agg = {v: min(1.0, max(0.0, p)) for v, p in agg.items()}
    ordered = sorted(agg.items(), key=lambda kv: -kv[1])[:top_k]
    return [(v, p) for v, p in ordered]


# --- вход: манифест и индекс кропов --------------------------------------------------------


def split_scan_ids(split: str, manifest: Path = MANIFEST) -> list[str]:
    """``scan_id`` сплита манифеста T02 (``split="all"`` — все сканы)."""
    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        return [
            r["scan_id"]
            for r in csv.DictReader(fh)
            if split == "all" or (r.get("split") or "").strip() == split
        ]


def load_crop_paths(
    index_csv: Path = INDEX_CSV, work_dir: Path = WORK_DIR
) -> dict[tuple[str, str], Path]:
    """``(scan_id, subfield) → путь кропа`` из индекса T10 (только успешно выровненные)."""
    out: dict[tuple[str, str], Path] = {}
    if not index_csv.exists():
        return out
    with index_csv.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("align_ok") or "True") != "True":
                continue
            out[(row["scan_id"], row["subfield"])] = work_dir / row["path"]
    return out


def output_path(split: str, variant: str, out_dir: Path = BASELINE_DIR) -> Path:
    return out_dir / f"{split}_{variant}.jsonl"


# --- прогон ---------------------------------------------------------------------------------


def predict_scan(
    scan_id: str,
    variant: str,
    crop_paths: dict[tuple[str, str], Path],
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: str,
    *,
    num_return_sequences: int = NUM_RETURN_SEQUENCES,
) -> dict[str, list[Candidate]]:
    """Предсказания по всем 17 подполям одного скана (подполя без кропа пропускаются)."""
    fields: dict[str, list[Candidate]] = {}
    for sub in SUBFIELDS:
        path = crop_paths.get((scan_id, sub))
        if path is None or not path.exists():
            continue
        image = prepare_image(load_crop_image(path), variant)
        beam = beam_predict(
            image, processor, model, device, num_return_sequences=num_return_sequences
        )
        cands = to_candidates(beam, top_k=num_return_sequences)
        if cands:
            fields[sub] = cands
    return fields


def infer_split(
    split: str,
    variant: str,
    *,
    manifest: Path = MANIFEST,
    index_csv: Path = INDEX_CSV,
    work_dir: Path = WORK_DIR,
    out_path: Path | None = None,
    device: str | None = None,
    limit: int | None = None,
    max_minutes: float | None = None,
    num_return_sequences: int = NUM_RETURN_SEQUENCES,
) -> Path:
    """Прогнать TrOCR-бейзлайн на сплите, дописывая JSONL (формат T13).

    Уже обработанные сканы (есть в существующем ``out_path``) пропускаются — прогон можно
    продолжать после остановки по ``--max-minutes`` или сбоя (п. 4 промпта: на CPU дольше
    9 минут — частями с дозаписью).
    """
    out = out_path or output_path(split, variant)
    ensure_dir(out.parent)
    done = {p.scan_id for p in read_predictions(out)} if out.exists() else set()
    scan_ids = [sid for sid in split_scan_ids(split, manifest) if sid not in done]
    if limit is not None:
        scan_ids = scan_ids[:limit]
    if not scan_ids:
        return out

    crop_paths = load_crop_paths(index_csv, work_dir)
    processor, model, dev = load_model(device)
    source = f"trocr_baseline_{variant}"

    t0 = time.perf_counter()
    n_done = 0
    with out.open("a", encoding="utf-8") as fh:
        for scan_id in scan_ids:
            if max_minutes is not None and (time.perf_counter() - t0) / 60 >= max_minutes:
                print(f"остановка по --max-minutes: обработано {n_done} из {len(scan_ids)}")
                break
            fields = predict_scan(
                scan_id, variant, crop_paths, processor, model, dev,
                num_return_sequences=num_return_sequences,
            )
            pred = ScanPrediction(scan_id=scan_id, source=source, fields=fields)
            fh.write(json.dumps(pred.to_json(), ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            if n_done % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {n_done}/{len(scan_ids)} сканов, {elapsed:.1f} с", flush=True)
    return out


def time_estimate(
    n: int,
    variant: str,
    *,
    device: str | None = None,
    index_csv: Path = INDEX_CSV,
    work_dir: Path = WORK_DIR,
    num_return_sequences: int = NUM_RETURN_SEQUENCES,
) -> dict[str, float | str | int]:
    """Время инференса на ``n`` кропах индекса (детерминированный срез) — оценка длительности."""
    crop_paths = load_crop_paths(index_csv, work_dir)
    items = list(crop_paths.items())[:n]
    processor, model, dev = load_model(device)
    t0 = time.perf_counter()
    for (_scan_id, _sub), path in items:
        image = prepare_image(load_crop_image(path), variant)
        beam_predict(image, processor, model, dev, num_return_sequences=num_return_sequences)
    elapsed = time.perf_counter() - t0
    per_item = elapsed / len(items) if items else 0.0
    return {"n": len(items), "device": dev, "seconds": elapsed, "seconds_per_crop": per_item}


# --- CLI ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr_lab.baseline_trocr", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_infer = sub.add_parser("infer", help="прогон TrOCR-бейзлайна на сплите")
    p_infer.add_argument("--split", choices=SPLITS, default="val")
    p_infer.add_argument("--variant", choices=VARIANTS, required=True)
    p_infer.add_argument("--device", default=None, help="cuda / cpu, по умолчанию — авто")
    p_infer.add_argument("--limit", type=int, default=None)
    p_infer.add_argument("--max-minutes", type=float, default=None)
    p_infer.add_argument("--out", type=Path, default=None)

    p_time = sub.add_parser("timing", help="оценить время на n кропах")
    p_time.add_argument("--n", type=int, default=50)
    p_time.add_argument("--variant", choices=VARIANTS, default="plain")
    p_time.add_argument("--device", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "infer":
        out = infer_split(
            args.split, args.variant, device=args.device, limit=args.limit,
            max_minutes=args.max_minutes, out_path=args.out,
        )
        print(f"записано: {out}")
    else:
        stats = time_estimate(args.n, args.variant, device=args.device)
        print(
            f"{stats['n']} кропов на {stats['device']}: {stats['seconds']:.1f} с "
            f"({stats['seconds_per_crop']:.3f} с/кроп)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
