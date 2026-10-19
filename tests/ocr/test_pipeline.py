"""Тесты T25 (`app/ocr/pipeline.py`): контракт `recognize_voucher`, без реальных моделей.

Модели цифр/номера подменяет `FakeRecognizer` (как в `tests/ocr/test_eval_pipeline.py`),
бланк и «скан» — синтетика `tests/ocr/synth.py`.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ocr import layouts as boxes
from app.ocr import pipeline as pl
from app.ocr.decoder import DecoderModel
from app.ocr.number_decoder import NumberPriors
from app.ocr.pair_decoder import PairPriors
from app.ocr.variants import load_layouts as load_variants
from app.ocr.variants import save_layout as save_variant
from tests.ocr.synth import PAGE_H, PAGE_W, make_form

VARIANT = "syn_v1"

TRUTH: dict[str, tuple[date, int, int]] = {
    "left_base": (date(2026, 5, 4), 8, 0),
    "started_work": (date(2026, 5, 4), 8, 40),
    "finished_work": (date(2026, 5, 4), 10, 0),
    "arrived_base": (date(2026, 5, 4), 10, 30),
}


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())


def _box_layout() -> boxes.Layout:
    """Тот же синтетический макет, что и в `test_eval_pipeline.py` (проверенная геометрия)."""
    items = [boxes.Box(boxes.VOUCHER_NUMBER, 900, 300, 1150, 400, "number")]
    for i, row in enumerate(boxes.DATE_ROWS):
        for j, part in enumerate(boxes.DATE_PARTS):
            x0, y0 = 300 + 200 * j, 1200 + 100 * i
            items.append(boxes.Box(f"{row}.{part}", x0, y0, x0 + 150, y0 + 90, "two_digit"))
    return boxes.Layout(VARIANT, "s", (PAGE_W, PAGE_H), tuple(items))


def _heads_output(number: int) -> np.ndarray:
    """Логиты трёх голов (`heads`), уверенно дающие `number`."""
    out = np.zeros((3, 11))
    digits = str(number).rjust(3)
    for i, ch in enumerate(digits):
        out[i, 10 if ch == " " else int(ch)] = 12.0
    return out


def _values_for_truth() -> dict[str, int]:
    out: dict[str, int] = {}
    for row, (d, hour, minute) in TRUTH.items():
        out[f"{row}.day"] = d.day
        out[f"{row}.month"] = d.month
        out[f"{row}.hour"] = hour
        out[f"{row}.minute"] = minute
    return out


class FakeRecognizer:
    """Распределения, уверенно дающие заданные значения, вместо `app.ocr.runtime`."""

    def __init__(self, values: Mapping[str, int] | None = None, number: int = 12) -> None:
        self.values = dict(values or {})
        self.number_value = number
        self.calls = 0

    def digits(self, crops: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, float]]]:
        self.calls += 1
        assert all(c.ndim == 2 for c in crops.values())
        return {n: [(self.values.get(n, 1), 0.98), (2, 0.02)] for n in crops}

    def number(self, crop: np.ndarray) -> tuple[np.ndarray, str, float]:
        return _heads_output(self.number_value), "heads", 1.0


@pytest.fixture()
def resources(tmp_path: Path) -> pl.PipelineResources:
    layouts_dir = tmp_path / "layouts"
    image, mask = make_form(layout_seed=0)
    save_variant(layouts_dir / VARIANT, image, mask, {"tug_code": "s"})
    return pl.PipelineResources(
        variants=load_variants(layouts_dir),
        boxes={VARIANT: _box_layout()},
        recognizer=FakeRecognizer(_values_for_truth(), number=12),
        decoder_model=DecoderModel.default(),
        number_priors=NumberPriors.default(),
        pair_priors=PairPriors.default(),
    )


def _scan(tmp_path: Path, name: str = "scan.png", fill_seed: int = 3) -> Path:
    scan, _ = make_form(layout_seed=0, fill_seed=fill_seed)
    path = tmp_path / name
    _write_png(path, scan)
    return path


def _context(**overrides: object) -> pl.RecognitionContext:
    base: dict[str, object] = {"year": 2026, "app_dt": datetime(2026, 5, 4, 6, 0)}
    base.update(overrides)
    return pl.RecognitionContext(**base)  # type: ignore[arg-type]


# --- контракт: обычный ход ------------------------------------------------------------------


def test_recognize_voucher_ok_contract(tmp_path: Path, resources: pl.PipelineResources) -> None:
    path = _scan(tmp_path, "2026_12s.png")
    result = pl.recognize_voucher(
        path, tug_hint="s", context=_context(), resources=resources, crops_dir=tmp_path / "crops",
    )
    assert result.status == pl.STATUS_OK and result.ok and result.reason == ""
    assert result.variant == VARIANT
    assert result.alignment is not None and result.alignment.ok

    # По каждой из 4 строк — top-3 вариантов записи с вероятностью.
    assert set(result.fields) == set(boxes.DATE_ROWS)
    for options in result.fields.values():
        assert 1 <= len(options) <= 3
        assert all(0.0 <= opt.p <= 1.0 for opt in options)

    # Номер — top-3 (здесь один уверенный кандидат).
    assert 1 <= len(result.voucher_number) <= 3
    assert result.voucher_number[0][0] == 12

    # Top-N записей целиком, со `score`/`p`, в формате строки бланка.
    assert result.records
    assert result.records[0].forms()["left_base"] == TRUTH["left_base"]
    assert 0.0 <= result.confidence <= 1.0
    assert result.margin >= 0.0

    # Кропы всех 17 подполей сохранены безопасными именами.
    assert set(result.crop_paths) == set(boxes.SUBFIELD_NAMES)
    for crop_path in result.crop_paths.values():
        assert crop_path.is_file()

    assert {"load", "align", "crops", "digits", "number", "decode", "total"} <= set(result.timing)
    assert result.timing["total"] < 5.0  # цель — не больше 2 с на реальном CPU


def test_recognize_voucher_auto_accept_follows_threshold(
    tmp_path: Path, resources: pl.PipelineResources
) -> None:
    path = _scan(tmp_path, "2026_12s.png")
    strict = pl.recognize_voucher(
        path, context=_context(), resources=resources, crops_dir=tmp_path / "c1",
        auto_accept_threshold=2.0,
    )
    assert strict.auto_accept is False  # ни одна вероятность не превышает 2.0
    loose = pl.recognize_voucher(
        path, context=_context(), resources=resources, crops_dir=tmp_path / "c2",
        auto_accept_threshold=0.0,
    )
    assert loose.auto_accept is True


def test_recognize_voucher_skips_crop_saving(
    tmp_path: Path, resources: pl.PipelineResources
) -> None:
    path = _scan(tmp_path)
    result = pl.recognize_voucher(
        path, context=_context(), resources=resources, save_crops=False,
    )
    assert result.status == pl.STATUS_OK
    assert result.crop_paths == {}


# --- деградация без падений ----------------------------------------------------------------


def test_recognize_voucher_no_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pl.reset_resources()
    monkeypatch.setenv("OCR_WORK_DIR", str(tmp_path / "empty"))
    path = _scan(tmp_path)
    result = pl.recognize_voucher(path, context=_context())
    assert result.status == pl.STATUS_NO_MODELS and result.reason
    assert result.records == () and result.crop_paths == {} and result.confidence == 0.0
    pl.reset_resources()


def test_recognize_voucher_bad_file_is_error_not_exception(
    resources: pl.PipelineResources,
) -> None:
    result = pl.recognize_voucher(
        Path("несуществующий_файл_ваучера.pdf"), context=_context(), resources=resources,
    )
    assert result.status == pl.STATUS_ERROR and result.reason
    assert result.records == ()


def test_recognize_voucher_corrupted_file_is_error(
    tmp_path: Path, resources: pl.PipelineResources
) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a real pdf")
    result = pl.recognize_voucher(bad, context=_context(), resources=resources)
    assert result.status == pl.STATUS_ERROR and result.reason


def test_recognize_voucher_align_failed(tmp_path: Path, resources: pl.PipelineResources) -> None:
    noise = np.random.default_rng(0).integers(0, 256, (PAGE_H, PAGE_W), dtype=np.uint8)
    path = tmp_path / "noise.png"
    _write_png(path, noise)
    result = pl.recognize_voucher(path, context=_context(), resources=resources)
    assert result.status == pl.STATUS_ALIGN_FAILED and result.reason
    assert result.records == () and result.crop_paths == {}


# --- пара (T20) -------------------------------------------------------------------------------


def test_recognize_voucher_partner_confirmed(
    tmp_path: Path, resources: pl.PipelineResources
) -> None:
    path = _scan(tmp_path, "2026_12s.png")
    ctx = _context(partner=pl.ConfirmedPartner(forms=dict(TRUTH)))
    result = pl.recognize_voucher(path, context=ctx, resources=resources, crops_dir=tmp_path / "c")
    assert result.status == pl.STATUS_OK
    assert "pair_same" in result.flags  # партнёр подтвердил ровно те же времена


def test_recognize_voucher_partner_pending(tmp_path: Path, resources: pl.PipelineResources) -> None:
    a_path = _scan(tmp_path, "a.png", fill_seed=3)
    b_path = _scan(tmp_path, "b.png", fill_seed=5)
    partner = pl.PendingPartner(path=b_path, context=_context(), tug_hint="s")
    result = pl.recognize_voucher(
        a_path, context=_context(partner=partner), resources=resources, crops_dir=tmp_path / "c",
    )
    assert result.status == pl.STATUS_OK
    assert "pair_same" in result.flags or "pair_differs" in result.flags


def test_recognize_voucher_partner_unavailable_falls_back_to_core(
    tmp_path: Path, resources: pl.PipelineResources
) -> None:
    a_path = _scan(tmp_path, "a.png")
    missing = pl.PendingPartner(path=tmp_path / "missing.png", context=_context())
    result = pl.recognize_voucher(
        a_path, context=_context(partner=missing), resources=resources, crops_dir=tmp_path / "c",
    )
    assert result.status == pl.STATUS_OK
    assert pl.FLAG_PARTNER_UNAVAILABLE in result.flags
    assert result.records  # запись ядра осталась, пайплайн не упал


# --- плашка и автоприём — по ядру, без пары (решение T21, T29 В7) ---------------------------


def test_recognize_voucher_confidence_is_core_confidence_even_with_partner(
    tmp_path: Path, resources: pl.PipelineResources
) -> None:
    path = _scan(tmp_path, "2026_12s.png")
    alone = pl.recognize_voucher(
        path, context=_context(), resources=resources, crops_dir=tmp_path / "c1",
    )
    paired = pl.recognize_voucher(
        path, context=_context(partner=pl.ConfirmedPartner(forms=dict(TRUTH))),
        resources=resources, crops_dir=tmp_path / "c2",
    )
    assert alone.pair_confidence is None
    assert paired.pair_confidence is not None
    # пара не влияет ни на уверенность, ни на автоприём
    assert paired.confidence == pytest.approx(alone.confidence)
    assert paired.margin == pytest.approx(alone.margin)
    for threshold in (0.0, 0.5, 2.0):
        a = pl.recognize_voucher(
            path, context=_context(), resources=resources, save_crops=False,
            auto_accept_threshold=threshold,
        )
        b = pl.recognize_voucher(
            path, context=_context(partner=pl.ConfirmedPartner(forms=dict(TRUTH))),
            resources=resources, save_crops=False, auto_accept_threshold=threshold,
        )
        assert a.auto_accept == b.auto_accept


def test_recognize_voucher_keeps_aligned_page_only_on_request(
    tmp_path: Path, resources: pl.PipelineResources
) -> None:
    path = _scan(tmp_path, "2026_12s.png")
    default = pl.recognize_voucher(path, context=_context(), resources=resources, save_crops=False)
    kept = pl.recognize_voucher(
        path, context=_context(), resources=resources, save_crops=False, keep_aligned=True,
    )
    assert default.aligned_page is None
    assert kept.aligned_page is not None
    assert kept.aligned_page.ndim == 2 and kept.aligned_page.dtype == np.uint8
    assert kept.aligned_page.shape == (PAGE_H, PAGE_W)
    assert "aligned_page" not in repr(kept)  # большой массив не попадает в журналы


# --- ресурсы: «нет моделей» запоминается на время (D4) ---------------------------------------


def test_unavailable_resources_are_remembered_for_a_while(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pl.reset_resources()
    builds: list[int] = []

    def failing_build(*args: object, **kwargs: object) -> pl.PipelineResources:
        builds.append(1)
        raise FileNotFoundError("нет эталонов")

    monkeypatch.setattr(pl, "_build_resources", failing_build)
    kwargs = {"models_dir": tmp_path / "m", "layouts_dir": tmp_path / "l", "model_version": "v1"}

    for _ in range(3):
        with pytest.raises(FileNotFoundError, match="нет эталонов"):
            pl.get_resources(**kwargs)
    assert len(builds) == 1  # не перегружаем эталоны на каждый ваучер

    pl.reset_resources()
    with pytest.raises(FileNotFoundError):
        pl.get_resources(**kwargs)
    assert len(builds) == 2  # reset_resources забывает и сбой

    monkeypatch.setattr(pl, "UNAVAILABLE_TTL_S", 0.0)
    with pytest.raises(FileNotFoundError):
        pl.get_resources(**kwargs)
    assert len(builds) == 3  # срок хранения сбоя истёк
    pl.reset_resources()


def test_unavailable_result_is_no_models_status() -> None:
    result = pl.unavailable_result("нет моделей")

    assert result.status == pl.STATUS_NO_MODELS and not result.ok
    assert result.reason == "нет моделей" and result.records == ()
    assert result.aligned_page is None and result.pair_confidence is None


# --- критерий приёмки: без torch/transformers -------------------------------------------------


def test_app_ocr_package_has_no_torch_imports() -> None:
    """`app/ocr/` не импортирует torch/transformers (критерий приёмки T25)."""
    pkg_dir = Path(pl.__file__).resolve().parent
    checked = 0
    for path in pkg_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        checked += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert "torch" not in names, f"{path}: импортирует torch"
            assert "transformers" not in names, f"{path}: импортирует transformers"
    assert checked > 5  # сверяем, что обошли реальные модули пакета, а не пустой каталог
