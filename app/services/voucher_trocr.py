"""OCR рукописных полей ваучера через TrOCR (kazars24/trocr-base-handwritten-ru).

Модель transformers VisionEncoderDecoder: процессор ресайзит кроп региона до
384x384, декодер генерирует строку посимвольно. Уверенность — средняя
вероятность выбранного токена на каждом шаге генерации.

Модель и процессор загружаются лениво и кэшируются на процесс. Если
torch/transformers не установлены или веса недоступны, функции возвращают
(None, None) — вызывающий код откатывается на Tesseract.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from PIL import Image

from app.config import settings

if TYPE_CHECKING:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_processor: TrOCRProcessor | None = None
_model: VisionEncoderDecoderModel | None = None
_device: str = ""
_load_failed = False


def _load_model() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel, str] | None:
    """Ленивая загрузка процессора и модели TrOCR (один раз на процесс)."""
    global _processor, _model, _device, _load_failed
    if _processor is not None and _model is not None:
        return _processor, _model, _device
    if _load_failed:
        return None
    with _lock:
        if _processor is not None and _model is not None:
            return _processor, _model, _device
        if _load_failed:
            return None
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            processor = TrOCRProcessor.from_pretrained(settings.trocr_model)
            model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model)
            device = settings.trocr_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model.to(device)
            model.eval()
            _processor, _model, _device = processor, model, device
            logger.info("TrOCR загружен: %s на %s", settings.trocr_model, device)
        except Exception as exc:  # pragma: no cover
            _load_failed = True
            logger.warning(
                "TrOCR недоступен (%s): рукописные поля пойдут в Tesseract", exc
            )
            return None
    return _processor, _model, _device


def _sequence_confidence(scores: tuple | None) -> float | None:
    """Средняя вероятность сгенерированных токенов (шаги generate)."""
    if not scores:
        return None
    import torch

    probs = [torch.softmax(step, dim=-1).max().item() for step in scores]
    return sum(probs) / len(probs) if probs else None


def trocr_image(image: Image.Image) -> tuple[str | None, float | None]:
    """Распознать рукописный текст на кропе. Возвращает (текст, confidence)."""
    if not settings.trocr_enabled:
        return None, None
    loaded = _load_model()
    if loaded is None:
        return None, None
    processor, model, device = loaded

    try:
        import torch

        pixel_values = processor(
            image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            output = model.generate(
                pixel_values,
                max_new_tokens=settings.trocr_max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
            )
        text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0]
        return text.strip() or None, _sequence_confidence(output.scores)
    except Exception as exc:  # pragma: no cover
        logger.warning("TrOCR региона завершился с ошибкой: %s", exc)
        return None, None
