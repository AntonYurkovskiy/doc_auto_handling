"""Синтетический бланк ваучера и искажения скана для тестов выравнивания.

Только синтетика: рамки, линии таблицы, печатный текст (``cv2.putText``) и «рукописные»
каракули, которые меняются от бланка к бланку. Раскладку задаёт ``layout_seed``,
рукопись — ``fill_seed``.
"""

from __future__ import annotations

import string

import cv2
import numpy as np

PAGE_W = 1654
PAGE_H = 2339

_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
)


def _word(rng: np.random.Generator, n: int) -> str:
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + "/-.:"
    return "".join(rng.choice(list(alphabet), size=n))


def make_form(layout_seed: int = 0, fill_seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Рисует бланк. Возвращает (серое изображение uint8, маска статичных зон uint8).

    Маска закрывает (нулём) поля, куда пишется рукопись: так выглядит и реальная маска
    эталона. При ``fill_seed=None`` рукописи нет — это «эталон».
    """
    rng = np.random.default_rng(layout_seed)
    img = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    mask = np.full((PAGE_H, PAGE_W), 255, np.uint8)

    # Рамки
    m = int(rng.integers(50, 90))
    cv2.rectangle(img, (m, m), (PAGE_W - m, PAGE_H - m), 0, 4)
    cv2.rectangle(img, (m + 12, m + 12), (PAGE_W - m - 12, PAGE_H - m - 12), 0, 1)

    # Шапка: крупный заголовок, «логотип», реквизиты
    y = m + 120
    cv2.putText(img, _word(rng, 14), (m + 220, y), _FONTS[3], 2.0, 0, 3, cv2.LINE_AA)
    cx, cy = m + 110, m + 110
    cv2.circle(img, (cx, cy), 70, 0, 5)
    cv2.circle(img, (cx, cy), 45, 0, 2)
    cv2.line(img, (cx - 50, cy), (cx + 50, cy), 0, 3)
    cv2.line(img, (cx, cy - 50), (cx, cy + 50), 0, 3)
    for i in range(4):
        cv2.putText(
            img, _word(rng, int(rng.integers(20, 40))), (m + 220, y + 50 + 38 * i),
            _FONTS[int(rng.integers(0, 3))], 0.8, 0, 1, cv2.LINE_AA,
        )

    # Таблица полей: подпись слева, поле для рукописи справа
    top = y + 250 + int(rng.integers(0, 60))
    n_rows = int(rng.integers(14, 18))
    row_h = int(rng.integers(70, 90))
    split = int(rng.integers(520, 700))
    x0, x1 = m + 40, PAGE_W - m - 40
    fields: list[tuple[int, int, int, int]] = []
    for r in range(n_rows + 1):
        yy = top + r * row_h
        cv2.line(img, (x0, yy), (x1, yy), 0, 2 if r % 4 == 0 else 1)
    cv2.line(img, (x0, top), (x0, top + n_rows * row_h), 0, 2)
    cv2.line(img, (x1, top), (x1, top + n_rows * row_h), 0, 2)
    cv2.line(img, (split, top), (split, top + n_rows * row_h), 0, 2)
    for r in range(n_rows):
        yy = top + r * row_h
        cv2.putText(
            img, _word(rng, int(rng.integers(8, 18))), (x0 + 15, yy + row_h // 2 + 10),
            _FONTS[int(rng.integers(0, 4))], 0.9, 0, 2, cv2.LINE_AA,
        )
        # Разбивка поля на ячейки «день/месяц/час/минута» в части строк
        if r % 3 == 1:
            for k in range(1, 4):
                xx = split + k * (x1 - split) // 4
                cv2.line(img, (xx, yy), (xx, yy + row_h), 0, 1)
        fields.append((split + 8, yy + 6, x1 - 8, yy + row_h - 6))

    # Нижний блок: мелкий текст и подписи
    yb = top + n_rows * row_h + 80
    for i in range(int(rng.integers(6, 10))):
        cv2.putText(
            img, _word(rng, int(rng.integers(30, 55))), (x0, yb + 34 * i),
            _FONTS[0], 0.7, 0, 1, cv2.LINE_AA,
        )
    ys = min(PAGE_H - m - 140, yb + 34 * 10 + 60)
    for xx in (x0, PAGE_W // 2 + 40):
        cv2.line(img, (xx, ys), (xx + 500, ys), 0, 2)
        cv2.putText(img, _word(rng, 10), (xx, ys + 40), _FONTS[1], 0.8, 0, 1, cv2.LINE_AA)
        fields.append((xx, ys - 90, xx + 500, ys - 4))

    for fx0, fy0, fx1, fy1 in fields:
        mask[fy0:fy1, fx0:fx1] = 0

    if fill_seed is not None:
        _scribble(img, fields, np.random.default_rng(fill_seed))
    return img, mask


def _scribble(img: np.ndarray, fields: list[tuple[int, int, int, int]], rng: np.random.Generator):
    """«Рукопись»: гладкие ломаные в части полей, толщина и наклон случайные."""
    for fx0, fy0, fx1, fy1 in fields:
        if rng.random() < 0.3:
            continue
        x = fx0 + int(rng.integers(5, 60))
        while x < fx1 - 60 and rng.random() < 0.9:
            glyph_w = int(rng.integers(20, 45))
            n = int(rng.integers(6, 14))
            t = np.linspace(0, 1, n)
            xs = x + t * glyph_w + rng.normal(0, 4, n)
            ys = fy0 + (fy1 - fy0) * (0.5 + 0.35 * np.sin(rng.uniform(1, 4) * np.pi * t
                                                          + rng.uniform(0, 6)))
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, int(rng.integers(0, 80)),
                          int(rng.integers(2, 4)), cv2.LINE_AA)
            x += glyph_w + int(rng.integers(4, 25))


def random_homography(
    rng: np.random.Generator,
    size: tuple[int, int] = (PAGE_W, PAGE_H),
    max_rot_deg: float = 4.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    max_shift: float = 40.0,
    max_persp: float = 2e-5,
) -> np.ndarray:
    """Случайная H «эталон → скан» вокруг центра страницы."""
    w, h = size
    a = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    s = rng.uniform(*scale_range)
    tx, ty = rng.uniform(-max_shift, max_shift, 2)
    c = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]], dtype=np.float64)
    rs = np.array(
        [[s * np.cos(a), -s * np.sin(a), 0], [s * np.sin(a), s * np.cos(a), 0], [0, 0, 1]]
    )
    p = np.eye(3)
    p[2, 0], p[2, 1] = rng.uniform(-max_persp, max_persp, 2)
    back = np.array([[1, 0, w / 2 + tx], [0, 1, h / 2 + ty], [0, 0, 1]], dtype=np.float64)
    H = back @ p @ rs @ c
    return H / H[2, 2]


def degrade(
    page: np.ndarray, H_ref_to_scan: np.ndarray, rng: np.random.Generator,
    noise_sigma: float = 6.0, jpeg_quality: int = 70,
) -> np.ndarray:
    """Имитация скана: гомография, блюр, шум, JPEG. Размер скана = размер страницы."""
    h, w = page.shape[:2]
    scan = cv2.warpPerspective(page, H_ref_to_scan, (w, h), flags=cv2.INTER_LINEAR,
                               borderValue=255)
    scan = cv2.GaussianBlur(scan, (3, 3), 0.8)
    noisy = scan.astype(np.float32) + rng.normal(0, noise_sigma, scan.shape).astype(np.float32)
    scan = np.clip(noisy, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", scan, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def corner_error(
    H_est_scan_to_ref: np.ndarray, H_true_ref_to_scan: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Ошибка углов эталона: эталон → скан (истина) → эталон (оценка), px."""
    w, h = size
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    on_scan = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_true_ref_to_scan)
    back = cv2.perspectiveTransform(on_scan, H_est_scan_to_ref).reshape(-1, 2)
    return np.linalg.norm(back - corners, axis=1)
