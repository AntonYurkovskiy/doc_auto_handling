"""Выравнивание скана ваучера по эталону бланка.

Путь по умолчанию: признаки ORB на уменьшенной копии → пары через ``knnMatch`` и тест Лоу →
гомография (USAC_MAGSAC, если есть, иначе RANSAC) → проверка правдоподобия →
уточнение ECC (``MOTION_HOMOGRAPHY``) на уменьшенной копии и, по желанию, на полной →
пост-проверка ``score`` по краям статичных зон. Если ORB не набрал инлайеров, пробуется
SIFT. Страница, перевёрнутая на 180°, распознаётся по углу гомографии (ORB и SIFT
инвариантны к повороту); если прямой путь не прошёл, пробуется явно повёрнутый скан.

Зависимости — только numpy и opencv. Все пороги собраны в :class:`AlignParams`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

_HAS_MAGSAC = hasattr(cv2, "USAC_MAGSAC")
_HAS_SIFT = hasattr(cv2, "SIFT")


@dataclass(frozen=True)
class AlignParams:
    """Параметры выравнивания и пороги ``ok``. Калибруются в T07."""

    # Признаки
    feature_max_side: int = 1200  # длинная сторона копии для поиска признаков, px
    orb_nfeatures: int = 5000
    orb_fast_threshold: int = 10
    sift_nfeatures: int = 4000
    use_sift_fallback: bool = True
    lowe_ratio: float = 0.75
    ransac_reproj_thresh: float = 3.0  # px на копии признаков
    ransac_max_iters: int = 5000
    ransac_confidence: float = 0.999

    # Правдоподобие гомографии
    min_scale: float = 0.5
    max_scale: float = 2.0
    max_anisotropy: float = 1.25  # отношение сингулярных чисел аффинной части
    max_rotation_deg: float = 15.0  # отклонение от 0° (или от 180° для перевёрнутой)
    max_perspective: float = 0.15  # |h31*W| + |h32*H|: относительное искажение по странице
    try_rotate180: bool = True

    # Уточнение ECC
    use_ecc: bool = True
    ecc_scale: float = 0.5  # масштаб грубого прохода ECC
    ecc_iters: int = 30
    ecc_eps: float = 1e-4
    ecc_full_res: bool = False  # второй проход на полном размере (дороже, точнее)
    ecc_full_iters: int = 10
    ecc_gauss_size: int = 5
    ecc_max_shift_px: float = 8.0  # допустимое расхождение ECC с начальной H по углам, px

    # Пост-проверка
    score_scale: float = 0.5
    score_dilate_px: int = 1  # радиус допуска совпадения краёв на копии score_scale, px
    canny_low: int = 50
    canny_high: int = 150

    # Пороги ok
    min_inliers: int = 40
    min_inlier_ratio: float = 0.15
    max_reproj_err: float = 3.0  # px на полном размере эталона
    min_score: float = 0.6


@dataclass(frozen=True, eq=False)
class _Features:
    pts: np.ndarray  # (N, 2) float32, координаты полного размера
    desc: np.ndarray | None


@dataclass(frozen=True, eq=False)
class Reference:
    """Эталон бланка: серое изображение, маска статичных зон и кэш признаков."""

    image: np.ndarray  # uint8, серое
    mask: np.ndarray  # uint8 0/255, размер эталона; 255 — статичная зона
    name: str
    orb: _Features
    sift: _Features | None
    ecc_image: np.ndarray  # float32, уменьшенная сглаженная копия для ECC
    ecc_mask: np.ndarray  # uint8, маска на копии ECC
    score_edges: np.ndarray  # uint8 0/255, края эталона на копии score_scale
    score_edges_dil: np.ndarray  # расширенные края эталона
    score_mask: np.ndarray  # uint8 0/255, маска на копии score_scale
    params: AlignParams = field(default_factory=lambda: AlignParams())

    @property
    def size(self) -> tuple[int, int]:
        """(ширина, высота) эталона."""
        return int(self.image.shape[1]), int(self.image.shape[0])


@dataclass(frozen=True, eq=False)
class AlignResult:
    ok: bool
    H: np.ndarray | None  # 3x3, скан -> эталон
    warped: np.ndarray | None  # скан в координатах эталона (размер эталона)
    inliers: int
    inlier_ratio: float
    reproj_err: float  # средняя ошибка на инлайерах, px полного размера эталона
    score: float  # совпадение краёв со статичными зонами, 0..1
    rotated180: bool
    method: str  # "orb+ecc", "orb", "sift+ecc", "sift"
    reason: str  # почему ok=False; пусто при ok=True


# --- утилиты -----------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"неподдерживаемая форма изображения: {image.shape}")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _resize(gray: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-9:
        return gray
    h, w = gray.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def _scale_mat(sx: float, sy: float | None = None) -> np.ndarray:
    return np.diag([sx, sx if sy is None else sy, 1.0])


def _feature_scale(gray: np.ndarray, params: AlignParams) -> float:
    return min(1.0, params.feature_max_side / float(max(gray.shape[:2])))


def _detect(gray: np.ndarray, params: AlignParams, kind: str) -> _Features:
    """Признаки на уменьшенной копии; точки возвращаются в координатах полного размера."""
    scale = _feature_scale(gray, params)
    small = _resize(gray, scale)
    detector: cv2.Feature2D
    if kind == "orb":
        detector = cv2.ORB.create(
            nfeatures=params.orb_nfeatures, fastThreshold=params.orb_fast_threshold
        )
    else:
        detector = cv2.SIFT.create(nfeatures=params.sift_nfeatures)
    kps, desc = detector.detectAndCompute(small, None)
    if not kps or desc is None:
        return _Features(np.empty((0, 2), np.float32), None)
    pts = np.array([kp.pt for kp in kps], dtype=np.float32) / scale
    return _Features(pts, desc)


def _edges(gray_small: np.ndarray, params: AlignParams) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray_small, (3, 3), 0)
    return cv2.Canny(blurred, params.canny_low, params.canny_high)


def _dilate(img: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return img
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(img, k)


def _mask_resize(mask: np.ndarray, scale: float) -> np.ndarray:
    h, w = mask.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def build_reference(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    name: str = "",
    params: AlignParams | None = None,
) -> Reference:
    """Готовит эталон: признаки ORB (и SIFT для запасного пути), копии для ECC и score.

    ``mask`` — статичные зоны бланка (ненулевые пиксели), того же размера, что эталон.
    Без маски статичной считается вся страница. ``params`` должны совпадать с теми,
    что потом передаются в :func:`align` (от них зависят масштабы кэша).
    """
    params = params or AlignParams()
    gray = _to_gray(image)
    if mask is None:
        mask_u8 = np.full(gray.shape, 255, np.uint8)
    else:
        if mask.shape[:2] != gray.shape:
            raise ValueError("маска должна совпадать по размеру с эталоном")
        mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)

    orb = _detect(gray, params, "orb")
    sift = _detect(gray, params, "sift") if params.use_sift_fallback and _HAS_SIFT else None

    ecc_small = _resize(gray, params.ecc_scale).astype(np.float32)
    ecc_mask = _mask_resize(mask_u8, params.ecc_scale)

    score_small = _resize(gray, params.score_scale)
    score_edges = _edges(score_small, params)
    score_mask = _mask_resize(mask_u8, params.score_scale)
    return Reference(
        image=gray,
        mask=mask_u8,
        name=name,
        orb=orb,
        sift=sift,
        ecc_image=ecc_small,
        ecc_mask=ecc_mask,
        score_edges=score_edges,
        score_edges_dil=_dilate(score_edges, params.score_dilate_px),
        score_mask=score_mask,
        params=params,
    )


# --- геометрия ---------------------------------------------------------------


def map_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Применяет гомографию к точкам формы (N, 2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def map_points_to_scan(points_ref: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Переводит точки из координат эталона в координаты исходного скана.

    ``H`` — гомография скан → эталон из :class:`AlignResult` (с учётом поворота на 180°).
    Нужна, чтобы резать боксы подполей по исходному скану без двойной интерполяции.
    """
    return map_points(points_ref, np.linalg.inv(np.asarray(H, dtype=np.float64)))


def _rotation_deg(H: np.ndarray, center: tuple[float, float]) -> float:
    """Угол поворота H в окрестности точки ``center`` (скан), градусы в (-180, 180]."""
    cx, cy = center
    p = map_points(np.array([[cx, cy], [cx + 100.0, cy]]), H)
    dx, dy = p[1] - p[0]
    return math.degrees(math.atan2(dy, dx))


def _check_plausible(
    H: np.ndarray, scan_size: tuple[int, int], params: AlignParams
) -> tuple[bool, bool, str]:
    """Проверка правдоподобия H (скан → эталон). Возвращает (ok, rotated180, reason)."""
    if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return False, False, "вырожденная гомография"
    Hn = H / H[2, 2]
    w, h = scan_size
    persp = abs(Hn[2, 0]) * w + abs(Hn[2, 1]) * h
    if persp > params.max_perspective:
        return False, False, f"велика перспектива ({persp:.3f})"
    A = Hn[:2, :2]
    if np.linalg.det(A) <= 0:
        return False, False, "гомография с отражением"
    s = np.linalg.svd(A, compute_uv=False)
    if not (params.min_scale <= s[1] and s[0] <= params.max_scale):
        return False, False, f"масштаб вне пределов ({s[1]:.2f}..{s[0]:.2f})"
    if s[0] / max(s[1], 1e-12) > params.max_anisotropy:
        return False, False, f"велика анизотропия ({s[0] / s[1]:.2f})"
    angle = _rotation_deg(Hn, (w / 2.0, h / 2.0))
    if abs(angle) <= params.max_rotation_deg:
        return True, False, ""
    if 180.0 - abs(angle) <= params.max_rotation_deg:
        return True, True, ""
    return False, False, f"поворот {angle:.1f}° вне допуска"


@dataclass
class _Estimate:
    H: np.ndarray  # скан -> эталон, полный размер
    inliers: int
    inlier_ratio: float
    src: np.ndarray  # точки скана (инлайеры)
    dst: np.ndarray  # точки эталона (инлайеры)
    method: str


def _match_and_estimate(
    scan_f: _Features, ref_f: _Features, kind: str, scan_scale: float, ref_scale: float,
    params: AlignParams,
) -> tuple[_Estimate | None, str]:
    if scan_f.desc is None or ref_f.desc is None or len(scan_f.pts) < 4 or len(ref_f.pts) < 4:
        return None, f"{kind}: мало признаков"
    if kind == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(scan_f.desc, ref_f.desc, k=2)
    good = [
        pair[0]
        for pair in knn
        if len(pair) == 2 and pair[0].distance < params.lowe_ratio * pair[1].distance
    ]
    if len(good) < 8:
        return None, f"{kind}: мало пар после теста Лоу ({len(good)})"
    src = scan_f.pts[[m.queryIdx for m in good]]
    dst = ref_f.pts[[m.trainIdx for m in good]]
    # Оценку ведём в координатах копий признаков: порог RANSAC в пикселях копии.
    src_s = (src * scan_scale).astype(np.float32)
    dst_s = (dst * ref_scale).astype(np.float32)
    method = cv2.USAC_MAGSAC if _HAS_MAGSAC else cv2.RANSAC
    H_s, inl = cv2.findHomography(
        src_s, dst_s, method, params.ransac_reproj_thresh,
        maxIters=params.ransac_max_iters, confidence=params.ransac_confidence,
    )
    if H_s is None or inl is None:
        return None, f"{kind}: гомография не найдена"
    inl_mask = inl.ravel().astype(bool)
    n_inl = int(inl_mask.sum())
    H = np.linalg.inv(_scale_mat(ref_scale)) @ H_s @ _scale_mat(scan_scale)
    return (
        _Estimate(H, n_inl, n_inl / len(good), src[inl_mask], dst[inl_mask], kind),
        "",
    )


def _refine_ecc(
    H: np.ndarray, scan_gray: np.ndarray, ref: Reference, params: AlignParams
) -> np.ndarray | None:
    """Уточняет H (скан → эталон) через ECC; ``None`` — если ECC не сошёлся или «уехал»."""
    s = params.ecc_scale
    scan_small = _resize(scan_gray, s).astype(np.float32)
    # ECC ищет W: эталон -> скан (warpPerspective с WARP_INVERSE_MAP).
    W = np.linalg.inv(H)
    W_small = _scale_mat(s) @ W @ np.linalg.inv(_scale_mat(s))
    W_small = (W_small / W_small[2, 2]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, params.ecc_iters, params.ecc_eps)
    try:
        _, W_small = cv2.findTransformECC(
            ref.ecc_image, scan_small, W_small, cv2.MOTION_HOMOGRAPHY, criteria,
            ref.ecc_mask, params.ecc_gauss_size,
        )
    except cv2.error:
        return None
    W = np.linalg.inv(_scale_mat(s)) @ W_small.astype(np.float64) @ _scale_mat(s)
    if params.ecc_full_res:
        criteria_full = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, params.ecc_full_iters, params.ecc_eps
        )
        W_full = (W / W[2, 2]).astype(np.float32)
        try:
            _, W_full = cv2.findTransformECC(
                ref.image.astype(np.float32), scan_gray.astype(np.float32), W_full,
                cv2.MOTION_HOMOGRAPHY, criteria_full, ref.mask, params.ecc_gauss_size,
            )
            W = W_full.astype(np.float64)
        except cv2.error:
            pass
    H_new = np.linalg.inv(W)
    H_new = H_new / H_new[2, 2]
    # ECC не должен далеко уводить от оценки по признакам: сверяем по углам эталона.
    w, h = ref.size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    shift = np.linalg.norm(
        map_points(corners, np.linalg.inv(H_new)) - map_points(corners, np.linalg.inv(H)), axis=1
    )
    if not np.all(np.isfinite(shift)) or shift.max() > params.ecc_max_shift_px:
        return None
    return H_new


def _score(
    warped_small: np.ndarray, valid_small: np.ndarray, ref: Reference, params: AlignParams
) -> float:
    """F1 совпадения краёв выровненного скана и эталона внутри маски статичных зон.

    Оба изображения — уже в координатах эталона на копии ``score_scale``.
    """
    region = (ref.score_mask > 0) & (valid_small > 0)
    e_ref = (ref.score_edges > 0) & region
    e_scan = (_edges(warped_small, params) > 0) & region
    n_ref = int(e_ref.sum())
    n_scan = int(e_scan.sum())
    if n_ref == 0 or n_scan == 0:
        return 0.0
    e_scan_dil = _dilate(e_scan.astype(np.uint8) * 255, params.score_dilate_px) > 0
    recall = float((e_ref & e_scan_dil).sum()) / n_ref
    precision = float((e_scan & (ref.score_edges_dil > 0)).sum()) / n_scan
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def alignment_score(
    scan: np.ndarray, H: np.ndarray, ref: Reference, params: AlignParams | None = None
) -> float:
    """``score`` для произвольной H (скан → эталон): совпадение краёв в статичных зонах, 0..1.

    Считается на копии ``score_scale``: скан сразу переводится в уменьшенные координаты
    эталона, полноразмерный warp не нужен.
    """
    params = params or ref.params
    gray = _to_gray(scan)
    s = params.score_scale
    h_small, w_small = ref.score_edges.shape
    H_small = _scale_mat(s) @ np.asarray(H, dtype=np.float64)
    warped = cv2.warpPerspective(
        gray, H_small, (w_small, h_small), flags=cv2.INTER_AREA, borderValue=255
    )
    valid = cv2.warpPerspective(
        np.full_like(gray, 255), H_small, (w_small, h_small), flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    return _score(warped, valid, ref, params)


def _reproj_err(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    if len(src) == 0:
        return float("inf")
    return float(np.linalg.norm(map_points(src, H) - dst, axis=1).mean())


# --- основной вход ------------------------------------------------------------


def _fail(reason: str, est: _Estimate | None, pre_rot: np.ndarray | None) -> AlignResult:
    """Отказ до уточнения; H (если есть) — диагностическая, в координатах исходного скана."""
    H = None
    if est is not None:
        H = est.H if pre_rot is None else est.H @ pre_rot
        H = H / H[2, 2]
    return AlignResult(
        ok=False,
        H=H,
        warped=None,
        inliers=0 if est is None else est.inliers,
        inlier_ratio=0.0 if est is None else est.inlier_ratio,
        reproj_err=float("inf") if est is None else _reproj_err(est.H, est.src, est.dst),
        score=0.0,
        rotated180=pre_rot is not None,
        method="" if est is None else est.method,
        reason=reason,
    )


def _align_once(scan: np.ndarray, scan_gray: np.ndarray, ref: Reference, params: AlignParams,
                pre_rot: np.ndarray | None) -> AlignResult:
    """Одна попытка; ``pre_rot`` — матрица поворота исходного скана в ``scan_gray``."""
    h_s, w_s = scan_gray.shape
    scan_scale = _feature_scale(scan_gray, params)
    ref_scale = _feature_scale(ref.image, params)

    reasons: list[str] = []
    est: _Estimate | None = None
    rotated = False
    kinds = ["orb"] + (["sift"] if params.use_sift_fallback and ref.sift is not None else [])
    for kind in kinds:
        ref_f = ref.orb if kind == "orb" else ref.sift
        assert ref_f is not None
        cand, why = _match_and_estimate(
            _detect(scan_gray, params, kind), ref_f, kind, scan_scale, ref_scale, params
        )
        if cand is None:
            reasons.append(why)
            continue
        if cand.inliers < params.min_inliers or cand.inlier_ratio < params.min_inlier_ratio:
            reasons.append(
                f"{kind}: мало инлайеров ({cand.inliers}, доля {cand.inlier_ratio:.2f})"
            )
            est = est if est is not None and est.inliers >= cand.inliers else cand
            continue
        plausible, rotated, why = _check_plausible(cand.H, (w_s, h_s), params)
        if not plausible:
            reasons.append(f"{kind}: {why}")
            est = est if est is not None and est.inliers >= cand.inliers else cand
            continue
        est = cand
        break
    else:
        return _fail("; ".join(reasons), est, pre_rot)

    H = est.H
    method = est.method
    if params.use_ecc:
        H_ecc = _refine_ecc(H, scan_gray, ref, params)
        if H_ecc is not None:
            H = H_ecc
            method += "+ecc"

    reproj = _reproj_err(H, est.src, est.dst)
    score = alignment_score(scan_gray, H, ref, params)
    w_r, h_r = ref.size

    H_total = H if pre_rot is None else H @ pre_rot
    H_total = H_total / H_total[2, 2]
    warped = cv2.warpPerspective(
        scan, H_total, (w_r, h_r), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 255),
    )
    fails: list[str] = []
    if reproj > params.max_reproj_err:
        fails.append(f"велика ошибка перепроецирования ({reproj:.2f} px)")
    if score < params.min_score:
        fails.append(f"низкий score совпадения со статичными зонами ({score:.2f})")
    return AlignResult(
        ok=not fails,
        H=H_total,
        warped=warped,
        inliers=est.inliers,
        inlier_ratio=est.inlier_ratio,
        reproj_err=reproj,
        score=score,
        rotated180=rotated != (pre_rot is not None),
        method=method,
        reason="; ".join(fails),
    )


def align(scan: np.ndarray, ref: Reference, params: AlignParams | None = None) -> AlignResult:
    """Выравнивает скан по эталону.

    ``scan`` — uint8, серый или BGR. ``params`` по умолчанию берутся из эталона.
    Возвращает :class:`AlignResult`; ``H`` переводит координаты исходного скана
    в координаты эталона (перевёрнутость на 180° уже учтена).
    """
    params = params or ref.params
    gray = _to_gray(scan)
    first = _align_once(scan, gray, ref, params, None)
    if first.ok or not params.try_rotate180:
        return first
    h, w = gray.shape
    # Поворот на 180°: (x, y) -> (w-1-x, h-1-y).
    rot = np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]])
    second = _align_once(scan, cv2.rotate(gray, cv2.ROTATE_180), ref, params, rot)
    if second.ok:
        return second
    best = max((first, second), key=lambda r: (r.score, r.inliers))
    reason = f"прямо: {first.reason or 'нет'}; повёрнутый: {second.reason or 'нет'}"
    return AlignResult(
        ok=False, H=best.H, warped=best.warped, inliers=best.inliers,
        inlier_ratio=best.inlier_ratio, reproj_err=best.reproj_err, score=best.score,
        rotated180=best.rotated180, method=best.method, reason=reason,
    )
