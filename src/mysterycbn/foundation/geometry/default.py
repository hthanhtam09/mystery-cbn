"""Default geometry kernel: reference implementations of MATH_SPEC §7–§9, §13.

Pure NumPy + stdlib; deterministic by construction (fixed iteration orders,
no RNG). Contract violations raise ``ValueError`` — the kernel is
stage-agnostic, so wrapping into ``StageError`` is the calling stage's job.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Sequence

import numpy as np

from mysterycbn.foundation.geometry.kernel import GeometryKernel
from mysterycbn.foundation.geometry.primitives import BezierChainData, PolylineData, Pt
from mysterycbn.foundation.geometry.types import BezierChain, Point, Polyline

# Corner-grid directions (du, dv): south, east, north, west.
_DIRS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _turn_left(d: tuple[int, int]) -> tuple[int, int]:
    return (-d[1], d[0])


def _turn_right(d: tuple[int, int]) -> tuple[int, int]:
    return (d[1], -d[0])


def _shoelace(coords: np.ndarray) -> float:
    """Signed area of a closed ring (implicit closing edge), y-down frame."""
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


class DefaultGeometryKernel(GeometryKernel):
    """Reference kernel. Every method is deterministic and side-effect-free.

    A future native (Numba/Rust) port must match this implementation exactly
    on the shared property-test suite (ARCHITECTURE.md §13.3).
    """

    # ------------------------------------------------------------------ cracks

    def trace_cracks(self, label_map: np.ndarray) -> Sequence[Polyline]:
        """Trace per-region boundary loops on the crack grid (ENGINE_SPEC §13).

        Each directed crack edge with an interior region on its left is
        traversed exactly once; every returned loop keeps one region on its
        left throughout. Coordinates are raster-frame (x, y) = (v − ½, u − ½).
        """
        labels = np.asarray(label_map)
        if labels.ndim != 2 or labels.size == 0:
            raise ValueError(f"label_map must be a non-empty 2-D array, got {labels.shape}")
        h, w = labels.shape

        def pixel(u: int, v: int) -> int:
            if 0 <= u < h and 0 <= v < w:
                return int(labels[u, v])
            return -1

        def left_right(u: int, v: int, d: tuple[int, int]) -> tuple[int, int]:
            """(left, right) pixel labels of the directed edge from corner (u, v)."""
            if d == (1, 0):  # south
                return pixel(u, v), pixel(u, v - 1)
            if d == (-1, 0):  # north
                return pixel(u - 1, v - 1), pixel(u - 1, v)
            if d == (0, 1):  # east
                return pixel(u - 1, v), pixel(u, v)
            return pixel(u, v - 1), pixel(u - 1, v - 1)  # west

        def is_crack(u: int, v: int, d: tuple[int, int]) -> bool:
            left, right = left_right(u, v, d)
            return left != right

        # All directed crack edges whose left side is an interior region,
        # in deterministic (corner, direction-index) order.
        starts: list[tuple[int, int, int]] = []
        for u in range(h + 1):
            for v in range(w + 1):
                for di, d in enumerate(_DIRS):
                    nu, nv = u + d[0], v + d[1]
                    if not (0 <= nu <= h and 0 <= nv <= w):
                        continue
                    left, right = left_right(u, v, d)
                    if left != right and left != -1:
                        starts.append((u, v, di))
        visited: set[tuple[int, int, int]] = set()
        loops: list[PolylineData] = []

        for su, sv, sdi in starts:
            if (su, sv, sdi) in visited:
                continue
            region = left_right(su, sv, _DIRS[sdi])[0]
            corners: list[tuple[int, int]] = []
            u, v, di = su, sv, sdi
            while True:
                visited.add((u, v, di))
                corners.append((u, v))
                d = _DIRS[di]
                u, v = u + d[0], v + d[1]
                # Turn priority left, straight, right, keeping `region` on the left.
                for cand in (_turn_left(d), d, _turn_right(d)):
                    nu, nv = u + cand[0], v + cand[1]
                    if not (0 <= nu <= h and 0 <= nv <= w):
                        continue
                    if is_crack(u, v, cand) and left_right(u, v, cand)[0] == region:
                        di = _DIRS.index(cand)
                        break
                else:
                    raise ValueError("crack walk stranded — invalid label map")
                if (u, v, di) == (su, sv, sdi):
                    break
            coords = np.array([(v - 0.5, u - 0.5) for (u, v) in corners], dtype=np.float64)
            loops.append(PolylineData(coords, is_closed=True))
        return tuple(loops)

    # ---------------------------------------------------------------- simplify

    def simplify_polyline(self, polyline: Polyline, tolerance: float) -> Polyline:
        """Visvalingam–Whyatt with effective-area threshold ``tolerance²`` (MATH_SPEC §8.1).

        Endpoints are pinned (index 0 also for closed rings — the anchor);
        closed rings keep ≥ 4 vertices, open chains ≥ 2. The cross-arc
        topology guard is a stage concern, not the kernel's.
        """
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be ≥ 0, got {tolerance}")
        pts = np.asarray(polyline.coords, dtype=np.float64)
        n = pts.shape[0]
        closed = polyline.is_closed
        min_keep = 4 if closed else 2
        if n <= min_keep:
            return PolylineData(pts, is_closed=closed)

        eps = tolerance * tolerance
        prev = np.arange(-1, n - 1)
        nxt = np.arange(1, n + 1)
        if closed:
            prev[0], nxt[n - 1] = n - 1, 0
        pinned = {0} if closed else {0, n - 1}
        alive = np.ones(n, dtype=bool)
        version = np.zeros(n, dtype=np.int64)

        def area(i: int) -> float:
            a, b, c = pts[prev[i]], pts[i], pts[nxt[i]]
            return 0.5 * abs(float((a[0] - b[0]) * (c[1] - b[1]) - (a[1] - b[1]) * (c[0] - b[0])))

        heap: list[tuple[float, int, int]] = [(area(i), i, 0) for i in range(n) if i not in pinned]
        heapq.heapify(heap)
        count = n
        while heap and count > min_keep:
            ea, i, ver = heapq.heappop(heap)
            if not alive[i] or ver != version[i]:
                continue
            if ea >= eps:
                break
            alive[i] = False
            count -= 1
            p, q = prev[i], nxt[i]
            nxt[p], prev[q] = q, p
            for j in (p, q):
                if alive[j] and j not in pinned:
                    version[j] += 1
                    heapq.heappush(heap, (area(int(j)), int(j), int(version[j])))
        return PolylineData(pts[alive], is_closed=closed)

    # ------------------------------------------------------------------ bezier

    def fit_bezier_chain(
        self, polyline: Polyline, max_error: float, corner_angle_deg: float
    ) -> BezierChain:
        """Schneider least-squares fitting with corner splitting (MATH_SPEC §9.2).

        G1 inside corner-free runs; C0 at corners. Chain endpoints interpolate
        the input endpoints exactly. Closed inputs are cut at their anchor
        (index 0), which is treated as a corner.
        """
        if max_error <= 0.0:
            raise ValueError(f"max_error must be positive, got {max_error}")
        pts = np.asarray(polyline.coords, dtype=np.float64)
        if polyline.is_closed:
            pts = np.vstack([pts, pts[:1]])
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("polyline degenerates to a single point")

        corners = self._corner_indices(pts, corner_angle_deg)
        segments: list[np.ndarray] = []
        bounds = [0, *corners, len(pts) - 1]
        for a, b in itertools.pairwise(bounds):
            run = pts[a : b + 1]
            t1 = self._end_tangent(run, at_start=True)
            t2 = self._end_tangent(run, at_start=False)
            segments.extend(_fit_cubic(run, t1, t2, max_error, depth=0))
        return BezierChainData(np.stack(segments))

    @staticmethod
    def _corner_indices(pts: np.ndarray, corner_angle_deg: float) -> list[int]:
        """Interior indices whose turn angle exceeds the corner threshold."""
        if len(pts) < 3:
            return []
        d = np.diff(pts, axis=0)
        cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
        dot = np.einsum("ij,ij->i", d[:-1], d[1:])
        turn = np.degrees(np.abs(np.arctan2(cross, dot)))
        return [int(i) + 1 for i in np.nonzero(turn > corner_angle_deg)[0]]

    @staticmethod
    def _end_tangent(run: np.ndarray, *, at_start: bool) -> np.ndarray:
        """Unit tangent from the average of up to 3 end chords (ENGINE_SPEC §18.2)."""
        chords = np.diff(run, axis=0)
        k = min(3, len(chords))
        vec = chords[:k].mean(axis=0) if at_start else -chords[-k:].mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:  # symmetric cancellation — fall back to the single end chord
            vec = chords[0] if at_start else -chords[-1]
            norm = float(np.linalg.norm(vec))
        return vec / norm

    # --------------------------------------------------------------- polylabel

    def pole_of_inaccessibility(self, boundary: Polyline) -> tuple[Point, float]:
        """Quadtree branch-and-bound pole of inaccessibility (MATH_SPEC §13.3).

        ``boundary`` must be a closed simple ring (holes are a stage concern:
        callers with holes take the min over hole distances themselves).
        Precision: 1/1024 of the longer bbox side.
        """
        if not boundary.is_closed:
            raise ValueError("pole_of_inaccessibility requires a closed ring")
        ring = np.asarray(boundary.coords, dtype=np.float64)
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        size = hi - lo
        precision = float(max(size[0], size[1])) / 1024.0
        if precision == 0.0:
            return Pt(float(lo[0]), float(lo[1])), 0.0

        seg_a = ring
        seg_b = np.roll(ring, -1, axis=0)

        def signed_dist(x: float, y: float) -> float:
            p = np.array([x, y])
            ap = p - seg_a
            ab = seg_b - seg_a
            tt = np.clip(np.einsum("ij,ij->i", ap, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
            d = float(np.min(np.linalg.norm(ap - tt[:, None] * ab, axis=1)))
            # Even-odd ray crossing for the sign.
            cond = (seg_a[:, 1] > y) != (seg_b[:, 1] > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi = seg_a[:, 0] + (y - seg_a[:, 1]) * ab[:, 0] / ab[:, 1]
            inside = bool(np.count_nonzero(cond & (x < xi)) % 2)
            return d if inside else -d

        half = float(max(size[0], size[1])) / 2.0
        cx0, cy0 = (lo + hi) / 2.0
        best_d = signed_dist(float(cx0), float(cy0))
        best = (float(cx0), float(cy0))
        counter = 0
        heap: list[tuple[float, int, float, float, float]] = []

        def push(cx: float, cy: float, hh: float) -> None:
            nonlocal counter
            d = signed_dist(cx, cy)
            heapq.heappush(heap, (-(d + hh * math.sqrt(2.0)), counter, cx, cy, hh))
            counter += 1

        push(float(cx0), float(cy0), half)
        while heap:
            neg_pot, _, cx, cy, hh = heapq.heappop(heap)
            if -neg_pot - best_d <= precision:
                break
            d = signed_dist(cx, cy)
            if d > best_d:
                best_d, best = d, (cx, cy)
            q = hh / 2.0
            for dx in (-q, q):
                for dy in (-q, q):
                    push(cx + dx, cy + dy, q)
        return Pt(best[0], best[1]), max(best_d, 0.0)

    def inscribed_circle_diameter(self, boundary: Polyline) -> float:
        """Largest-inscribed-circle diameter = 2·r* of the pole (MATH_SPEC §17.1)."""
        _, radius = self.pole_of_inaccessibility(boundary)
        return 2.0 * radius

    # -------------------------------------------------------------- watertight

    def is_watertight(self, polylines: Sequence[Polyline], page_area: float) -> bool:
        """Re-prove the partition identity |Σ signed areas| = page_area (MATH_SPEC §6.1).

        Holes carry opposite orientation to outer loops, so the signed sum over
        all per-region loops nets to the covered area. Relative tolerance 1e−4
        (the QM-02 band); every input must be a closed ring.
        """
        if page_area <= 0.0:
            raise ValueError(f"page_area must be positive, got {page_area}")
        total = 0.0
        for line in polylines:
            if not line.is_closed:
                return False
            total += _shoelace(np.asarray(line.coords, dtype=np.float64))
        return abs(abs(total) - page_area) <= 1e-4 * page_area


# ------------------------------------------------------------- Schneider fit


def _bernstein(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = 1.0 - u
    return v**3, 3.0 * u * v**2, 3.0 * u**2 * v, u**3


def _bezier_eval(ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    b0, b1, b2, b3 = _bernstein(u)
    return (
        np.outer(b0, ctrl[0])
        + np.outer(b1, ctrl[1])
        + np.outer(b2, ctrl[2])
        + np.outer(b3, ctrl[3])
    )


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return np.asarray(d / d[-1])


def _generate_bezier(pts: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Least-squares α solve (MATH_SPEC §9.2) with the Wu/Schneider fallback."""
    first, last = pts[0], pts[-1]
    b0, b1, b2, b3 = _bernstein(u)
    a_l = np.outer(b1, t1)
    a_r = np.outer(b2, t2)
    tmp = pts - np.outer(b0 + b1, first) - np.outer(b2 + b3, last)
    c00 = float(np.einsum("ij,ij->", a_l, a_l))
    c01 = float(np.einsum("ij,ij->", a_l, a_r))
    c11 = float(np.einsum("ij,ij->", a_r, a_r))
    x0 = float(np.einsum("ij,ij->", a_l, tmp))
    x1 = float(np.einsum("ij,ij->", a_r, tmp))
    det = c00 * c11 - c01 * c01
    alpha_l = alpha_r = -1.0
    if abs(det) > 1e-12 * max(c00, c11, 1e-300) ** 2:
        alpha_l = (x0 * c11 - x1 * c01) / det
        alpha_r = (c00 * x1 - c01 * x0) / det
    chord = float(np.linalg.norm(last - first))
    if alpha_l <= 0.0 or alpha_r <= 0.0 or alpha_l > 3.0 * chord or alpha_r > 3.0 * chord:
        alpha_l = alpha_r = chord / 3.0
    return np.stack([first, first + alpha_l * t1, last + alpha_r * t2, last])


def _max_error(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    dist2 = np.sum((_bezier_eval(ctrl, u) - pts) ** 2, axis=1)
    interior = dist2[1:-1]
    if interior.size == 0:
        return 0.0, len(pts) // 2
    idx = int(np.argmax(interior)) + 1
    return float(np.sqrt(dist2[idx])), idx


def _reparameterize(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Newton–Raphson step per parameter (MATH_SPEC §9.2), clamped to [0, 1]."""
    d1 = 3.0 * np.diff(ctrl, axis=0)
    d2 = 2.0 * np.diff(d1, axis=0)
    q = _bezier_eval(ctrl, u) - pts
    v = 1.0 - u
    qp = np.outer(v**2, d1[0]) + np.outer(2.0 * u * v, d1[1]) + np.outer(u**2, d1[2])
    qpp = np.outer(v, d2[0]) + np.outer(u, d2[1])
    num = np.einsum("ij,ij->i", q, qp)
    den = np.einsum("ij,ij->i", qp, qp) + np.einsum("ij,ij->i", q, qpp)
    step = np.where(np.abs(den) > 1e-12, num / np.where(den == 0.0, 1.0, den), 0.0)
    out = np.asarray(np.clip(u - step, 0.0, 1.0))
    out[0], out[-1] = 0.0, 1.0
    return out


def _fit_cubic(
    pts: np.ndarray, t1: np.ndarray, t2: np.ndarray, err: float, depth: int
) -> list[np.ndarray]:
    if len(pts) == 2:
        d = float(np.linalg.norm(pts[1] - pts[0])) / 3.0
        return [np.stack([pts[0], pts[0] + d * t1, pts[1] + d * t2, pts[1]])]
    u = _chord_params(pts)
    ctrl = _generate_bezier(pts, u, t1, t2)
    max_e, split = _max_error(pts, ctrl, u)
    if max_e <= err:
        return [ctrl]
    if max_e <= 4.0 * err:
        for _ in range(4):
            u = _reparameterize(pts, ctrl, u)
            ctrl = _generate_bezier(pts, u, t1, t2)
            max_e, split = _max_error(pts, ctrl, u)
            if max_e <= err:
                return [ctrl]
    if depth >= 32:  # bounded recursion: per-edge line segments always succeed
        return [
            np.stack([a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0, b])
            for a, b in itertools.pairwise(pts)
        ]
    center = pts[split - 1] - pts[split + 1]
    norm = float(np.linalg.norm(center))
    if norm < 1e-12:
        center = pts[split - 1] - pts[split]
        norm = float(np.linalg.norm(center))
    center = center / norm
    left = _fit_cubic(pts[: split + 1], t1, center, err, depth + 1)
    right = _fit_cubic(pts[split:], -center, t2, err, depth + 1)
    return left + right
