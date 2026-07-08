"""Renders one bound artifact to a viewable preview (PNG bytes or text) plus
a downloadable byte blob, for the HTML report (docs/VISUAL_DEBUGGER.md §3).

Every artifact already carries enough to reconstruct a picture of itself
without re-running any engine algorithm: label maps/region graphs color by
integer id, arc graphs/curve sets are traced as SVG polylines, and the
final render artifacts (svg/pdf/png_previews) are already-produced bytes
pulled straight from the context. No engine code is modified or duplicated
beyond simple colorization, which is presentation, not computation.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from mysterycbn.model.layout import LabelPlan, Legend
from mysterycbn.model.records import ImageStats, LabelMap, Palette, RasterImage, RegionGraph
from mysterycbn.model.vector import ArcGraph, CurveSet
from mysterycbn.render.png import PngPreviews
from mysterycbn.render.svg import SvgDocument
from mysterycbn.stages.raster.load import SourceBytes

_MAX_PREVIEW_SIDE = 640


@dataclass(frozen=True)
class ArtifactView:
    """One artifact's rendered form for the report."""

    kind: str  # "image" | "text"
    preview_png: bytes | None
    text: str | None
    download_bytes: bytes
    download_filename: str
    summary: str


def _encode_png(rgb_u8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_u8, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _downscale_for_preview(rgb_u8: np.ndarray) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    scale = min(1.0, _MAX_PREVIEW_SIDE / max(h, w))
    if scale >= 1.0:
        return rgb_u8
    img = Image.fromarray(rgb_u8, mode="RGB")
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.NEAREST)
    return np.asarray(img)


def _categorical_colors(n: int, *, seed: int = 0) -> np.ndarray:
    """Deterministic, visually distinct RGB colors for up to ``n`` integer
    ids -- an HSV wheel, not a perceptual palette (this is a debug view,
    not an engine artifact)."""
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) / max(n, 1) + rng.uniform(0, 1)) % 1.0
    hsv = np.stack([hues, np.full(n, 0.65), np.full(n, 0.95)], axis=1)
    img = Image.fromarray((hsv[None, :, :] * 255).astype(np.uint8), mode="HSV")
    rgb: np.ndarray = np.asarray(img.convert("RGB"))[0]
    return rgb


def view_source_bytes(artifact: SourceBytes) -> ArtifactView:
    try:
        img = Image.open(io.BytesIO(artifact.data)).convert("RGB")
        preview = _encode_png(_downscale_for_preview(np.asarray(img)))
    except Exception:
        preview = None
    return ArtifactView(
        kind="image",
        preview_png=preview,
        text=None,
        download_bytes=artifact.data,
        download_filename="original.bin",
        summary=f"{len(artifact.data):,} bytes",
    )


def _view_raster_image(artifact: RasterImage, *, filename: str) -> ArtifactView:
    rgb_u8 = np.clip(artifact.pixels * 255.0 + 0.5, 0, 255).astype(np.uint8)
    full_png = _encode_png(rgb_u8)
    preview = _encode_png(_downscale_for_preview(rgb_u8))
    h, w = rgb_u8.shape[:2]
    return ArtifactView(
        kind="image",
        preview_png=preview,
        text=None,
        download_bytes=full_png,
        download_filename=filename,
        summary=f"{w}x{h}, work_scale={artifact.work_scale:.4f}",
    )


def view_raster_source(artifact: RasterImage) -> ArtifactView:
    return _view_raster_image(artifact, filename="raster_source.png")


def view_raster_working(artifact: RasterImage) -> ArtifactView:
    return _view_raster_image(artifact, filename="raster_working.png")


def view_image_stats(artifact: ImageStats) -> ArtifactView:
    text = (
        f"colorfulness: {artifact.colorfulness:.3f}\n"
        f"edge_density: {artifact.edge_density:.3f}\n"
        f"brightness (L*): {artifact.brightness:.2f}\n"
        f"contrast (L* std): {artifact.contrast:.2f}\n"
        f"saturation (mean C*): {artifact.saturation:.2f}\n"
        f"entropy_bits: {artifact.entropy_bits:.3f}\n"
        f"lab_mean: {tuple(round(v, 2) for v in artifact.lab_mean)}\n"
        f"lab_std: {tuple(round(v, 2) for v in artifact.lab_std)}\n"
    )
    return ArtifactView(
        kind="text",
        preview_png=None,
        text=text,
        download_bytes=text.encode("utf-8"),
        download_filename="image_stats.txt",
        summary=(
            f"colorfulness={artifact.colorfulness:.2f}, edge_density={artifact.edge_density:.2f}"
        ),
    )


def view_label_map(artifact: LabelMap, *, palette: Palette | None = None) -> ArtifactView:
    labels = artifact.labels
    n = int(labels.max()) + 1
    if palette is not None and palette.size >= n:
        rgb = np.array(
            [
                np.clip(np.array(c.srgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
                for c in palette.colors[:n]
            ]
        )
    else:
        rgb = _categorical_colors(n)
    colorized = rgb[labels]
    full_png = _encode_png(colorized)
    preview = _encode_png(_downscale_for_preview(colorized))
    return ArtifactView(
        kind="image",
        preview_png=preview,
        text=None,
        download_bytes=full_png,
        download_filename="label_map.png",
        summary=f"{labels.shape[1]}x{labels.shape[0]}, {n} labels",
    )


def view_palette(artifact: Palette) -> ArtifactView:
    swatch_side = 48
    n = artifact.size
    strip = np.zeros((swatch_side, swatch_side * n, 3), dtype=np.uint8)
    for i, color in enumerate(artifact.colors):
        rgb = np.clip(np.array(color.srgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        strip[:, i * swatch_side : (i + 1) * swatch_side] = rgb
    png = _encode_png(strip)
    return ArtifactView(
        kind="image",
        preview_png=png,
        text=None,
        download_bytes=png,
        download_filename="palette.png",
        summary=f"{n} colors, min_delta_e={artifact.min_delta_e:.2f}",
    )


def view_region_graph(artifact: RegionGraph) -> ArtifactView:
    cmap = artifact.component_map
    n = len(artifact.regions)
    colors = _categorical_colors(n)
    colorized = colors[cmap]
    full_png = _encode_png(colorized)
    preview = _encode_png(_downscale_for_preview(colorized))
    return ArtifactView(
        kind="image",
        preview_png=preview,
        text=None,
        download_bytes=full_png,
        download_filename="region_graph.png",
        summary=f"{n} regions, {len(artifact.edges)} adjacency edges",
    )


def _arcs_to_svg(arcs: list[tuple[np.ndarray, bool]], *, width: float, height: float) -> bytes:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.1f} {height:.1f}" '
        f'width="{min(width, _MAX_PREVIEW_SIDE):.0f}">',
        f'<rect x="0" y="0" width="{width:.1f}" height="{height:.1f}" fill="white"/>',
    ]
    for points, closed in arcs:
        d = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        if closed:
            d += " Z"
        parts.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="0.5"/>')
    parts.append("</svg>")
    return "".join(parts).encode("utf-8")


def view_arc_graph(artifact: ArcGraph) -> ArtifactView:
    all_points = np.concatenate([arc.points for arc in artifact.arcs], axis=0)
    width = float(all_points[:, 0].max()) + 10.0
    height = float(all_points[:, 1].max()) + 10.0
    svg_bytes = _arcs_to_svg(
        [(arc.points, arc.closed) for arc in artifact.arcs], width=width, height=height
    )
    return ArtifactView(
        kind="image",
        preview_png=None,
        text=svg_bytes.decode("utf-8"),
        download_bytes=svg_bytes,
        download_filename="arc_graph.svg",
        summary=f"{len(artifact.arcs)} arcs, {len(artifact.faces)} faces",
    )


def view_curve_set(artifact: CurveSet) -> ArtifactView:
    polylines = []
    for curve in artifact.curves:
        pts = [curve.segments[0].control[0]]
        for seg in curve.segments:
            pts.append(seg.control[3])
        polylines.append((np.array(pts), False))
    all_points = np.concatenate([p for p, _ in polylines], axis=0)
    width = float(all_points[:, 0].max()) + 10.0
    height = float(all_points[:, 1].max()) + 10.0
    svg_bytes = _arcs_to_svg(polylines, width=width, height=height)
    return ArtifactView(
        kind="image",
        preview_png=None,
        text=svg_bytes.decode("utf-8"),
        download_bytes=svg_bytes,
        download_filename="curve_set.svg",
        summary=f"{len(artifact.curves)} curves, {len(artifact.faces)} faces",
    )


def view_label_plan(artifact: LabelPlan) -> ArtifactView:
    lines = [
        f"region {label.region_id}: #{label.printed_number} at "
        f"({label.anchor[0]:.1f}, {label.anchor[1]:.1f}) {label.font_size_pt:.1f}pt "
        f"mode={label.mode}"
        for label in artifact.labels
    ]
    text = "\n".join(lines) + "\n"
    return ArtifactView(
        kind="text",
        preview_png=None,
        text=text,
        download_bytes=text.encode("utf-8"),
        download_filename="label_plan.txt",
        summary=f"{len(artifact.labels)} labels placed",
    )


def view_legend(artifact: Legend) -> ArtifactView:
    lines = [
        f"palette_index {chip[0]}: printed #{artifact.permutation[chip[0]] + 1} "
        f"at ({chip[1][0]:.1f}, {chip[1][1]:.1f}) side={chip[2]:.1f}pt"
        for chip in artifact.chips
    ]
    text = "\n".join(lines) + "\n"
    return ArtifactView(
        kind="text",
        preview_png=None,
        text=text,
        download_bytes=text.encode("utf-8"),
        download_filename="legend.txt",
        summary=f"{len(artifact.chips)} legend chips",
    )


def view_svg_document(artifact: SvgDocument) -> ArtifactView:
    return ArtifactView(
        kind="text",
        preview_png=None,
        text=artifact.data.decode("utf-8"),
        download_bytes=artifact.data,
        download_filename="page.svg",
        summary=f"{len(artifact.data):,} bytes",
    )


def view_png_previews(artifact: PngPreviews) -> dict[str, ArtifactView]:
    views = {}
    for name, data in artifact.previews.items():
        views[name] = ArtifactView(
            kind="image",
            preview_png=data,
            text=None,
            download_bytes=data,
            download_filename=f"preview_{name}.png",
            summary=f"{len(data):,} bytes",
        )
    return views
