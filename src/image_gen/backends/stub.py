"""Stub backend: no model, no GPU, no torch.

Draws a placeholder that *describes itself*. A static placeholder image would
make every scene look identical, which defeats the point -- you could not tell
whether the right file landed in the right place, whether the skybox really is
2:1, or whether the trigger phrase was applied. Drawing the parameters onto the
image means all three are visible at a glance, in Unity or in a file browser.

Everything else -- paths, metadata, manifests, failure records -- is identical
to the real backend, because it lives in BaseGenerator.
"""

from __future__ import annotations

import colorsys
import hashlib
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

from ..config import AppConfig
from ..generator import BaseGenerator
from ..types import GenerateRequest, ResolvedParams

if TYPE_CHECKING:  # pragma: no cover
    from PIL.ImageFont import FreeTypeFont, ImageFont as BitmapFont

    AnyFont = FreeTypeFont | BitmapFont


def _stable_hue(kind: str, seed: int) -> float:
    """A hue in [0, 1) that depends only on kind and seed.

    hashlib rather than hash(): Python randomises string hashing per process,
    so hash() would give a different colour on every run and break the "same
    seed always looks the same" property this is here to demonstrate.
    """
    digest = hashlib.md5(f"{kind}:{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def _palette(kind: str, seed: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    hue = _stable_hue(kind, seed)
    bg = colorsys.hsv_to_rgb(hue, 0.45, 0.30)
    accent = colorsys.hsv_to_rgb((hue + 0.5) % 1.0, 0.55, 0.95)
    to_rgb = lambda c: tuple(int(v * 255) for v in c)  # noqa: E731
    return to_rgb(bg), to_rgb(accent)


def _font(size: int) -> "AnyFont":
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover - older Pillow
        pass
    for name in ("DejaVuSans.ttf", "arial.ttf"):  # pragma: no cover
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()  # pragma: no cover


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Measure-based wrapping, so it holds up whichever font was available."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


class StubGenerator(BaseGenerator):
    backend_name = "stub"

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)

    def loaded_kinds(self) -> list[str]:
        # Nothing to load, so every configured kind is instantly servable.
        return sorted(self.config.kinds)

    def _render(self, req: GenerateRequest, params: ResolvedParams) -> Image.Image:
        kind_cfg = self.config.kind(req.kind)

        if kind_cfg.stub_file is not None:
            path = self.config.resolve_path(kind_cfg.stub_file)
            with Image.open(path) as src:
                # Resized so the dimension guarantee holds even when the
                # override image is some other size.
                return src.convert("RGB").resize(
                    (params.width, params.height), Image.LANCZOS
                )

        return self._draw(req, params)

    def _draw(self, req: GenerateRequest, params: ResolvedParams) -> Image.Image:
        w, h = params.width, params.height
        bg, accent = _palette(req.kind, req.seed)
        kind_cfg = self.config.kind(req.kind)

        image = Image.new("RGB", (w, h), bg)
        draw = ImageDraw.Draw(image)

        grid = tuple(min(255, c + 22) for c in bg)
        step = max(32, w // 24)
        for x in range(step, w, step):
            draw.line([(x, 0), (x, h)], fill=grid, width=1)
        for y in range(step, h, step):
            draw.line([(0, y), (w, y)], fill=grid, width=1)

        is_equirect = kind_cfg.aspect_ratio == 2.0
        if is_equirect:
            # Horizon, so a wrong vertical orientation is obvious in the headset.
            draw.line([(0, h // 2), (w, h // 2)], fill=accent, width=max(1, h // 256))
            # Seam markers: identical bars hugging both edges. If the equirect
            # wraps correctly they meet behind the viewer as one continuous bar.
            bar_w = max(8, w // 64)
            bar_h = max(8, h // 32)
            for box in (
                (0, h // 2 - bar_h, bar_w, h // 2 + bar_h),
                (w - bar_w, h // 2 - bar_h, w, h // 2 + bar_h),
            ):
                draw.rectangle(box, fill=accent)

        margin = max(16, w // 40)
        title_size = max(16, h // 18)
        body_size = max(11, h // 40)
        title_font = _font(title_size)
        body_font = _font(body_size)

        y = margin
        draw.text((margin, y), req.kind.upper(), font=title_font, fill=accent)
        y += title_size + body_size // 2

        facts = [
            f"scene: {req.scene_id}",
            f"seed: {req.seed}",
            f"{w}x{h}  -  {params.steps} steps  -  cfg {params.guidance:g}",
            "STUB - no model was run",
        ]
        for line in facts:
            draw.text((margin, y), line, font=body_font, fill=(235, 235, 235))
            y += int(body_size * 1.45)

        y += body_size
        text_width = w - 2 * margin
        for label, value in (
            ("prompt", params.final_prompt),
            ("negative", params.negative_prompt or "(none)"),
        ):
            draw.text((margin, y), f"{label}:", font=body_font, fill=accent)
            y += int(body_size * 1.45)
            for line in _wrap(draw, value, body_font, text_width)[:6]:
                draw.text((margin, y), line, font=body_font, fill=(215, 215, 215))
                y += int(body_size * 1.35)
            y += body_size // 2

        return image
