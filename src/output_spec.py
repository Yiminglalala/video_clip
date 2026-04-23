import os
from dataclasses import dataclass
from typing import Dict, List


LANDSCAPE_RESOLUTION_CHOICES = ("1920x1080", "1080x1440", "1080x1920")
DEFAULT_LANDSCAPE_RESOLUTION = "1920x1080"
SUBTITLE_FONT_FAMILY = "ZiYuWenYunTi"
STANDARD_VIDEO_OUTPUT_ARGS = [
    "-pix_fmt", "yuv420p",
    "-color_range", "tv",
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
]


@dataclass(frozen=True)
class OutputResolutionSpec:
    width: int
    height: int
    orientation: str
    label: str


def escape_ffmpeg_filter_path(path: str) -> str:
    return str(path or "").replace("\\", "/").replace(":", "\\:")


def get_subtitle_font_dir() -> str:
    candidates: List[str] = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")


def build_ass_filter_value(ass_path: str, post_filter: str = "") -> str:
    ass_escaped = escape_ffmpeg_filter_path(ass_path)
    font_dir = escape_ffmpeg_filter_path(get_subtitle_font_dir())
    base = f"ass='{ass_escaped}':fontsdir='{font_dir}'"
    if post_filter:
        return f"{base},{post_filter}"
    return base


def normalize_landscape_resolution_choice(choice: str) -> str:
    normalized = str(choice or "").strip().lower().replace("×", "x")
    if normalized in LANDSCAPE_RESOLUTION_CHOICES:
        return normalized
    return DEFAULT_LANDSCAPE_RESOLUTION


def resolve_output_resolution_spec(input_orientation: str, landscape_choice: str) -> OutputResolutionSpec:
    orientation = "portrait" if str(input_orientation or "").strip().lower() == "portrait" else "landscape"
    if orientation == "portrait":
        return OutputResolutionSpec(
            width=1080,
            height=1920,
            orientation="portrait",
            label="1080x1920",
        )

    choice = normalize_landscape_resolution_choice(landscape_choice)
    size_map: Dict[str, tuple[int, int]] = {
        "1920x1080": (1920, 1080),
        "1080x1440": (1080, 1440),
        "1080x1920": (1080, 1920),
    }
    width, height = size_map[choice]
    target_orientation = "portrait" if height > width else "landscape"
    return OutputResolutionSpec(
        width=width,
        height=height,
        orientation=target_orientation,
        label=choice,
    )


def build_cover_crop_filter(width: int, height: int, extra_filter: str = "") -> str:
    parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={width}:{height}:(in_w-{width})/2:(in_h-{height})/2",
        "format=yuv420p",
    ]
    if extra_filter:
        parts.append(extra_filter)
    return ",".join(parts)
