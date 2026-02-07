from __future__ import annotations

from pathlib import Path
import re
from typing import Any, List, Tuple

import streamlit as st
from ruamel.yaml import YAML

BASE_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = BASE_DIR / "config" / "settings.yaml"
TG_CHANNELS_PATH = BASE_DIR / "config" / "telegram_channels.txt"

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
TG_RE = re.compile(r"^[A-Za-z0-9_]{3,64}$")


yaml = YAML()
yaml.preserve_quotes = True

def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.load(f) or {}
    return data


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


def _get_rss_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    sources = data.get("sources") or {}
    rss = sources.get("rss") or []
    if not isinstance(rss, list):
        return []
    out: list[dict[str, Any]] = []
    for item in rss:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": str(item.get("name") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return out


def _set_rss_list(data: dict[str, Any], rss_list: list[dict[str, Any]]) -> None:
    if "sources" not in data or not isinstance(data.get("sources"), dict):
        data["sources"] = {}
    data["sources"]["rss"] = rss_list


def _validate_rss_list(rss_list: list[dict[str, Any]]) -> Tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(rss_list, start=1):
        name = str(item.get("name") or "").strip()
        url = str(item.get("url") or "").strip()
        enabled = bool(item.get("enabled", True))
        if not name:
            errors.append(f"RSS строка {i}: пустое поле name")
        if not url or not URL_RE.match(url):
            errors.append(f"RSS строка {i}: некорректный url")
        if name and url and URL_RE.match(url):
            cleaned.append({"name": name, "url": url, "enabled": enabled})
    return errors, cleaned


def _load_tg_channels(path: Path) -> list[str]:
    if not path.exists():
        return []
    channels: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            ch = raw.lstrip("@").strip()
            if ch:
                channels.append(ch)
    # de-dup while preserving order
    seen = set()
    out: list[str] = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out


def _save_tg_channels(path: Path, channels: list[str]) -> None:
    lines = ["# Telegram channels (one per line)"]
    lines.extend(channels)
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _parse_tg_text(text: str) -> Tuple[list[str], list[str]]:
    channels: list[str] = []
    errors: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        ch = raw.lstrip("@").strip()
        if not TG_RE.match(ch):
            errors.append(f"TG строка {i}: некорректный формат "
                         f"(допустимы A-Z, 0-9, _, длина 3-64)")
            continue
        channels.append(ch)
    # de-dup
    seen = set()
    out: list[str] = []
    for ch in channels:
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out, errors


def _rss_summary(rss_list: list[dict[str, Any]]) -> str:
    total = len(rss_list)
    enabled = sum(1 for r in rss_list if r.get("enabled"))
    return f"Всего: {total} · Включено: {enabled}"


st.set_page_config(page_title="Media Intelligence Hub — Sources", layout="wide")

st.title("Media Intelligence Hub — Sources")
st.caption("Локальный интерфейс для настройки RSS и Telegram источников")

# RSS section
st.subheader("RSS источники")

try:
    settings_data = _load_yaml(SETTINGS_PATH)
    rss_list = _get_rss_list(settings_data)
except Exception as exc:
    st.error(f"Не удалось прочитать settings.yaml: {exc}")
    st.stop()

st.write(_rss_summary(rss_list))

edited_rss = st.data_editor(
    rss_list,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("name", required=True),
        "url": st.column_config.TextColumn("url", required=True),
        "enabled": st.column_config.CheckboxColumn("enabled"),
    },
)

if st.button("Сохранить RSS", type="primary"):
    errors, cleaned = _validate_rss_list(edited_rss)
    if errors:
        st.error("\n".join(errors))
    else:
        _set_rss_list(settings_data, cleaned)
        _save_yaml(SETTINGS_PATH, settings_data)
        st.success("RSS источники сохранены")

st.divider()

# Telegram section
st.subheader("Telegram каналы")

channels = _load_tg_channels(TG_CHANNELS_PATH)

st.write(f"Каналов: {len(channels)}")

channels_text = st.text_area(
    "Введите каналы, по одному на строку (без @)",
    value="\n".join(channels),
    height=240,
)

if st.button("Сохранить Telegram", type="primary"):
    parsed, errors = _parse_tg_text(channels_text)
    if errors:
        st.error("\n".join(errors))
    else:
        _save_tg_channels(TG_CHANNELS_PATH, parsed)
        st.success("Telegram каналы сохранены")

st.caption(f"Файлы: {SETTINGS_PATH} · {TG_CHANNELS_PATH}")
