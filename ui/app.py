from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import json
import re
import shutil
import subprocess
import sys
from urllib.parse import urlparse
import textwrap

import pandas as pd
import streamlit as st
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = BASE_DIR / "config" / "settings.yaml"
TG_PATH = BASE_DIR / "config" / "telegram_channels.txt"


# -----------------------------
# Helpers
# -----------------------------

def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _save_yaml(path: Path, data: dict) -> None:
    if path.exists():
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
        )


def _normalize_rss_rows(df: pd.DataFrame) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    invalid_urls: list[str] = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        name = ("" if pd.isna(row.get("name")) else str(row.get("name")).strip())
        url = ("" if pd.isna(row.get("url")) else str(row.get("url")).strip())
        enabled = bool(row.get("enabled", True))

        if not name and not url:
            continue

        if url:
            if not (url.startswith("http://") or url.startswith("https://")):
                invalid_urls.append(url)
                continue
            if not name:
                parsed = urlparse(url)
                host = (parsed.netloc or "").lower().lstrip("www.")
                name = host or url
        else:
            invalid_urls.append("(missing url)")
            continue

        url_key = url.lower()
        if url_key in seen:
            continue
        seen.add(url_key)

        rows.append({"name": name, "url": url, "enabled": bool(enabled)})

    return rows, invalid_urls


def _load_tg_channels(path: Path) -> list[str]:
    if not path.exists():
        return []
    channels: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            s = s.lstrip("@").strip()
            if s:
                channels.append(s)
    # Deduplicate, keep order
    seen: set[str] = set()
    out: list[str] = []
    for c in channels:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _validate_tg_channels(channels: list[str]) -> list[str]:
    invalid: list[str] = []
    for c in channels:
        if not re.fullmatch(r"[A-Za-z0-9_]{3,32}", c):
            invalid.append(c)
    return invalid


def _save_tg_channels(path: Path, channels: list[str]) -> None:
    if path.exists():
        shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
    with path.open("w", encoding="utf-8") as f:
        f.write("# Telegram channels (one per line, without @)\n")
        f.write(f"# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for c in channels:
            f.write(f"{c}\n")


def _run_cmd(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = proc.stdout or ""
    err = proc.stderr or ""
    if err:
        out = (out + "\n\n[stderr]\n" + err).strip()
    return proc.returncode, out.strip()


def _extract_saved_paths(text: str) -> list[str]:
    paths: list[str] = []
    for line in text.splitlines():
        m = re.search(r"Saved .* to (/.+)$", line)
        if m:
            p = m.group(1).strip()
            if p and Path(p).exists():
                paths.append(p)
    # de-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _latest_files(limit: int = 10) -> list[dict]:
    specs = [
        ("raw", "data/raw/articles_*_rss_combined.json"),
        ("raw", "data/raw/articles_*_rss_*.json"),
        ("raw", "data/raw/articles_*_telegram*.json"),
        ("raw_enriched", "data/raw_enriched/*.json"),
        ("gold", "data/gold/*.parquet"),
    ]
    files: list[tuple[str, Path]] = []
    for kind, pat in specs:
        files.extend([(kind, p) for p in BASE_DIR.glob(pat)])
    # de-dup
    uniq: dict[str, tuple[str, Path]] = {}
    for kind, p in files:
        uniq[str(p.resolve())] = (kind, p)
    paths = list(uniq.values())
    paths.sort(key=lambda kp: kp[1].stat().st_mtime, reverse=True)
    out: list[dict] = []
    for kind, p in paths[:limit]:
        stt = p.stat()
        out.append(
            {
                "kind": kind,
                "path": str(p),
                "size_kb": round(stt.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stt.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    return out


def _parse_rss_stats(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        m = re.search(
            r"\[RSS\] raw_text stats for (.+?): total=(\d+), empty=(\d+), same_as_title=(\d+)",
            line,
        )
        if not m:
            continue
        src, total, empty, same = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        rows.append(
            {
                "source": src,
                "total": total,
                "empty": empty,
                "same_as_title": same,
                "same_ratio": round((same / total) if total else 0, 3),
            }
        )
    return rows


def _load_json_records(path: str, limit: int = 50) -> tuple[list[dict], str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return [], "Unsupported JSON structure."
        out = [d for d in data if isinstance(d, dict)]
        return out[:limit], ""
    except Exception as e:
        return [], f"Failed to read JSON: {type(e).__name__}: {e}"


def _load_parquet_records(path: str, limit: int = 50) -> tuple[list[dict], str]:
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return [], ""
        df = df.head(limit)
        out = df.to_dict(orient="records")
        return out, ""
    except Exception as e:
        return [], f"Failed to read Parquet: {type(e).__name__}: {e}"


def _load_records(path: str, limit: int = 50) -> tuple[list[dict], str]:
    if path.endswith(".parquet"):
        return _load_parquet_records(path, limit=limit)
    if path.endswith(".json"):
        return _load_json_records(path, limit=limit)
    return [], "Unsupported file type."


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _format_preview(text: str, width: int = 100) -> str:
    if not text:
        return ""
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False))


def _shorten_text(s: str, max_len: int = 200) -> str:
    if not s:
        return ""
    s = _normalize_text(str(s))
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _latest_gold_file() -> Path | None:
    paths = list(BASE_DIR.glob("data/gold/*.parquet"))
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def _pick_text_field(rec: dict) -> tuple[str, str, bool]:
    keys = ["raw_text", "description", "summary", "content", "body", "rss_text", "text"]
    title = _normalize_text(str(rec.get("title") or ""))
    best_text = ""
    best_key = ""
    fallback_text = ""
    fallback_key = ""

    for k in keys:
        v = rec.get(k)
        if not isinstance(v, str):
            continue
        t = _normalize_text(v)
        if not t:
            continue
        if title and t == title:
            if not fallback_text:
                fallback_text, fallback_key = t, k
            continue
        if len(t) > len(best_text):
            best_text, best_key = t, k

    if best_text:
        return best_text, best_key, False
    if fallback_text:
        return fallback_text, fallback_key, True
    return "", "", False


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Media Intel Hub - Sources", layout="wide")

st.title("Media Intelligence Hub - Sources")
st.caption("Local configuration for RSS and Telegram sources. No data leaves your machine.")

settings = _load_yaml(SETTINGS_PATH)
settings.setdefault("sources", {})
settings["sources"].setdefault("rss", [])

rss_list = settings.get("sources", {}).get("rss", []) or []

rss_df = pd.DataFrame(rss_list)
if rss_df.empty:
    rss_df = pd.DataFrame(columns=["name", "url", "enabled"])
else:
    for col in ["name", "url", "enabled"]:
        if col not in rss_df.columns:
            rss_df[col] = "" if col != "enabled" else True
    rss_df = rss_df[["name", "url", "enabled"]]

channels = _load_tg_channels(TG_PATH)


rss_tab, tg_tab = st.tabs(["RSS Sources", "Telegram Channels"])

with rss_tab:
    st.subheader("RSS feeds")
    st.write("Edit the list of RSS sources. You can add rows or disable a feed.")

    edited_df = st.data_editor(
        rss_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name", width="medium"),
            "url": st.column_config.TextColumn("URL", width="large"),
            "enabled": st.column_config.CheckboxColumn("Enabled", width="small"),
        },
    )

    normalized, invalid_urls = _normalize_rss_rows(edited_df)

    if invalid_urls:
        st.error("Invalid RSS URLs detected: " + ", ".join(sorted(set(invalid_urls))))

    cols = st.columns([1, 3, 3])
    with cols[0]:
        if st.button("Save RSS", type="primary", disabled=bool(invalid_urls)):
            settings.setdefault("sources", {})
            settings["sources"]["rss"] = normalized
            _save_yaml(SETTINGS_PATH, settings)
            st.success(f"Saved {len(normalized)} RSS sources to {SETTINGS_PATH}")
    with cols[1]:
        st.caption("Saving will create a backup (settings.yaml.bak). Comments in settings.yaml will be removed.")

with tg_tab:
    st.subheader("Telegram channels")
    st.write("One channel per line, without @. Example: `ostorozhno_novosti`")

    tg_text = st.text_area(
        "Channels",
        value="\n".join(channels),
        height=300,
        help="Lines starting with # are ignored. Spaces are trimmed.",
    )

    raw_lines = [ln.strip() for ln in tg_text.splitlines()]
    parsed = [ln.lstrip("@").strip() for ln in raw_lines if ln and not ln.startswith("#")]

    # Deduplicate, keep order
    seen: set[str] = set()
    tg_clean: list[str] = []
    for c in parsed:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        tg_clean.append(c)

    invalid_tg = _validate_tg_channels(tg_clean)
    if invalid_tg:
        st.error("Invalid channel names: " + ", ".join(invalid_tg))

    cols = st.columns([1, 3, 3])
    with cols[0]:
        if st.button("Save Telegram", type="primary", disabled=bool(invalid_tg)):
            _save_tg_channels(TG_PATH, tg_clean)
            st.success(f"Saved {len(tg_clean)} channels to {TG_PATH}")
    with cols[1]:
        st.caption("Saving will create a backup (telegram_channels.txt.bak). File header will be regenerated.")

st.divider()
st.subheader("Run collection")
st.caption("Runs collectors locally using the current config files.")

source_choice = st.selectbox("Source", ["RSS", "Telegram", "RSS + Telegram"])
save_local_only = st.checkbox("Save raw locally only (RAW_BACKEND=local)", value=True)
rss_combined = st.checkbox("RSS combined output (single file)", value=True)

if source_choice in {"Telegram", "RSS + Telegram"}:
    st.info("Telegram collection requires a working browser driver (Selenium) and may take longer.")

if st.button("Run collection", type="primary"):
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    if save_local_only:
        env["RAW_BACKEND"] = "local"

    commands: list[tuple[str, list[str]]] = []
    if source_choice in {"RSS", "RSS + Telegram"}:
        cmd = [sys.executable, "-m", "src.collectors.rss_collector"]
        if rss_combined:
            cmd.append("--combined")
        commands.append(("RSS", cmd))
    if source_choice in {"Telegram", "RSS + Telegram"}:
        cmd = [
            sys.executable,
            "-m",
            "src.collectors.telegram_scraper",
            "--channels-file",
            str(TG_PATH),
        ]
        commands.append(("Telegram", cmd))

    for label, cmd in commands:
        with st.status(f"Running {label} collector...", expanded=True) as status:
            code, out = _run_cmd(cmd, env=env)
            if out:
                st.code(out)
                stats = _parse_rss_stats(out)
                if stats:
                    st.info("RSS raw_text stats (title-only detection):")
                    st.dataframe(stats, use_container_width=True, hide_index=True)
                    for row in stats:
                        if row["total"] and row["same_ratio"] >= 0.8:
                            st.warning(
                                f"{row['source']}: raw_text mostly equals title. "
                                "Use body enrichment (raw -> raw_enriched) for full text."
                            )
                saved = _extract_saved_paths(out)
                if saved:
                    st.success("Saved files:")
                    for p in saved:
                        st.code(p)
            if code == 0:
                status.update(label=f"{label} collector finished", state="complete")
            else:
                status.update(label=f"{label} collector failed (exit {code})", state="error")

st.subheader("NLP (gold) quick view")
gold_path = _latest_gold_file()
col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("Run NLP for latest data (RSS)", type="primary"):
        env = os.environ.copy()
        env["PYTHON"] = sys.executable
        env["SOURCE"] = "rss"
        cmd = [sys.executable, "-m", "src.pipeline.etl_latest"]
        with st.status("Running NLP pipeline (raw/silver -> gold)...", expanded=True) as status:
            code, out = _run_cmd(cmd, env=env)
            if out:
                st.code(out)
            if code == 0:
                status.update(label="NLP pipeline finished", state="complete")
            else:
                status.update(label=f"NLP pipeline failed (exit {code})", state="error")

with col_b:
    if gold_path:
        st.caption("Latest gold file:")
        st.code(str(gold_path))
    else:
        st.info("No gold files yet. Run the NLP pipeline to generate data/gold/*.parquet.")

gold_path = _latest_gold_file()
if gold_path:
    records, err = _load_records(str(gold_path), limit=50)
    if err:
        st.error(err)
    elif records:
        df_gold = pd.DataFrame(records)
        show_cols = [
            c
            for c in [
                "title",
                "source",
                "published_at",
                "summary",
                "sentiment_label",
                "sentiment_score",
                "entities_persons",
                "entities_orgs",
                "entities_geo",
                "keywords",
            ]
            if c in df_gold.columns
        ]
        df_show = df_gold[show_cols].copy()
        for c in ["summary", "entities_persons", "entities_orgs", "entities_geo", "keywords"]:
            if c in df_show.columns:
                df_show[c] = df_show[c].map(lambda v: _shorten_text(v, 240))
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        def _gold_label(i: int) -> str:
            r = records[i]
            t = str(r.get("title") or r.get("link") or f"row {i}")
            t = t.replace("\n", " ").strip()
            return f"{i}: {t[:120]}"

        idx = st.selectbox("NLP record", list(range(len(records))), format_func=_gold_label, key="nlp_record")
        st.json(records[idx])
    else:
        st.info("Gold file is empty.")

st.subheader("Latest files (raw / raw_enriched / gold)")
latest = _latest_files(limit=10)
if latest:
    st.dataframe(latest, use_container_width=True, hide_index=True)
    st.subheader("Preview file")
    paths = [row["path"] for row in latest]
    selected = st.selectbox("File", paths)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        limit = st.slider("Rows to preview", min_value=5, max_value=200, value=50, step=5)
    with col_b:
        force_fetch = st.checkbox("Force fetch body", value=True, help="Fetch body even if raw_text exists.")
        enrich_limit = st.number_input("Enrich limit", min_value=1, max_value=5000, value=200, step=50)

    can_enrich = selected.endswith(".json")
    if st.button("Run body enrichment for selected file", disabled=not can_enrich):
        env = os.environ.copy()
        env["PYTHON"] = sys.executable
        cmd = [
            sys.executable,
            "-m",
            "src.pipeline.enrich_raw_with_body_local",
            "--input",
            selected,
            "--limit",
            str(int(enrich_limit)),
        ]
        if force_fetch:
            cmd.append("--force-fetch")
        with st.status("Running body enrichment...", expanded=True) as status:
            code, out = _run_cmd(cmd, env=env)
            if out:
                st.code(out)
                saved = _extract_saved_paths(out)
                if saved:
                    st.success("Enriched files:")
                    for p in saved:
                        st.code(p)
            if code == 0:
                status.update(label="Body enrichment finished", state="complete")
            else:
                status.update(label=f"Body enrichment failed (exit {code})", state="error")

    if not can_enrich:
        st.caption("Body enrichment works only for raw JSON files.")

    records, err = _load_records(selected, limit=limit)
    if err:
        st.error(err)
    elif records:
        df_prev = pd.DataFrame(records)
        cols = [c for c in ["title", "source", "published_at", "link"] if c in df_prev.columns]
        nlp_cols = [
            c
            for c in [
                "summary",
                "keywords",
                "entities_persons",
                "entities_orgs",
                "entities_geo",
                "sentiment_label",
                "sentiment_score",
            ]
            if c in df_prev.columns
        ]
        mode = st.radio("View mode", ["Table", "Cards"], horizontal=True)

        if mode == "Table":
            show_nlp = st.checkbox("Show NLP columns", value=bool(nlp_cols))
            show_cols = cols + (nlp_cols if show_nlp else [])
            st.dataframe(df_prev[show_cols], use_container_width=True, hide_index=True)

            def _label(i: int) -> str:
                r = records[i]
                t = str(r.get("title") or r.get("link") or f"row {i}")
                t = t.replace("\n", " ").strip()
                return f"{i}: {t[:120]}"

            idx = st.selectbox("Row", list(range(len(records))), format_func=_label)
            rec = records[idx]

            st.subheader("Details")
            link = rec.get("link") if isinstance(rec.get("link"), str) else ""
            if link:
                st.link_button("Open link", link)
            st.json(rec)

            st.subheader("Text preview")
            n_chars = st.slider("Preview length (chars)", min_value=200, max_value=4000, value=800, step=200)
            text, field, same_as_title = _pick_text_field(rec)
            if text:
                st.caption(f"Text source: {field}" + (" (same as title)" if same_as_title else ""))
                if same_as_title:
                    st.info(
                        "This RSS source provides only the title in raw_text. "
                        "To fetch full article text, run body enrichment (raw -> raw_enriched)."
                    )
                st.text(_format_preview(text[:n_chars], width=100))
            else:
                st.info("No text field found in this record.")
        else:
            max_cards = min(200, len(records))
            if max_cards <= 1:
                cards_n = 1
                st.caption("Showing 1 card (only one record available).")
            else:
                min_cards = 1
                step_cards = 1 if max_cards < 5 else 5
                default_cards = min(20, max_cards)
                cards_n = st.slider(
                    "Cards to show",
                    min_value=min_cards,
                    max_value=max_cards,
                    value=default_cards,
                    step=step_cards,
                )
            n_chars = st.slider(
                "Card text preview (chars)",
                min_value=50,
                max_value=4000,
                value=300,
                step=50,
                help="Applies to the text field (raw_text/description/etc). Titles are always full.",
            )
            for rec in records[:cards_n]:
                title = str(rec.get("title") or "").strip()
                source = str(rec.get("source") or "").strip()
                published_at = str(rec.get("published_at") or "").strip()
                link = rec.get("link") if isinstance(rec.get("link"), str) else ""
                text, field, same_as_title = _pick_text_field(rec)
                text_len = len(text) if isinstance(text, str) else 0

                with st.container(border=True):
                    st.markdown(f"**{title or 'Untitled'}**")
                    meta = " | ".join([v for v in [source, published_at] if v])
                    if meta:
                        st.caption(meta)
                    if link:
                        st.link_button("Open link", link)
                    if text:
                        caption = f"Text source: {field} | {text_len} chars"
                        if same_as_title:
                            caption += " | same as title"
                        st.caption(caption)
                        if same_as_title:
                            st.caption("Full article text requires body enrichment.")
                        st.text(_format_preview(text[:n_chars], width=100))
                    else:
                        st.caption("No text preview.")
    else:
        st.info("No records to preview.")
else:
    st.info("No raw files found yet. After a successful run, results will appear in data/raw.")

st.caption("Tip: keep this app running while you update sources. Changes apply to the next pipeline run.")
