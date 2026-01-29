# src/processing/stopwords_build.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, List


@dataclass
class BuildStopwordsConfig:
    # Базовые списки
    manual_ru_path: Path = Path("config/stopwords/stopwords_ru.txt")
    auto_ru_path: Path = Path("config/stopwords/stopwords_auto_ru.txt")

    # Guardrails
    whitelist_ru_path: Path = Path("config/stopwords/whitelist_ru.txt")

    blacklist_global_path: Path = Path("config/stopwords/blacklist_global.txt")
    blacklist_ru_tg_path: Path = Path("config/stopwords/blacklist_ru_source_tg.txt")
    blacklist_ru_rss_path: Path = Path("config/stopwords/blacklist_ru_source_rss.txt")

    # Авто-источниковые стоп-листы (генерятся пайплайном)
    blacklist_ru_tg_auto_path: Path = Path("config/stopwords/blacklist_ru_source_tg_auto.txt")
    blacklist_ru_rss_auto_path: Path = Path("config/stopwords/blacklist_ru_source_rss_auto.txt")

    # Авто-доменные стоп-листы RSS (генерятся пайплайном)
    # ВАЖНО (Вариант B): эти списки НЕ должны смешиваться с глобальными стоп-словами.
    # Они применяются только для своего домена на этапе очистки текста.
    rss_domains_dir: Path = Path("config/stopwords/rss_domains")


def _read_wordlist(path: Path) -> Set[str]:
    """
    Читает txt со словами построчно.
    - пустые строки игнорируем
    - строки с # в начале считаем комментами
    - всё приводим к lower()
    """
    if not path.exists():
        return set()

    out: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.add(s.lower())
    return out


def _read_wordlists_from_dir(dir_path: Path, glob_pat: str) -> Set[str]:
    """
    Читает все списки из директории по glob-шаблону и объединяет в один set.
    Полезно для доменных авто-списков, но НЕ использовать для глобального стоп-листа (вариант B).
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return set()

    out: Set[str] = set()
    for p in sorted(dir_path.glob(glob_pat)):
        out |= _read_wordlist(p)
    return out


def build_stopwords_ru_v001(cfg: BuildStopwordsConfig) -> List[str]:
    """
    v001: (manual + auto + базовые blacklists) - whitelist
    (без auto source-stopwords и без доменных auto)
    """
    manual = _read_wordlist(cfg.manual_ru_path)
    auto = _read_wordlist(cfg.auto_ru_path)

    whitelist = _read_wordlist(cfg.whitelist_ru_path)

    black = set()
    black |= _read_wordlist(cfg.blacklist_global_path)
    black |= _read_wordlist(cfg.blacklist_ru_tg_path)
    black |= _read_wordlist(cfg.blacklist_ru_rss_path)

    final = (manual | auto | black) - whitelist
    return sorted(final)


def build_stopwords_ru_v002(cfg: BuildStopwordsConfig) -> List[str]:
    """
    v002 (Вариант B): (manual + auto + базовые blacklists + auto source-blacklists) - whitelist

    Доменные RSS auto-blacklists НЕ добавляем в общий список:
    они применяются только для конкретного домена при очистке текста.
    """
    manual = _read_wordlist(cfg.manual_ru_path)
    auto = _read_wordlist(cfg.auto_ru_path)

    whitelist = _read_wordlist(cfg.whitelist_ru_path)

    black = set()
    black |= _read_wordlist(cfg.blacklist_global_path)
    black |= _read_wordlist(cfg.blacklist_ru_tg_path)
    black |= _read_wordlist(cfg.blacklist_ru_rss_path)

    # авто source-stopwords
    black |= _read_wordlist(cfg.blacklist_ru_tg_auto_path)
    black |= _read_wordlist(cfg.blacklist_ru_rss_auto_path)

    # ВАЖНО (Вариант B): НЕ смешиваем доменные RSS stopwords с глобальными
    # black |= _read_wordlists_from_dir(cfg.rss_domains_dir, "blacklist_ru_rss_*_auto.txt")

    final = (manual | auto | black) - whitelist
    return sorted(final)


def write_stopwords(path: Path, words: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    words_list = list(words)
    content = "\n".join(words_list) + ("\n" if words_list else "")
    path.write_text(content, encoding="utf-8")

def build_stopwords_ru_for_source(cfg: BuildStopwordsConfig, source: str) -> List[str]:
    """
    Вариант B:
    - базовые стоп-слова = v002 (без доменных)
    - + доменный auto-лист ТОЛЬКО для конкретного RSS источника (если файл есть)
    """
    base = set(build_stopwords_ru_v002(cfg))

    s = (source or "").strip().lower()
    if s.startswith("rss:"):
        dom = s[len("rss:"):]  # например "tvzvezda.ru"
        dom_path = cfg.rss_domains_dir / f"blacklist_ru_rss_{dom}_auto.txt"
        base |= _read_wordlist(dom_path)

    return sorted(base)