#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Telegram channel scraper for https://t.me/s/<channel>

Особенности:
- Автоопределение направления скролла (вверх/вниз) по числовым ID сообщений.
- Переход по пагинации ?before=<id> / ?after=<id> при отсутствии прогресса.
- Уникальность сообщений по data-post (резервы: URL, хеш outerHTML).
- Допускает пустой/короткий текст (медиа-посты не теряются).
- Явное ожидание роста числа карточек в DOM после скролла.
- Обрабатывает только действительно новые ID.
- CLI параметры: --channel, --target, --headless.

Зависимости:
  pip install selenium webdriver-manager
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# --- Selenium imports ---
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
except ImportError:
    print("Требуется: pip install selenium")
    sys.exit(1)

# --- WebDriver Manager (необязательно, но удобно) ---
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False


class RobustTelegramScraper:
    def __init__(
        self,
        channel: str,
        target_posts: int = 300,
        headless: bool = False,
        out_dir: str = "data/raw",
        debug_dir: str = "data/debug/telegram",
        mih_only: bool = False,
        mih_schema: str = "bundle",
        batch_id: str = None,
    ):
        self.channel = channel
        self.url = f"https://t.me/s/{self.channel}"
        self.target_posts = int(target_posts)
        self.headless = headless

        self.out_dir = out_dir
        self.debug_dir = debug_dir
        self.mih_only = mih_only
        self.mih_schema = mih_schema
        self.batch_id = batch_id or os.getenv("BATCH_ID") or datetime.now().isoformat()

        self.driver = None
        self.posts = []
        self.seen_ids = set()        # data-post / URL / hash
        self.processed_urls = set()  # для статистики
        self.batch_size = 100        # верхняя граница обработки за итерацию
        self._last_dom_count = 0
        self.scroll_direction = "down"  # auto: up/down

    # -------------------- Driver setup & page load --------------------

    def setup_driver(self) -> bool:
        print("[1/4] Настройка браузера...")
        options = Options()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1600,1200")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        if self.headless:
            options.add_argument("--headless=new")

        try:
            if USE_WEBDRIVER_MANAGER:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)

            self.driver.implicitly_wait(15)
            self.driver.set_page_load_timeout(90)
            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            print("Браузер запущен")
            return True
        except Exception as e:
            print(f"Ошибка запуска браузера: {e}")
            return False

    def load_page(self) -> bool:
        print(f"[2/4] Загрузка страницы канала @{self.channel} ...")
        try:
            self.driver.get(self.url)
            WebDriverWait(self.driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(4)  # небольшой буфер под сторонние скрипты

            html = self.driver.page_source
            if "This channel doesn't exist" in html:
                print("Ошибка: канал не существует")
                return False
            if "This channel is private" in html:
                print("Ошибка: канал приватный")
                return False

            # ждём появления карточек постов (иначе дальше скроллить бессмысленно)
            WebDriverWait(self.driver, 30).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, "div.tgme_widget_message")) > 0
            )

            # принудительно уходим в самый низ (в зону последних постов)
            self._force_scroll_to_bottom()

            title = self.driver.title
            size = len(self.driver.page_source)
            print(f"Страница: {title}")
            print(f"Размер HTML: {size:,} символов")
            return True
        except Exception as e:
            print(f"Ошибка загрузки страницы: {e}")
            return False

    # -------------------- Small helpers --------------------

    def is_driver_alive(self) -> bool:
        try:
            _ = self.driver.current_url
            return True
        except Exception:
            return False

    def get_posts_elements(self):
        try:
            return self.driver.find_elements(By.CSS_SELECTOR, "div.tgme_widget_message")
        except Exception:
            return []

    def _force_scroll_to_bottom(self):
        """Надёжно уводит страницу в самый низ (к последним постам)."""
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
        except Exception:
            pass

    def _get_id_and_url(self, element):
        """
        Возвращает (post_id, post_url). Предпочтительно берём data-post.
        Резервы: URL из .tgme_widget_message_date, либо хеш outerHTML.
        """
        post_id = ""
        post_url = ""

        try:
            data_post = element.get_attribute("data-post") or ""
            if data_post:
                try:
                    post_id = data_post.split("/")[-1].strip()
                except Exception:
                    post_id = data_post.strip()
        except Exception:
            pass

        if not post_id:
            try:
                link = element.find_element(By.CSS_SELECTOR, "a.tgme_widget_message_date")
                post_url = link.get_attribute("href") or ""
                if post_url:
                    post_id = post_url.rstrip("/").split("/")[-1]
            except Exception:
                pass

        if not post_id:
            try:
                outer = element.get_attribute("outerHTML")
                post_id = str(hash(outer))
            except Exception:
                post_id = str(id(element))

        if not post_url and post_id.isdigit():
            post_url = f"https://t.me/{self.channel}/{post_id}"

        return post_id, post_url

    def _get_message_text(self, element) -> str:
        """Аккуратно берём текст из .tgme_widget_message_text. Пустой текст допустим."""
        try:
            node = element.find_element(By.CSS_SELECTOR, ".tgme_widget_message_text")
            txt = node.text
            return txt.strip() if txt else ""
        except Exception:
            return ""

    def _make_title(self, text: str, max_len: int = 120) -> str:
        """Стабильный title для дальнейшего пайплайна (если текста нет — заглушка)."""
        if not text:
            return "Telegram пост"
        first_line = ""
        for ln in text.splitlines():
            ln = (ln or "").strip()
            if ln:
                first_line = ln
                break
        if not first_line:
            first_line = text.strip()

        if len(first_line) > max_len:
            return first_line[: max_len - 1] + "…"
        return first_line

    def _numeric_id_from_element(self, el):
        pid, _ = self._get_id_and_url(el)
        try:
            return int(pid)
        except Exception:
            return None

    def detect_scroll_direction(self) -> str:
        """
        Определяет, где "старее": сверху или снизу.
        Если снизу id меньше (старее), скроллим вниз; если сверху меньше — вверх.
        """
        elems = self.get_posts_elements()
        if len(elems) < 3:
            return "down"

        top_id = None
        for el in elems[:5]:
            top_id = self._numeric_id_from_element(el)
            if top_id is not None:
                break

        bottom_id = None
        for el in reversed(elems[-5:]):
            bottom_id = self._numeric_id_from_element(el)
            if bottom_id is not None:
                break

        if top_id is None or bottom_id is None:
            return "down"

        return "down" if bottom_id < top_id else "up"

    def _wait_for_dom_growth(self, timeout: int = 20) -> bool:
        """Ждём реального увеличения числа карточек сообщений в DOM."""
        start = time.time()
        initial = len(self.get_posts_elements())
        while time.time() - start < timeout:
            cnt = len(self.get_posts_elements())
            if cnt > initial:
                self._last_dom_count = cnt
                return True
            time.sleep(0.5)
        return False

    def _scroll_to_bottom(self):
        elems = self.get_posts_elements()
        if elems:
            try:
                _ = elems[-1].location_once_scrolled_into_view
            except Exception:
                pass
        try:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except Exception:
            pass

    def _scroll_to_top_chunk(self):
        elems = self.get_posts_elements()
        if elems:
            try:
                _ = elems[0].location_once_scrolled_into_view
            except Exception:
                pass
        try:
            self.driver.execute_script("window.scrollBy(0, -Math.max(800, window.innerHeight));")
        except Exception:
            pass

    def _scroll_more(self):
        if self.scroll_direction == "up":
            self._scroll_to_top_chunk()
        else:
            self._scroll_to_bottom()

    def _jump_via_before(self) -> bool:
        """
        Пагинация назад: ...?before=<min_collected_id> (старее).
        """
        numeric_ids = [int(p["post_id"]) for p in self.posts if str(p.get("post_id", "")).isdigit()]
        if not numeric_ids:
            return False

        min_id = min(numeric_ids)
        if min_id <= 1:
            return False  # тупик: дальше "до 1" уходить некуда

        jump_url = f"{self.url}?before={min_id}"
        print(f"Переход по пагинации: {jump_url}")
        try:
            self.driver.get(jump_url)
            time.sleep(4)
            self._last_dom_count = len(self.get_posts_elements())
            return True
        except Exception:
            return False

    def _jump_via_after(self) -> bool:
        """
        Пагинация вперёд: ...?after=<max_collected_id> (новее).
        Полезно, если мы случайно оказались слишком близко к началу канала.
        """
        numeric_ids = [int(p["post_id"]) for p in self.posts if str(p.get("post_id", "")).isdigit()]
        if not numeric_ids:
            return False

        max_id = max(numeric_ids)
        jump_url = f"{self.url}?after={max_id}"
        print(f"Переход по пагинации: {jump_url}")
        try:
            self.driver.get(jump_url)
            time.sleep(4)
            self._last_dom_count = len(self.get_posts_elements())
            return True
        except Exception:
            return False

    # -------------------- Extraction --------------------

    def extract_post_data(self, element):
        try:
            if not self.is_driver_alive():
                return None

            post_id, post_url = self._get_id_and_url(element)
            if post_id in self.seen_ids:
                return None

            text = self._get_message_text(element)

            datetime_str = ""
            human_date = "Неизвестно"
            try:
                t = element.find_element(By.CSS_SELECTOR, "time")
                datetime_str = t.get_attribute("datetime") or ""
                human_date = (t.text or "").strip() or human_date
            except Exception:
                try:
                    t2 = element.find_element(By.CSS_SELECTOR, ".tgme_widget_message_date")
                    human_date = (t2.text or "").strip() or human_date
                except Exception:
                    pass

            views = "0"
            try:
                v = element.find_element(By.CSS_SELECTOR, ".tgme_widget_message_views")
                views = (v.text or "").strip() or "0"
            except Exception:
                pass

            try:
                outer = (element.get_attribute("outerHTML") or "").lower()
                has_video = "video" in outer
                has_photo = ("tgme_widget_message_photo" in outer) or ("img" in outer)
            except Exception:
                has_video = False
                has_photo = False

            self.seen_ids.add(post_id)
            if post_url:
                self.processed_urls.add(post_url)

            title = self._make_title(text)
            source = f"telegram:{self.channel}"
            uid = f"tg:{self.channel}:{post_id}"
            link = post_url

            return {
                "id": len(self.posts) + 1,
                "uid": uid,
                "source": source,
                "channel": self.channel,
                "link": link,
                "title": title,
                "published_at": datetime_str,
                "content": text or "",
                "post_id": post_id,
                "datetime": datetime_str,
                "human_date": human_date,
                "text": text or "",
                "views": views,
                "has_video": has_video,
                "has_photo": has_photo,
                "text_length": len(text or ""),
                "post_url": post_url,
                "extracted_at": datetime.now().isoformat(),
            }
        except Exception as e:
            print(f"Ошибка извлечения: {e}")
            return None

    # -------------------- Main scroll loop --------------------

    def scroll_and_extract(self) -> int:
        print("[3/4] Прокрутка и извлечение данных...")

        # На всякий случай ещё раз уходим в низ: снижаем шанс старта "в начале ленты"
        self._force_scroll_to_bottom()

        print("Определение направления скролла...")
        self.scroll_direction = self.detect_scroll_direction()
        print(f"Скроллим: {self.scroll_direction.upper()}")

        attempts = 0
        max_attempts = 400
        failures = 0
        max_failures = 20

        self._last_dom_count = len(self.get_posts_elements())

        while attempts < max_attempts and failures < max_failures and len(self.posts) < self.target_posts:
            if not self.is_driver_alive():
                print("Браузер недоступен — остановка")
                break

            attempts += 1
            collected_before = len(self.posts)
            elems = self.get_posts_elements()
            print(f"Итерация #{attempts}: элементов в DOM = {len(elems)}, собрано = {collected_before}")

            to_process = []
            for el in elems:
                pid, _ = self._get_id_and_url(el)
                if pid not in self.seen_ids:
                    to_process.append(el)

            # Ограничим размер партии: при UP важнее верх, при DOWN — низ
            if len(to_process) > self.batch_size:
                if self.scroll_direction == "up":
                    to_process = to_process[: self.batch_size]
                else:
                    to_process = to_process[-self.batch_size :]

            new_count = 0
            for el in to_process:
                item = self.extract_post_data(el)
                if item:
                    self.posts.append(item)
                    new_count += 1
                    if len(self.posts) % 25 == 0:
                        self.save_intermediate_results()
                    if len(self.posts) >= self.target_posts:
                        print(f"Достигнута цель: {len(self.posts)} постов")
                        return len(self.posts)

            print(f"Новых постов за итерацию: {new_count}")

            self._scroll_more()
            grew = self._wait_for_dom_growth(timeout=25 if len(self.posts) < 100 else 15)

            if new_count == 0 and not grew:
                failures += 1
                print(f"Нет прогресса ({failures}/{max_failures}). Пробуем альтернативы.")
                try:
                    delta = 6000 if self.scroll_direction == "down" else -6000
                    self.driver.execute_script(f"window.scrollBy(0, {delta});")
                    time.sleep(2.0)
                except Exception:
                    pass

                if failures >= 3:
                    numeric_ids = [int(p["post_id"]) for p in self.posts if str(p.get("post_id", "")).isdigit()]
                    min_id = min(numeric_ids) if numeric_ids else None

                    if min_id is not None and min_id <= 5:
                        jumped = self._jump_via_after()
                    else:
                        jumped = self._jump_via_before()

                    if jumped:
                        failures = 0
            else:
                failures = 0

            time.sleep(1.2 if len(self.posts) >= 100 else 2.0)

        print(
            f"Прокрутка завершена. Попыток: {attempts}, собрано: {len(self.posts)}, подряд без прогресса: {failures}"
        )
        if failures >= max_failures:
            print("Остановлено по лимиту отсутствия прогресса")
        return len(self.posts)

    # -------------------- Save & stats --------------------

    def save_intermediate_results(self):
        if self.mih_only:
            return

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.debug_dir, exist_ok=True)
            path = os.path.join(self.debug_dir, f"temp_results_{len(self.posts)}_posts_{ts}.json")
            payload = {
                "intermediate_save": True,
                "timestamp": datetime.now().isoformat(),
                "posts_count": len(self.posts),
                "posts": self.posts,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Промежуточное сохранение: {path}")
        except Exception as e:
            print(f"Ошибка промежуточного сохранения: {e}")

    def save_mih_raw(self, ts: str):
        """Сохраняет raw JSON в формате, удобном для MIH (raw → silver).

        ВАЖНО: добавляем raw_text/raw_title/raw_url, потому что текущий raw→silver
        ожидает колонку raw_text.

        Схемы:
        - list: просто список records
        - bundle: объект с метаданными и ключом items
        """
        os.makedirs(self.out_dir, exist_ok=True)

        records = []
        now_iso = datetime.now().isoformat()

        for p in self.posts:
            content = p.get("content") or p.get("text") or ""
            title = p.get("title") or self._make_title(content)
            link = p.get("link") or p.get("post_url") or ""
            published_at = p.get("published_at") or p.get("datetime") or ""

            records.append(
                {
                    # Универсальные поля (как было)
                    "uid": p.get("uid"),
                    "source": p.get("source") or f"telegram:{self.channel}",
                    "channel": p.get("channel") or self.channel,
                    "link": link,
                    "title": title,
                    "published_at": published_at,
                    "content": content,
                    "batch_id": self.batch_id,

                    # Контракт под текущий raw→silver
                    "raw_text": content,
                    "raw_title": title,
                    "raw_url": link,
                    "raw_published_at": published_at,

                    "meta": {
                        "tg_post_id": p.get("post_id"),
                        "views": p.get("views"),
                        "has_photo": p.get("has_photo"),
                        "has_video": p.get("has_video"),
                        "human_date": p.get("human_date"),
                        "extracted_at": p.get("extracted_at"),
                        "post_url": p.get("post_url"),
                    },
                }
            )

        out_name = f"articles_{ts}_telegram_{self.channel}.json"
        out_path = os.path.join(self.out_dir, out_name)

        if str(self.mih_schema).lower() == "list":
            payload = records
        else:
            payload = {
                "batch_id": self.batch_id,
                "ingested_at": now_iso,
                "source": "telegram",
                "channel": self.channel,
                "url": self.url,
                "items_count": len(records),
                "items": records,
            }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"MIH RAW: {out_path}")
        return out_path

    def save_final_results(self):
        if not self.posts:
            print("Нет постов для сохранения")
            return None, None, None

        print("[4/4] Сохранение результатов...")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        mih_path = self.save_mih_raw(ts)

        if self.mih_only:
            self.show_statistics()
            return mih_path, None, None

        os.makedirs(self.debug_dir, exist_ok=True)
        base = f"{self.channel}_robust_{len(self.posts)}_posts_{ts}"

        result = {
            "scraping_info": {
                "timestamp": datetime.now().isoformat(),
                "total_posts": len(self.posts),
                "target_posts": self.target_posts,
                "scraper_version": "v1.5",
                "url": self.url,
                "scroll_direction": self.scroll_direction,
                "extraction_method": "scroll + ?before/?after fallback",
            },
            "posts": self.posts,
        }

        json_path = os.path.join(self.debug_dir, f"{base}.json")
        txt_path = os.path.join(self.debug_dir, f"{base}.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"КАНАЛ @{self.channel}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Дата сбора: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Всего постов: {len(self.posts)}\n")
            f.write(f"Метод: {result['scraping_info']['extraction_method']}\n")
            f.write(f"Направление скролла: {self.scroll_direction}\n")
            f.write("=" * 60 + "\n\n")
            for p in self.posts:
                f.write(f"ПОСТ #{p['id']:3d} | {p.get('human_date', '')}\n")
                if p.get("post_id"):
                    f.write(f"ID: {p['post_id']}\n")
                if p.get("post_url"):
                    f.write(f"Ссылка: {p['post_url']}\n")
                f.write(f"Просмотры: {p.get('views')}\n")
                f.write(f"Длина: {p.get('text_length')} символов\n")
                media = []
                if p.get("has_video"):
                    media.append("Видео")
                if p.get("has_photo"):
                    media.append("Фото")
                if media:
                    f.write(f"Медиа: {' '.join(media)}\n")
                f.write("\n")
                f.write((p.get("text") or "") + "\n")
                f.write("-" * 60 + "\n\n")

        print(f"JSON: {json_path}")
        print(f"TXT:  {txt_path}")

        self.show_statistics()
        return mih_path, json_path, txt_path

    def show_statistics(self):
        if not self.posts:
            return
        print("\nСтатистика")
        print("=" * 40)
        total = len(self.posts)
        avg_len = int(sum(p.get("text_length", 0) for p in self.posts) / total) if total else 0
        max_len = max((p.get("text_length", 0) for p in self.posts), default=0)
        min_len = min((p.get("text_length", 0) for p in self.posts), default=0)
        print(f"Всего постов: {total}")
        print(f"Средняя длина: {avg_len} символов")
        print(f"Диапазон: {min_len} — {max_len} символов")
        videos = sum(1 for p in self.posts if p.get("has_video"))
        photos = sum(1 for p in self.posts if p.get("has_photo"))
        print(f"С видео: {videos} ({videos/total*100:.1f}%)")
        print(f"С фото:  {photos} ({photos/total*100:.1f}%)")
        with_dates = sum(1 for p in self.posts if p.get("datetime"))
        with_views = sum(1 for p in self.posts if p.get("views") != "0")
        with_urls = sum(1 for p in self.posts if p.get("post_url"))
        print(f"С датами:   {with_dates} ({with_dates/total*100:.1f}%)")
        print(f"С просмотрами: {with_views} ({with_views/total*100:.1f}%)")
        print(f"Со ссылками: {with_urls} ({with_urls/total*100:.1f}%)")
        print(f"Уникальных URL: {len(self.processed_urls)}")

    def cleanup(self):
        try:
            if self.driver:
                print("\nЗакрытие браузера...")
                self.driver.quit()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description="Robust Telegram scraper for t.me/s/<channel>")

    p.add_argument(
        "--channel",
        type=str,
        default=os.getenv("TG_CHANNEL", "make_your_style"),
        help="Имя канала без @ (fallback, если не задано --channels/--channels-file)",
    )

    p.add_argument(
        "--channels",
        type=str,
        default=os.getenv("TG_CHANNELS", ""),
        help="Список каналов без @ через запятую (например: 'lenta_ru,meduza')",
    )

    p.add_argument(
        "--channels-file",
        type=str,
        default=os.getenv("TG_CHANNELS_FILE", ""),
        help="Файл со списком каналов (по одному на строку, поддерживает комментарии #)",
    )

    p.add_argument(
        "--target",
        type=int,
        default=int(os.getenv("TG_TARGET", "300")),
        help="Целевое число постов на канал (по умолчанию: 300)",
    )

    p.add_argument("--headless", action="store_true", help="Запуск браузера в headless режиме")

    p.add_argument(
        "--out-dir",
        type=str,
        default=os.getenv("TG_OUT_DIR", "data/raw"),
        help="Куда сохранять raw JSON для MIH (по умолчанию: data/raw)",
    )

    p.add_argument(
        "--debug-dir",
        type=str,
        default=os.getenv("TG_DEBUG_DIR", "data/debug/telegram"),
        help="Куда сохранять debug JSON/TXT (по умолчанию: data/debug/telegram)",
    )

    p.add_argument("--mih-only", action="store_true", help="Сохранять только MIH raw (без debug JSON/TXT)")

    p.add_argument(
        "--mih-schema",
        type=str,
        default=os.getenv("TG_MIH_SCHEMA", "bundle"),
        choices=["bundle", "list"],
        help="Формат MIH raw: bundle (с метаданными и items) или list (просто список записей)",
    )

    p.add_argument("--combined", action="store_true", help="Дополнительно сохранить общий MIH raw файл по всем каналам")

    return p.parse_args()


def _resolve_channels(args) -> list:
    channels = []

    if args.channels:
        parts = [p.strip().lstrip("@") for p in args.channels.split(",")]
        channels.extend([c for c in parts if c])

    if args.channels_file:
        try:
            with open(args.channels_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line or line.startswith("#"):
                        continue
                    channels.append(line.lstrip("@"))
        except Exception as e:
            print(f"Не удалось прочитать channels-file: {e}")

    if not channels:
        channels = [args.channel.lstrip("@")]

    uniq = []
    seen = set()
    for c in channels:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    return uniq


def main():
    args = parse_args()
    channels = _resolve_channels(args)

    print("Robust Telegram Scraper (t.me/s/<channel>)")
    print(f"Каналов: {len(channels)}")
    print("Каналы: " + ", ".join([f"@{c}" for c in channels]))
    print(f"Цель: {args.target} постов на канал")
    print(f"Headless: {args.headless}")
    print(f"MIH out-dir: {args.out_dir}")
    print(f"Debug dir:  {args.debug_dir}")
    print(f"MIH only:   {args.mih_only}")
    print("=" * 60)

    batch_id = os.getenv("BATCH_ID") or datetime.now().isoformat()
    mih_paths = []

    try:
        for ch in channels:
            print("=" * 60)
            print(f"[START] @{ch}")

            scraper = RobustTelegramScraper(
                channel=ch,
                target_posts=args.target,
                headless=args.headless,
                out_dir=args.out_dir,
                debug_dir=args.debug_dir,
                mih_only=args.mih_only,
                mih_schema=args.mih_schema,
                batch_id=batch_id,
            )

            try:
                if not scraper.setup_driver():
                    continue
                if not scraper.load_page():
                    continue

                count = scraper.scroll_and_extract()

                if count > 0:
                    mih_path, debug_json, debug_txt = scraper.save_final_results()
                    if mih_path:
                        mih_paths.append(mih_path)
                else:
                    print("Не удалось извлечь посты")

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Ошибка по каналу @{ch}: {e}")
            finally:
                scraper.cleanup()

    except KeyboardInterrupt:
        print("\nПрервано пользователем")

    if args.combined and mih_paths:
        combined = []
        for pth in mih_paths:
            try:
                with open(pth, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    if isinstance(obj, dict) and "items" in obj:
                        combined.extend(obj.get("items") or [])
                    elif isinstance(obj, list):
                        combined.extend(obj)
                    else:
                        print(f"Неожиданный формат в {pth}: {type(obj)}")
            except Exception as e:
                print(f"Не удалось прочитать {pth}: {e}")

        if combined:
            os.makedirs(args.out_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(args.out_dir, f"articles_{ts}_telegram_combined.json")
            with open(out, "w", encoding="utf-8") as f:
                if str(args.mih_schema).lower() == "list":
                    payload = combined
                else:
                    payload = {
                        "batch_id": batch_id,
                        "ingested_at": datetime.now().isoformat(),
                        "source": "telegram",
                        "channels": channels,
                        "items_count": len(combined),
                        "items": combined,
                    }
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"COMBINED MIH RAW: {out}")


if __name__ == "__main__":
    main()