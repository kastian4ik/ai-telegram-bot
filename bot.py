import os
import io
import re
import time
import sqlite3
import logging
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, date
from urllib.parse import quote
from contextlib import asynccontextmanager

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    LabeledPrice,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    PreCheckoutQueryHandler,
    filters,
)

from google import genai


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "bot.db"

load_dotenv(ENV_PATH)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")

TEXT_MODEL = os.getenv("TEXT_MODEL", "gemini-2.5-flash-lite")
FILE_MODEL = os.getenv("FILE_MODEL", "gemini-2.5-flash-lite")
VOICE_MODEL = os.getenv("VOICE_MODEL", "gemini-2.5-flash-lite")
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты полезный AI-ассистент в Telegram. Отвечай понятно, дружелюбно и по делу. "
    "Не начинай каждый ответ с приветствия или имени пользователя без необходимости."
)

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
ADMIN_ID = os.getenv("ADMIN_ID")

# антиспам
MIN_SECONDS_BETWEEN_AI_REQUESTS = float(os.getenv("MIN_SECONDS_BETWEEN_AI_REQUESTS", "2"))
MIN_SECONDS_BETWEEN_USER_MESSAGES = float(os.getenv("MIN_SECONDS_BETWEEN_USER_MESSAGES", "2"))

# free лимиты
FREE_TEXT_LIMIT = int(os.getenv("FREE_TEXT_LIMIT", "60"))
FREE_VOICE_LIMIT = int(os.getenv("FREE_VOICE_LIMIT", "8"))
FREE_IMAGE_LIMIT = int(os.getenv("FREE_IMAGE_LIMIT", "8"))
FREE_FILE_LIMIT = int(os.getenv("FREE_FILE_LIMIT", "5"))

# pro лимиты
PRO_TEXT_LIMIT = int(os.getenv("PRO_TEXT_LIMIT", "999999"))
PRO_VOICE_LIMIT = int(os.getenv("PRO_VOICE_LIMIT", "25"))
PRO_IMAGE_LIMIT = int(os.getenv("PRO_IMAGE_LIMIT", "25"))
PRO_FILE_LIMIT = int(os.getenv("PRO_FILE_LIMIT", "15"))

# кэш
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "21600"))

# stars подписка
PRO_PLAN_NAME = "pro"
PRO_PRICE_STARS = int(os.getenv("PRO_PRICE_STARS", "199"))
PRO_DURATION_DAYS = int(os.getenv("PRO_DURATION_DAYS", "30"))

if not TELEGRAM_TOKEN:
    raise ValueError("Не найден TELEGRAM_TOKEN")
if not GEMINI_API_KEY:
    raise ValueError("Не найден GEMINI_API_KEY")
if not WEBHOOK_URL:
    raise ValueError("Не найден WEBHOOK_URL")

logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

last_ai_request_time = 0.0
ai_request_lock = asyncio.Lock()
user_cooldowns: dict[int, float] = {}


# =========================
# LIMIT PLANS
# =========================
FREE_LIMITS = {
    "text": FREE_TEXT_LIMIT,
    "voice": FREE_VOICE_LIMIT,
    "image": FREE_IMAGE_LIMIT,
    "file": FREE_FILE_LIMIT,
}

PRO_LIMITS = {
    "text": PRO_TEXT_LIMIT,
    "voice": PRO_VOICE_LIMIT,
    "image": PRO_IMAGE_LIMIT,
    "file": PRO_FILE_LIMIT,
}


# =========================
# HELPERS
# =========================
def today_str() -> str:
    return date.today().isoformat()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_retry_delay(error_text: str) -> tuple[int | None, str | None]:
    match = re.search(r"retryDelay[:=]\s*['\"]?(\d+)s['\"]?", error_text)
    if not match:
        return None, None
    seconds = int(match.group(1))
    retry_time = datetime.now() + timedelta(seconds=seconds)
    return seconds, retry_time.strftime("%H:%M:%S")


def format_retry_message(error_text: str) -> str:
    seconds, retry_at = extract_retry_delay(error_text)
    if seconds and retry_at:
        return (
            f"⚠️ Лимит AI временно исчерпан.\n"
            f"Попробуй через {seconds} сек.\n"
            f"Примерно в {retry_at}."
        )
    return "⚠️ Лимит AI временно исчерпан.\nПопробуй через 30–60 секунд."


async def wait_for_rate_limit():
    global last_ai_request_time

    async with ai_request_lock:
        now = time.monotonic()
        diff = now - last_ai_request_time

        if diff < MIN_SECONDS_BETWEEN_AI_REQUESTS:
            await asyncio.sleep(MIN_SECONDS_BETWEEN_AI_REQUESTS - diff)

        last_ai_request_time = time.monotonic()


def user_is_rate_limited(user_id: int) -> int | None:
    now = time.monotonic()
    last_time = user_cooldowns.get(user_id)

    if last_time is None:
        user_cooldowns[user_id] = now
        return None

    diff = now - last_time
    if diff < MIN_SECONDS_BETWEEN_USER_MESSAGES:
        return max(1, int(MIN_SECONDS_BETWEEN_USER_MESSAGES - diff + 0.999))

    user_cooldowns[user_id] = now
    return None


async def safe_ai_call(func, *args):
    await wait_for_rate_limit()
    return await asyncio.to_thread(func, *args)


# =========================
# DATABASE
# =========================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        chat_id TEXT PRIMARY KEY,
        username TEXT,
        first_name TEXT,
        last_name TEXT,
        language_code TEXT,
        mode TEXT DEFAULT 'assistant',
        memory_notes TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS usage_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT NOT NULL,
        usage_date TEXT NOT NULL,
        kind TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0,
        UNIQUE(chat_id, usage_date, kind)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS response_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cache_key TEXT NOT NULL UNIQUE,
        response_text TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_subscriptions (
        chat_id TEXT PRIMARY KEY,
        plan_name TEXT NOT NULL,
        expires_at INTEGER NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def is_admin(user_id: int) -> bool:
    return bool(ADMIN_ID) and str(user_id) == str(ADMIN_ID)


def upsert_user(update: Update):
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return

    chat_id = str(chat.id)

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO users (chat_id, username, first_name, last_name, language_code, updated_at)
    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(chat_id) DO UPDATE SET
        username=excluded.username,
        first_name=excluded.first_name,
        last_name=excluded.last_name,
        language_code=excluded.language_code,
        updated_at=CURRENT_TIMESTAMP
    """, (
        chat_id,
        user.username,
        user.first_name,
        user.last_name,
        user.language_code
    ))
    conn.commit()
    conn.close()


def get_user(chat_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()
    conn.close()
    return row


def set_mode(chat_id: str, mode: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    UPDATE users
    SET mode = ?, updated_at = CURRENT_TIMESTAMP
    WHERE chat_id = ?
    """, (mode, chat_id))
    conn.commit()
    conn.close()


def add_memory_note(chat_id: str, note: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT memory_notes FROM users WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()

    old_notes = row["memory_notes"] if row and row["memory_notes"] else ""
    old_notes = old_notes.strip()
    new_notes = (old_notes + "\n- " + note).strip() if old_notes else "- " + note

    cur.execute("""
    UPDATE users
    SET memory_notes = ?, updated_at = CURRENT_TIMESTAMP
    WHERE chat_id = ?
    """, (new_notes, chat_id))

    conn.commit()
    conn.close()


def add_message(chat_id: str, role: str, content: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO messages (chat_id, role, content)
    VALUES (?, ?, ?)
    """, (chat_id, role, content))
    conn.commit()
    conn.close()


def get_recent_history(chat_id: str, limit: int = 20):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT role, content
    FROM messages
    WHERE chat_id = ?
    ORDER BY id DESC
    LIMIT ?
    """, (chat_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def clear_history(chat_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.commit()
    conn.close()


def get_total_stats():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM users")
    users_count = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) AS cnt FROM messages")
    messages_count = cur.fetchone()["cnt"]
    conn.close()
    return users_count, messages_count


def get_users_list(limit: int = 50):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT chat_id, username, first_name FROM users LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_top_users(limit: int = 10):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT chat_id, COUNT(*) as cnt
        FROM messages
        GROUP BY chat_id
        ORDER BY cnt DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def clear_all_messages():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages")
    conn.commit()
    conn.close()


# =========================
# SUBSCRIPTIONS
# =========================
def get_subscription(chat_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT plan_name, expires_at
    FROM user_subscriptions
    WHERE chat_id = ?
    """, (chat_id,))
    row = cur.fetchone()
    conn.close()
    return row


def set_subscription(chat_id: str, plan_name: str, expires_at: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO user_subscriptions (chat_id, plan_name, expires_at, updated_at)
    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(chat_id) DO UPDATE SET
        plan_name=excluded.plan_name,
        expires_at=excluded.expires_at,
        updated_at=CURRENT_TIMESTAMP
    """, (chat_id, plan_name, expires_at))
    conn.commit()
    conn.close()


def has_active_pro(chat_id: str) -> bool:
    row = get_subscription(chat_id)
    if not row:
        return False
    return row["plan_name"] == PRO_PLAN_NAME and int(row["expires_at"]) > int(time.time())


def is_unlimited_user(user_id: int, chat_id: str) -> bool:
    if is_admin(user_id):
        return True
    return has_active_pro(chat_id)


def get_user_limits(user_id: int, chat_id: str) -> dict:
    if is_admin(user_id):
        return {
            "text": 999999,
            "voice": 999999,
            "image": 999999,
            "file": 999999,
        }

    if has_active_pro(chat_id):
        return PRO_LIMITS

    return FREE_LIMITS


# =========================
# USAGE LIMITS
# =========================
def get_usage_count(chat_id: str, kind: str) -> int:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT count
    FROM usage_stats
    WHERE chat_id = ? AND usage_date = ? AND kind = ?
    """, (chat_id, today_str(), kind))
    row = cur.fetchone()
    conn.close()
    return row["count"] if row else 0


def increment_usage(chat_id: str, kind: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO usage_stats (chat_id, usage_date, kind, count)
    VALUES (?, ?, ?, 1)
    ON CONFLICT(chat_id, usage_date, kind)
    DO UPDATE SET count = count + 1
    """, (chat_id, today_str(), kind))
    conn.commit()
    conn.close()


def check_daily_limit(chat_id: str, user_id: int, kind: str) -> tuple[bool, int, int]:
    limits = get_user_limits(user_id, chat_id)
    used = get_usage_count(chat_id, kind)
    limit = limits.get(kind, 0)
    return used < limit, used, limit


# =========================
# CACHE
# =========================
def build_cache_key(chat_id: str, mode: str, user_text: str) -> str:
    return f"{chat_id}:{mode}:{normalize_text(user_text)}"


def get_cached_response(cache_key: str) -> str | None:
    if not ENABLE_CACHE:
        return None

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    SELECT response_text, created_at
    FROM response_cache
    WHERE cache_key = ?
    """, (cache_key,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    created_at = int(row["created_at"])
    if int(time.time()) - created_at > CACHE_TTL_SECONDS:
        return None

    return row["response_text"]


def save_cached_response(cache_key: str, response_text: str):
    if not ENABLE_CACHE or not response_text.strip():
        return

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO response_cache (cache_key, response_text, created_at)
    VALUES (?, ?, ?)
    ON CONFLICT(cache_key) DO UPDATE SET
        response_text=excluded.response_text,
        created_at=excluded.created_at
    """, (cache_key, response_text, int(time.time())))
    conn.commit()
    conn.close()


# =========================
# PROMPTS
# =========================
def mode_prompt(mode: str) -> str:
    prompts = {
        "assistant": "Ты полезный AI-ассистент. Отвечай понятно, дружелюбно и по делу.",
        "coder": "Ты сильный программист. Пиши рабочий код и объясняй пошагово.",
        "translator": "Ты профессиональный переводчик. Переводи естественно и точно.",
        "teacher": "Ты терпеливый преподаватель. Объясняй простыми словами и с примерами.",
    }
    return prompts.get(mode, prompts["assistant"])


def build_prompt(chat_id: str, user_text: str) -> str:
    user = get_user(chat_id)
    history = get_recent_history(chat_id, MAX_HISTORY)

    mode = user["mode"] if user and user["mode"] else "assistant"
    memory_notes = user["memory_notes"] if user and user["memory_notes"] else ""
    first_name = user["first_name"] if user and user["first_name"] else ""

    parts = [
        f"System instruction: {SYSTEM_PROMPT}",
        "Rule: do not start every reply with greetings like 'Привет' or the user's name unless it is really needed.",
        f"Current mode: {mode_prompt(mode)}",
    ]

    if first_name:
        parts.append(
            f"User first name: {first_name}. Используй имя только когда это действительно уместно."
        )

    if memory_notes:
        parts.append(f"Known user notes:\n{memory_notes}")

    if history:
        parts.append("Conversation history:")
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{role}: {msg['content']}")

    parts.append(f"User: {user_text}")
    parts.append("Assistant:")
    return "\n".join(parts)


# =========================
# QUICK ANSWERS WITHOUT AI
# =========================
def quick_answer(chat_id: str, user_text: str) -> str | None:
    text = normalize_text(user_text)
    user = get_user(chat_id)
    memory_notes = user["memory_notes"] if user and user["memory_notes"] else ""

    if text in {"привет", "ку", "здарова", "здравствуйте", "добрый день"}:
        return "Привет! Чем могу помочь?"

    if text in {"как дела", "как настроение", "как насроения"}:
        return "Всё хорошо 🙂 Чем помочь?"

    if text in {"что ты умеешь", "что ты можешь"}:
        return (
            "Я умею отвечать на вопросы, помнить переписку, запоминать факты о пользователе, "
            "работать в разных режимах, распознавать голосовые, читать файлы и генерировать картинки."
        )

    if text in {"как меня зовут", "моё имя", "мое имя"}:
        if memory_notes:
            match = re.search(r"меня зовут\s+([^\n\r-]+)", memory_notes, re.IGNORECASE)
            if match:
                return f"Тебя зовут {match.group(1).strip()} 🙂"
        if user and user["first_name"]:
            return f"У тебя в Telegram указано имя: {user['first_name']}."
        return "Я пока не знаю, как тебя зовут. Напиши: /remember меня зовут ..."

    return None


# =========================
# GEMINI HELPERS
# =========================
def ask_gemini(chat_id: str, user_text: str) -> str:
    prompt = build_prompt(chat_id, user_text)

    response = gemini_client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt
    )

    text = getattr(response, "text", None)
    if text:
        return text.strip()

    try:
        return response.candidates[0].content.parts[0].text.strip()
    except Exception:
        return ""


def transcribe_audio_file(file_path: str) -> str:
    uploaded = gemini_client.files.upload(file=file_path)

    response = gemini_client.models.generate_content(
        model=VOICE_MODEL,
        contents=[
            "Сделай точную расшифровку речи из этого аудио. Верни только текст расшифровки без комментариев.",
            uploaded,
        ]
    )

    return (response.text or "").strip()


def summarize_document(file_path: str) -> str:
    uploaded = gemini_client.files.upload(file=file_path)

    response = gemini_client.models.generate_content(
        model=FILE_MODEL,
        contents=[
            uploaded,
            "Сделай краткое и понятное резюме этого файла на русском языке."
        ]
    )

    return (response.text or "").strip()


# =========================
# IMAGE HELPER
# =========================
def generate_image_with_pollinations(prompt: str) -> bytes:
    encoded_prompt = quote(prompt, safe="")
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if content_type and not content_type.startswith("image/"):
        raise ValueError(f"Сервис не вернул изображение. Content-Type: {content_type}")

    return response.content


def get_document_suffix(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".pdf", ".txt", ".md", ".csv", ".json"}:
        return ext
    return ".bin"


# =========================
# KEYBOARDS
# =========================
def main_keyboard():
    return ReplyKeyboardMarkup(
        [
            ["🧠 Новый чат", "🧹 Очистить память"],
            ["💻 Coder", "🌍 Translator"],
            ["📚 Teacher", "🤖 Assistant"],
        ],
        resize_keyboard=True
    )


def admin_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📊 Статистика", callback_data="admin_stats"),
            InlineKeyboardButton("👥 Users", callback_data="admin_users"),
        ],
        [
            InlineKeyboardButton("🏆 Top", callback_data="admin_top"),
            InlineKeyboardButton("🗑 Clear DB", callback_data="admin_clear_db"),
        ],
        [
            InlineKeyboardButton("❌ Закрыть", callback_data="admin_close"),
        ],
    ])


# =========================
# USER COMMANDS
# =========================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    text = (
        "Привет! Я AI-бот.\n\n"
        "Умею:\n"
        "- отвечать на вопросы\n"
        "- помнить диалог\n"
        "- запоминать информацию о пользователе\n"
        "- работать в разных режимах\n"
        "- понимать голосовые\n"
        "- читать PDF/TXT\n"
        "- генерировать картинки\n\n"
        "Команды:\n"
        "/start\n/help\n/clear\n/me\n/remember текст\n"
        "/mode assistant|coder|translator|teacher\n"
        "/image описание\n"
        "/upgrade — купить PRO\n"
        "/pro — статус подписки"
    )
    await update.message.reply_text(text, reply_markup=main_keyboard())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Просто пиши сообщения.\n\n"
        "/remember — сохранить факт о себе\n"
        "/me — показать сохранённую память\n"
        "/clear — очистить историю\n"
        "/mode — сменить режим\n"
        "/image — сделать картинку\n"
        "/upgrade — купить PRO\n"
        "/pro — статус подписки\n\n"
        "Админу доступна /admin",
        reply_markup=main_keyboard()
    )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    clear_history(chat_id)
    await update.message.reply_text("История диалога очищена.", reply_markup=main_keyboard())


async def me_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user = get_user(chat_id)

    if not user:
        await update.message.reply_text("О пользователе пока ничего не сохранено.")
        return

    username = f"@{user['username']}" if user["username"] else "-"
    sub = get_subscription(chat_id)

    sub_text = "FREE"
    if sub and int(sub["expires_at"]) > int(time.time()):
        expire_date = datetime.fromtimestamp(int(sub["expires_at"])).strftime("%d.%m.%Y %H:%M")
        sub_text = f"{sub['plan_name']} до {expire_date}"

    text = (
        f"Имя: {user['first_name'] or '-'}\n"
        f"Username: {username}\n"
        f"Язык: {user['language_code'] or '-'}\n"
        f"Режим: {user['mode'] or 'assistant'}\n"
        f"Тариф: {sub_text}\n\n"
        f"Память:\n{user['memory_notes'] or 'Пока пусто'}"
    )
    await update.message.reply_text(text)


async def remember_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    text = " ".join(context.args).strip()

    if not text:
        await update.message.reply_text("После /remember напиши, что сохранить.")
        return

    add_memory_note(chat_id, text)
    await update.message.reply_text("Запомнил.")


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    upsert_user(update)

    allowed = {"assistant", "coder", "translator", "teacher"}

    if not context.args:
        user = get_user(chat_id)
        current_mode = user["mode"] if user else "assistant"
        await update.message.reply_text(f"Текущий режим: {current_mode}")
        return

    new_mode = context.args[0].strip().lower()
    if new_mode not in allowed:
        await update.message.reply_text("Доступные режимы: assistant, coder, translator, teacher")
        return

    set_mode(chat_id, new_mode)
    await update.message.reply_text(f"Режим переключён на: {new_mode}")


async def pro_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    user_id = update.effective_user.id

    if is_admin(user_id):
        await update.message.reply_text("👑 Ты админ. Для тебя все лимиты отключены.")
        return

    sub = get_subscription(chat_id)
    if sub and int(sub["expires_at"]) > int(time.time()):
        expire_date = datetime.fromtimestamp(int(sub["expires_at"])).strftime("%d.%m.%Y %H:%M")
        await update.message.reply_text(
            f"⭐ PRO активен\n"
            f"Тариф: {sub['plan_name']}\n"
            f"До: {expire_date}"
        )
        return

    await update.message.reply_text(
        f"У тебя сейчас FREE.\n\n"
        f"PRO: {PRO_PRICE_STARS} ⭐ на {PRO_DURATION_DAYS} дней\n"
        f"Команда: /upgrade"
    )


async def upgrade_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prices = [LabeledPrice(f"PRO на {PRO_DURATION_DAYS} дней", PRO_PRICE_STARS)]

    await update.message.reply_invoice(
        title="AI Bot PRO",
        description=(
            f"⭐ PRO подписка на {PRO_DURATION_DAYS} дней\n\n"
            f"• без лимита на текст\n"
            f"• {PRO_IMAGE_LIMIT} картинок в день\n"
            f"• {PRO_VOICE_LIMIT} голосовых в день\n"
            f"• {PRO_FILE_LIMIT} файлов в день\n"
            f"• приоритетный доступ"
        ),
        payload="pro_30_days",
        currency="XTR",
        prices=prices,
        provider_token="",
        start_parameter="pro-subscription",
    )


async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.pre_checkout_query

    if query.invoice_payload != "pro_30_days":
        await query.answer(ok=False, error_message="Неизвестный платёж.")
        return

    await query.answer(ok=True)


async def successful_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payment = update.message.successful_payment
    chat_id = str(update.effective_chat.id)

    if payment.invoice_payload != "pro_30_days":
        return

    now_ts = int(time.time())
    current_sub = get_subscription(chat_id)

    if current_sub and int(current_sub["expires_at"]) > now_ts:
        new_expires = int(current_sub["expires_at"]) + PRO_DURATION_DAYS * 24 * 60 * 60
    else:
        new_expires = now_ts + PRO_DURATION_DAYS * 24 * 60 * 60

    set_subscription(chat_id, PRO_PLAN_NAME, new_expires)

    expire_date = datetime.fromtimestamp(new_expires).strftime("%d.%m.%Y %H:%M")

    await update.message.reply_text(
        f"🎉 PRO активирован!\n\n"
        f"Действует до: {expire_date}\n"
        f"Спасибо за поддержку бота ⭐"
    )


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user(update)
    chat_id = str(update.effective_chat.id)
    user_id = update.effective_user.id

    allowed, used, limit = check_daily_limit(chat_id, user_id, "image")
    if not allowed:
        await update.message.reply_text(
            f"⚠️ Дневной лимит картинок исчерпан.\nСегодня использовано: {used}/{limit}.\n\n"
            f"Для большего лимита: /upgrade"
        )
        return

    prompt = " ".join(context.args).strip()
    if not prompt:
        await update.message.reply_text("После /image напиши описание картинки.")
        return

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_PHOTO)
        image_bytes = generate_image_with_pollinations(prompt)

        bio = io.BytesIO(image_bytes)
        bio.name = "generated.jpg"
        bio.seek(0)

        increment_usage(chat_id, "image")
        await update.message.reply_photo(photo=bio, caption=f"Готово.\nПромпт: {prompt}")

    except Exception as e:
        logger.exception("Ошибка генерации картинки")
        await update.message.reply_text(f"Ошибка при генерации картинки.\n\n{type(e).__name__}: {e}")


# =========================
# ADMIN COMMANDS
# =========================
async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Эта команда только для админа.")
        return
    await update.message.reply_text("Админ-панель:", reply_markup=admin_keyboard())


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Эта команда только для админа.")
        return
    users_count, messages_count = get_total_stats()
    await update.message.reply_text(f"Пользователей: {users_count}\nСообщений в базе: {messages_count}")


async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Команда только для админа.")
        return

    rows = get_users_list(50)
    if not rows:
        await update.message.reply_text("Пользователей нет.")
        return

    text = "👥 Пользователи:\n\n"
    for row in rows:
        username = f"@{row['username']}" if row["username"] else "-"
        first_name = row["first_name"] or "-"
        text += f"{first_name} ({username}) | chat_id: {row['chat_id']}\n"

    await update.message.reply_text(text[:4000])


async def top_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Команда только для админа.")
        return

    rows = get_top_users(10)
    if not rows:
        await update.message.reply_text("Нет данных.")
        return

    text = "🏆 Самые активные пользователи:\n\n"
    for i, row in enumerate(rows, start=1):
        text += f"{i}. {row['chat_id']} — {row['cnt']} сообщений\n"

    await update.message.reply_text(text)


async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Команда только для админа.")
        return

    message = " ".join(context.args).strip()
    if not message:
        await update.message.reply_text("Напиши сообщение после /broadcast")
        return

    rows = get_users_list(100000)
    sent = 0
    failed = 0

    for row in rows:
        try:
            await context.bot.send_message(chat_id=row["chat_id"], text=message)
            sent += 1
        except Exception:
            failed += 1

    await update.message.reply_text(f"Рассылка завершена.\nОтправлено: {sent}\nОшибок: {failed}")


async def clear_db_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("Команда только для админа.")
        return
    clear_all_messages()
    await update.message.reply_text("База сообщений очищена.")


async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return

    if not is_admin(query.from_user.id):
        await query.answer("Только для админа", show_alert=True)
        return

    await query.answer()
    data = query.data

    if data == "admin_stats":
        users_count, messages_count = get_total_stats()
        await query.message.edit_text(
            f"📊 Статистика\n\nПользователей: {users_count}\nСообщений в базе: {messages_count}",
            reply_markup=admin_keyboard()
        )

    elif data == "admin_users":
        rows = get_users_list(30)
        text = "👥 Пользователи:\n\n" if rows else "👥 Пользователей нет."
        for row in rows:
            username = f"@{row['username']}" if row["username"] else "-"
            first_name = row["first_name"] or "-"
            text += f"{first_name} ({username})\n"
        await query.message.edit_text(text[:4000], reply_markup=admin_keyboard())

    elif data == "admin_top":
        rows = get_top_users(10)
        text = "🏆 Самые активные:\n\n" if rows else "🏆 Нет данных."
        for i, row in enumerate(rows, start=1):
            text += f"{i}. {row['chat_id']} — {row['cnt']} сообщений\n"
        await query.message.edit_text(text[:4000], reply_markup=admin_keyboard())

    elif data == "admin_clear_db":
        clear_all_messages()
        await query.message.edit_text("🗑 База сообщений очищена.", reply_markup=admin_keyboard())

    elif data == "admin_close":
        await query.message.edit_text("Админ-панель закрыта.")


# =========================
# BUTTONS
# =========================
async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not update.message or not update.message.text:
        return False

    text = update.message.text.strip()
    chat_id = str(update.effective_chat.id)

    if text == "🧠 Новый чат":
        clear_history(chat_id)
        await update.message.reply_text("Начали новый диалог.", reply_markup=main_keyboard())
        return True

    if text == "🧹 Очистить память":
        clear_history(chat_id)
        await update.message.reply_text("Память очищена.", reply_markup=main_keyboard())
        return True

    if text == "💻 Coder":
        set_mode(chat_id, "coder")
        await update.message.reply_text("Режим: coder", reply_markup=main_keyboard())
        return True

    if text == "🌍 Translator":
        set_mode(chat_id, "translator")
        await update.message.reply_text("Режим: translator", reply_markup=main_keyboard())
        return True

    if text == "📚 Teacher":
        set_mode(chat_id, "teacher")
        await update.message.reply_text("Режим: teacher", reply_markup=main_keyboard())
        return True

    if text == "🤖 Assistant":
        set_mode(chat_id, "assistant")
        await update.message.reply_text("Режим: assistant", reply_markup=main_keyboard())
        return True

    return False


# =========================
# HANDLERS
# =========================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    upsert_user(update)

    user_id = update.effective_user.id
    chat_id = str(update.effective_chat.id)
    user_text = update.message.text.strip()

    if not is_admin(user_id):
        wait_seconds = user_is_rate_limited(user_id)
        if wait_seconds is not None:
            await update.message.reply_text(
                f"⏳ Слишком быстро. Подожди {wait_seconds} сек. перед следующим сообщением."
            )
            return

    if await handle_buttons(update, context):
        return

    if not user_text:
        return

    quick = quick_answer(chat_id, user_text)
    if quick:
        add_message(chat_id, "user", user_text)
        add_message(chat_id, "assistant", quick)
        await update.message.reply_text(quick, reply_markup=main_keyboard())
        return

    allowed, used, limit = check_daily_limit(chat_id, user_id, "text")
    if not allowed:
        await update.message.reply_text(
            f"⚠️ Дневной AI-лимит исчерпан.\nСегодня использовано: {used}/{limit}.\n\n"
            f"Для большего лимита: /upgrade"
        )
        return

    add_message(chat_id, "user", user_text)

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        user = get_user(chat_id)
        mode = user["mode"] if user and user["mode"] else "assistant"
        cache_key = build_cache_key(chat_id, mode, user_text)

        cached = get_cached_response(cache_key)
        if cached:
            add_message(chat_id, "assistant", cached)
            await update.message.reply_text(cached, reply_markup=main_keyboard())
            return

        ai_text = await safe_ai_call(ask_gemini, chat_id, user_text)
        if not ai_text:
            ai_text = "Я не смог сформировать ответ."

        increment_usage(chat_id, "text")
        save_cached_response(cache_key, ai_text)
        add_message(chat_id, "assistant", ai_text)

        for i in range(0, len(ai_text), 4000):
            await update.message.reply_text(ai_text[i:i + 4000], reply_markup=main_keyboard())

    except Exception as e:
        error_text = str(e)

        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            await update.message.reply_text(format_retry_message(error_text))
            return

        if "503" in error_text or "UNAVAILABLE" in error_text:
            await update.message.reply_text("⚠️ Сервер AI сейчас перегружен.\nПопробуй через 10–20 секунд.")
            return

        logger.exception("Ошибка текстового ответа")
        await update.message.reply_text(f"Ошибка при обработке текста.\n\n{type(e).__name__}: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.voice:
        return

    upsert_user(update)

    user_id = update.effective_user.id
    chat_id = str(update.effective_chat.id)
    voice = update.message.voice

    if not is_admin(user_id):
        wait_seconds = user_is_rate_limited(user_id)
        if wait_seconds is not None:
            await update.message.reply_text(
                f"⏳ Слишком быстро. Подожди {wait_seconds} сек. перед следующим сообщением."
            )
            return

    allowed, used, limit = check_daily_limit(chat_id, user_id, "voice")
    if not allowed:
        await update.message.reply_text(
            f"⚠️ Дневной лимит голосовых исчерпан.\nСегодня использовано: {used}/{limit}.\n\n"
            f"Для большего лимита: /upgrade"
        )
        return

    if voice.duration and voice.duration > 30:
        await update.message.reply_text("Голосовое слишком длинное. Отправь до 30 секунд.")
        return

    temp_path = None
    processing_msg = None

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        processing_msg = await update.message.reply_text("🎤 Обрабатываю голосовое...")

        tg_file = await update.message.voice.get_file()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            temp_path = tmp.name

        await tg_file.download_to_drive(temp_path)

        transcript = ""
        last_error = None

        for attempt in range(2):
            try:
                transcript = await asyncio.wait_for(
                    safe_ai_call(transcribe_audio_file, temp_path),
                    timeout=60
                )
                transcript = transcript.strip()
                if transcript:
                    break
            except Exception as e:
                last_error = e
                if attempt == 0:
                    await asyncio.sleep(3)

        if processing_msg:
            try:
                await processing_msg.delete()
            except Exception:
                pass

        if not transcript:
            if last_error:
                error_text = str(last_error)

                if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
                    await update.message.reply_text(format_retry_message(error_text))
                    return

                if "503" in error_text or "UNAVAILABLE" in error_text:
                    await update.message.reply_text("⚠️ AI перегружен при обработке голосового.\nПопробуй через 10–20 секунд.")
                    return

                await update.message.reply_text(f"Не удалось распознать голосовое.\n\n{type(last_error).__name__}: {last_error}")
            else:
                await update.message.reply_text("Не удалось распознать голосовое сообщение.")
            return

        increment_usage(chat_id, "voice")
        add_message(chat_id, "user", f"[voice] {transcript}")

        ai_text = await asyncio.wait_for(
            safe_ai_call(
                ask_gemini,
                chat_id,
                f"Пользователь прислал голосовое сообщение. Текст расшифровки: {transcript}"
            ),
            timeout=60
        )

        if not ai_text:
            ai_text = "Я не смог сформировать ответ."

        increment_usage(chat_id, "text")
        add_message(chat_id, "assistant", ai_text)

        await update.message.reply_text(f"📝 Расшифровка:\n{transcript}")
        for i in range(0, len(ai_text), 4000):
            await update.message.reply_text(ai_text[i:i + 4000], reply_markup=main_keyboard())

    except asyncio.TimeoutError:
        if processing_msg:
            try:
                await processing_msg.delete()
            except Exception:
                pass
        await update.message.reply_text("⏳ Голосовое обрабатывалось слишком долго. Попробуй ещё раз или отправь короче.")

    except Exception as e:
        error_text = str(e)

        if processing_msg:
            try:
                await processing_msg.delete()
            except Exception:
                pass

        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            await update.message.reply_text(format_retry_message(error_text))
            return

        if "503" in error_text or "UNAVAILABLE" in error_text:
            await update.message.reply_text("⚠️ AI перегружен при обработке голосового.\nПопробуй через 10–20 секунд.")
            return

        logger.exception("Ошибка обработки голосового")
        await update.message.reply_text(f"Ошибка при обработке голосового.\n\n{type(e).__name__}: {e}")

    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.document:
        return

    upsert_user(update)

    user_id = update.effective_user.id
    chat_id = str(update.effective_chat.id)
    doc = update.message.document
    filename = doc.file_name or "file"
    suffix = get_document_suffix(filename)

    if not is_admin(user_id):
        wait_seconds = user_is_rate_limited(user_id)
        if wait_seconds is not None:
            await update.message.reply_text(
                f"⏳ Слишком быстро. Подожди {wait_seconds} сек. перед следующим сообщением."
            )
            return

    allowed, used, limit = check_daily_limit(chat_id, user_id, "file")
    if not allowed:
        await update.message.reply_text(
            f"⚠️ Дневной лимит файлов исчерпан.\nСегодня использовано: {used}/{limit}.\n\n"
            f"Для большего лимита: /upgrade"
        )
        return

    if suffix not in {".pdf", ".txt", ".md", ".csv", ".json"}:
        await update.message.reply_text("Пока поддерживаю PDF, TXT, MD, CSV и JSON.")
        return

    temp_path = None
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

        tg_file = await doc.get_file()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name

        await tg_file.download_to_drive(temp_path)

        summary = await safe_ai_call(summarize_document, temp_path)
        if not summary:
            summary = "Не удалось получить краткое резюме файла."

        increment_usage(chat_id, "file")
        add_message(chat_id, "user", f"[document] {filename}")
        add_message(chat_id, "assistant", f"[summary of {filename}] {summary}")

        await update.message.reply_text(
            f"📄 Файл: {filename}\n\nКраткое резюме:\n{summary}",
            reply_markup=main_keyboard()
        )

    except Exception as e:
        error_text = str(e)

        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            await update.message.reply_text(format_retry_message(error_text))
            return

        if "503" in error_text or "UNAVAILABLE" in error_text:
            await update.message.reply_text("⚠️ AI перегружен при обработке файла.\nПопробуй через 10–20 секунд.")
            return

        logger.exception("Ошибка обработки документа")
        await update.message.reply_text(f"Ошибка при обработке файла.\n\n{type(e).__name__}: {e}")

    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


# =========================
# ERROR HANDLER
# =========================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Telegram error", exc_info=context.error)


# =========================
# PTB APP
# =========================
ptb_app: Application = ApplicationBuilder().token(TELEGRAM_TOKEN).updater(None).build()

ptb_app.add_handler(CommandHandler("start", start_command))
ptb_app.add_handler(CommandHandler("help", help_command))
ptb_app.add_handler(CommandHandler("clear", clear_command))
ptb_app.add_handler(CommandHandler("me", me_command))
ptb_app.add_handler(CommandHandler("remember", remember_command))
ptb_app.add_handler(CommandHandler("mode", mode_command))
ptb_app.add_handler(CommandHandler("image", image_command))
ptb_app.add_handler(CommandHandler("upgrade", upgrade_command))
ptb_app.add_handler(CommandHandler("pro", pro_command))

ptb_app.add_handler(CommandHandler("admin", admin_command))
ptb_app.add_handler(CommandHandler("stats", stats_command))
ptb_app.add_handler(CommandHandler("users", users_command))
ptb_app.add_handler(CommandHandler("top", top_command))
ptb_app.add_handler(CommandHandler("broadcast", broadcast_command))
ptb_app.add_handler(CommandHandler("clear_db", clear_db_command))

ptb_app.add_handler(CallbackQueryHandler(admin_callback_handler, pattern="^admin_"))
ptb_app.add_handler(PreCheckoutQueryHandler(precheckout_callback))
ptb_app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))

ptb_app.add_handler(MessageHandler(filters.VOICE, handle_voice))
ptb_app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
ptb_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

ptb_app.add_error_handler(error_handler)


# =========================
# FASTAPI APP
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    await ptb_app.initialize()
    await ptb_app.start()

    webhook_url = f"{WEBHOOK_URL}/webhook/{TELEGRAM_TOKEN}"
    await ptb_app.bot.set_webhook(url=webhook_url)
    logger.info("Webhook set to %s", webhook_url)

    yield

    try:
        await ptb_app.bot.delete_webhook()
    except Exception:
        pass
    await ptb_app.stop()
    await ptb_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return PlainTextResponse("Bot is running")


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    update = Update.de_json(data, ptb_app.bot)
    await ptb_app.process_update(update)
    return JSONResponse({"ok": True})