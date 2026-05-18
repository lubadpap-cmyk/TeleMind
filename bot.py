"""
Telegram AI Userbot — An intelligent, human-like automated reply assistant
powered by Google Gemini, Groq (Llama 3.3), or local LLMs (via Ollama/LM Studio).
"""
import sys
import os
import logging
import asyncio
import time
import random
import json
from collections import defaultdict
from dotenv import load_dotenv

# Reconfigure console streams to UTF-8 on Windows to prevent UnicodeEncodeError
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Settings ---
load_dotenv()

API_ID_RAW = os.getenv("API_ID")
API_ID = API_ID_RAW.strip() if API_ID_RAW else None
API_HASH_RAW = os.getenv("API_HASH")
API_HASH = API_HASH_RAW.strip() if API_HASH_RAW else None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Support multiple API keys for Gemini (comma-separated or single)
GEMINI_KEYS_RAW = os.getenv("GEMINI_API_KEYS", GEMINI_API_KEY)
GEMINI_KEYS = [k.strip() for k in GEMINI_KEYS_RAW.split(",") if k.strip()] if GEMINI_KEYS_RAW else []
if not GEMINI_KEYS and GEMINI_API_KEY:
    GEMINI_KEYS = [GEMINI_API_KEY.strip()]

MY_NAME = os.getenv("MY_NAME", "YourName")

# Response mode (instant reply / human-like simulation with delays and typos)
INSTANT_REPLY = os.getenv("INSTANT_REPLY", "True").lower() == "true"

# AI Engine selection (gemini / local / groq)
AI_ENGINE = os.getenv("AI_ENGINE", "gemini").lower()
if os.getenv("USE_LOCAL_AI", "False").lower() == "true":
    AI_ENGINE = "local"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Support multiple API keys for Groq (comma-separated or single)
GROQ_KEYS_RAW = os.getenv("GROQ_API_KEYS", GROQ_API_KEY)
GROQ_KEYS = [k.strip() for k in GROQ_KEYS_RAW.split(",") if k.strip()] if GROQ_KEYS_RAW else []
if not GROQ_KEYS and GROQ_API_KEY:
    GROQ_KEYS = [GROQ_API_KEY.strip()]

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LOCAL_AI_BASE_URL = os.getenv("LOCAL_AI_BASE_URL", "http://localhost:11434/v1")
LOCAL_AI_MODEL = os.getenv("LOCAL_AI_MODEL", "llama3.2:1b")

if not API_ID or not API_HASH:
    raise ValueError("❌ Please specify API_ID and API_HASH in your .env file")
if AI_ENGINE == "gemini" and not GEMINI_KEYS:
    raise ValueError("❌ Please specify GEMINI_API_KEY or GEMINI_API_KEYS in your .env file for Gemini")
if AI_ENGINE == "groq" and not GROQ_KEYS:
    raise ValueError("❌ Please specify GROQ_API_KEY or GROQ_API_KEYS in your .env file for Groq")

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- AI Clients Initialization ---
gemini_clients = []
gemini_cooldowns = []
groq_clients = []
groq_cooldowns = []
local_ai_client = None
MODEL = "gemini-flash-latest"

# 1. Initialize Google Gemini clients for all keys
if GEMINI_KEYS:
    try:
        from google import genai
        for key in GEMINI_KEYS:
            try:
                client_instance = genai.Client(api_key=key)
                masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "..."
                gemini_clients.append((client_instance, masked))
                gemini_cooldowns.append(0.0)
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Google Gemini client for key {key[:6]}...: {e}")
        
        if gemini_clients:
            logger.info(f"🔌 Initialized {len(gemini_clients)} Google Gemini API clients")
    except Exception as e:
        logger.warning(f"⚠️ Failed to import genai / initialize Google Gemini: {e}")

# 2. Initialize Groq & Local AI clients if openai is installed
try:
    from openai import AsyncOpenAI
    
    if GROQ_KEYS:
        for key in GROQ_KEYS:
            try:
                client_instance = AsyncOpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=key
                )
                masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "..."
                groq_clients.append((client_instance, masked))
                groq_cooldowns.append(0.0)
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Groq client for key {key[:6]}...: {e}")
                
        if groq_clients:
            logger.info(f"🔌 Initialized {len(groq_clients)} Groq API clients")
    
    local_ai_client = AsyncOpenAI(
        base_url=LOCAL_AI_BASE_URL,
        api_key="local-ai"
    )
    logger.info(f"🔌 Initialized Local AI client (Model: {LOCAL_AI_MODEL})")

except ImportError:
    # Only raise fatal error if the user set a default boot engine that requires openai
    if AI_ENGINE in ("local", "groq"):
        print(f"❌ Error: Selected boot engine is {AI_ENGINE}, but the 'openai' package is not installed.")
        print("👉 Please run in terminal: pip install openai")
        sys.exit(1)
    else:
        logger.warning("⚠️ 'openai' package not found. Groq and Local AI engines will not be available. Install with: pip install openai")

# Backward compatibility pointers for commands and validations
gemini_client = gemini_clients[0][0] if gemini_clients else None
groq_client = groq_clients[0][0] if groq_clients else None

# --- AI Persona & Chat System Prompt ---
# NOTE: The persona rules and examples below are tailored to chat naturally
# with Russian friends. You can translate this entire system prompt and 
# the few-shot dialogue examples to any language (e.g. English, Spanish) 
# to match your local chatting style!
SYSTEM_PROMPT = f"""Ты — {MY_NAME}. Общаешься в Telegram с друзьями.
Ты обычный парень, твой стиль общения — максимально расслабленный, ленивый и неформальный.
Ты отвечаешь логично, естественно и строго по смыслу, очень кратко (обычно от 1 до 5 слов).

ГЛАВНЫЕ ПРАВИЛА:
1. Пиши ТОЛЬКО строчными (маленькими) буквами. Никаких заглавных букв!
2. Никаких точек или запятых на самом конце твоих сообщений.
3. Не используй скобочки (типа ), ( ) и эмодзи. Вообще никаких картинок и смайликов. Пиши только буквами.
4. Пиши естественно и просто, как обычный человек в реальных чатах. Твоя речь должна быть живой.
5. НЕ пихай сленговые слова (норм, спс, го, хз, лан) в каждое сообщение! Используй их РЕДКО и только тогда, когда они реально подходят по смыслу. Не будь попугаем!
6. Отвечай строго по контексту переписки. Если тебе прислали простое "ок", ответь "ага", "угу" или "лан", не нужно выдумывать бред.

Примеры твоих идеальных ответов:
- Собеседник: привет
- Ты: о ку че делаешь

- Собеседник: Нормасик го в кс Го
- Ты: го ща зайду

- Собеседник: На улицу
- Ты: давай гулять

- Собеседник: Идём
- Ты: да ща выйду

- Собеседник: ты че тупишь
- Ты: да занят был просто

- Собеседник: ты где щас?
- Ты: да дома сижу

- Собеседник: сможешь помочь вечером?
- Ты: позже напишу

- Собеседник: спасибо за помощь!
- Ты: да не за что

- Собеседник: ок
- Ты: ага
"""

# --- Active State ---
auto_reply_enabled = True
my_user_id = None
my_username = None

# chat_id -> message history list [{role, content}, ...]
chat_histories: dict[int, list[dict]] = defaultdict(list)
MAX_HISTORY = 20  # Keep a moving window of the last 20 messages for context
HISTORY_FILE = "chat_histories.json"


def load_histories():
    """Load cached chat histories from local JSON file at boot."""
    global chat_histories
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                chat_histories.clear()
                for k, v in data.items():
                    chat_histories[int(k)] = v
            logger.info("💾 Successfully loaded chat histories cache from disk")
        except Exception as e:
            logger.error(f"⚠️ Error loading histories cache: {e}")


def save_histories():
    """Save updated chat histories to local JSON file."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            # Convert int keys to string for JSON serialization
            data = {str(k): v for k, v in chat_histories.items()}
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"⚠️ Error saving histories cache: {e}")


def get_history(chat_id: int) -> list[dict]:
    """Retrieve chat history trimmed to MAX_HISTORY length."""
    history = chat_histories[chat_id]
    if len(history) > MAX_HISTORY:
        chat_histories[chat_id] = history[-MAX_HISTORY:]
    return chat_histories[chat_id]


def add_message(chat_id: int, role: str, text: str):
    """Add a message to the history and save to disk."""
    chat_histories[chat_id].append({"role": role, "content": text})
    save_histories()


# --- AI Text Generation with Self-Healing Fallback ---

# Track engine cooldowns (engine_name -> float (timestamp when cooldown ends))
engine_cooldowns = {}
COOLDOWN_DURATION = 90  # Put a failed API key / engine on cooldown for 90 seconds

def mark_engine_failed(engine_name: str):
    """Mark an engine as failed to log the incident."""
    logger.warning(f"📉 Engine {engine_name.upper()} has completely exhausted all available keys/retries.")

def mark_client_failed(engine_name: str, client_idx: int):
    """Mark a specific client key as failed and put it on cooldown for 90s."""
    now = time.time()
    cooldown_expiry = now + COOLDOWN_DURATION
    if engine_name == "gemini":
        gemini_cooldowns[client_idx] = cooldown_expiry
        masked = gemini_clients[client_idx][1]
        logger.warning(f"📉 Gemini API Key #{client_idx+1} ({masked}) on cooldown for {COOLDOWN_DURATION}s due to rate limit/error.")
    elif engine_name == "groq":
        groq_cooldowns[client_idx] = cooldown_expiry
        masked = groq_clients[client_idx][1]
        logger.warning(f"📉 Groq API Key #{client_idx+1} ({masked}) on cooldown for {COOLDOWN_DURATION}s due to rate limit/error.")

def is_engine_healthy(engine_name: str) -> bool:
    """Check if the engine has at least one healthy (not on cooldown) API key."""
    now = time.time()
    if engine_name == "gemini":
        if not gemini_clients:
            return False
        return any(now > t for t in gemini_cooldowns)
    elif engine_name == "groq":
        if not groq_clients:
            return False
        return any(now > t for t in groq_cooldowns)
    elif engine_name == "local":
        cooldown_until = engine_cooldowns.get("local", 0)
        return now > cooldown_until
    return False

def get_engine_attempts() -> list[str]:
    """Returns a list of configured engines, prioritizing healthy ones, with failed ones as last resort."""
    available = []
    if AI_ENGINE == "gemini":
        available = ["gemini", "groq", "local"]
    elif AI_ENGINE == "groq":
        available = ["groq", "gemini", "local"]
    else:
        available = ["local", "gemini", "groq"]

    # Filter out unconfigured clients
    configured = []
    for eng in available:
        if eng == "gemini" and gemini_clients:
            configured.append(eng)
        elif eng == "groq" and groq_clients:
            configured.append(eng)
        elif eng == "local" and local_ai_client is not None:
            configured.append(eng)

    # Put healthy engines first, then cooldown ones as a last-resort backup
    healthy = [eng for eng in configured if is_engine_healthy(eng)]
    unhealthy = [eng for eng in configured if not is_engine_healthy(eng)]
    
    return healthy + unhealthy

async def query_engine(engine_name: str, history: list[dict]) -> str:
    """Queries a specific AI engine, smart-rotating through all configured API keys if rate-limited."""
    now = time.time()

    if engine_name == "gemini":
        if not gemini_clients:
            raise ValueError("No Gemini clients are initialized")
        
        # Sort indices: healthy keys first, then cooldown ones (least remaining cooldown first)
        healthy_indices = [i for i in range(len(gemini_clients)) if now > gemini_cooldowns[i]]
        cooldown_indices = sorted([i for i in range(len(gemini_clients)) if now <= gemini_cooldowns[i]], key=lambda i: gemini_cooldowns[i])
        client_indices = healthy_indices + cooldown_indices

        prompt = SYSTEM_PROMPT + "\n\nChat History:\n"
        for msg in history:
            prefix = "Friend" if msg["role"] == "user" else MY_NAME
            prompt += f"{prefix}: {msg['content']}\n"
        prompt += f"\nReply as {MY_NAME}:"

        last_err = None
        for idx in client_indices:
            client_inst, masked = gemini_clients[idx]
            # Fast retry loop per key
            for attempt in range(2):
                try:
                    if idx > 0 or now <= gemini_cooldowns[idx]:
                        logger.info(f"🔑 Querying Gemini API Key #{idx+1} ({masked})")

                    response = await client_inst.aio.models.generate_content(
                        model=MODEL,
                        contents=[prompt],
                    )
                    return response.text.strip().replace("(", "").replace(")", "")
                except Exception as e:
                    if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"⚠️ Gemini API Key #{idx+1} Rate Limit. Waiting {wait_time}s... (Attempt {attempt+1}/2)")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    logger.warning(f"⚠️ Gemini API Key #{idx+1} ({masked}) failed: {e}")
                    mark_client_failed("gemini", idx)
                    last_err = e
                    break  # Go to the next API key immediately
        
        raise last_err if last_err else ValueError("All Gemini keys failed")

    elif engine_name == "groq":
        if not groq_clients:
            raise ValueError("No Groq clients are initialized")

        healthy_indices = [i for i in range(len(groq_clients)) if now > groq_cooldowns[i]]
        cooldown_indices = sorted([i for i in range(len(groq_clients)) if now <= groq_cooldowns[i]], key=lambda i: groq_cooldowns[i])
        client_indices = healthy_indices + cooldown_indices

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        last_err = None
        for idx in client_indices:
            client_inst, masked = groq_clients[idx]
            for attempt in range(2):
                try:
                    if idx > 0 or now <= groq_cooldowns[idx]:
                        logger.info(f"🔑 Querying Groq API Key #{idx+1} ({masked})")

                    response = await client_inst.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=messages,
                        temperature=0.7,
                    )
                    return response.choices[0].message.content.strip().replace("(", "").replace(")", "")
                except Exception as e:
                    if ("429" in str(e) or "rate" in str(e).lower()) and attempt < 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"⚠️ Groq API Key #{idx+1} Rate Limit. Waiting {wait_time}s... (Attempt {attempt+1}/2)")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    logger.warning(f"⚠️ Groq API Key #{idx+1} ({masked}) failed: {e}")
                    mark_client_failed("groq", idx)
                    last_err = e
                    break
        
        raise last_err if last_err else ValueError("All Groq keys failed")

    elif engine_name == "local":
        if not local_ai_client:
            raise ValueError("Local AI client is not configured")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        try:
            response = await local_ai_client.chat.completions.create(
                model=LOCAL_AI_MODEL,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip().replace("(", "").replace(")", "")
        except Exception as e:
            engine_cooldowns["local"] = time.time() + COOLDOWN_DURATION
            raise e
    else:
        raise ValueError(f"Unsupported AI Engine: {engine_name}")

async def generate_response(chat_id: int, user_message: str) -> str:
    """Queries the AI engines with automatic self-healing fallback and rate-limit protection."""
    add_message(chat_id, "user", user_message)
    history = get_history(chat_id)

    engines = get_engine_attempts()
    if not engines:
        return "сорри, ни один ИИ-движок не настроен в .env 😅"

    for idx, engine in enumerate(engines):
        try:
            if engine != AI_ENGINE:
                logger.info(f"🔄 Fallback attempt: Using {engine.upper()} (Primary {AI_ENGINE.upper()} unavailable/on cooldown)")
            else:
                logger.info(f"🧠 Querying active engine: {engine.upper()}")

            reply = await query_engine(engine, history)
            add_message(chat_id, "assistant", reply)
            return reply

        except Exception as e:
            logger.error(f"❌ {engine.upper()} query failed: {e}")
            mark_engine_failed(engine)

    # All engines failed
    logger.critical("🚨 All configured AI engines failed to generate a response!")
    return "сорри, все нейросети прилегли, напиши чуть позже 😅"


# --- Telethon Client Instantiation ---
from telethon import TelegramClient, events

client = TelegramClient(
    "my_account",
    int(API_ID),
    API_HASH,
    system_version="4.16.30-x64",
    device_model="Windows PC",
    app_version="1.0.0",
    lang_code="en",
    system_lang_code="en-US"
)


# --- Self-Admin Commands (Executed by typing to yourself or Saved Messages) ---

@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai (on|off)$"))
async def toggle_ai(event):
    """Enable or disable auto-reply. Usage: .ai on / .ai off"""
    global auto_reply_enabled
    command = event.pattern_match.group(1)

    if command == "on":
        auto_reply_enabled = True
        try:
            await event.edit("✅ AI Auto-Reply Enabled")
        except Exception:
            pass
    else:
        auto_reply_enabled = False
        try:
            await event.edit("⛔ AI Auto-Reply Disabled")
        except Exception:
            pass

    logger.info(f"AI Auto-Reply state: {'ENABLED' if auto_reply_enabled else 'DISABLED'}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai instant (on|off)$"))
async def toggle_instant(event):
    """Enable or disable instant replies. Usage: .ai instant on / .ai instant off"""
    global INSTANT_REPLY
    command = event.pattern_match.group(1)

    if command == "on":
        INSTANT_REPLY = True
        try:
            await event.edit("⚡ AI Instant Reply Mode Enabled (No delays, no typos)")
        except Exception:
            pass
    else:
        INSTANT_REPLY = False
        try:
            await event.edit("👤 AI Human Simulation Mode Enabled (With delays and typos)")
        except Exception:
            pass

    logger.info(f"AI Instant Reply state: {'ENABLED' if INSTANT_REPLY else 'DISABLED'}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai engine (gemini|groq|local)$"))
async def switch_engine(event):
    """Switch active AI engine. Usage: .ai engine gemini / .ai engine groq / .ai engine local"""
    global AI_ENGINE
    new_engine = event.pattern_match.group(1).lower()

    if new_engine == "gemini" and not gemini_clients:
        try:
            await event.edit("❌ Gemini clients are not initialized (check GEMINI_API_KEYS in .env)")
        except Exception:
            pass
        return
    elif new_engine == "groq" and not groq_clients:
        try:
            await event.edit("❌ Groq clients are not initialized (check GROQ_API_KEYS in .env)")
        except Exception:
            pass
        return
    elif new_engine == "local" and not local_ai_client:
        try:
            await event.edit("❌ Local AI client is not initialized")
        except Exception:
            pass
        return

    AI_ENGINE = new_engine
    engine_cooldowns.pop(AI_ENGINE, None)  # Reset cooldown if manually selected
    # Reset all key cooldowns for the selected engine
    if AI_ENGINE == "gemini":
        for i in range(len(gemini_cooldowns)):
            gemini_cooldowns[i] = 0.0
    elif AI_ENGINE == "groq":
        for i in range(len(groq_cooldowns)):
            groq_cooldowns[i] = 0.0
    active_model = MODEL if AI_ENGINE == "gemini" else (GROQ_MODEL if AI_ENGINE == "groq" else LOCAL_AI_MODEL)
    try:
        await event.edit(f"🔌 AI Engine switched to **{AI_ENGINE.upper()}** (Model: `{active_model}`)")
    except Exception:
        pass
    logger.info(f"AI Engine manually switched to: {AI_ENGINE.upper()}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai model (.+)$"))
async def switch_model(event):
    """Switch the model for the active engine. Usage: .ai model <model_name>"""
    global MODEL, GROQ_MODEL, LOCAL_AI_MODEL
    new_model = event.pattern_match.group(1).strip()

    if AI_ENGINE == "gemini":
        MODEL = new_model
    elif AI_ENGINE == "groq":
        GROQ_MODEL = new_model
    else:
        LOCAL_AI_MODEL = new_model

    try:
        await event.edit(f"🧠 Active model for **{AI_ENGINE.upper()}** switched to: `{new_model}`")
    except Exception:
        pass
    logger.info(f"Model for {AI_ENGINE.upper()} switched to: {new_model}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai clear$"))
async def clear_context(event):
    """Wipe history context for the current chat. Usage: .ai clear"""
    chat_id = event.chat_id
    chat_histories[chat_id] = []
    save_histories()
    try:
        await event.edit("🧹 Chat Context Cleared")
    except Exception:
        pass


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai status$"))
async def ai_status(event):
    """Display active bot stats and engine health. Usage: .ai status"""
    status = "✅ ENABLED" if auto_reply_enabled else "⛔ DISABLED"
    mode_status = "⚡ INSTANT" if INSTANT_REPLY else "👤 HUMAN-LIKE"
    chats_count = len(chat_histories)
    now = time.time()

    def get_engine_status_str(name, label, clients_list, cooldowns_list, model_val):
        if not clients_list:
            return f"  • {label}: ❌ Not Configured"
        
        status_lines = [f"  • {label}: `{model_val}`"]
        for idx, (client_inst, masked) in enumerate(clients_list):
            cooldown_until = cooldowns_list[idx]
            remaining = int(cooldown_until - now)
            if remaining > 0:
                status_lines.append(f"      Key #{idx+1} ({masked}): ⏳ Cooldown ({remaining}s)")
            else:
                status_lines.append(f"      Key #{idx+1} ({masked}): 🟢 Healthy")
        return "\n".join(status_lines)

    gemini_info = get_engine_status_str("gemini", "Google Gemini", gemini_clients, gemini_cooldowns, MODEL)
    groq_info = get_engine_status_str("groq", "Groq Cloud", groq_clients, groq_cooldowns, GROQ_MODEL)
    
    if not local_ai_client:
        local_info = "  • Local Ollama: ❌ Not Configured"
    else:
        cooldown_until = engine_cooldowns.get("local", 0)
        remaining = int(cooldown_until - now)
        if remaining > 0:
            local_info = f"  • Local Ollama: `{LOCAL_AI_MODEL}` ⏳ Cooldown ({remaining}s)"
        elif AI_ENGINE == "local":
            local_info = f"  • Local Ollama: `{LOCAL_AI_MODEL}` 🟢 Active & Healthy"
        else:
            local_info = f"  • Local Ollama: `{LOCAL_AI_MODEL}` 🔵 Ready & Healthy"

    try:
        await event.edit(
            f"🤖 **AI Auto-Reply Status:** {status}\n"
            f"⚡ **Response Mode:** {mode_status}\n"
            f"💬 **Active Chats Cache:** {chats_count}\n"
            f"📚 **Context Window:** {MAX_HISTORY} messages\n\n"
            f"🔌 **AI Engines & Status:**\n"
            f"{gemini_info}\n"
            f"{groq_info}\n"
            f"{local_info}"
        )
    except Exception:
        pass


# --- Incoming Message Handlers & Human Simulation ---

# Active message accumulation tasks: chat_id -> Task
active_accumulators = {}
# Chat queues for combining fast consecutive messages: chat_id -> List[str]
message_buffers = defaultdict(list)


@client.on(events.NewMessage(incoming=True))
async def handle_incoming(event):
    """Fires on incoming messages, filtering DMs and queuing them to simulate human behavior."""
    if not auto_reply_enabled:
        return

    # Strictly process private direct messages (DMs) only
    if not event.is_private:
        return

    # Ignore media-only messages or developer bot commands
    if not event.text or event.text.startswith((".", "/")):
        return

    # Critical bug fix: Ignore official Telegram bots and self-messages to prevent infinite loops
    try:
        sender = await event.get_sender()
        if sender and (getattr(sender, 'bot', False) or sender.id == my_user_id):
            return
    except Exception as e:
        logger.warning(f"Failed to get sender info: {e}")

    chat_id = event.chat_id
    
    # Store incoming text in the accumulation buffer
    message_buffers[chat_id].append(event.text)

    # If the user is typing a stream of messages, reset the response wait timer
    if chat_id in active_accumulators:
        active_accumulators[chat_id].cancel()

    # Wait for 2.0 seconds of silence before packing and generating a combined reply
    task = asyncio.create_task(wait_and_respond(chat_id, event))
    active_accumulators[chat_id] = task


async def wait_and_respond(chat_id: int, event):
    """Waits for consecutive texts to conclude and passes the combined message to the reply routine."""
    try:
        # In instant mode, wait for a very short debounce (0.2s) instead of 2.0s to feel immediate
        wait_delay = 0.2 if INSTANT_REPLY else 2.0
        await asyncio.sleep(wait_delay)
        
        texts = message_buffers[chat_id]
        message_buffers[chat_id] = []
        
        active_accumulators.pop(chat_id, None)

        combined_text = " ".join(texts)

        # Trigger reply generation, realistic delay, typing and delivery
        await process_reply(chat_id, combined_text, event)

    except asyncio.CancelledError:
        # Task was cancelled because a new message was received. Exit quietly.
        pass


def introduce_typo(text: str) -> tuple[str, str | None]:
    """
    Randomly injects a realistic keyboard typo in the text with a 15% probability.
    Returns a tuple: (typo_text, clean_correction_word).
    If no typo is introduced, returns (original_text, None).
    """
    # 15% chance to make a typo
    if random.random() > 0.15:
        return text, None
        
    words = text.split()
    # Find Russian words of length >= 5, supporting letter "ё" and "Ё" correctly
    eligible_indices = []
    cyrillic_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    for idx, w in enumerate(words):
        clean_w = "".join(c for c in w if c.isalpha())
        if len(clean_w) >= 5 and any(c.lower() in cyrillic_chars for c in clean_w):
            eligible_indices.append(idx)
            
    if not eligible_indices:
        return text, None
        
    target_idx = random.choice(eligible_indices)
    original_word = words[target_idx]
    
    # Swap two adjacent letters in the middle of the selected word
    word_chars = list(original_word)
    letter_indices = [i for i, c in enumerate(original_word) if c.isalpha()]
    
    if len(letter_indices) < 4:
        return text, None
        
    swap_pos = random.randint(1, len(letter_indices) - 3)
    idx1 = letter_indices[swap_pos]
    idx2 = letter_indices[swap_pos + 1]
    
    word_chars[idx1], word_chars[idx2] = word_chars[idx2], word_chars[idx1]
    typo_word = "".join(word_chars)
    
    words[target_idx] = typo_word
    typo_text = " ".join(words)
    
    # Extract the clean correction word
    clean_correction = "".join(c for c in original_word if c.isalpha())
    
    return typo_text, clean_correction


async def process_reply(chat_id: int, combined_text: str, event):
    """Handles response generation, read delay, typing state, typos, and delivery."""
    sender_name = "Friend"
    if event.sender:
        sender_name = event.sender.first_name or "Friend"

    logger.info(f"📩 Incoming DM from {sender_name} ({chat_id}): {combined_text}")

    if INSTANT_REPLY:
        # Show typing status while generating to indicate immediate activity
        try:
            async with client.action(event.input_chat, "typing"):
                original_reply = await generate_response(chat_id, combined_text)
        except Exception:
            original_reply = await generate_response(chat_id, combined_text)
        
        logger.info(f"📤 Sent reply instantly → ({chat_id}): {original_reply}")
        await event.respond(original_reply)
        return

    # Generate response
    original_reply = await generate_response(chat_id, combined_text)

    # Introduce a typo with a 15% probability
    typo_reply, correction = introduce_typo(original_reply)

    # ⏳ "Reading Effect" Delay (Simulate the user reading the message before typing)
    read_delay = random.uniform(1.2, 2.5)
    logger.info(f"👀 Simulating reading... ({read_delay:.2f}s)")
    await asyncio.sleep(read_delay)

    # Calculate realistic typing speed (approx 0.05s per character + random variance)
    typing_speed = random.uniform(0.04, 0.07)
    delay = len(typo_reply) * typing_speed
    delay = max(1.0, min(delay, 4.0))

    logger.info(f"⏳ Simulating typing... ({delay:.2f}s)")
    
    # Display the "typing..." status in chat
    try:
        async with client.action(event.input_chat, "typing"):
            await asyncio.sleep(delay)
    except Exception as e:
        logger.warning(f"Failed to send typing status: {e}")
        await asyncio.sleep(delay)

    logger.info(f"📤 Sent reply → ({chat_id}): {typo_reply}")

    # Send the response (with typo if one occurred)
    await event.respond(typo_reply)

    # If a typo was made, send a self-correction after a brief human-like pause
    if correction:
        correction_delay = random.uniform(1.2, 2.2)
        logger.info(f"✏️ Typo made! Sending correction in {correction_delay:.2f}s...")
        await asyncio.sleep(correction_delay)
        
        try:
            async with client.action(event.input_chat, "typing"):
                await asyncio.sleep(0.6)
        except Exception:
            await asyncio.sleep(0.6)
            
        correction_text = f"*{correction}"
        logger.info(f"📤 Sent correction → ({chat_id}): {correction_text}")
        await event.respond(correction_text)


# --- Startup Sequence ---

async def my_password_callback():
    """Handles Telegram 2FA Cloud Password prompt securely in console."""
    env_password = os.getenv("TG_PASSWORD")
    if env_password:
        logger.info("🔑 Utilizing Telegram 2FA cloud password from .env file")
        return env_password

    print("\n" + "═" * 60)
    print("🔑 Telegram requested your 2FA Cloud Password.")
    print("⚠️  IMPORTANT: Characters will NOT show on the screen when you type or paste!")
    print("   This is standard secure terminal behavior.")
    print("   Type/paste your password and press ENTER.")
    print("═" * 60 + "\n")

    import getpass
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, getpass.getpass, "Enter Cloud Password: ")


async def main():
    """Main application loop."""
    print(f"🚀 Telegram AI Userbot is booting up...")
    if AI_ENGINE == "local":
        print(f"🤖 Engine: LOCAL LLM (Ollama/LM Studio)")
        print(f"📡 API URL: {LOCAL_AI_BASE_URL}")
        print(f"🧠 Model: {LOCAL_AI_MODEL}")
    elif AI_ENGINE == "groq":
        print(f"🤖 Engine: Groq Cloud API")
        print(f"🧠 Model: {GROQ_MODEL}")
    else:
        print(f"🤖 Engine: Google Gemini API")
        print(f"🧠 Model: {MODEL}")
    print(f"📚 Context Window Limit: {MAX_HISTORY} messages")
    print("─" * 50)
    print("Self-Admin Commands (type these in any chat):")
    print("  .ai on              — Enable auto-reply")
    print("  .ai off             — Disable auto-reply")
    print("  .ai instant on      — Enable instant replies (no delay, no typos)")
    print("  .ai instant off     — Disable instant replies (human-like simulation)")
    print("  .ai engine <engine> — Switch active engine (gemini, groq, local)")
    print("  .ai model <model>   — Switch active model for the current engine")
    print("  .ai clear           — Wipe chat history cache for the current chat")
    print("  .ai status          — View current running statistics")
    print("─" * 50)

    # Load cache
    load_histories()

    global my_user_id, my_username
    await client.start(password=my_password_callback)
    me = await client.get_me()
    my_user_id = me.id
    my_username = me.username or ""
    print(f"✅ Successfully logged in as: {me.first_name} (@{me.username})")
    print("⏳ Listening for incoming direct messages...")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
