"""
Telegram AI Userbot — An intelligent, human-like automated reply assistant
powered by Google Gemini, Groq (Llama 3.3), or local LLMs (via Ollama/LM Studio).
"""
import sys
import os
import logging
import asyncio
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

API_ID = os.getenv("API_ID")
API_HASH = os.getenv("API_HASH")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MY_NAME = os.getenv("MY_NAME", "YourName")

# AI Engine selection (gemini / local / groq)
AI_ENGINE = os.getenv("AI_ENGINE", "gemini").lower()
if os.getenv("USE_LOCAL_AI", "False").lower() == "true":
    AI_ENGINE = "local"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LOCAL_AI_BASE_URL = os.getenv("LOCAL_AI_BASE_URL", "http://localhost:11434/v1")
LOCAL_AI_MODEL = os.getenv("LOCAL_AI_MODEL", "llama3.2:1b")

if not API_ID or not API_HASH:
    raise ValueError("❌ Please specify API_ID and API_HASH in your .env file")
if AI_ENGINE == "gemini" and not GEMINI_API_KEY:
    raise ValueError("❌ Please specify GEMINI_API_KEY in your .env file for Gemini")
if AI_ENGINE == "groq" and not GROQ_API_KEY:
    raise ValueError("❌ Please specify GROQ_API_KEY in your .env file for Groq")

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- AI Clients Initialization ---
gemini_client = None
local_ai_client = None
MODEL = "gemini-flash-latest"

if AI_ENGINE in ("local", "groq"):
    try:
        from openai import AsyncOpenAI
        if AI_ENGINE == "groq":
            local_ai_client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY
            )
            logger.info(f"🔌 Initialized ultra-fast Groq API client (Model: {GROQ_MODEL})")
        else:
            local_ai_client = AsyncOpenAI(
                base_url=LOCAL_AI_BASE_URL,
                api_key="local-ai"
            )
            logger.info(f"🔌 Initialized Local AI client (Model: {LOCAL_AI_MODEL})")
    except ImportError:
        print(f"❌ Error: Selected engine is {AI_ENGINE}, but the 'openai' package is not installed.")
        print("👉 Please run in terminal: pip install openai")
        sys.exit(1)
else:
    from google import genai
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("🔌 Initialized Google Gemini API client")

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
            import json
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
        import json
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


# --- AI Text Generation ---

async def generate_response(chat_id: int, user_message: str) -> str:
    """Queries the active AI engine (Gemini, Groq, or Local) and returns a generated response."""
    add_message(chat_id, "user", user_message)
    history = get_history(chat_id)

    if AI_ENGINE in ("local", "groq"):
        # Format structured message list for OpenAI-compatible clients
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        active_model = LOCAL_AI_MODEL if AI_ENGINE == "local" else GROQ_MODEL

        try:
            response = await local_ai_client.chat.completions.create(
                model=active_model,
                messages=messages,
                temperature=0.7,
            )
            reply = response.choices[0].message.content.strip().replace("(", "").replace(")", "")
            add_message(chat_id, "assistant", reply)
            return reply
        except Exception as e:
            logger.error(f"AI Client Error ({AI_ENGINE}): {e}")
            return "сорри, нейросеть прилегла, напиши чуть позже 😅"
    else:
        # Generate content with Gemini API
        prompt = SYSTEM_PROMPT + "\n\nChat History:\n"
        for msg in history:
            prefix = "Friend" if msg["role"] == "user" else MY_NAME
            prompt += f"{prefix}: {msg['content']}\n"
        prompt += f"\nReply as {MY_NAME}:"

        # Simple retry logic in case of Google rate-limiting (429)
        for attempt in range(3):
            try:
                response = await gemini_client.aio.models.generate_content(
                    model=MODEL,
                    contents=[prompt],
                )
                reply = response.text.strip().replace("(", "").replace(")", "")
                add_message(chat_id, "assistant", reply)
                return reply

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"⚠️ Gemini Rate Limit (429). Waiting {wait_time}s... (Attempt {attempt+1}/3)")
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Gemini API Error: {e}")
                break
                
        return "сорри, что-то пошло не так, напиши позже 😅"


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
        await event.edit("✅ AI Auto-Reply Enabled")
    else:
        auto_reply_enabled = False
        await event.edit("⛔ AI Auto-Reply Disabled")

    logger.info(f"AI Auto-Reply state: {'ENABLED' if auto_reply_enabled else 'DISABLED'}")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai clear$"))
async def clear_context(event):
    """Wipe history context for the current chat. Usage: .ai clear"""
    chat_id = event.chat_id
    chat_histories[chat_id] = []
    save_histories()
    await event.edit("🧹 Chat Context Cleared")


@client.on(events.NewMessage(outgoing=True, pattern=r"^\.ai status$"))
async def ai_status(event):
    """Display active bot stats. Usage: .ai status"""
    status = "✅ ENABLED" if auto_reply_enabled else "⛔ DISABLED"
    chats_count = len(chat_histories)
    if AI_ENGINE == "local":
        engine_name = "Local Ollama"
        model_name = LOCAL_AI_MODEL
    elif AI_ENGINE == "groq":
        engine_name = "Groq Cloud"
        model_name = GROQ_MODEL
    else:
        engine_name = "Google Gemini"
        model_name = MODEL

    await event.edit(
        f"🤖 **AI Auto-Reply Status:** {status}\n"
        f"💬 **Active Chats Cache:** {chats_count}\n"
        f"🔌 **Active Engine:** {engine_name}\n"
        f"🧠 **Model:** `{model_name}`\n"
        f"📚 **Context Window:** {MAX_HISTORY} messages"
    )


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

    chat_id = event.chat_id
    
    # Store incoming text in the accumulation buffer
    message_buffers[chat_id].append(event.text)

    # If the user is typing a stream of messages, reset the response wait timer
    if chat_id in active_accumulators:
        active_accumulators[chat_id].cancel()

    # Wait for 2.0 seconds of silence before packing and generating a combined reply
    loop = asyncio.get_event_loop()
    task = loop.create_task(wait_and_respond(chat_id, event))
    active_accumulators[chat_id] = task


async def wait_and_respond(chat_id: int, event):
    """Waits for consecutive texts to conclude and passes the combined message to the reply routine."""
    try:
        await asyncio.sleep(2.0)
        
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
    import random
    
    # 15% chance to make a typo
    if random.random() > 0.15:
        return text, None
        
    words = text.split()
    # Find Russian words of length >= 5
    eligible_indices = []
    for idx, w in enumerate(words):
        clean_w = "".join(c for c in w if c.isalpha())
        if len(clean_w) >= 5 and any(1040 <= ord(c) <= 1103 for c in clean_w):
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

    # Generate response
    original_reply = await generate_response(chat_id, combined_text)

    # Introduce a typo with a 15% probability
    typo_reply, correction = introduce_typo(original_reply)

    import random
    import asyncio

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
    print("  .ai on     — Enable auto-reply")
    print("  .ai off    — Disable auto-reply")
    print("  .ai clear  — Wipe chat history cache for the current chat")
    print("  .ai status — View current running statistics")
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
    import asyncio
    asyncio.run(main())
