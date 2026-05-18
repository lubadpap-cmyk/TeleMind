# 🧠 TeleMind

An advanced, human-like automated reply assistant designed for personal Telegram accounts. It works as a **Userbot**—meaning it replies to direct messages (DMs) from your personal profile, simulating natural human behavior.

Powered by **Google Gemini**, **Groq Cloud (Llama 3.3)**, or **local LLMs** (via Ollama or LM Studio).

---

## ✨ Premium Features

*   **🔌 Triple Engine Support**: Switch seamlessly between **Google Gemini API**, ultra-fast **Groq Cloud API** (Llama 3.3 70B), or fully offline **Local models** (via Ollama/LM Studio).
*   **🧠 Message Accumulator (Consecutive stream packing)**: Solves the bot-spam problem. If a friend sends multiple short messages in a row (e.g. *“hey”*, *“you there?”*, *“let's play CS”*), the bot waits for a 2-second silence, combines them into a single prompt, and generates **one cohesive, natural reply**.
*   **⏳ "Reading Effect" Delay**: Simulates natural reaction times. The bot waits for a random period (1.2s to 2.5s) *before* showing the "typing..." status in chat.
*   **✏️ Dynamic Human Typos & Self-Correction**: Implements a 15% chance to accidentally swap two letters in a long word (simulating fast keyboard typing), sends the message, and then follows up 1.5s later with a corrected word prefixed by `*` (e.g., sends *“let's posmторим”* followed by *“*посмотрим”*).
*   **💾 Context Persistence (Autosave)**: Saves conversation history to a local `chat_histories.json` database. History is automatically restored upon bot reboot so it never suffers from amnesia.
*   **🔒 Strict DM-Only Guard**: Safe and private. The bot completely ignores groups, channels, and other bots to protect your rate limits and account standing.

---

## 🛠 Self-Admin Chat Commands

Type these commands in **any chat** (or send them to yourself in "Saved Messages") to manage your AI bot:

| Command | Action |
| :--- | :--- |
| `.ai on` | Enable AI auto-reply |
| `.ai off` | Disable AI auto-reply |
| `.ai clear` | Wipe context history for the current chat |
| `.ai status` | View active stats (engine, model, cache size, state) |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Obtain Telegram API Credentials
1. Visit [my.telegram.org](https://my.telegram.org) and log in.
2. Go to **API development tools**.
3. Create a new application to obtain your `API_ID` and `API_HASH`.

### 3. Clone & Install Dependencies
Clone the repository, navigate into the project directory, and install the required dependencies:
```bash
# Clone the repository
git clone https://github.com/lubadpap-cmyk/TeleMind.git

# Navigate into the project folder
cd TeleMind

# Install required Python packages
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root folder of the project (you can copy `.env.example` as a template):
```bash
# Telegram Credentials
API_ID=your_telegram_api_id
API_HASH=your_telegram_api_hash

# 2FA Password (Optional: enter your cloud password to bypass manual console prompts at login)
TG_PASSWORD=your_telegram_2fa_cloud_password

# Persona details
MY_NAME=

# Active Engine (gemini / groq / local)
AI_ENGINE=groq

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key

# Groq Cloud Configuration
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

# Local Ollama / LM Studio Configuration
LOCAL_AI_BASE_URL=http://localhost:11434/v1
LOCAL_AI_MODEL=llama3.2:1b
```

---

## 🚀 Running the Bot

To start your userbot, execute:
```bash
python bot.py
```

*   **First Run Authorization**: On your first boot, Telegram will request your phone number (including country code, e.g. `+7...` or `+380...`) and a login confirmation code sent via Telegram.
*   **2FA Prompt**: If you have two-factor authentication enabled, it will request your cloud password (you can fill it in your `.env` under `TG_PASSWORD` to authorize automatically).
*   **Session File**: A file named `my_account.session` will be created in your directory. **Keep this file strictly confidential**, as it grants access to your account.

---

## 📁 Repository Structure

```
├── bot.py                  # Main Telethon script with AI integrations
├── requirements.txt        # Required python packages
├── .gitignore              # Tells git to ignore session keys and .env secrets
├── .env.example            # Configuration boilerplate
└── README.md               # English documentation (This file)
```

---

## ⚠️ Security Note
Your `.env` and `my_account.session` files contain secret credentials. **Never commit them to GitHub** or share them online. A pre-configured `.gitignore` is provided to ensure these files remain safe and local.

---

## 👤 Author
Created with ❤️ by [@lubadpap-cmyk](https://github.com/lubadpap-cmyk)
