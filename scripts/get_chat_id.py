"""Find your Telegram chat ID, and explain it when it cannot be found.

    $env:TELEGRAM_BOT_TOKEN = "123456789:AAF..."
    python scripts/get_chat_id.py

The plain getUpdates call in most setup guides fails silently in three situations
that look identical from the outside, so this checks each one and says which
applies. The token is read from the environment and is never printed.
"""

from __future__ import annotations

import os
import sys

import requests

API = "https://api.telegram.org"
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"


def call(token: str, method: str, **params):
    response = requests.get(f"{API}/bot{token}/{method}", params=params, timeout=20)
    try:
        return response.json()
    except ValueError:
        return {"ok": False, "description": f"HTTP {response.status_code}, not JSON"}


def main() -> int:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        print(f"Set {TOKEN_ENV} first, then re-run. The token is never printed.")
        return 2

    # 1. Is the token even valid?
    me = call(token, "getMe")
    if not me.get("ok"):
        print(f"✗ The token was rejected: {me.get('description')}")
        print("  Re-copy it from @BotFather — it must include the digits, the colon,")
        print("  and the long suffix, with no spaces.")
        return 1
    bot = me["result"]
    print(f"✓ Token valid. Bot is @{bot.get('username')} ({bot.get('first_name')}).")
    print(f"  Make sure that is the bot you messaged — not another one in your list.")

    # 2. A webhook silently disables getUpdates. This is the usual culprit.
    hook = call(token, "getWebhookInfo")
    hook_url = (hook.get("result") or {}).get("url") if hook.get("ok") else None
    if hook_url:
        print(f"\n✗ A webhook is set ({hook_url}).")
        print("  Telegram will not deliver updates by BOTH webhook and getUpdates, so")
        print("  getUpdates returns empty no matter how many messages you send.")
        print("  Remove it with:")
        print(f'    python -c "import os,requests;requests.get(f\'{API}/bot\'+'
              f'os.environ[\'{TOKEN_ENV}\']+\'/deleteWebhook\')"')
        print("  then message the bot again and re-run this script.")
        return 1
    print("✓ No webhook set, so getUpdates should work.")

    # 3. Any pending updates?
    updates = call(token, "getUpdates", timeout=0)
    if not updates.get("ok"):
        print(f"\n✗ getUpdates failed: {updates.get('description')}")
        return 1

    chats = {}
    for update in updates.get("result", []):
        message = (update.get("message") or update.get("edited_message")
                   or update.get("channel_post") or {})
        chat = message.get("chat") or {}
        if chat.get("id") is not None:
            chats[chat["id"]] = chat

    if not chats:
        print("\n✗ No messages waiting.")
        print(f"  Open Telegram, find @{bot.get('username')}, and send it any message")
        print("  (a bot cannot start the conversation, so it has no chat until you do).")
        print("  Then re-run this script within 24 hours — Telegram discards older updates.")
        print("\n  Shortcut: message @userinfobot instead. It replies with your numeric")
        print("  user ID, which IS your chat ID for a one-to-one chat with your bot.")
        return 1

    print("\n✓ Found:")
    for chat_id, chat in chats.items():
        kind = chat.get("type", "?")
        who = chat.get("username") or chat.get("title") or chat.get("first_name") or ""
        print(f"    TELEGRAM_CHAT_ID = {chat_id}    ({kind}, {who})")
    print("\nAdd that as the TELEGRAM_CHAT_ID Actions secret.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
