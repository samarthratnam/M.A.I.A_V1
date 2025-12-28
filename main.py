import requests
import json
from dotenv import load_dotenv
import os
import sys, time, threading

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Spinner control flag
done = False

# Spinner function
def spinner():
    chars = "|/-\\"
    i = 0
    while not done:
        sys.stdout.write(f"\rThinking {chars[i % len(chars)]}")
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)

def ask_openrouter(prompt, history=None, stream=True):
    global done

    if history is None:
        history = []

    history.append({"role": "user", "content": prompt})

    # Start spinner in background
    done = False
    t = threading.Thread(target=spinner)
    t.start()

    start_time = time.time()

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "openai/gpt-oss-120b:free",
            "messages": history,
            "reasoning": {"enabled": True},
            "data_policy": "public",
            "stream": stream  # enables streaming from API side
        }),
        stream=stream  # enables streaming in Python
    )

    # Stop spinner once API responds
    done = True
    t.join()

    elapsed = round(time.time() - start_time, 2)
    print(f"\rConnected… ({elapsed}s)\n")

    # STREAMING OUTPUT
    if stream:
        print("Assistant: ", end="", flush=True)
        full_reply = ""

        for chunk in response.iter_lines():
            if chunk:
                try:
                    decoded = json.loads(chunk.decode().replace("data: ", ""))
                    if "choices" in decoded and len(decoded["choices"]) > 0:
                        delta = decoded["choices"][0].get("delta", {}).get("content")
                        if delta:
                            print(delta, end="", flush=True)
                            full_reply += delta
                except:
                    continue  # ignore keep-alive or non-JSON chunks

        print("\n")
        history.append({"role": "assistant", "content": full_reply})
        return full_reply, history

    # NORMAL (non-stream) fallback
    data = response.json()
    reply = data["choices"][0]["message"]["content"]
    print(reply, "\n")
    history.append({"role": "assistant", "content": reply})
    return reply, history


# Interactive loop
chat_history = []

while True:
    user_prompt = input("Enter prompt (or 'exit'): ")
    if user_prompt.lower() == "exit":
        print("Exiting...")
        break

    ask_openrouter(user_prompt, chat_history, stream=True)
