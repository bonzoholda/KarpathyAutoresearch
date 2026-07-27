# main.py
import os
import time
import requests
from openai import OpenAI
import evaluator

# Menggunakan Groq / OpenRouter API untuk mutasi cepat
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", "your_groq_key")
)

OPENCLAW_WEBHOOK_URL = os.getenv("OPENCLAW_WEBHOOK_URL", "")
OPENCLAW_TOKEN = os.getenv("OPENCLAW_GATEWAY_TOKEN", "")

BEST_SHARPE_SCORE = 1.2  # Baseline awal

def mutate_strategy_code():
    """Menggunakan LLM untuk menulis ulang strategy_candidate.py"""
    with open("strategy_candidate.py", "r") as f:
        current_code = f.read()

    prompt = f"""
    Berikut adalah kode strategi Python saat ini:
    ```python
    {current_code}
    ```
    Tugas Anda: Buat variasi/mutasi logika baru untuk fungsi `generate_signals(df)`.
    Gunakan kombinasi teknik seperti EMA, RSI, Bollinger Bands, ATR, atau Volume Spikes.
    Aturan: Output HARUS murni kode Python valid tanpa penjelasan atau markdown.
    """

    res = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    new_code = res.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    return new_code

def notify_openclaw(new_code, score):
    """Mengirim strategi baru ke OpenClaw Webhook Listener"""
    if not OPENCLAW_WEBHOOK_URL:
        print("Webhook URL empty. Skipping notification.")
        return

    payload = {
        "token": OPENCLAW_TOKEN,
        "event": "STRATEGY_MUTATION_SUCCESS",
        "sharpe_score": score,
        "strategy_code": new_code,
        "instructions": f"Gunakan indikator teknikal berikut sebagai acuan utama analisis: {new_code}"
    }
    try:
        r = requests.post(OPENCLAW_WEBHOOK_URL, json=payload, timeout=10)
        print(f"⚡ Pushed to OpenClaw: Status {r.status_code}")
    except Exception as e:
        print(f"Failed to notify OpenClaw: {e}")

def research_loop():
    global BEST_SHARPE_SCORE
    print("🚀 Starting Autoresearch Self-Healing Loop...")
    
    while True:
        print("\n[STEP] Mutating strategy code via LLM...")
        try:
            candidate = mutate_strategy_code()
            
            # Simpan sementara kode baru
            with open("strategy_candidate.py", "w") as f:
                f.write(candidate)
                
            # Evaluasi Kinerja
            score = evaluator.run_evaluation()
            print(f"[EVAL] Candidate Sharpe Score: {score} vs Best: {BEST_SHARPE_SCORE}")
            
            # Jika LEBIH BAGUS -> Simpan sebagai Champion & Notify OpenClaw
            if score > BEST_SHARPE_SCORE:
                BEST_SHARPE_SCORE = score
                print(f"🔥 NEW CHAMPION FOUND! Sharpe: {score}")
                notify_openclaw(candidate, score)
            else:
                print("❌ Performance lower or invalid. Reverting...")
                
        except Exception as e:
            print(f"Loop Exception: {e}")
            
        time.sleep(120)  # Siklus eksperimen setiap 2 menit

if __name__ == "__main__":
    research_loop()
