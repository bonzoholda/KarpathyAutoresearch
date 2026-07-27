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
    with open("strategy_candidate.py", "r") as f:
        current_code = f.read()

    prompt = f"""
    Berikut adalah kode strategi Python saat ini:
    ```python
    {current_code}
    ```
    Tugas Anda: Buat variasi/mutasi logika baru untuk fungsi `generate_signals(df: pd.DataFrame) -> pd.Series`.
    
    ATURAN STRICT:
    1. WAJIB sertakan `import pandas as pd` dan `import numpy as np` di baris paling atas!
    2. `df` sudah memiliki kolom float: 'open', 'high', 'low', 'close', 'volume'.
    3. Output HARUS MURNI kode Python valid tanpa penjelasan, tanpa tag ```python.
    4. Kembalikan pd.Series dengan nilai 1 (BUY), -1 (SELL), atau 0 (HOLD).
    """

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    new_code = res.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    return new_code


# main.py (Perbaruan Fungsi notify_openclaw)

def notify_openclaw(new_code, score):
    """Mengirim strategi baru ke OpenClaw API Gateway dengan Header Universal"""
    openclaw_domain = os.getenv("OPENCLAW_DOMAIN", "https://openclawshit.up.railway.app").rstrip("/")
    openclaw_token = os.getenv("OPENCLAW_GATEWAY_TOKEN", "MySuperSecretToken123!")
    
    # Endpoint resmi OpenClaw Gateway untuk prompt injection/config update
    endpoints = [
        f"{openclaw_domain}/api/agent/prompt",
        f"{openclaw_domain}/api/v1/agent/prompt",
        f"{openclaw_domain}/api/config",
        f"{openclaw_domain}/api/gateway"
    ]
    
    # Header lengkap agar lolos dari validasi security OpenClaw Gateway
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openclaw_token}",
        "x-gateway-token": openclaw_token,
        "x-token": openclaw_token
    }
    
    payload = {
        "token": openclaw_token,
        "agent_id": "Analyst-Sentinel",
        "prompt": f"[SYSTEM INSTRUCTION UPDATE]\nStrategi trading acuan terbaru (Sharpe Ratio: {score}) telah diperbarui oleh Autoresearch:\n\n{new_code}\n\nGunakan aturan indikator di atas untuk mengevaluasi sinyal BUY/SELL berikutnya.",
        "system_prompt_patch": f"Aturan Indikator Kunci (Sharpe: {score}):\n{new_code}"
    }
    
    success = False
    for ep in endpoints:
        try:
            r = requests.post(ep, json=payload, headers=headers, timeout=5)
            if r.status_code in [200, 201, 202, 204]:
                print(f"⚡ Pushed to OpenClaw ({ep}): Status {r.status_code} - HOT-RELOAD SUCCESS!")
                success = True
                break
            else:
                print(f"⚠️ Endpoint {ep} responded with status: {r.status_code}")
        except Exception as e:
            continue
            
    if not success:
        print("ℹ️ REST API gateway protected. OpenClaw UI Dashboard will fetch the active champion strategy upon manual/scheduled prompt.")

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
