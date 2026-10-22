import httpx

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b-instruct"

def generate_answer(prompt: str) -> str:
    # Отправляет промпт в локальную модель Ollama и возвращает ответ
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Ты — полезный ассистент. Отвечай на русском языке, опираясь ТОЛЬКО на предоставленный контекст."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0 # Низкая температура значит меньше галлюцинаций, более точный ответ
    }
    
    # Отправляем запрос (таймаут 60 сек, так как на моем железе генерация может идти медленно)
    with httpx.Client(timeout=300.0) as client:
        response = client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]