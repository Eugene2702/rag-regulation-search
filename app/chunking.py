import re
from transformers import AutoTokenizer

# Загружаем токенизатор от модели e5 (чтобы точно считать токены)
# Это не скачивает веса модели, только словарь токенов.
tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def recursive_chunk(text: str, max_tokens: int = 400, overlap_tokens: int = 50) -> list[str]:
    
    # Режет текст на чанки, не превышающие max_tokens.
    # Старается резать по абзацам, затем по предложениям.
    # Базовый случай: если текст уже влезает - возвращаем как есть
    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    # Пытаемся разделить по абзацам
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Если абзац сам по себе больше лимита, режем его по предложениям
        if count_tokens(para) > max_tokens:
            # Простая регулярка для разбиения по точкам, восклицательным и вопросительным знакам
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if count_tokens(current_chunk + " " + sentence) <= max_tokens:
                    current_chunk += " " + sentence
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # Начинаем новый чанк (учитывая overlap - берем кусок предыдущего)
                    # Для простоты пока просто начинаем с текущего предложения
                    current_chunk = sentence
        else:
            # Если абзац влезает
            if count_tokens(current_chunk + "\n\n" + para) <= max_tokens:
                current_chunk += "\n\n" + para
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks