import json
import httpx

API_URL = "http://127.0.0.1:8000/ask"

def evaluate():
    with open("testset.json", "r", encoding="utf-8") as f:
        testset = json.load(f)

    hit_count = 0
    answerable_count = 0
    refusal_count = 0
    unanswerable_count = 0

    print(f"Запуск оценки на {len(testset)} вопросах")
    
    for item in testset:
        print(f"\nВопрос {item['id']}: {item['question']}")
        
        # Передаем роль user по умолчанию
        payload = {"text": item["question"], "top_k": 3, "user_role": "user"}
        
        try:
            # Ждем до 5 минут, так как модель на CPU думает долго
            response = httpx.post(API_URL, json=payload, timeout=300.0)
            response.raise_for_status()
            data = response.json()
            
            answer = data["answer"].lower()
            sources_text = " ".join(data["sources"]).lower()
            
            if item["answerable"]:
                answerable_count += 1
                # Проверяем Hit@5: есть ли ключевые слова в источниках
                is_hit = any(kw in sources_text for kw in item["expected_keywords"])
                if is_hit:
                    print("Нашли в источниках")
                    hit_count += 1
                else:
                    print(f"Не нашли ключевые слова: {item['expected_keywords']}")
            else:
                unanswerable_count += 1
                # Проверяем Correct Refusal
                if "нет информации" in answer or "не найдено" in answer or "отсутствует" in answer:
                    print(f"Correct Refusal (Модель честно отказалась)")
                    refusal_count += 1
                else:
                    print(f"Модель выдумала ответ: {answer[:50]}...")
                    
        except Exception as e:
            print(f"Ошибка при запросе: {e}")

    hit_rate = (hit_count / answerable_count * 100) if answerable_count > 0 else 0
    refusal_rate = (refusal_count / unanswerable_count * 100) if unanswerable_count > 0 else 0

    print("\n" + "="*40)
    print("ИТОГОВЫЕ МЕТРИКИ:")
    print(f"Hit@5: {hit_rate:.1f}% ({hit_count} из {answerable_count})")
    print(f"Correct Refusal: {refusal_rate:.1f}% ({refusal_count} из {unanswerable_count})")
    print("="*40)

if __name__ == "__main__":
    evaluate()