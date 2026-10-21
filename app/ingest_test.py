from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Подключаемся к Qdrant
client = QdrantClient("localhost", port=6333)
COLLECTION_NAME = "regulations"

# Создаем коллекцию, если ее еще нет
# Вектор у e5-base имеет размер 768
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
    print(f"Коллекция {COLLECTION_NAME} создана.")

# Загружаем модель эмбеддингов
print("Загрузка модели эмбеддингов...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Тестовый документ (регламент)
document_text = """
Правила подключения тарифов.
Заявка на подключение рассматривается 3 рабочих дня. 
Если клиент является VIP-персоной, заявка рассматривается 1 рабочий день.
Для подключения необходим паспорт. Без паспорта подключение невозможно.
Стоимость подключения базового тарифа составляет 500 рублей.
"""

# Простой чанкинг (режем по абзацам/предложениям) для тестов
# В реальном проекте напишем рекурсивный чанкинг
chunks = [line.strip() for line in document_text.strip().split("\n") if line.strip()]

print(f"Нарезано чанков: {len(chunks)}")

# Превращаем в эмбеддинги и готовим к отправке
points = []
for i, chunk in enumerate(chunks):
    
    # Модель e5 требует, чтобы к тексту спереди добавляли специальные префиксы:
    # Для документов, которые кладем в базу, нужен префикс "passage: ".
    # Для вопросов, которые будем искать, нужен префикс "query: ".
    # Если этого не делать, качество поиска упадет.
    text_to_embed = f"passage: {chunk}"
    
    # Получаем вектор (список чисел)
    vector = model.encode(text_to_embed).tolist()
    
    # Формируем точку для Qdrant: ID, вектор, полезная нагрузка (метаданные)
    points.append(
        PointStruct(
            id=str(uuid.uuid4()), # Уникальный ID
            vector=vector,
            payload={
                "text": chunk, # Сам текст
                "doc_id": "test_doc_id", # ID документа
                "section": f"part_{i}", # Раздел
                "acl": ["admin", "user"], # Права доступа (пока просто тест)
            }
        )
    )
       
# Заносим в Qdrant    
client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=points
)

print("Данные успешно занесены в Qdrant.")