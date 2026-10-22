from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from app.chunking import recursive_chunk # Импортируем наш чанкер
import uuid

client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "regulations"

# Пересоздаем коллекцию с нуля, чтобы почистить старые данные
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

model = SentenceTransformer("intfloat/multilingual-e5-base")

document_text = """
Правила подключения тарифов.
Заявка на подключение рассматривается 3 рабочих дня. 
Если клиент является VIP-персоной, заявка рассматривается 1 рабочий день.
Для подключения необходим паспорт. Без паспорта подключение невозможно.
Стоимость подключения базового тарифа составляет 500 рублей.
"""

# Используем умный чанкер
chunks = recursive_chunk(document_text, max_tokens=100) # Ставим лимит 100 для теста

print(f"Нарезано чанков: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"Чанк {i}: {c[:50]}...")

points = []
for i, chunk in enumerate(chunks):
    vector = model.encode(f"passage: {chunk}").tolist()
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk, "doc_id": "test_doc_1", "acl": ["admin", "user"]}
        )
    )

client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
print("Данные успешно залиты!")