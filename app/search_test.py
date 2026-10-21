from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Подключаемся к базе
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "regulations"

# Загружаем модель
print("Загрузка модели эмбеддингов...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Наш вопрос, сформулированый по другому
question = "Сколько времени нужно ждать, чтобы подключили тариф?"

# Для вопросов используем префикс "query: "
question_to_embed = f"query: {question}"
question_vector = model.encode(question_to_embed).tolist()

# Ищем в Qdrant 2 самых похожих чанка (top_k=2).
# В последствии поменял limit на 5,
# потому что в найденных 2-х чанках не нашлось нужного предложения
print(f"\nЗадаем вопрос: {question}")
print("Результаты поиска:")
print("-" * 40)

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=question_vector, # вектор вопроса
    limit=5,  # Сколько кусков достать (top_k)
    with_payload=True # Просим вернуть не только векторы, но и текст с метаданными
)

# Выводим результаты
for i, hit in enumerate(results.points, 1):
    print(f"Результат {i}:")
    print(f"Точность (score): {hit.score:.4f}")
    print(f"Текст: {hit.payload['text']}")
    print(f"Права доступа: {hit.payload['acl']}")
    print("-" * 40)