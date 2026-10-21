from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager  # <-- Новый импорт

# Модели данных
class SearchQuery(BaseModel):
    text: str
    top_k: int = 3

class SearchResult(BaseModel):
    score: float
    text: str
    acl: list[str]

# Функция жизненного цикла (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # код до yield выполняется при запуске сервера
    print("Инициализация Qdrant и модели...")
    app.state.qdrant = QdrantClient(host="localhost", port=6333)
    app.state.model = SentenceTransformer("intfloat/multilingual-e5-base")
    print("Сервис готов к работе!")
    
    yield  # сервер работает и принимает запросы
    
    # код после yield выполняется при выключении сервера
    print("Выключение сервера. Очистка ресурсов...")
    # если бы были открытые соединения с БД, закрыли бы их здесь

# Инициализация приложения с передачей lifespan
app = FastAPI(title="RAG Regulations Search", version="0.1.0", lifespan=lifespan)

# Эндпоинты
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/search", response_model=list[SearchResult])
def search(query: SearchQuery):
    # Достаем модель и клиент из состояния приложения (app.state)
    model = app.state.model
    qdrant = app.state.qdrant
    
     # Кодируем вопрос (с префиксом query:)
    query_vector = model.encode(f"query: {query.text}").tolist()
    
     # Ищем в Qdrant
    results = qdrant.query_points(
        collection_name="regulations",
        query=query_vector,
        limit=query.top_k,
        with_payload=True
    )
    
    # Форматируем ответ
    response = []
    for hit in results.points:
        response.append(
            SearchResult(
                score=hit.score,
                text=hit.payload["text"],
                acl=hit.payload.get("acl", [])
            )
        )
    return response