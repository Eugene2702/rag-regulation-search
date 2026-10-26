from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue  # Формируем фильтр для Qdrant (искать только там, где есть наша роль)
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from app.llm import generate_answer
import os

# Модели данных
class SearchQuery(BaseModel):
    text: str
    top_k: int = 3
    user_role: str = "user"  # Добавлено для ACL

class SearchResult(BaseModel):
    score: float
    text: str
    acl: list[str]

# Добавляем поле user_role в запрос
class AskQuery(BaseModel):
    text: str
    top_k: int = 3
    user_role: str = "user"  # По умолчанию обычный пользователь

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

# Функция жизненного цикла (Lifespan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Инициализация Qdrant и модели")
    app.state.qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
    app.state.model = SentenceTransformer("intfloat/multilingual-e5-base")
    print("Сервис готов к работе")
    
    yield  # Сервер работает и принимает запросы
    
    print("Выключение сервера. Очистка ресурсов")

app = FastAPI(title="RAG Regulations Search", version="0.1.0", lifespan=lifespan)

# Эндпоинты
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/search", response_model=list[SearchResult])
def search(query: SearchQuery):
    model = app.state.model
    qdrant = app.state.qdrant
    
    query_vector = model.encode(f"query: {query.text}").tolist()
    
    # Фильтр по правам доступа (ACL)
    acl_filter = Filter(
        must=[
            FieldCondition(
                key="acl",
                match=MatchValue(value=query.user_role)
            )
        ]
    )
    
    results = qdrant.query_points(
        collection_name="regulations",
        query=query_vector,
        limit=query.top_k,
        with_payload=True,
        query_filter=acl_filter  # Применяем фильтр
    )
    
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

@app.post("/ask", response_model=AskResponse)
def ask(query: AskQuery):
    model = app.state.model
    qdrant = app.state.qdrant
    
    # Ищем контекст (с учетом ACL)
    query_vector = model.encode(f"query: {query.text}").tolist()
    
    acl_filter = Filter(
        must=[
            FieldCondition(
                key="acl",  # Имя поля в payload
                match=MatchValue(value=query.user_role) # Ищем нашу роль
            )
        ]
    )
    
    # Передаем этот фильтр в метод поиска
    results = qdrant.query_points(
        collection_name="regulations",
        query=query_vector,
        limit=query.top_k,
        with_payload=True,
        query_filter=acl_filter  # Применяем фильтр
    )
    
    # Собираем найденные куски в один текст
    context_texts = [hit.payload["text"] for hit in results.points]
    context = "\n\n".join(context_texts)
    
    if not context_texts:
         return AskResponse(answer="В документах нет информации.", sources=[])

    # Формируем улучшенный промпт (боремся с over-refusal)
    prompt = f"""Ты - полезный ассистент службы поддержки. Твоя задача - ответить на вопрос пользователя, опираясь на предоставленный контекст.
Внимательно прочитай контекст. Если ответ можно вывести из контекста (даже если слова сформулированы иначе, например "ждать" и "рассматривается"), обязательно дай этот ответ.

Если в контексте ВООБЩЕ нет нужной информации, напиши ровно одну фразу: "В документах нет информации."

Контекст:
{context}

Вопрос пользователя: {query.text}

Ответ:"""
    
    # Идем в LLM
    print("Генерация ответа")
    answer = generate_answer(prompt)
    
    return AskResponse(answer=answer, sources=context_texts)