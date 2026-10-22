from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
from app.llm import generate_answer # Импортируем нашу функцию

class AskQuery(BaseModel):
    text: str
    top_k: int = 3

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Инициализация Qdrant и модели...")
    app.state.qdrant = QdrantClient(host="localhost", port=6333)
    app.state.model = SentenceTransformer("intfloat/multilingual-e5-base")
    print("Сервис готов к работе!")
    yield
    print("Выключение сервера...")

app = FastAPI(title="RAG Regulations Search", version="0.1.0", lifespan=lifespan)

@app.post("/ask", response_model=AskResponse)
def ask(query: AskQuery):
    model = app.state.model
    qdrant = app.state.qdrant
    
    # Ищем контекст
    query_vector = model.encode(f"query: {query.text}").tolist()
    results = qdrant.query_points(
        collection_name="regulations",
        query=query_vector,
        limit=query.top_k,
        with_payload=True
    )
    
    # Собираем найденные куски в один текст
    context_texts = [hit.payload["text"] for hit in results.points]
    context = "\n\n".join(context_texts)
    
    if not context_texts:
         return AskResponse(answer="В документах нет информации по этому вопросу.", sources=[])

    # Формируем промпт по правилам Prompt Engineering
    prompt = f"""Ты - полезный ассистент службы поддержки. Твоя задача - ответить на вопрос пользователя, опираясь на предоставленный контекст.
        Внимательно прочитай контекст. Если ответ можно вывести из контекста (даже если слова сформулированы иначе, например "ждать" и "рассматривается"), обязательно дай этот ответ.

        Если в контексте ВООБЩЕ нет нужной информации, напиши ровно одну фразу: "В документах нет информации."

        Контекст:
        {context}

        Вопрос пользователя: {query.text}

        Ответ:
    """
    
    # Идем в LLM
    print("Генерация ответа...")
    answer = generate_answer(prompt)
    
    return AskResponse(answer=answer, sources=context_texts)