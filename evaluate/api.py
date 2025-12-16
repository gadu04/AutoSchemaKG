"""
Simple FastAPI Backend for Medical QA
Uses Think on Graph retrieval directly.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os

# Import ToG components directly
from chat_interactive import (
    TogV3Retriever, 
    LLMGenerator, 
    EmbeddingModel, 
    InferenceConfig,
    load_kg_from_neo4j
)

app = FastAPI(title="Medical QA API", version="1.0.0")

# Global retriever instance (initialized on startup)
retriever: Optional[TogV3Retriever] = None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    llm_answer: str
    tog_answer: str
    triples: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize retriever on startup."""
    global retriever
    
    print("Initializing ToG retriever...")
    
    # Load KG from Neo4j
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    
    try:
        kg = load_kg_from_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print(f"✓ Loaded KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    except Exception as e:
        print(f"✗ Failed to load KG from Neo4j: {e}")
        return
    
    # Initialize LLM
    use_real_llm = os.getenv("USE_REAL_LLM", "true").lower() == "true"
    llm_generator = LLMGenerator(use_real_llm=use_real_llm)
    
    # Initialize embedding model - FIXED: remove use_docker parameter
    # EmbeddingModel in chat_interactive.py only takes model_name parameter
    embedding_model = EmbeddingModel(model_name="BAAI/bge-m3")
    
    # Read ToG config from environment
    max_depth = int(os.getenv("TOG_MAX_DEPTH", "3"))
    inference_config = InferenceConfig(Dmax=max_depth)
    
    # Create retriever with config
    use_qdrant = os.getenv("USE_QDRANT", "true").lower() == "true"
    
    retriever = TogV3Retriever(
        KG=kg,
        llm_generator=llm_generator,
        sentence_encoder=embedding_model,
        inference_config=inference_config,
        use_qdrant=use_qdrant,
        qdrant_url="http://localhost:6333"
    )
    
    print(f"✓ ToG retriever initialized (Dmax={max_depth})")


def get_llm_answer(question: str) -> str:
    """Get answer from LLM only."""
    return ask_lmstudio(question)


def get_tog_answer(question: str) -> tuple:
    """Get answer from Think on Graph retriever."""
    global retriever
    
    if retriever is None:
        return "Error: ToG retriever not initialized", []
    
    try:
        # Read topN from environment
        top_paths = int(os.getenv("TOG_TOP_PATHS", "10"))
        answer, triples = retriever.retrieve(question, topN=top_paths)
        return answer, triples
    except Exception as e:
        return f"ToG Error: {str(e)}", []


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "message": "Medical QA API is running",
        "endpoints": ["/api/ask", "/api/llm", "/api/tog"]
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    llm_answer = ask_lmstudio(question)
    tog_answer, triples = get_tog_answer(question)

    return AnswerResponse(
        llm_answer=llm_answer,
        tog_answer=tog_answer,
        triples=triples
    )


@app.post("/api/llm")
async def llm_only(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = ask_lmstudio(question)
    return {"answer": answer}


@app.post("/api/tog")
async def tog_only(request: QuestionRequest):
    """Get answer from ToG only."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    answer, triples = get_tog_answer(question)
    return {"answer": answer, "triples": triples}


import requests

def ask_lmstudio(question: str) -> str:
    """
    Query local LLM running in LM Studio (OpenAI-compatible API).
    """
    try:
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "local-model",
                "messages": [
                    {"role": "user", "content": question}
                ],
                "temperature": 0.3
            }
        )

        data = response.json()

        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]

        return "LM Studio: no valid response"

    except Exception as e:
        return f"LM Studio Error: {str(e)}"


if __name__ == "__main__":
    import uvicorn
    print("Starting Medical QA API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)