import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
import shutil
import pypdf
import re
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, Any
import logging

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough

# Used for running synchronous code in the async endpoint
from starlette.concurrency import run_in_threadpool

load_dotenv()

# Configure logging: write detailed logs to file; don't expose logs in API responses
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")
LOG_FILE = os.getenv("LOG_FILE", "app.log")
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()] if DEBUG else [logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger(__name__)

# Global dictionary for models
ml_models: Dict[str, Any] = {}

# --- FastAPI Lifespan (Startup/Shutdown Events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads ML models (Embeddings and LLM) on application startup."""
    logger.info("Server startup: loading models...")

    # 1. Embedding model (HuggingFace)
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        ml_models["embeddings"] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        logger.info(f"Embedding model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load embedding model")
        ml_models["embeddings"] = None

    # 2. LLM (Gemini)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        logger.warning("GOOGLE_API_KEY not found in environment; LLM will not be available.")
        ml_models["llm"] = None
    else:
        try:
            ml_models["llm"] = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            logger.info("LLM loaded successfully.")
        except Exception:
            logger.exception("Error loading Gemini LLM")
            ml_models["llm"] = None

    yield

    logger.info("Server shutdown: clearing models.")
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:4200")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Constants ---
UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY", "WarrantyDocuments")
VECTOR_STORE_DIRECTORY = os.getenv("VECTOR_STORE_DIRECTORY", "VectorStore")

# --- RAG Prompt Definition ---
SYSTEM_PROMPT_TEMPLATE = """
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise.

Context: {context}
"""
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# --- Helper Functions ---

def clean_text(text: str) -> str:
    """Cleans up extracted PDF text by normalizing whitespace."""
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return "\n".join([line.strip() for line in text.split("\n")]).strip()


def extract_text_sync(path: str) -> str:
    """Synchronous function to extract text from a PDF path."""
    text = ""
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def create_and_save_vector_store(docs: list[Document], model: HuggingFaceEmbeddings, filename: str) -> str:
    """Synchronous function to create and save the FAISS vector store."""
    vector_store = FAISS.from_documents(docs, model)
    index_name = os.path.splitext(filename)[0]
    save_path = os.path.join(VECTOR_STORE_DIRECTORY, index_name)
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    return save_path


def load_vector_store_sync(path: str, model: HuggingFaceEmbeddings) -> FAISS:
    """Synchronous function to load the FAISS vector store."""
    return FAISS.load_local(path, model, allow_dangerous_deserialization=True)


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved documents into a single string for the prompt context."""
    return "\n\n".join(doc.page_content for doc in docs)


# --- API Endpoints ---
@app.post("/create_warranty/")
async def create_warranty(details: str = Form(...), pdf_file: UploadFile = File(...)):
    """
    Uploads a PDF, extracts text, chunks it, embeds it, and saves the FAISS vector store.

    Returns a structured JSON response (no raw logs). The created policy_id is returned so the frontend
    can use it for later queries.
    """
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIRECTORY, exist_ok=True)

    # Normalize index name (no spaces, safe characters)
    index_name = os.path.splitext(pdf_file.filename)[0].replace(" ", "_")
    file_path = os.path.join(UPLOAD_DIRECTORY, pdf_file.filename)

    # 1. Save File
    try:
        with open(file_path, "wb") as buffer:
            await run_in_threadpool(shutil.copyfileobj, pdf_file.file, buffer)
    except Exception as e:
        logger.exception("Failed to save uploaded file")
        return {"error": "Could not save file."}
    finally:
        pdf_file.file.close()

    # 2. Extract Text
    try:
        raw_text = await run_in_threadpool(extract_text_sync, file_path)
        cleaned_text = clean_text(raw_text)
    except Exception as e:
        logger.exception("Failed to extract text from PDF")
        return {"error": "Could not extract text from PDF."}

    # 3. Chunk Text
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        text_chunks = await run_in_threadpool(text_splitter.split_text, cleaned_text)
    except Exception:
        logger.exception("Text chunking failed")
        return {"error": "Could not chunk text."}

    # 4. Convert to Documents
    documents = [
        Document(page_content=chunk, metadata={"source": pdf_file.filename, "chunk_index": i})
        for i, chunk in enumerate(text_chunks)
    ]

    # 5. Create and Save Vector Store
    try:
        embeddings_model = ml_models.get("embeddings")
        if embeddings_model is None:
            logger.error("Embedding model not available when creating vector store")
            return {"error": "Embedding model not loaded on server."}

        save_path = await run_in_threadpool(
            create_and_save_vector_store, documents, embeddings_model, pdf_file.filename
        )
    except Exception:
        logger.exception("Failed to create or save vector store")
        return {"error": "Could not create or save vector store."}

    return {
        "message": "Warranty created and vector store saved successfully",
        "policy_id": index_name,
        "warranty_details": details,
        "file_saved_at": file_path,
        "vector_store_saved_at": save_path,
        "chunk_count": len(text_chunks),
    }


class QueryRequest(BaseModel):
    policy_id: str
    question: str


@app.post("/query_warranty_doc/")
async def query_warranty_doc(request: QueryRequest):
    """
    Queries the vector store for a specific policy ID and uses RAG to answer the question.

    Returns structured JSON: {"answer": "..."} on success, or {"error": "..."} on failure.
    """
    try:
        if ml_models.get("llm") is None:
            logger.warning("LLM not available for query")
            return {"error": "LLM not loaded. Check server configuration."}
        if ml_models.get("embeddings") is None:
            logger.warning("Embedding model not available for query")
            return {"error": "Embedding model not loaded."}

        # Ensure policy index exists
        index_path = os.path.join(VECTOR_STORE_DIRECTORY, request.policy_id)
        if not os.path.exists(index_path):
            logger.warning("Policy id not found: %s", request.policy_id)
            return {"error": f"Policy ID '{request.policy_id}' not found. Have you created it?"}

        # 1. Load Vector Store
        embeddings_model = ml_models["embeddings"]
        vector_store = await run_in_threadpool(load_vector_store_sync, index_path, embeddings_model)

        llm = ml_models["llm"]
        retriever = vector_store.as_retriever()

        # 2. LCEL RAG Chain Definition
        qa_chain = (
            {
                "context": (lambda x: x["question"]) | retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
        )

        # 3. Invoke the chain synchronously in a threadpool
        def run_chain_sync(chain, inputs):
            result = chain.invoke(inputs)
            # Try to return a string safely
            try:
                return getattr(result, "content", str(result))
            except Exception:
                return str(result)

        answer = await run_in_threadpool(run_chain_sync, qa_chain, {"question": request.question})

        # Final structured response (no logs)
        return {"answer": answer}

    except Exception:
        logger.exception("Error during query processing")
        return {"error": "Internal server error during query processing."}
