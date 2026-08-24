import os
import re
from pathlib import Path
from typing import Any

import requests
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from app.logging_config import get_logger

logger = get_logger("onboardai.rag")

REPO_RAW = "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial"
PAGES = [
    "appetite.rst", "interpreter.rst", "introduction.rst", "controlflow.rst",
    "datastructures.rst", "modules.rst", "inputoutput.rst", "errors.rst",
    "classes.rst", "stdlib.rst", "stdlib2.rst", "venv.rst",
    "floatingpoint.rst", "interactive.rst", "whatnow.rst",
]

IDK = "I don't have enough information to answer that."
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "1.20"))
TOP_K = 4

class RAGService:
    def __init__(self):
        self.ready = False
        self.document_count = 0
        self.chunk_count = 0
        self.chunking_strategy = "recursive"
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.docs_dir = Path(os.getenv("DOCS_DIR", "docs"))

    async def initialize(self):
        self._download_docs()
        raw_docs = self._load_docs()
        self.document_count = len(raw_docs)

        # Same winning strategy used in the assignment:
        # 800-character recursive chunks with 120-character overlap.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)
        self.chunk_count = len(chunks)

        logger.info(
            "chunks_created",
            extra={"documents": self.document_count, "chunks": self.chunk_count}
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("google_api_key_missing_generation_disabled")
        else:
            model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)
            logger.info("llm_ready", extra={"model": model})

        self.ready = True

    def _download_docs(self):
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        for page in PAGES:
            path = self.docs_dir / page
            if path.exists() and path.stat().st_size > 100:
                continue
            response = requests.get(f"{REPO_RAW}/{page}", timeout=30)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
        logger.info("documents_ready", extra={"count": len(PAGES)})

    def _load_docs(self):
        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.rst",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = Path(doc.metadata["source"]).name
        return docs

    def answer(self, question: str, top_k: int = TOP_K) -> dict[str, Any]:
        if not self.ready:
            return {"status": "not_ready", "answer": "Service is still starting.", "sources": []}

        scored = self.vectorstore.similarity_search_with_score(question, k=top_k)

        if not scored or scored[0][1] > SCORE_THRESHOLD:
            return {
                "status": "insufficient_context",
                "answer": IDK,
                "sources": [],
            }

        retrieved = [doc for doc, score in scored]
        sources = [
            {
                "file": doc.metadata["source"],
                "snippet": re.sub(r"\s+", " ", doc.page_content[:300]).strip(),
            }
            for doc in retrieved
        ]

        if self.llm is None:
            return {
                "status": "retrieval_only",
                "answer": "Generation is disabled because GOOGLE_API_KEY is not configured.",
                "sources": sources,
            }

        context = "\n\n".join(
            f"[source: {doc.metadata['source']}]\n{doc.page_content}"
            for doc in retrieved
        )
        prompt = f"""Answer ONLY using the context below.
Cite the source file name in square brackets for every claim, for example [errors.rst].
If the context does not contain enough information, say exactly:
"{IDK}"
Do not guess and do not use outside knowledge.

Context:
{context}

Question: {question}
"""
        response = self.llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            answer = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            answer = str(content)

        return {
            "status": "answered",
            "answer": answer,
            "sources": sources,
        }
