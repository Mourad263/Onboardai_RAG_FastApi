import os
import re
from pathlib import Path
from typing import Any

import requests
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logging_config import get_logger


logger = get_logger("onboardai.rag")


REPO_RAW = "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial"

PAGES = [
    "appetite.rst",
    "interpreter.rst",
    "introduction.rst",
    "controlflow.rst",
    "datastructures.rst",
    "modules.rst",
    "inputoutput.rst",
    "errors.rst",
    "classes.rst",
    "stdlib.rst",
    "stdlib2.rst",
    "venv.rst",
    "floatingpoint.rst",
    "interactive.rst",
    "whatnow.rst",
]

IDK = "I don't have enough information to answer that."

# Keep this configurable from Railway / .env.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "1.20"))

TOP_K = int(os.getenv("TOP_K", "4"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


class RAGService:
    def __init__(self):
        self.ready = False
        self.document_count = 0
        self.chunk_count = 0
        self.chunking_strategy = "recursive"

        self.vectorstore = None
        self.llm = None
        self.embeddings = None

        self.docs_dir = Path(
            os.getenv("DOCS_DIR", "docs")
        )

    async def initialize(self):
        """
        Build the complete RAG pipeline.

        This method intentionally remains async because the FastAPI
        startup wrapper runs it in a background thread.
        """

        try:
            logger.info("rag_initialization_begin")

            # ---------------------------------------------------------
            # 1. Download / refresh source documents
            # ---------------------------------------------------------
            self._download_docs()

            # ---------------------------------------------------------
            # 2. Load documents
            # ---------------------------------------------------------
            raw_docs = self._load_docs()

            if not raw_docs:
                raise RuntimeError(
                    f"No documents found in {self.docs_dir.resolve()}"
                )

            self.document_count = len(raw_docs)

            # ---------------------------------------------------------
            # 3. Split documents
            # ---------------------------------------------------------
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=120,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )

            chunks = splitter.split_documents(raw_docs)

            if not chunks:
                raise RuntimeError("Document splitting produced zero chunks.")

            self.chunk_count = len(chunks)

            logger.info(
                "chunks_created",
                extra={
                    "documents": self.document_count,
                    "chunks": self.chunk_count,
                },
            )

            # ---------------------------------------------------------
            # 4. Embeddings
            # ---------------------------------------------------------
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
            )

            # ---------------------------------------------------------
            # 5. Vector store
            # ---------------------------------------------------------
            self.vectorstore = FAISS.from_documents(
                chunks,
                self.embeddings,
            )

            logger.info(
                "vectorstore_ready",
                extra={
                    "documents": self.document_count,
                    "chunks": self.chunk_count,
                },
            )

            # ---------------------------------------------------------
            # 6. LLM
            # ---------------------------------------------------------
            api_key = os.getenv("GOOGLE_API_KEY")

            if not api_key:
                self.llm = None

                logger.warning(
                    "google_api_key_missing_generation_disabled"
                )

            else:
                self.llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL,
                    temperature=0,
                )

                logger.info(
                    "llm_ready",
                    extra={
                        "model": GEMINI_MODEL,
                    },
                )

            self.ready = True

            logger.info(
                "rag_initialization_complete",
                extra={
                    "documents": self.document_count,
                    "chunks": self.chunk_count,
                    "generation_enabled": self.llm is not None,
                },
            )

        except Exception:
            self.ready = False
            logger.exception("rag_initialization_failed")
            raise

    def _download_docs(self):
        self.docs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        downloaded = 0
        existing = 0

        for page in PAGES:
            path = self.docs_dir / page

            # Do not download again if the local document already exists.
            if path.exists() and path.stat().st_size > 100:
                existing += 1
                continue

            logger.info(
                "downloading_document",
                extra={
                    "page": page,
                },
            )

            response = requests.get(
                f"{REPO_RAW}/{page}",
                timeout=30,
            )

            response.raise_for_status()

            path.write_text(
                response.text,
                encoding="utf-8",
            )

            downloaded += 1

        logger.info(
            "documents_ready",
            extra={
                "count": len(PAGES),
                "downloaded": downloaded,
                "existing": existing,
            },
        )

    def _load_docs(self):
        loader = DirectoryLoader(
            str(self.docs_dir),
            glob="**/*.rst",
            loader_cls=TextLoader,
            loader_kwargs={
                "encoding": "utf-8",
            },
        )

        docs = loader.load()

        if not docs:
            return []

        for doc in docs:
            source = doc.metadata.get("source", "")

            doc.metadata["source"] = Path(
                source
            ).name

        return docs

    def _retrieve(
        self,
        question: str,
        top_k: int,
    ):
        if self.vectorstore is None:
            return []

        scored = self.vectorstore.similarity_search_with_score(
            question,
            k=top_k,
        )

        logger.info(
            "retrieval_results",
            extra={
                "question": question,
                "scores": [
                    round(float(score), 4)
                    for _, score in scored
                ],
                "sources": [
                    doc.metadata.get("source", "unknown")
                    for doc, _ in scored
                ],
            },
        )

        return scored

    def answer(
        self,
        question: str,
        top_k: int = TOP_K,
    ) -> dict[str, Any]:

        if not self.ready:
            return {
                "status": "not_ready",
                "answer": "Service is still starting.",
                "sources": [],
            }

        if not question or not question.strip():
            return {
                "status": "invalid_question",
                "answer": "Please provide a valid question.",
                "sources": [],
            }

        # Keep top_k inside a safe range.
        top_k = max(1, min(int(top_k), 8))

        scored = self._retrieve(
            question=question.strip(),
            top_k=top_k,
        )

        if not scored:
            return {
                "status": "insufficient_context",
                "answer": IDK,
                "sources": [],
            }

        retrieved = [
            doc
            for doc, _ in scored
        ]

        sources = [
            {
                "file": doc.metadata.get(
                    "source",
                    "unknown",
                ),
                "snippet": re.sub(
                    r"\s+",
                    " ",
                    doc.page_content[:300],
                ).strip(),
            }
            for doc in retrieved
        ]

        # -------------------------------------------------------------
        # Retrieval threshold
        #
        # Don't immediately discard the sources. We keep them visible
        # for debugging and evaluation.
        # -------------------------------------------------------------
        best_score = float(scored[0][1])

        logger.info(
            "best_retrieval_score",
            extra={
                "question": question,
                "best_score": round(
                    best_score,
                    4,
                ),
                "threshold": SCORE_THRESHOLD,
            },
        )

        if best_score > SCORE_THRESHOLD:
            return {
                "status": "insufficient_context",
                "answer": IDK,
                "sources": sources,
            }

        # -------------------------------------------------------------
        # Retrieval-only mode
        # -------------------------------------------------------------
        if self.llm is None:
            return {
                "status": "retrieval_only",
                "answer": (
                    "Generation is disabled because "
                    "GOOGLE_API_KEY is not configured."
                ),
                "sources": sources,
            }

        # -------------------------------------------------------------
        # Build context
        # -------------------------------------------------------------
        context_parts = []

        for doc in retrieved:
            source = doc.metadata.get(
                "source",
                "unknown",
            )

            context_parts.append(
                f"[source: {source}]\n"
                f"{doc.page_content}"
            )

        context = "\n\n".join(
            context_parts
        )

        # -------------------------------------------------------------
        # Prompt
        # -------------------------------------------------------------
        prompt = f"""
You are a Python documentation assistant.

Answer the user's question ONLY from the retrieved context below.

Rules:
1. Use only information contained in the context.
2. Do not invent or add outside knowledge.
3. You may combine information from multiple retrieved chunks.
4. You may explain or summarize information from the context in your own words.
5. Cite the source filename in square brackets for factual claims.
6. If the context genuinely does not contain enough information to answer
   the question, respond with exactly:

{IDK}

Do not return the fallback sentence when the context contains enough
information to provide a useful answer.

Retrieved context:
{context}

Question:
{question}
""".strip()

        # -------------------------------------------------------------
        # Generate
        # -------------------------------------------------------------
        try:
            response = self.llm.invoke(
                prompt
            )

        except Exception:
            logger.exception(
                "llm_generation_failed"
            )

            return {
                "status": "generation_error",
                "answer": (
                    "An error occurred while generating the answer."
                ),
                "sources": sources,
            }

        content = response.content

        if isinstance(content, list):
            answer = "".join(
                block.get("text", "")
                for block in content
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                )
            )
        else:
            answer = str(content)

        answer = answer.strip()

        # -------------------------------------------------------------
        # Prevent misleading:
        #
        # status=answered + answer=I don't have enough information
        # -------------------------------------------------------------
        if answer == IDK:
            return {
                "status": "insufficient_context",
                "answer": IDK,
                "sources": sources,
            }

        return {
            "status": "answered",
            "answer": answer,
            "sources": sources,
        }