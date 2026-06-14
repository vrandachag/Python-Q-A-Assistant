import os
import json
import logging
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import List, Dict

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

CHROMA_DIR = "./chroma_db"
DATA_PATH = r"C:\Users\chagv\Downloads\archive\python-qa-assistant\python-qa-assistant\data/processed_qa.json"

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a Python programming expert helping data science learners.
Use the retrieved Stack Overflow Q&A pairs below to answer the question accurately.
If the context does not contain a clear answer, say so honestly and provide general guidance.
Always include working code examples where relevant.

Context:
{context}

Question: {question}

Answer (be concise, accurate, and include code if helpful):"""
)


class RAGPipeline:
    def __init__(self):
        #self.embeddings = OpenAIEmbeddings(
           # model="text-embedding-3-small",
          #  openai_api_key=os.getenv("OPENAI_API_KEY")
        #)
        self.embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
        #self.llm = ChatOpenAI(
           # model="gpt-3.5-turbo",
            #temperature=0.2,
            #openai_api_key=os.getenv("OPENAI_API_KEY")
        #)
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None

    def load_or_build_index(self):
        if Path(CHROMA_DIR).exists():
            logger.info("Loading existing Chroma index...")
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Building new Chroma index from data...")
            self._build_index()

        #self.qa_chain = RetrievalQA.from_chain_type(
         #   llm=self.llm,
          #  chain_type="stuff",
           # retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            #chain_type_kwargs={"prompt": SYSTEM_PROMPT},
            #return_source_documents=True
        #)
        logger.info("QA chain ready.")

    def _build_index(self):
        docs = self._load_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        logger.info(f"Indexing {len(chunks)} chunks...")
        self.vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DIR
        )
        self.vectorstore.persist()
        logger.info("Index built and persisted.")

    def _load_documents(self) -> List[Document]:
        with open(DATA_PATH, "r") as f:
            records = json.load(f)

        docs = []
        for r in records:
            content = f"Question: {r['title']}\n\n{r['body']}\n\nAnswer: {r['answer']}"
            docs.append(Document(
                page_content=content,
                metadata={"title": r["title"], "score": r.get("score", 0)}
            ))
        logger.info(f"Loaded {len(docs)} Q&A documents.")
        return docs

    def answer(self, question: str):
        docs = self.vectorstore.similarity_search(question, k=3)

        sources = []

        for doc in docs:
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "score": float(doc.metadata.get("score", 0)),
                "snippet": doc.page_content[:300]
            })
        #print("Vectorstore count:", self.vectorstore._collection.count())

        answer = docs[0].page_content if docs else "No answer found."

        return {
            "answer": answer,
            "sources": sources
        }