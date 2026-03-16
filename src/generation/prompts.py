"""Prompt templates for the RAG system."""

from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context below. If the context doesn't contain
enough information to answer the question, say "I don't have enough information
in the provided documents to answer this question."

Always cite which source document your answer comes from.

Context:
{context}

Question: {question}

Answer:""",
)

CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:""",
)