"""Evaluation pipeline using RAGAS metrics."""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.generation.llm_chain import ask_question


def create_eval_dataset(
    questions: list[str],
    ground_truths: list[str],
    search_type: str = "hybrid",
) -> Dataset:
    """Run the RAG pipeline on evaluation questions and build a RAGAS dataset."""
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for question, truth in zip(questions, ground_truths):
        result = ask_question(question, search_type=search_type)
        data["question"].append(question)
        data["answer"].append(result["answer"])
        data["contexts"].append(
            [doc.page_content for doc in result["context_documents"]]
        )
        data["ground_truth"].append(truth)

    return Dataset.from_dict(data)


def run_evaluation(
    questions: list[str],
    ground_truths: list[str],
    search_type: str = "hybrid",
) -> dict:
    """Run RAGAS evaluation and return scores.

    Metrics:
        - Faithfulness: Is the answer grounded in the context?
        - Answer Relevancy: Does the answer address the question?
        - Context Precision: Are the retrieved docs relevant?
        - Context Recall: Did we retrieve all necessary info?
    """
    dataset = create_eval_dataset(questions, ground_truths, search_type)

    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return {
        "faithfulness": results["faithfulness"],
        "answer_relevancy": results["answer_relevancy"],
        "context_precision": results["context_precision"],
        "context_recall": results["context_recall"],
    }


# Example usage / quick test
if __name__ == "__main__":
    # Replace with questions relevant to YOUR test documents
    test_questions = [
        "What is the main topic of the document?",
        "What are the key findings?",
    ]
    test_ground_truths = [
        "The document discusses...",  # Fill in real ground truths
        "The key findings are...",
    ]
    scores = run_evaluation(test_questions, test_ground_truths)
    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")