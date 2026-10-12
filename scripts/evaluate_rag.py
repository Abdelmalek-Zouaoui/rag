"""Automated RAG Evaluation using Ragas Framework.

Evaluates the RAG pipeline on the RAG Triad:
1. Faithfulness (Hallucination metric: is the answer grounded in context?)
2. Answer Relevance (Is the answer directly answering the question?)
3. Context Precision (Are relevant chunks ranked higher?)
4. Context Recall (Did retrieval capture all necessary facts?)

Outputs:
- Console table summary
- reports/evaluation_results.json (raw per-sample metrics)
- reports/evaluation_summary.md (GitHub & LinkedIn ready markdown report)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure 'src' is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from langchain_groq import ChatGroq
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from langchain_core.embeddings import Embeddings
from rag.chain import ask, retrieve_relevant_docs
from rag.config import CHROMA_PERSIST_DIR, DATA_DIR, EMBEDDING_MODEL, GROQ_API_KEY, LLM_MODEL
from rag.ingest import build_index
from rag.models import get_embeddings


class FastEmbedRagasWrapper(Embeddings):
    """Wraps FastEmbed to provide a clean string model name for Ragas usage telemetry."""

    def __init__(self, emb, model_name: str = EMBEDDING_MODEL):
        self.emb = emb
        self.model = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.emb.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.emb.embed_query(text)


def load_dataset(dataset_path: Path) -> list[dict]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_dataset(eval_samples: list[dict], data_dir: str, persist_dir: str):
    print(f"\n[1/3] Running RAG Pipeline over {len(eval_samples)} test samples...")
    ragas_samples: list[SingleTurnSample] = []

    for i, item in enumerate(eval_samples, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"  Sample {i}/{len(eval_samples)}: '{question[:55]}...'")

        # 1. Retrieve contexts
        docs = retrieve_relevant_docs(question, data_dir=data_dir, persist_dir=persist_dir)
        retrieved_contexts = [doc.page_content for doc in docs]

        # 2. Generate answer
        answer = ask(question, data_dir=data_dir, persist_dir=persist_dir)

        # 3. Create Ragas SingleTurnSample
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            response=answer,
            reference=ground_truth,
        )
        ragas_samples.append(sample)

    return EvaluationDataset(samples=ragas_samples)


def run_evaluation(data_dir: str = str(DATA_DIR), persist_dir: str = str(CHROMA_PERSIST_DIR), dataset_path: Path | None = None):
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY is not set. Please set it in your environment or .env file.")
        sys.exit(1)

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    if dataset_path is None:
        dataset_path = ROOT_DIR / "data" / "eval_dataset.json"

    # Step 0: Ensure index is built
    print(f"[0/3] Ensuring index is built for '{data_dir}'...")
    build_index(data_dir=data_dir, persist_dir=persist_dir)

    # Step 1: Collect pipeline outputs
    eval_data = load_dataset(dataset_path)
    dataset = run_pipeline_on_dataset(eval_data, data_dir, persist_dir)

    # Step 2: Configure Ragas Evaluator (Judge LLM + Embeddings)
    print("\n[2/3] Configuring Ragas Evaluators (Groq LLM-as-a-Judge + FastEmbed embeddings)...")
    evaluator_llm = LangchainLLMWrapper(ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0.0))
    evaluator_embeddings = FastEmbedRagasWrapper(get_embeddings())

    # Groq supports single completion (n=1) for generating sample variants
    if hasattr(answer_relevancy, "strictness"):
        answer_relevancy.strictness = 1

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # Step 3: Run Ragas Evaluation
    print("\n[3/3] Calculating Ragas Metrics (Faithfulness, Relevance, Precision, Recall)...")
    start_time = time.time()
    run_config = RunConfig(max_workers=1, timeout=120, max_retries=3)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )
    elapsed = time.time() - start_time
    print(f"Evaluation completed in {elapsed:.2f} seconds.")

    # Step 4: Parse & Output Results
    df_result = result.to_pandas()
    import math

    summary_scores = {}
    for col in df_result.columns:
        if col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            val = float(df_result[col].mean())
            summary_scores[col] = 0.0 if math.isnan(val) else val

    # Console display (use safe ASCII characters compatible with all Windows terminals)
    print("\n" + "=" * 65)
    print("           RAGAS QUANTITATIVE EVALUATION SUMMARY")
    print("=" * 65)
    for metric, score in summary_scores.items():
        filled = max(0, min(20, int(round(score * 20))))
        bar = "#" * filled + "-" * (20 - filled)
        print(f" {metric.replace('_', ' ').title():<22} | {score:0.4f} | [{bar}]")
    print("=" * 65)

    # Save detailed JSON report
    json_path = reports_dir / "evaluation_results.json"
    detailed_records = df_result.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "llm_model": LLM_MODEL,
            "summary_scores": summary_scores,
            "samples": detailed_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed JSON results saved to: {json_path}")

    # Generate Markdown Summary for LinkedIn / README / Portfolio
    md_path = reports_dir / "evaluation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Enterprise RAG Evaluation Report (Ragas Benchmark)\n\n")
        f.write(f"- **Evaluated At:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **LLM Evaluator / Generator:** `{LLM_MODEL}` (Groq)\n")
        f.write(f"- **Retrieval Strategy:** Hybrid (BM25 + ChromaDB) with Cross-Encoder Reranker\n\n")
        f.write("## 1. Quantitative Benchmark Summary\n\n")
        f.write("| Metric | Score | Target | Status |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for metric, score in summary_scores.items():
            status = "✅ Pass" if score >= 0.80 else "⚠️ Needs Improvement"
            f.write(f"| **{metric.replace('_', ' ').title()}** | `{score:.4f}` | `> 0.80` | {status} |\n")
        f.write("\n## 2. Metric Descriptions (The RAG Triad)\n\n")
        f.write("- **Faithfulness**: Measures hallucination. High score indicates the answer strictly reflects retrieved facts.\n")
        f.write("- **Answer Relevancy**: Measures how directly the response addresses the user query.\n")
        f.write("- **Context Precision**: Measures whether the most relevant chunks are ranked at top positions by the reranker.\n")
        f.write("- **Context Recall**: Measures whether all ground-truth facts were retrieved.\n")

    print(f"Markdown report generated at: {md_path}")
    return summary_scores


if __name__ == "__main__":
    run_evaluation()
