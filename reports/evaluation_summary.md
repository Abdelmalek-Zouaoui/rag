# Enterprise RAG Evaluation Report (Ragas Benchmark)

- **Evaluated At:** 2026-10-12 17:49:21
- **LLM Evaluator / Generator:** `openai/gpt-oss-20b` (Groq)
- **Retrieval Strategy:** Hybrid (BM25 + ChromaDB) with Cross-Encoder Reranker

## 1. Quantitative Benchmark Summary

| Metric | Score | Target | Status |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | `0.7667` | `> 0.80` | ⚠️ Needs Improvement |
| **Answer Relevancy** | `0.8782` | `> 0.80` | ✅ Pass |
| **Context Precision** | `1.0000` | `> 0.80` | ✅ Pass |
| **Context Recall** | `1.0000` | `> 0.80` | ✅ Pass |

## 2. Metric Descriptions (The RAG Triad)

- **Faithfulness**: Measures hallucination. High score indicates the answer strictly reflects retrieved facts.
- **Answer Relevancy**: Measures how directly the response addresses the user query.
- **Context Precision**: Measures whether the most relevant chunks are ranked at top positions by the reranker.
- **Context Recall**: Measures whether all ground-truth facts were retrieved.
