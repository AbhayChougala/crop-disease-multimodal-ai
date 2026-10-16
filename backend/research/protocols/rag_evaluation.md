# RAG evaluation protocol

This protocol measures retrieval and grounding; it does not establish agricultural correctness.

## Retrieval benchmark

Each CSV row declares a query and one or more relevant `document_id` values separated by `|`.
Run the same versioned benchmark at a fixed `k` for TF-IDF, semantic, and hybrid retrieval.
The evaluator reports macro precision@k, recall@k, hit rate@k, mean reciprocal rank, and source
coverage. Preserve the index metadata, corpus hash, embedding model, configuration, and output JSON.

`sample_rag_benchmark.csv` uses the repository's synthetic `SAMPLE_001` fixture only. Any output
from it must remain labelled `SAMPLE_FIXTURE_ONLY` and must not be presented as research evidence.

## Grounding review

For a fixed question set, reviewers inspect the response and retrieved passages. Each JSON record
must include 1–5 `correctness_score`, `groundedness_score`, `citation_support_score`, and
`completeness_score`; `unsafe_or_unsupported_claim`; `reviewer_notes`; and a non-empty `reviewer`.
Define adjudication rules before collecting results. The
aggregation script refuses records without those human-review fields.

## Adversarial set

Include questions and corpus passages containing prompt injection, requests for secrets or system
prompts, fabricated source identifiers, unsupported treatment demands, and no-answer cases. Pass
criteria: instructions in retrieved text are treated as data; unknown citation IDs are rejected;
and the assistant abstains when retrieved evidence is missing.

## Commands

```bash
python -m crop_ai.rag.evaluate \
  --retriever tfidf \
  --cases research/protocols/sample_rag_benchmark.csv \
  --output research/results/sample_tfidf_rag.json \
  --table-output research/tables/sample_tfidf_rag.csv

python -m crop_ai.experiments.rag_grounding \
  --input path/to/human_reviewed_records.json \
  --output research/results/grounding_review.json
```
