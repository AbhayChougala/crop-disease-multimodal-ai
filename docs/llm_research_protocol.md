# LLM and RAG research protocol

Use one frozen, manually reviewed question set and the same provider/model/configuration across G0
(LLM only, research condition), G1 (TF-IDF), G2 (semantic), and G3 (hybrid). Production remains
retrieval-gated.

Record the question, answer, retrieved IDs, accepted citations, latency, token usage when available,
provider, model, retriever/index metadata, timestamp, and configuration snapshot. Reviewers score
correctness, groundedness, citation support, and completeness on a predeclared 1–5 rubric, plus mark
unsafe/unsupported claims and notes. Do not use model self-evaluation as factual validation.

The retrieval protocol and sample format are in `research/protocols/rag_evaluation.md`. Add
ANSWERABLE, PARTIALLY ANSWERABLE, and UNANSWERABLE cases. For abstention, measure correct abstention,
unsupported citations, and fabricated citations. For injection, include malicious retrieved text and
verify policy, prompt, secret, and citation protections. Save only measured artifacts.
