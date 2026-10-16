# Phase 3 presentation demo

## 5–7 minute sequence

1. Open Home and identify the vision, retrieval, and language subsystems.
2. Upload a leaf image on Diagnose; point out the explicit DEMO/REAL state.
3. Show OpenCV stages, top-k confidence, and abstention.
4. Show Grad-CAM if available: classifier influence, not lesion segmentation or severity.
5. Show severity status; experimental values are not validated agronomic measurements.
6. Review sources; an empty list means the explanation is not grounded.
7. Open Ask CropAI and ask, “What should I check before trusting this result?”
8. Open a citation card and distinguish visual evidence from retrieved knowledge.
9. Ask an unsupported question and show insufficient-evidence behavior.
10. Download HTML/PDF/JSON reports and show provenance and limitations.
11. Open Research and explain TF-IDF vs semantic vs hybrid and pending results.

## Likely viva questions

**Why keep TF-IDF?** It is a cheap, interpretable lexical baseline for controlled comparison.

**Why FAISS?** It performs local nearest-neighbor search over normalized dense vectors while metadata
stays in a separate validated representation.

**What makes this RAG?** Supported knowledge claims are conditioned on retrieved manifest-approved
chunks, and citation IDs are validated against those chunks.

**How is prompt injection handled?** Retrieved text is marked untrusted, separated from policy, and
cannot contribute arbitrary citation IDs; adversarial fixtures test the boundary.

**Does Grad-CAM locate disease?** No. It shows influential classifier regions and is not a validated
segmentation mask.

**Is severity validated?** Not yet. Numerical severity is withheld in reports unless evidence status
is explicitly validated.

**Where is chat persisted?** SQLite stores bounded text history and citation metadata; image artifacts
are separate and API secrets are never stored.
