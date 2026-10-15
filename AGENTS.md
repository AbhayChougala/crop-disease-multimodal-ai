# Crop Disease Detection and Diagnosis Using Multimodal AI

This repository is an academic B.Tech AI/ML research project.

## Core rules

- Never fabricate model metrics, experiment results, citations, predictions, or research findings.
- Any mock inference must be explicitly marked `is_mock=true` and the UI must visibly show DEMO MODE.
- Never commit secrets, `.env`, datasets, model checkpoints, vector stores, or private source documents.
- Keep all ML code device-agnostic: CUDA > MPS > CPU.
- Never hard-code `.cuda()`.
- Computer-vision preprocessing used for inference must match the selected training configuration.
- Grad-CAM is explainability, not a disease segmentation ground-truth mask.
- Severity produced from heuristic lesion segmentation must be labelled experimental unless validated against annotated masks.
- LLM agricultural statements must be grounded in retrieved sources where possible.
- The system is decision support, not a replacement for an agricultural expert.
- Keep frontend, backend, training, evaluation, and research code reproducible.
- Run relevant tests, lint, and build checks after changes.
- Do not perform long model training or download large datasets unless explicitly requested.
- Prefer maintainable implementations suitable for a student research project over unnecessary complexity.

## Project documentation

Use:

- `docs/architecture.md` for system boundaries.
- `docs/research_protocol.md` for experiments.
- `docs/gpu_handoff.md` for GPU-machine instructions.
- `README.md` for quick start.

Use current official OpenAI developer documentation whenever implementing or modifying OpenAI API integration.
