# Phase 2 progress

Last updated: 2026-10-15

## Audit and architecture

- [x] Verified the Git root is this cloned repository.
- [x] Audited the existing backend, frontend, CV modules, schemas, tests, and documentation.
- [x] Confirmed no commit or push is authorized for this task.
- [x] Identified `DiagnosisService` as the runtime integration point and `src/crop_ai` as the research library boundary.

## Models, data, and training

- [x] Implement the ResNet50, EfficientNet-B0, and MobileNetV2 registry.
- [x] Implement safe state-dict checkpoints and structured inference with abstention.
- [x] Implement manifest-based dataset preparation, validation, splitting, leakage checks, and fingerprinting.
- [x] Implement training losses, metrics, reproducibility, trainer, evaluation, and CLI entry points.
- [x] Add a synthetic one-batch training/checkpoint smoke test.

## Computer vision and explainability

- [x] Expose leaf-isolated and overlay artifacts from preprocessing.
- [x] Implement Grad-CAM for all three supported classifiers.
- [x] Implement U-Net, segmentation loss/metrics, training, inference, and checkpoint interfaces.
- [x] Strengthen severity status and mask validation.

## RAG and language system

- [x] Implement manifest validation, chunking, local TF-IDF indexing, retrieval, evaluation, and citations.
- [x] Add mock/OpenAI/Ollama provider handling and diagnosis-context chat foundations.

## Integration and research

- [x] Add runtime model management without silent mock fallback.
- [x] Extend API capabilities and frontend result fields without redesigning the UI.
- [x] Add experiment metadata/result utilities and research CLI scaffolds.
- [x] Update setup, dataset, training, RAG, GPU, and handoff documentation.

No real agricultural model, dataset, metric, citation corpus, or segmentation validation is present yet.
Semantic embedding/FAISS retrieval, full experiment orchestration/plotting, durable chat persistence,
and real-provider streaming remain documented follow-up work.
