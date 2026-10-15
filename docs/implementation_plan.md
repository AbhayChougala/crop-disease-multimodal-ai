# Implementation plan

## Phase 1 - Runnable foundation

- Repository safety rules and environment configuration
- Typed FastAPI API and browser client
- Deterministic mock inference with visible demo labeling
- Modular OpenCV preprocessing, leaf mask, lesion-color baseline, and experimental severity
- Upload diagnosis and throttled live-camera workflow
- Tests, build checks, and architecture documentation

## Phase 2 - ML and explainability

- Shared preprocessing contract for training and inference
- ResNet50, EfficientNet-B0, and MobileNetV2 model factory
- Dataset inspection/splitting and configurable training/evaluation
- State-dict artifact loading with label and preprocessing metadata
- Grad-CAM for the supported final convolution layers

## Phase 3 - Grounded knowledge subsystem

- Source manifest validation and PDF/text ingestion
- Chunking with page/source metadata
- Embeddings, persistent retrieval, and tiny-corpus tests
- Mock, OpenAI, and local provider interfaces
- Session-scoped follow-up chat and citation validation

## Phase 4 - Research evaluation and handoff

- Model comparison, preprocessing ablation, and cross-domain tools
- Severity validation and Grad-CAM qualitative protocol
- RAG/no-RAG and realtime latency evaluation
- GPU smoke-test, training, export, and deployment handoff

No phase may publish a metric until it was produced by a recorded experiment on real data.
