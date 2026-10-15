# Architecture

The system separates HTTP/UI concerns from reusable research code.

```text
React upload or camera
        |
        v
FastAPI routes -> diagnosis/realtime services
        |
        +-> OpenCV decode and configurable preprocessing
        +-> leaf mask -> independent lesion mask -> explicit severity status
        +-> one-time model manager (explicit demo or safe state-dict checkpoint)
        +-> class-specific Grad-CAM and typed provenance-rich evidence
        +-> local retrieval -> mock/OpenAI/Ollama provider
        |
        v
Typed response with artifacts, sources, timings, warnings, and is_mock
```

`src/crop_ai` contains framework-independent CV, model, evidence, retrieval, and generation code. `backend/app` owns transport, validation, configuration, and orchestration. `frontend` consumes the API contract and never receives credentials.

Classification training/evaluation, U-Net segmentation, RAG indexing, and experiment aggregation run as
offline CLIs. Generated datasets, checkpoints, vector indexes, and results stay outside Git.

Grad-CAM explains classifier influence. It remains separate from the experimental lesion mask and is
never used as severity ground truth. Real model mode fails closed; it does not silently produce demo
predictions when a checkpoint is missing or invalid.
