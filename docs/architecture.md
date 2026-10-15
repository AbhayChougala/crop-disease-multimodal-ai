# Architecture

The system separates HTTP/UI concerns from reusable research code.

```text
React upload or camera
        |
        v
FastAPI routes -> diagnosis/realtime services
        |
        +-> OpenCV decode and configurable preprocessing
        +-> leaf mask -> independent lesion mask -> experimental severity
        +-> classifier (mock now; state-dict model later)
        +-> structured visual evidence
        +-> retrieval (planned) -> provider-based LLM (mock now)
        |
        v
Typed response with artifacts, sources, timings, warnings, and is_mock
```

`src/crop_ai` contains framework-independent CV, model, evidence, retrieval, and generation code. `backend/app` owns transport, validation, configuration, and orchestration. `frontend` consumes the API contract and never receives credentials.

Grad-CAM will explain classifier influence. It will remain separate from the experimental lesion mask and will never be used as severity ground truth.
