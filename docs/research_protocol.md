# Research protocol

All experiments must record a timestamp, seed, git commit when available, hardware/device, dataset identifier, preprocessing configuration, model artifact, and raw machine-readable results.

1. Compare ResNet50, EfficientNet-B0, and MobileNetV2 using accuracy, macro precision/recall/F1, weighted F1, model size, and measured latency.
2. Compare raw, standard OpenCV, and leaf-segmented inputs using the same split and classifier configuration.
3. Evaluate the selected model separately on controlled and explicitly mapped field datasets.
4. Review representative correct, incorrect, well-focused, and poorly-focused Grad-CAM cases.
5. Validate color-based lesion severity against annotated masks before reporting segmentation metrics or interpreting bands agriculturally.
6. Compare an LLM-only condition with retrieval-grounded generation using a fixed question set and source-support rubric.
7. Measure static and live preprocessing, inference, end-to-end latency, warmup, and steady-state throughput.

No mock output may enter an experiment result directory.
