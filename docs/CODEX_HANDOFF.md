# Phase 2 continuation handoff

The implemented vertical slices are working and tested. No agricultural dataset, trained checkpoint,
authoritative knowledge corpus, API key, or genuine experiment result is present, so those dependent
outcomes remain intentionally unavailable.

Next work, in order:

1. Add legally usable agricultural extension documents and their hashes to the source manifest.
2. Add local PlantVillage data, prepare the manifest, inspect duplicate/leakage reports, and train the
   three classifier configurations on the same split.
3. Evaluate held-out and mapped PlantDoc field classes; never import metrics from incompatible labels.
4. Add lesion-mask annotations, run U-Net training/evaluation, and design a reviewed process for
   promoting a checkpoint from `validated=false`.
5. Add optional sentence-transformer/FAISS providers if semantic retrieval is required.
6. Extend experiment commands to directly orchestrate all controlled runs and add plot generation
   after real results exist.
7. Add token-usage persistence and optional streaming for real LLM providers.
8. Run browser accessibility/responsive QA with the backend and frontend active.

Do not mark these items complete until the required external inputs and measured outputs exist.
