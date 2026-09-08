# huggingface.co — Leaderboards

- Hugging Face Spaces host the leaderboards that matter for picking a model; go to the leaderboard for the *category* of model needed rather than searching generically, since rankings and available checkpoints shift often and general knowledge goes stale fast
- Text embeddings / retrieval / reranking: `mteb/leaderboard` (Massive Text Embedding Benchmark) ranks models across retrieval, classification, clustering, STS, and reranking tasks — filter by task type first, then by model size, since the top overall model is rarely the best fit for a size-constrained deployment; the RTEB (Retrieval-focused Text Embedding Benchmark) tab isolates retrieval-only performance when that is the actual use case (e.g. RAG)
- General-purpose / instruction-following LLMs: check for the current community leaderboard Space in this category (the original `open-llm-leaderboard` was retired — search Spaces for its active successor) and cross-reference with a human-preference arena (e.g. LMArena / Chatbot Arena) since static benchmarks and human preference rank models differently and both signals matter
- Speech-to-text (ASR): `hf-audio/open_asr_leaderboard` ranks by word error rate but also reports RTF (real-time factor) — RTF is the hardware-relevant column, since a lower-WER model with RTF too high for the available compute is not actually usable
- Text-to-speech (TTS): `TTS-AGI/TTS-Arena` ranks by human preference (ELO), not automated metrics — treat top-ranked models as a shortlist to sanity-check for latency/hardware fit rather than a final answer
- Vision-language models (VLMs): `opencompass/open_vlm_leaderboard` covers multimodal benchmarks across model sizes — filter by parameter count before comparing scores
- Code generation: leaderboards tied to BigCode / EvalPlus-style benchmarks rank by pass@k on HumanEval/MBPP-style tasks — check the parameter count column against available hardware before trusting the ranking
- Every leaderboard above exposes a parameter-count (or model-size) column and usually a license column — filter to the sizes and licenses that are actually usable *before* comparing scores, rather than picking the top-ranked model and discovering afterward that it doesn't fit
- To translate a leaderboard entry into a hardware feasibility check: open the model's repo, check "Files and versions" for actual weight file sizes, and estimate memory as roughly 2 bytes/parameter at FP16, ~1 byte/parameter at INT8, and ~0.5–0.6 bytes/parameter at 4-bit quantization (GGUF/AWQ/GPTQ) — then add headroom (roughly 15–20%) for KV cache and activations rather than treating the raw weight size as the full requirement
- When the full-precision checkpoint doesn't fit, search the same model name for quantized community re-uploads (GGUF, AWQ, GPTQ variants) before ruling the model out entirely — a leaderboard-topping model is frequently usable on constrained hardware once quantized, at a small, documented cost to quality
- Leaderboard rank and hardware fit are independent axes — always resolve both before recommending a model, since the best-fitting model is the best *ranked model that also fits*, not simply the best-ranked model

## Metadata

- parent domain: huggingface.co
