# Decision Log: MEP Drawing Interpretation System

This document captures the key architectural decisions, tradeoffs, and design justifications for the upgrades implemented in the MEP Drawing Interpretation System.

---

## 1. Context Retrieval vs. Full OCR Injection
* **Decision**: Implement a `ContextRetriever` and `QuestionAnalyzer` rather than injecting the entire page's OCR elements.
* **Justification**: Large D-size drawings contain thousands of OCR text tokens. Passing all tokens to LLMs is expensive, causes high latency, and triggers rate-limiting errors (`429 Resource Exhausted`) on standard API tiers.
* **Solution**: The `QuestionAnalyzer` classifies query intents (7 classes) to route search queries. The `ContextRetriever` fetches only localized notes, legends, or adjacent coordinates of target keywords, reducing context to 10-20 clean elements.

---

## 2. Multi-Tier Model Orchestration Hierarchy
* **Decision**: Adopt a three-tier model hierarchy.
  - **Tier 1 (NVIDIA API Model Provider)**: Serves as the primary advanced reasoning model provider, supporting any NVIDIA-hosted vision model (configured via `.env` without code changes).
  - **Tier 2 (Gemini 2.5 Flash)**: Acts as the secondary high-speed fallback model.
  - **Tier 3 (Local Qwen2.5-VL)**: Acts as the offline fallback and verifier, automatically routing to SmolVLM or SmolVLM2 if Qwen is not loaded in Ollama.
* **Justification**: Protects system availability and provides provider-independent model flexibility. If the primary NVIDIA endpoint experiences rate limits or downtime, Gemini takes over. If cloud access is completely cut off, local offline models process the drawing transparently.

---

## 3. Multi-Model Agreement & Verification
* **Decision**: Introduce a `ModelVerifier` comparing two model responses.
  - Textual answers are compared via token-based Jaccard similarity.
  - Grounding coordinates (bounding boxes) are compared via Intersection-over-Union (IoU) overlap.
* **Justification**: A single model's confidence rating is self-reported and prone to overconfidence. Corroborating the response with a second, independent model provides a factual metric of model agreement, raising or lowering the final confidence rating.

---

## 4. Counting Verification and Confidence Capping
* **Decision**: Enforce a cap on counting questions. The confidence score is capped at `0.79` (Medium) unless:
  1. Multiple models are in high agreement (agreement score $\ge$ 0.80).
  2. OCR search occurrences support the target item count (OCR support $\ge$ 0.80).
* **Justification**: Vision-language models frequently hallucinate quantities of dense small symbols. Applying a strict ceiling prevents the system from reporting counting results with false certainty.

---

## 5. Evidence-Based Confidence Score
* **Decision**: Adopt a weighted multi-factor formula:
  $$\text{Confidence} = 0.30 \times \text{OCR Match} + 0.25 \times \text{Agreement} + 0.20 \times \text{BBox} + 0.15 \times \text{Density} + 0.10 \times \text{Model Confidence}$$
* **Justification**: This formula moves away from pure LLM self-confidence to objective, evidence-based metrics. It factors in layout presence, verified OCR words, multi-model consensus, and local element density.

---

## 6. UI: Non-Suppressed Low Confidence Output
* **Decision**: Always display the model's answer, even if the final confidence is low or fails validation, but pair it with a bright red, pulsing warning banner indicating `HUMAN REVIEW REQUIRED`.
* **Justification**: In V1, low-confidence responses were completely suppressed and replaced with the phrase "HUMAN REVIEW REQUIRED". Reviewers and engineers requested to see what the AI calculated to understand its reasoning, while keeping the warning banner prominent to indicate that the answer must be manually checked.

---

## 7. Explainability: Debug Panel and Debrief Toggles
* **Decision**: Implement a 5-Tab Debug Panel and an expandable Explainability Debrief.
* **Justification**: Standard QA systems are black boxes. Engineers require full visibility into what layout regions were targeted, what OCR text was retrieved, how independent models compared, and the exact weights contributing to the confidence rating.
