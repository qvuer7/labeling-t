"""The transformers model-server (the `[models]` extra; runs on the GPU pod).

Imported ONLY by the `labeling-t-models serve` entrypoint — never by the thin
client / CLI — so `import labeling_t` stays torch-free. Heavy model deps
(torch/transformers/sam2) are imported lazily inside each ModelAdapter, not at
package import time.
"""
