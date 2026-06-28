"""Web UI for the labeling pipeline — a FastAPI adapter over the CLI stages.

A thin orchestration layer: it reuses the same pipeline functions
(ingest_images, prelabel_cloud, import_to_label_studio, pull_verified) and the
Storage/DatasetLayout core, exposing them as a small JSON API + a static SPA so
the upload -> auto-label -> Label Studio -> verify loop can be driven from a
browser. Run with `labeling-t-web`.
"""
