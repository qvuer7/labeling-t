#!/usr/bin/env python
"""DEBUG (run ON the GPU box): reveal the transformers OWLv2 post-process API and
reproduce the /infer failure with a FULL traceback. The model server returns a
bare 500; this surfaces the real exception + the actual API shape so the adapter
can be fixed against reality instead of guessed.

The running pod's image already has everything installed:
    cd /app && .venv/bin/python /app/src/../scripts/owlv2_debug.py <presigned-url>
or with the repo cloned fresh:
    uv sync --extra models && uv run python scripts/owlv2_debug.py <presigned-url>

Pass a presigned S3 frame URL as arg 1 to reproduce real inference; omit it to
just introspect the API.
"""

import io
import sys
import traceback


def main() -> int:
    import torch
    import transformers

    print(f"transformers {transformers.__version__} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    import inspect

    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    hf = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(hf)
    print("post_process methods:", [m for m in dir(proc) if "post_process" in m])
    for name in ("post_process_grounded_object_detection", "post_process_object_detection"):
        fn = getattr(proc, name, None)
        if fn:
            print(f"  {name}{inspect.signature(fn)}")

    url = sys.argv[1] if len(sys.argv) > 1 else None
    if not url:
        print("\n(pass a presigned frame URL to reproduce detect())")
        return 0

    import httpx
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Owlv2ForObjectDetection.from_pretrained(hf).to(dev).eval()
    img = Image.open(io.BytesIO(httpx.get(url, timeout=60).content)).convert("RGB")
    w, h = img.size
    queries = ["player", "ball", "referee"]
    inputs = proc(text=[queries], images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    side = max(w, h)
    ts = torch.tensor([[side, side]], device=dev)
    try:
        res = proc.post_process_grounded_object_detection(
            out, threshold=0.1, target_sizes=ts, text_labels=[queries]
        )[0]
        print("\nresult keys:", list(res.keys()))
        print("n boxes:", len(res["boxes"]))
        print("sample:", {k: (v[:2].tolist() if hasattr(v, "tolist") else v[:2]) for k, v in res.items()})
    except Exception:
        print("\n=== post_process FAILED — real traceback ===")
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
