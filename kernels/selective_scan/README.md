# selective_scan kernel (optional)

CUDA selective-scan kernel (`selective_scan_cuda_core`, "core" variant only) used by
the DMSM branch in `d2mamba/models/defmamba.py`. Adapted from VMamba (MIT), derived
from Mamba (Apache-2.0).

```bash
pip install . --no-build-isolation
```

If this extension is not installed, `defmamba.py` falls back to `selective_scan_cuda`
from `mamba-ssm`, so installing either one is enough.
