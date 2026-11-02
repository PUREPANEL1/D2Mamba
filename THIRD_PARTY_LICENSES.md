# Third-party code and licenses

The original D²Mamba code in this repository (SDM fusion, CCLR decoder, DDR
dataset/preprocessing/loss/metric, configs, tools, tests) is licensed under the
Apache License 2.0 (see `LICENSE`). The files below are adapted from other
projects and remain subject to their original terms.

| File(s) | Upstream | Upstream license |
| --- | --- | --- |
| `d2mamba/models/vssd_backbone.py` (Backbone_VMAMBA2, Mamba2/NC-SSD; D²Mamba adds the deformable branch and `γ` fusion) | [VSSD](https://github.com/YuHengsss/VSSD) | **No license file published upstream** |
| `d2mamba/models/mamba_util.py` | [VSSD](https://github.com/YuHengsss/VSSD) | **No license file published upstream** |
| `d2mamba/models/defmamba.py` | [DefMamba](https://github.com/leiyeliu/DefMamba) | **No license file published upstream** |
| `kernels/selective_scan/` | [VMamba](https://github.com/MzeroMiko/VMamba) (derived from Mamba) | MIT (VMamba), Apache-2.0 (Mamba) |
| `tools/train.py`, `tools/test.py`, `tools/dist_train.sh`, `tools/dist_test.sh` | [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) | Apache-2.0 |

## Note on VSSD and DefMamba

At the time of this release the VSSD and DefMamba repositories did not include
a license file. Without a license, the default copyright of their authors
applies, and the Apache-2.0 license of this repository does **not** extend to
the adapted files listed above. They are included for research
reproducibility of the D²Mamba paper only. If you redistribute or use them
beyond that, please check the current upstream terms or contact the original
authors.

## MIT License (VMamba)

```
MIT License

Copyright (c) 2024 MzeroMiko

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apache License 2.0 (MMSegmentation, Mamba)

MMSegmentation — Copyright 2020 The MMSegmentation Authors.
Mamba — Copyright 2023 Tri Dao, Albert Gu.
Full text: see `LICENSE` in this repository (identical Apache-2.0 text).
