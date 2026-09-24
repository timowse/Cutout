# Third-party notices

This project's own source code is released under the [MIT License](LICENSE).
The deployed website also distributes the following third-party components.
All of them are under permissive licenses that allow free use, modification
and redistribution, including commercial use.

## BiRefNet (AI model weights)

- **What:** `BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx` ("BiRefNet_lite",
  general use, Swin-T backbone) from the author's GitHub release
  <https://github.com/ZhengPeng7/BiRefNet/releases/tag/v1>
- **SHA-256 of the source file:** `5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333`
- **License:** MIT License, Copyright (c) 2024 ZhengPeng
- **Modifications:** converted by [`tools/model/convert_birefnet.py`](tools/model/convert_birefnet.py)
  (memory-efficient but mathematically equivalent deformable convolutions,
  patch rearrangement as Reshape/Transpose, sigmoid output, float16 weight
  storage, split into chunks). The full MIT license text is shipped next to the
  model files (`models/birefnet-lite/LICENSE.txt`).
- **Paper:** Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen,
  Wanli Ouyang, Nicu Sebe: *Bilateral Reference for High-Resolution
  Dichotomous Image Segmentation*, CAAI Artificial Intelligence Research, 2024.

The BiRefNet architecture uses a Swin Transformer backbone
(<https://github.com/microsoft/Swin-Transformer>, MIT License, Copyright (c)
Microsoft Corporation).

Note on training data: the weights were trained by the BiRefNet author on
public research datasets (listed in the BiRefNet README), some of which have
their own terms of use. The author releases the resulting weights under the
MIT License; this project relies on that license.

```
MIT License

Copyright (c) 2024 ZhengPeng

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

## ONNX Runtime Web 1.30.0 (inference runtime)

- **What:** `onnxruntime-web` JavaScript bundles and WebAssembly binaries
  (<https://github.com/microsoft/onnxruntime>)
- **License:** MIT License, Copyright (c) Microsoft Corporation — full text in
  [`public/licenses/onnxruntime-LICENSE.txt`](public/licenses/onnxruntime-LICENSE.txt)
- The WebAssembly binaries contain further open-source components (e.g.
  Abseil, Eigen, ONNX, Protocol Buffers, XNNPACK, Dawn). Their notices are in
  ONNX Runtime's
  [`ThirdPartyNotices.txt`](public/licenses/onnxruntime-ThirdPartyNotices.txt)
  (copied from the v1.30.0 release tag), which is also deployed with the site
  at `licenses/onnxruntime-ThirdPartyNotices.txt`.

## Algorithms (own implementations)

- Foreground colour estimation follows M. Forte and F. Pitié, *Approximate
  Fast Foreground Colour Estimation*, IEEE ICIP 2021. Implemented from the paper
  in [`src/shared/foreground.ts`](src/shared/foreground.ts).
- Image resampling follows the separable-filter scheme popularised by Pillow;
  implemented independently in [`src/shared/resample.ts`](src/shared/resample.ts).

## Test images

The images in `tests/fixtures/images/` come from the
[scikit-image](https://scikit-image.org) data set and are free of copyright
restrictions (see [`tests/fixtures/images/README.md`](tests/fixtures/images/README.md)).

## Development tools

Build, test and lint tools (Vite, TypeScript, Vitest, Playwright, ESLint and
their dependencies) are only used during development and are not part of the
deployed website. Their licenses are listed in `package-lock.json`.
