# Third-party research and license notices

VibeScope core source, compact estimator, training code and synthetic examples are original work. No upstream neural model source or pretrained checkpoint is distributed in the application. The optional baseline runner uses an external EulerMormer checkout. Retain its upstream LICENSE with any copy or redistribution of those sources or weights. This file preserves the explicit upstream notices for the inspected MIT foundations. Bibliographic attribution is in [docs/FOUNDATIONS.md](docs/FOUNDATIONS.md).

Axial-MM was reviewed at `f6d2ce387d6f7f4e4840905d35019cc809c6ce00`; no repository LICENSE file was present. Its source and weights are not bundled, adapted or redistributed. Obtain appropriate permission before incorporating them.

## EulerMormer

Source: https://github.com/VUT-HFUT/EulerMormer

```text
MIT License

Copyright (c) 2024 Fei Wang

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

## STB-VMM

Source: https://github.com/RLado/STB-VMM

```text
MIT License

Copyright (c) 2022 Ricard Lado

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

## ViMag

Source: https://github.com/RLado/ViMag

```text
MIT License

Copyright (c) 2022 Ricard Lado

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

## Runtime packages

The application depends on separately installed packages including NumPy (BSD), SciPy (BSD), OpenCV (Apache 2.0), FastAPI (MIT), Uvicorn (BSD), Matplotlib (PSF-based license), Pillow (HPND), and python-multipart (Apache 2.0). Optional research dependencies include PyTorch (BSD-style) and einops (MIT). Test dependencies include pytest (MIT) and HTTPX (BSD). Their installed distributions retain their own complete license files and transitive notices. This list is informational and does not replace those files.

FFmpeg is supplied by the separately installed imageio-ffmpeg distribution for generated video exports; ffprobe is an optional separately installed tool. VibeScope source does not embed an FFmpeg binary. The actual FFmpeg license depends on that build and enabled components (including libx264); retain the distributor's license and notices. No FFmpeg executable is included in this repository.
