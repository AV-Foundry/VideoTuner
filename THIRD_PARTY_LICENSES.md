# Third-Party Licenses

VideoTuner bundles the following third-party software components. This document provides attribution and license information as required by each component's license terms.

## Summary

| Component | License | Source |
| --------- | ------- | ------ |
| x264 Patman's Mod | GPL-2.0-or-later | <https://github.com/Patman86/x264-Mod-by-Patman> |
| x265 Patman's Mod | GPL-2.0-or-later | <https://github.com/Patman86/x265-Mod-by-Patman> |
| VapourSynth | LGPL-2.1-or-later | <https://github.com/vapoursynth/vapoursynth> |
| vapoursynth-zip | MIT | <https://github.com/dnjulek/vapoursynth-zip> |
| FFMS2 | GPL-3.0 (binary) | <https://github.com/FFMS/ffms2> |
| LSMASHSource | ISC + LGPL-2.1 | <https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works> |
| L-SMASH | ISC | <https://github.com/l-smash/l-smash> |
| 7-Zip | LGPL-2.1 | <https://7-zip.org> |
| Python | PSF-2.0 | <https://python.org> |

## Source Code Availability

For GPL and LGPL licensed components, source code is available at the URLs listed above. You may also request source code by contacting the VideoTuner maintainers.

---

## x264 Patman's Mod

**Location:** `tools/x264.exe`\
**License:** GNU General Public License v2.0 or later\
**Copyright:** Copyright (C) 2003-2024 x264 project contributors

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

Full license text: [licenses/GPL-2.0.txt](licenses/GPL-2.0.txt)

---

## x265 Patman's Mod

**Location:** `tools/x265.exe`\
**License:** GNU General Public License v2.0 or later\
**Copyright:** Copyright (C) 2013-2024 MulticoreWare, Inc

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

Full license text: [licenses/GPL-2.0.txt](licenses/GPL-2.0.txt)

---

## VapourSynth

**Location:** `vapoursynth-portable/`\
**License:** GNU Lesser General Public License v2.1 or later\
**Copyright:** Copyright (C) 2012-2024 Fredrik Mellbin

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

Full license text: [licenses/LGPL-2.1.txt](licenses/LGPL-2.1.txt)

---

## vapoursynth-zip (vszip)

**Location:** `vapoursynth-portable/vs-plugins/vszip.dll`\
**License:** MIT License\
**Copyright:** Copyright (c) 2024 Julek

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Full license text: [licenses/MIT.txt](licenses/MIT.txt)

---

## FFMS2

**Location:** `vapoursynth-portable/vs-plugins/ffms2.dll`, `ffmsindex.exe`\
**License:** GNU General Public License v3.0 (for binaries linked with GPL FFmpeg)\
**Copyright:** Copyright (C) 2007-2024 FFMS2 contributors

The FFMS2 source code is MIT licensed, but Windows binaries are distributed under GPL v3 due to being linked with GPL-licensed FFmpeg components.

Full license text: [licenses/GPL-3.0.txt](licenses/GPL-3.0.txt)

---

## LSMASHSource

**Location:** `vapoursynth-portable/vs-plugins/LSMASHSource.dll`\
**License:** ISC License (L-SMASH portions) + LGPL-2.1 (VapourSynth portions)\
**Copyright:** Copyright (C) 2011-2015 L-SMASH Works project contributors

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Full license texts: [licenses/ISC.txt](licenses/ISC.txt), [licenses/LGPL-2.1.txt](licenses/LGPL-2.1.txt)

---

## 7-Zip

**Location:** `vapoursynth-portable/7z.dll`, `7z.exe`\
**License:** GNU Lesser General Public License v2.1\
**Copyright:** Copyright (C) 1999-2024 Igor Pavlov

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.

Full license text: [licenses/LGPL-2.1.txt](licenses/LGPL-2.1.txt)

---

## Python

**Location:** `vapoursynth-portable/python*.dll`, `python.exe`\
**License:** Python Software Foundation License Version 2\
**Copyright:** Copyright (C) 2001-2024 Python Software Foundation

Python is distributed under the PSF License, which permits redistribution provided that copyright notices are retained. See `vapoursynth-portable/LICENSE.txt` for the complete license text.

Full license text: [vapoursynth-portable/LICENSE.txt](vapoursynth-portable/LICENSE.txt)

---

## Additional Bundled Components

The VapourSynth portable distribution includes additional components with their own licenses:

- **OpenSSL** (libcrypto, libssl): Apache License 2.0
- **SQLite**: Public Domain
- **Microsoft Visual C++ Runtime**: Redistributable under Microsoft's terms (see `vapoursynth-portable/LICENSE.txt`)
- **libffi**: MIT License
- **bzip2**: BSD-style license

These licenses are documented in `vapoursynth-portable/LICENSE.txt`.
