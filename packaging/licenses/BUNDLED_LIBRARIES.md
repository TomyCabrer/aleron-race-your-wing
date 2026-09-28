# Native libraries bundled with pygame

pygame's wheels carry these libraries next to it (macOS: `pygame/.dylibs`;
Windows: `pygame/*.dll`; Linux: `pygame.libs/`). The pygame wheel does not
include their licence texts, so they are listed here. Every Python package in
the build has its own licence files in this folder (see README.txt).

| library | licence |
|---|---|
| SDL2, SDL2_image, SDL2_mixer, SDL2_ttf | zlib (text below) |
| zlib / zlib-ng | zlib (text below) |
| FreeType | the FreeType Project License (FTL), chosen over GPL-2: "Portions of this software are copyright © The FreeType Project (www.freetype.org). All rights reserved." |
| libpng | libpng License (PNG Reference Library License v2) |
| libjpeg / libjpeg-turbo | IJG License / BSD-3-Clause: "This software is based in part on the work of the Independent JPEG Group." |
| libtiff | libtiff License (BSD-like) |
| libwebp, sharpyuv | BSD-3-Clause (Google) |
| brotli | MIT |
| ogg, vorbis, opus, opusfile, FLAC | BSD-3-Clause (Xiph.Org Foundation) |
| libmodplug | public domain |
| portmidi | MIT-style (PortMidi license) |
| FluidSynth, GLib, GThread, libintl, mpg123, libsndfile | **LGPL-2.1 or later**: full text in `LGPL-2.1.txt`, source offer in `SOURCE_OFFER.md` |

## zlib licence (SDL2 and its satellite libraries, zlib)

```
This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

SDL2: Copyright (C) 1997-2023 Sam Lantinga <slouken@libsdl.org>.
