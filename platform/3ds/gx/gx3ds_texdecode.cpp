// GameCube texture decoding (tiled GX formats -> RGBA8, row-major, top row first).
//
// Copied from sms-pc-port's platform/gx/src/gx_texture.cpp (helpers and decodeTexture), which
// is under the MIT License, Copyright (c) 2026 SoftwareAura LLC; see ext/sms-pc-port/LICENSE.
// Only the decoder is taken: that file's GL texture cache is replaced by gx3ds_backend.cpp.
#include "gx_internal.h"

#include <string.h>

namespace gx {

static inline uint16_t be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }

static inline void rgb565(uint16_t v, uint8_t* o) {
    o[0] = uint8_t((v >> 11) << 3 | (v >> 13));
    o[1] = uint8_t(((v >> 5) & 63) << 2 | ((v >> 9) & 3));
    o[2] = uint8_t((v & 31) << 3 | ((v >> 2) & 7));
    o[3] = 255;
}
static inline void rgb5a3(uint16_t v, uint8_t* o) {
    if (v & 0x8000) {
        o[0] = uint8_t(((v >> 10) & 31) << 3 | ((v >> 12) & 7));
        o[1] = uint8_t(((v >> 5) & 31) << 3 | ((v >> 7) & 7));
        o[2] = uint8_t((v & 31) << 3 | ((v >> 2) & 7));
        o[3] = 255;
    } else {
        o[0] = uint8_t(((v >> 8) & 15) * 17);
        o[1] = uint8_t(((v >> 4) & 15) * 17);
        o[2] = uint8_t((v & 15) * 17);
        o[3] = uint8_t(((v >> 12) & 7) * 255 / 7);
    }
}
static inline void ia8(uint16_t v, uint8_t* o) {
    o[0] = o[1] = o[2] = uint8_t(v & 0xFF);
    o[3] = uint8_t(v >> 8);
}

static const uint8_t kTileW[16] = {8, 8, 8, 4, 4, 4, 4, 0, 8, 8, 4, 0, 0, 0, 8, 0};
static const uint8_t kTileH[16] = {8, 4, 4, 4, 4, 4, 4, 0, 8, 4, 4, 0, 0, 0, 8, 0};
static const uint8_t kTileBytes[16] = {32, 32, 32, 32, 32, 32, 64, 0, 32, 32, 32, 0, 0, 0, 32, 0};

uint32_t texLevelBytes(uint32_t fmt, uint32_t w, uint32_t h) {
    fmt &= 15;
    if (!kTileW[fmt]) return 0;
    uint32_t cols = (w + kTileW[fmt] - 1) / kTileW[fmt];
    uint32_t rows = (h + kTileH[fmt] - 1) / kTileH[fmt];
    return cols * rows * kTileBytes[fmt];
}

static void palette(const uint8_t* tlut, uint32_t tlutFmt, uint32_t idx, uint8_t* o) {
    uint16_t v = be16(tlut + 2 * idx);
    if (tlutFmt == 0) ia8(v, o);
    else if (tlutFmt == 1) rgb565(v, o);
    else rgb5a3(v, o);
}

// Decodes one mip level into out (w*h*4 bytes, row-major, top row first).
void decodeTexture(const uint8_t* src, uint32_t fmt, uint32_t w, uint32_t h, const uint8_t* tlut, uint32_t tlutFmt,
                   uint8_t* out) {
    fmt &= 15;
    uint32_t tw = kTileW[fmt], th = kTileH[fmt];
    if (!tw) {
        memset(out, 0xFF, size_t(w) * h * 4);
        return;
    }
    uint32_t cols = (w + tw - 1) / tw, rows = (h + th - 1) / th;
    const uint8_t* p = src;
    for (uint32_t ty = 0; ty < rows; ty++) {
        for (uint32_t tx = 0; tx < cols; tx++) {
            uint32_t x0 = tx * tw, y0 = ty * th;
            auto put = [&](uint32_t x, uint32_t y, const uint8_t* c) {
                if (x < w && y < h) memcpy(out + (size_t(y) * w + x) * 4, c, 4);
            };
            uint8_t c[4];
            switch (fmt) {
            case 0:  // I4
                for (uint32_t i = 0; i < 64; i++) {
                    uint8_t b = p[i >> 1];
                    uint8_t v = (i & 1) ? (b & 15) : (b >> 4);
                    c[0] = c[1] = c[2] = c[3] = uint8_t(v * 17);
                    put(x0 + (i & 7), y0 + (i >> 3), c);
                }
                break;
            case 1:  // I8
                for (uint32_t i = 0; i < 32; i++) {
                    c[0] = c[1] = c[2] = c[3] = p[i];
                    put(x0 + (i & 7), y0 + (i >> 3), c);
                }
                break;
            case 2:  // IA4
                for (uint32_t i = 0; i < 32; i++) {
                    c[0] = c[1] = c[2] = uint8_t((p[i] & 15) * 17);
                    c[3] = uint8_t((p[i] >> 4) * 17);
                    put(x0 + (i & 7), y0 + (i >> 3), c);
                }
                break;
            case 3:  // IA8
                for (uint32_t i = 0; i < 16; i++) {
                    ia8(be16(p + 2 * i), c);
                    put(x0 + (i & 3), y0 + (i >> 2), c);
                }
                break;
            case 4:  // RGB565
                for (uint32_t i = 0; i < 16; i++) {
                    rgb565(be16(p + 2 * i), c);
                    put(x0 + (i & 3), y0 + (i >> 2), c);
                }
                break;
            case 5:  // RGB5A3
                for (uint32_t i = 0; i < 16; i++) {
                    rgb5a3(be16(p + 2 * i), c);
                    put(x0 + (i & 3), y0 + (i >> 2), c);
                }
                break;
            case 6:  // RGBA8: AR plane then GB plane
                for (uint32_t i = 0; i < 16; i++) {
                    c[3] = p[2 * i];
                    c[0] = p[2 * i + 1];
                    c[1] = p[32 + 2 * i];
                    c[2] = p[32 + 2 * i + 1];
                    put(x0 + (i & 3), y0 + (i >> 2), c);
                }
                break;
            case 8:  // C4
                for (uint32_t i = 0; i < 64; i++) {
                    uint8_t b = p[i >> 1];
                    palette(tlut, tlutFmt, (i & 1) ? (b & 15) : (b >> 4), c);
                    put(x0 + (i & 7), y0 + (i >> 3), c);
                }
                break;
            case 9:  // C8
                for (uint32_t i = 0; i < 32; i++) {
                    palette(tlut, tlutFmt, p[i], c);
                    put(x0 + (i & 7), y0 + (i >> 3), c);
                }
                break;
            case 10:  // C14X2
                for (uint32_t i = 0; i < 16; i++) {
                    palette(tlut, tlutFmt, be16(p + 2 * i) & 0x3FFF, c);
                    put(x0 + (i & 3), y0 + (i >> 2), c);
                }
                break;
            case 14:  // CMPR: 2x2 DXT1 sub-blocks
                for (uint32_t sb = 0; sb < 4; sb++) {
                    const uint8_t* b = p + sb * 8;
                    uint16_t c0 = be16(b), c1 = be16(b + 2);
                    uint8_t pal[4][4];
                    rgb565(c0, pal[0]);
                    rgb565(c1, pal[1]);
                    if (c0 > c1) {
                        for (int k = 0; k < 3; k++) {
                            pal[2][k] = uint8_t((2 * pal[0][k] + pal[1][k]) / 3);
                            pal[3][k] = uint8_t((pal[0][k] + 2 * pal[1][k]) / 3);
                        }
                        pal[2][3] = pal[3][3] = 255;
                    } else {
                        // Index 3 is transparent but keeps index 2's colour
                        // (the hardware's, as Dolphin decodes it), not black:
                        // a material that ignores texture alpha shows it.
                        for (int k = 0; k < 3; k++) {
                            pal[2][k] = uint8_t((pal[0][k] + pal[1][k]) / 2);
                            pal[3][k] = pal[2][k];
                        }
                        pal[2][3] = 255;
                        pal[3][3] = 0;
                    }
                    uint32_t sx = x0 + (sb & 1) * 4, sy = y0 + (sb >> 1) * 4;
                    for (uint32_t r = 0; r < 4; r++) {
                        uint8_t bits = b[4 + r];
                        for (uint32_t q = 0; q < 4; q++) put(sx + q, sy + r, pal[(bits >> (6 - 2 * q)) & 3]);
                    }
                }
                break;
            default:
                break;
            }
            p += kTileBytes[fmt];
        }
    }
}

}  // namespace gx
