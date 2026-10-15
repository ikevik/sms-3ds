// 3DS back end for the sms_gx front half (gx_api.cpp, gx_fifo.cpp, gx_vert.cpp, gx_trace.cpp).
//
// The front half turns every GX/GD call, write-gather-pipe byte and display list into register
// writes (g, the BP/CP/XF file) and decodes each primitive's vertices into a packed batch. It
// reaches the renderer only through the functions defined here (see gx_internal.h). This file
// is written for the PICA200:
//   * drawBatch() transforms each batch on the CPU as sms-pc-port's vertex shader does
//     (position matrix from XF memory, GX projection, GX viewport) into EFB pixels and hands
//     the triangles to platform/3ds/pica (citro3d);
//   * the rasterised colour is the channel's material or vertex colour, shaded by the normal
//     when lighting is on; depth test and blending follow the BP registers;
//   * GX textures are decoded and cached, and the TEV stages are translated into PICA texture
//     combiner stages (translateTev);
//   * display copies present the frame.
// Interface is sms-pc-port's (MIT), the implementation is independent.
#include "gx_internal.h"
#include "../pica/pica_draw.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <initializer_list>
#include <unordered_map>
#include <vector>

namespace gx {

// ---- globals the front half expects from its renderer -------------------------------------
uint32_t g_displayFrames = 0;
double g_presentWaited = 0, g_presentDrain = 0, g_flushSeconds = 0;
bool g_gxStats = false;

// Counters, readable from the 3DS side (platform/3ds/gx/gx3ds_backend.h).
struct Stats {
	uint32_t frames, batches, vertices, triangles, lines, points, copies;
};
static Stats s_stats;
static Stats s_frame;  // the frame in progress

// ---- the batch --------------------------------------------------------------------------
static bool s_ready;
static std::vector<uint8_t> s_vdata;   // packed vertices (layout: vtxFmtLayout(s_bfmt))
static std::vector<uint32_t> s_idx;    // indices into s_vdata
static uint32_t s_bcount;              // vertices in the batch
static uint32_t s_bstride;
static uint32_t s_bfmt;
static PrimClass s_bclass = PRIM_TRIS;

static inline bool batchHasVertices() { return !s_idx.empty(); }

// ---- stage 1 renderer: CPU transform + flat colour ---------------------------------------
static inline float xff(int r) {
	float f;
	memcpy(&f, &g.xfReg[r], 4);
	return f;
}
static inline const float* xfMtx(uint32_t word) {  // XF memory as floats
	return reinterpret_cast<const float*>(&g.xfMem[word]);
}

// GX compare functions (NEVER, LESS, EQUAL, LEQUAL, GREATER, NEQUAL, GEQUAL, ALWAYS) as GPU_TESTFUNC.
static const uint8_t kTestFunc[8] = {0, 4, 2, 5, 6, 3, 7, 1};
// GX blend factors as GPU_BLENDFACTOR; the colour slots differ between source and destination.
static const uint8_t kBlendSrc[8] = {0, 1, 4, 5, 6, 7, 8, 9};  // ZERO ONE DSTCLR INVDSTCLR SRCA INVSRCA DSTA INVDSTA
static const uint8_t kBlendDst[8] = {0, 1, 2, 3, 6, 7, 8, 9};  // ZERO ONE SRCCLR INVSRCCLR SRCA INVSRCA DSTA INVDSTA

static void currentPicaState(PicaState& st) {
	uint32_t z = g.bp[BP_ZMODE], c = g.bp[BP_CMODE0];
	st.depth_test = z & 1;
	st.depth_func = kTestFunc[(z >> 1) & 7];
	st.depth_write = (z & 1) && ((z >> 4) & 1);
	st.blend = c & 1;
	st.blend_src = kBlendSrc[(c >> 8) & 7];
	st.blend_dst = kBlendDst[(c >> 5) & 7];
	if ((c >> 11) & 1) st.blend = 0;  // subtractive blending: not mapped yet
	st.color_write = (c >> 3) & 1;
}

static inline uint8_t clamp255(float v) { return v <= 0.f ? 0 : v >= 255.f ? 255 : uint8_t(v + 0.5f); }

// Debug colours (GX3DS_SetDebugColors, SELECT on the 3DS side): each draw gets a colour from a
// hash of what it draws (format, size, first vertex, matrix), so the same model keeps its
// colour from frame to frame, and every surface is shaded by its view-space normal. Alpha and
// blending are left alone so fades stay transparent.
static volatile bool s_debugColors;
// Draws before and after the frame's first EFB-to-texture copy, and the copies (status screen).
static bool s_afterTexCopy;
static uint32_t s_drawsBeforeCopy, s_drawsAfterCopy, s_texCopies;
static uint32_t s_lastBefore, s_lastAfter, s_lastCopies;

static void debugColor(uint32_t pm, const uint8_t* v0, uint32_t stride, uint32_t nidx, uint8_t out[3]) {
	uint32_t h = 2166136261u;
	auto mix = [&](uint32_t x) { h = (h ^ x) * 16777619u; };
	mix(s_bfmt);
	mix(nidx);
	mix(pm);
	for (uint32_t i = 0; i < stride && i < 16; i++) mix(v0[i]);
	// A bright colour: one channel high, one low, one from the hash.
	static const uint8_t kOrder[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
	const uint8_t* o = kOrder[(h >> 8) % 6];
	out[o[0]] = 255;
	out[o[1]] = uint8_t(40 + (h >> 16) % 80);
	out[o[2]] = uint8_t(h >> 24);
}

// ---- stage 2: textures ---------------------------------------------------------------------
// GX textures are decoded to RGBA8 (gx3ds_texdecode.cpp), resampled to a power of two from 8
// to kMaxTexSide, and kept in a cache keyed like sms-pc-port's (address, format, size, TLUT).
// The data is hashed once per frame to catch changes. Textures that are EFB copies are not
// supported yet: draws sampling them are skipped.
void decodeTexture(const uint8_t* src, uint32_t fmt, uint32_t w, uint32_t h, const uint8_t* tlut, uint32_t tlutFmt,
                   uint8_t* out);

static const uint32_t kMaxTexSide = 256;
static const uint32_t kTexBudget = 8u << 20;

struct TexKey {
	const uint8_t* ptr;
	uint32_t fmt, w, h, tlutAddr, tlutFmt;
	bool operator==(const TexKey& o) const {
		return ptr == o.ptr && fmt == o.fmt && w == o.w && h == o.h && tlutAddr == o.tlutAddr && tlutFmt == o.tlutFmt;
	}
};
struct TexKeyHash {
	size_t operator()(const TexKey& k) const {
		return size_t(uintptr_t(k.ptr)) ^ (k.fmt << 28) ^ (k.w << 16) ^ k.h ^ (k.tlutAddr << 8);
	}
};
struct TexEntry {
	PicaTex* tex = nullptr;
	uint64_t dataHash = 0, tlutHash = 0;
	uint32_t checkedFrame = ~0u, usedFrame = 0;
};
static std::unordered_map<TexKey, TexEntry, TexKeyHash> s_texCache;
static uint32_t s_texBytes;
static std::vector<uint8_t> s_decodeBuf, s_scaleBuf;
static uint32_t s_texUploads, s_copyTexDraws;

// EFB copies: each destination address the game copies the EFB to gets a whole-EFB snapshot
// texture (platform/3ds/pica), with the source rectangle of its latest copy. A draw sampling
// that address maps its texture coordinates through the rectangle into the snapshot.
struct EfbCopy {
	const void* dest;
	PicaTex* tex;
	float x, y, w, h;  // source rectangle in EFB pixels
	uint32_t usedFrame;
};
static std::vector<EfbCopy> s_efbCopies;
static const size_t kMaxEfbCopies = 6;

// Logs each distinct kind of EFB copy once (destination, rectangle, format, flags).
static void logCopyKind(const EfbCopy* c, uint32_t ctrl, uint32_t pix) {
	static std::vector<uint64_t> seen;
	const uint32_t fmt = ((ctrl >> 3) & 1) << 3 | ((ctrl >> 4) & 7);
	uint64_t key = (uint64_t(uintptr_t(c->dest)) << 32) ^ (uint64_t(ctrl) << 8) ^ pix ^
	               (uint64_t(uint32_t(c->w)) << 20) ^ (uint64_t(uint32_t(c->h)) << 40);
	for (uint64_t k : seen)
		if (k == key) return;
	if (seen.size() < 64) seen.push_back(key);
	fprintf(stderr,
	        "[copy] dest %p rect %d,%d %dx%d fmt %u intensity %d half %d clear %d efb-pixfmt %u%s\n",
	        c->dest, int(c->x), int(c->y), int(c->w), int(c->h), fmt, ((ctrl >> 15) & 3) == 3, (ctrl >> 9) & 1,
	        (ctrl >> 11) & 1, pix, pix == 3 ? " (Z copy: not supported, colour copied)" : "");
}

static EfbCopy* findEfbCopy(const void* p) {
	for (EfbCopy& c : s_efbCopies)
		if (c.dest == p) return &c;
	return nullptr;
}

static inline uint32_t pow2Side(uint32_t v) {
	uint32_t p = 8;
	while (p < v && p < kMaxTexSide) p <<= 1;
	return p;
}

static void evictTextures() {
	while (s_texBytes > kTexBudget) {
		auto victim = s_texCache.end();
		for (auto it = s_texCache.begin(); it != s_texCache.end(); ++it)
			if (it->second.usedFrame != g_displayFrames &&
			    (victim == s_texCache.end() || it->second.usedFrame < victim->second.usedFrame))
				victim = it;
		if (victim == s_texCache.end()) return;  // everything is in use this frame
		s_texBytes -= pica_tex_bytes(victim->second.tex);
		pica_tex_free(victim->second.tex);
		s_texCache.erase(victim);
	}
}

// The PICA texture for GX texture map `map`, or null (no data, out of memory). *copy is set
// when the map is an EFB copy.
static PicaTex* textureForMap(int map, const EfbCopy** copy) {
	const uint32_t img0 = g.bp[bpTexReg(BP_TX_IMAGE0, map)], tlutReg = g.bp[bpTexReg(BP_TX_TLUT, map)];
	const uint32_t w = (img0 & 0x3FF) + 1, h = ((img0 >> 10) & 0x3FF) + 1, fmt = (img0 >> 20) & 15;
	const uint8_t* ptr = g.texImage[map];
	if (!ptr) return nullptr;
	*copy = nullptr;
	if (EfbCopy* c = findEfbCopy(ptr)) {
		c->usedFrame = g_displayFrames;
		s_copyTexDraws++;
		*copy = c;
		return c->tex;
	}
	const bool ci = fmt == 8 || fmt == 9 || fmt == 10;
	TexKey key{ptr, fmt, w, h, ci ? (tlutReg & 0x3FF) : 0, ci ? ((tlutReg >> 10) & 3) : 0};
	const uint8_t* tlut = ci ? g.tlutMem + ((tlutReg & 0x3FF) << 9) : nullptr;
	uint32_t tlutBytes = fmt == 8 ? 32 : fmt == 9 ? 512 : 0x8000;
	if (tlut && ((tlutReg & 0x3FF) << 9) + tlutBytes > TMEM_TLUT_SIZE) tlutBytes = TMEM_TLUT_SIZE - ((tlutReg & 0x3FF) << 9);

	TexEntry& e = s_texCache[key];
	e.usedFrame = g_displayFrames;
	if (e.tex && e.checkedFrame == g_displayFrames) return e.tex;
	const uint32_t bytes = texLevelBytes(fmt, w, h);
	const uint64_t dh = hashBytes(ptr, bytes), th = tlut ? hashBytes(tlut, tlutBytes) : 0;
	e.checkedFrame = g_displayFrames;
	if (e.tex && dh == e.dataHash && th == e.tlutHash) return e.tex;

	const uint32_t pw = pow2Side(w), ph = pow2Side(h);
	if (!e.tex) {
		e.tex = pica_tex_create(pw, ph);
		if (!e.tex) {
			s_texCache.erase(key);
			return nullptr;
		}
		s_texBytes += pica_tex_bytes(e.tex);
	}
	s_decodeBuf.resize(size_t(w) * h * 4);
	decodeTexture(ptr, fmt, w, h, tlut, key.tlutFmt, s_decodeBuf.data());
	const uint8_t* src = s_decodeBuf.data();
	if (pw != w || ph != h) {  // nearest resample to the power-of-two size
		s_scaleBuf.resize(size_t(pw) * ph * 4);
		for (uint32_t y = 0; y < ph; y++) {
			const uint8_t* row = s_decodeBuf.data() + size_t(y * h / ph) * w * 4;
			uint8_t* o = s_scaleBuf.data() + size_t(y) * pw * 4;
			for (uint32_t x = 0; x < pw; x++) memcpy(o + x * 4, row + (x * w / pw) * 4, 4);
		}
		src = s_scaleBuf.data();
	}
	pica_tex_upload(e.tex, src);
	e.dataHash = dh;
	e.tlutHash = th;
	s_texUploads++;
	evictTextures();
	return e.tex;
}

// ---- stage 3: TEV -> PICA texture combiners ------------------------------------------------
// The GX TEV state is read from BP registers as sms-pc-port's shader generator reads it
// (gx_shader.cpp: buildKey/genFS) and translated into PICA combiner stages. Each GX stage
// computes, per channel (colour, alpha), r = d +/- lerp(a, b, c) + bias, times a scale, into
// one of four registers (PREV, C0-C2); the PICA has six stages that see the previous stage's
// output, a combiner buffer (one stage delayed), one constant colour per stage, the vertex
// colour and three textures. The translation tracks where each GX register's channel lives:
//   * a constant known now (the BP_TEV_REG value it starts with, or a stage whose inputs are
//     all constant, folded on the CPU);
//   * PREVIOUS, the output of the last PICA stage;
//   * the combiner buffer, where a live value is saved before a stage overwrites PREVIOUS.
// Each GX stage channel becomes one or two PICA operations (REPLACE / MODULATE / ADD /
// ADD_SIGNED / SUBTRACT / INTERPOLATE / MULTIPLY_ADD, and the stage scale); with two, the
// first one's result is kept in PREVIOUS for the second, and a GX stage takes two PICA
// stages. A setup that cannot be expressed falls back to the stage-2 rule (first texture x
// vertex colour) and is logged once.
enum { CH_RGB = 0, CH_A = 1 };
enum : uint8_t { SRC_PRIMARY = 0x00, SRC_TEX0 = 0x03, SRC_BUFFER = 0x0D, SRC_CONSTANT = 0x0E, SRC_PREVIOUS = 0x0F };
enum : uint8_t {
	FN_REPLACE = 0, FN_MODULATE = 1, FN_ADD = 2, FN_ADD_SIGNED = 3, FN_INTERPOLATE = 4, FN_SUBTRACT = 5,
	FN_MULTIPLY_ADD = 8
};

// A combiner input for one channel: a constant (rgb, or v[0] for alpha) or a PICA source with
// the component it reads (0-3: r, g, b, a replicated; 4: the colour as is, rgb only). mid: the
// result of the channel's first operation (PREVIOUS in its second).
struct TevSrc {
	bool k;
	int16_t v[3];
	uint8_t src, comp;
	bool inv;  // 1 - x
	bool mid;
};
static TevSrc tsConst(int r, int g, int b) {
	return TevSrc{true, {int16_t(r), int16_t(g), int16_t(b)}, 0, 0, false, false};
}
static TevSrc tsConst1(int a) { return tsConst(a, a, a); }
static TevSrc tsSrc(uint8_t src, uint8_t comp) { return TevSrc{false, {0, 0, 0}, src, comp, false, false}; }
static bool tsSame(const TevSrc& a, const TevSrc& b, int nc) {
	if (a.k != b.k || a.inv != b.inv || a.mid != b.mid) return false;
	if (a.k) {
		for (int i = 0; i < nc; i++)
			if (a.v[i] != b.v[i]) return false;
		return true;
	}
	return a.src == b.src && a.comp == b.comp;
}
static bool tsAll(const TevSrc& s, int nc, int v) {
	if (!s.k) return false;
	for (int i = 0; i < nc; i++)
		if (s.v[i] != v) return false;
	return true;
}

struct TevOp {
	uint8_t func, n, scale;
	TevSrc s[3];
};
// What one channel of a GX stage becomes: folded to a constant, or one or two PICA operations.
struct ChanPlan {
	bool folded;
	int16_t fv[3];
	int nops;
	TevOp op[2];
	const char* fail;
	bool softScale;  // scale 1/2, drawn at 1
};

static inline int tevLerp(int a, int b, int c) {
	int c2 = c + (c >> 7);
	return (a * (256 - c2) + b * c2 + 128) >> 8;
}

// GX stage arithmetic on constants (as genFS computes it).
static int tevFold(int a, int b, int c, int d, uint32_t bias, uint32_t sub, uint32_t scale, uint32_t clampv) {
	int r = sub ? d - tevLerp(a & 255, b & 255, c & 255) : d + tevLerp(a & 255, b & 255, c & 255);
	r += bias == 1 ? 128 : bias == 2 ? -128 : 0;
	if (scale == 1) r *= 2;
	else if (scale == 2) r *= 4;
	else if (scale == 3) r >>= 1;
	if (clampv) return r < 0 ? 0 : r > 255 ? 255 : r;
	return r < -1024 ? -1024 : r > 1023 ? 1023 : r;
}

static ChanPlan planChannel(TevSrc A, TevSrc B, TevSrc C, TevSrc D, int nc, uint32_t bias, uint32_t sub,
                            uint32_t scale, uint32_t clampv) {
	ChanPlan p;
	memset(&p, 0, sizeof p);
	if (bias == 3) {
		p.fail = "compare stage";
		return p;
	}
	for (TevSrc* s : {&A, &B, &C})
		if (s->k)
			for (int i = 0; i < 3; i++) s->v[i] &= 255;
	// Inputs the lerp ignores.
	if (tsAll(C, nc, 0)) B = tsConst1(0);
	else if (tsAll(C, nc, 255)) A = tsConst1(0);
	if (A.k && B.k && C.k && D.k) {
		p.folded = true;
		for (int i = 0; i < nc; i++) p.fv[i] = int16_t(tevFold(A.v[i], B.v[i], C.v[i], D.v[i], bias, sub, scale, clampv));
		return p;
	}
	const uint8_t outScale = scale == 1 ? 1 : scale == 2 ? 2 : 0;
	p.softScale = scale == 3;

	// lerp(a, b, c) as: nothing, X, X * Y, or a real interpolation.
	enum { LZ, LX, LXY, LERP } lf;
	TevSrc X{}, Y{};
	if (tsAll(C, nc, 0)) { lf = tsAll(A, nc, 0) ? LZ : LX; X = A; }
	else if (tsAll(C, nc, 255)) { lf = tsAll(B, nc, 0) ? LZ : LX; X = B; }
	else if (tsAll(A, nc, 0) && tsAll(B, nc, 0)) lf = LZ;
	else if (tsAll(A, nc, 0)) { lf = LXY; X = B; Y = C; }
	else if (tsAll(B, nc, 0)) { lf = LXY; X = A; Y = C; Y.inv = !Y.inv; }
	else if (tsSame(A, B, nc)) { lf = LX; X = A; }
	else lf = LERP;
	auto kval = [](const TevSrc& s, int i) { return s.inv ? 255 - s.v[i] : s.v[i]; };
	if (lf == LXY && X.k && Y.k) {  // a constant product
		TevSrc K = tsConst1(0);
		for (int i = 0; i < nc; i++) K.v[i] = int16_t(tevLerp(0, kval(X, i), kval(Y, i)));
		lf = LX;
		X = K;
	}

	auto op = [&](uint8_t fn, std::initializer_list<TevSrc> srcs) {
		TevOp& o = p.op[p.nops++];
		o.func = fn;
		o.n = 0;
		for (const TevSrc& s : srcs) o.s[o.n++] = s;
	};
	TevSrc mid = tsSrc(SRC_PREVIOUS, uint8_t(nc == 3 ? 4 : 3));
	mid.mid = true;
	// The lerp alone, as one operation (for LXY and LERP).
	auto lerpOp = [&] {
		if (lf == LXY) op(FN_MODULATE, {X, Y});
		else op(FN_INTERPOLATE, {B, A, C});
	};
	auto fold0 = [&] {
		p.folded = true;
		memset(p.fv, 0, sizeof p.fv);
	};
	auto finish = [&]() -> ChanPlan {
		if (p.nops) p.op[p.nops - 1].scale = outScale;
		// Constants under 1 - x are folded.
		for (int k = 0; k < p.nops; k++)
			for (int j = 0; j < p.op[k].n; j++) {
				TevSrc& s = p.op[k].s[j];
				if (s.k && s.inv) {
					for (int i = 0; i < 3; i++) s.v[i] = int16_t(255 - s.v[i]);
					s.inv = false;
				}
			}
		return p;
	};
	auto failed = [&](const char* why) {
		p.fail = why;
		return p;
	};
	const TevSrc half = tsConst1(128);

	int biasv = bias == 1 ? 128 : bias == 2 ? -128 : 0;
	if (D.k) {  // the bias joins a constant d
		bool neg = false, pos = false, over = false;
		for (int i = 0; i < nc; i++) {
			int v = D.v[i] + biasv;
			D.v[i] = int16_t(v > 255 ? 255 : v);
			if (v < 0) neg = true;
			if (v > 0) pos = true;
			if (v > 255) over = true;
		}
		biasv = 0;
		if (over && sub) return failed("constant d over 1 minus x");
		if (neg) {  // lerp - k
			if (pos || sub) return failed("constant d of mixed sign");
			TevSrc K = D;
			for (int i = 0; i < nc; i++) K.v[i] = int16_t(-K.v[i]);
			if (lf == LZ) fold0();
			else if (lf == LX) op(FN_SUBTRACT, {X, K});
			else {
				lerpOp();
				op(FN_SUBTRACT, {mid, K});
			}
			return finish();
		}
	}
	const bool dz = tsAll(D, nc, 0);
	if (biasv) {  // a varying d with a bias
		if (lf == LZ) {
			op(biasv > 0 ? FN_ADD : FN_SUBTRACT, {D, half});
		} else if (!sub && biasv < 0) {  // d + lerp - 1/2
			if (lf == LX) op(FN_ADD_SIGNED, {X, D});
			else {
				lerpOp();
				op(FN_ADD_SIGNED, {mid, D});
			}
		} else if (sub && biasv > 0) {  // d - lerp + 1/2 ~ d + (1 - lerp) - 1/2
			if (lf == LX) {
				TevSrc nx = X;
				nx.inv = !nx.inv;
				op(FN_ADD_SIGNED, {D, nx});
			} else {
				lerpOp();
				TevSrc nm = mid;
				nm.inv = true;
				op(FN_ADD_SIGNED, {D, nm});
			}
		} else if (lf == LX) {  // d + x + 1/2, d - x - 1/2: the first result's clamp is harmless
			op(sub ? FN_SUBTRACT : FN_ADD, sub ? std::initializer_list<TevSrc>{D, X} : std::initializer_list<TevSrc>{X, D});
			op(sub ? FN_SUBTRACT : FN_ADD, {mid, half});
		} else {
			return failed("bias with a varying d");
		}
		return finish();
	}
	if (!sub) {
		switch (lf) {
		case LZ: if (dz) fold0(); else op(FN_REPLACE, {D}); break;
		case LX: if (dz) op(FN_REPLACE, {X}); else op(FN_ADD, {X, D}); break;
		case LXY: if (dz) op(FN_MODULATE, {X, Y}); else op(FN_MULTIPLY_ADD, {X, Y, D}); break;
		case LERP:
			op(FN_INTERPOLATE, {B, A, C});
			if (!dz) op(FN_ADD, {mid, D});
			break;
		}
	} else {
		switch (lf) {
		case LZ: if (dz) fold0(); else op(FN_REPLACE, {D}); break;
		case LX: if (dz) fold0(); else op(FN_SUBTRACT, {D, X}); break;
		case LXY:
		case LERP:
			if (dz) {
				fold0();
				break;
			}
			lerpOp();
			op(FN_SUBTRACT, {D, mid});
			break;
		}
	}
	return finish();
}

// Logs each distinct unsupported TEV pattern once.
static void tevLogOnce(const char* why, uint32_t stage, uint32_t nStages, uint32_t ce, uint32_t ae) {
	static std::vector<uint64_t> seen;
	uint64_t key = (uint64_t(uintptr_t(why)) << 32) ^ (uint64_t(ce) << 8) ^ (uint64_t(ae) << 24) ^ stage;
	for (uint64_t k : seen)
		if (k == key) return;
	if (seen.size() >= 256) return;
	seen.push_back(key);
	fprintf(stderr, "[tev] %s: stage %u of %u, colour env %06x, alpha env %06x\n", why, stage, nStages, ce, ae);
}

// The GX TEV state of the draw, decoded.
struct TevSetup {
	uint32_t numStages;
	uint32_t colorEnv[16], alphaEnv[16], order[16], ksel[16], swap[4];
	int unit[16];          // PICA texture unit each stage samples (-1: none)
	int nUnits;
	int unitMap[PICA_MAX_TEX], unitCoord[PICA_MAX_TEX];
	int rasChan;           // colour channel the vertex colour carries (0, 1)
	bool tooManyTex;
};

static void readTev(TevSetup& t) {
	memset(&t, 0, sizeof t);
	t.numStages = ((g.bp[BP_GENMODE] >> 10) & 15) + 1;
	for (uint32_t s = 0; s < t.numStages; s++) {
		t.colorEnv[s] = g.bp[BP_TEV_COLOR_ENV + 2 * s] & 0xFFFFFF;
		t.alphaEnv[s] = g.bp[BP_TEV_COLOR_ENV + 2 * s + 1] & 0xFFFFFF;
		t.order[s] = (g.bp[BP_TREF + (s >> 1)] >> ((s & 1) * 12)) & 0x3FF;
		const uint32_t ks = g.bp[BP_TEV_KSEL + (s >> 1)];
		t.ksel[s] = (s & 1) ? ((ks >> 14) & 0x3FF) : ((ks >> 4) & 0x3FF);
	}
	for (int i = 0; i < 4; i++)
		t.swap[i] = (g.bp[BP_TEV_KSEL + 2 * i] & 15) | (g.bp[BP_TEV_KSEL + 2 * i + 1] & 15) << 4;
	t.rasChan = -1;
	for (uint32_t s = 0; s < t.numStages; s++) {
		const uint32_t ord = t.order[s], map = ord & 7, coord = (ord >> 3) & 7, chan = (ord >> 7) & 7;
		t.unit[s] = -1;
		if ((ord >> 6) & 1) {
			int u = 0;
			while (u < t.nUnits && (t.unitMap[u] != int(map) || t.unitCoord[u] != int(coord))) u++;
			if (u == t.nUnits) {
				if (t.nUnits == PICA_MAX_TEX) {
					t.tooManyTex = true;
					u = 0;
				} else {
					t.unitMap[u] = int(map);
					t.unitCoord[u] = int(coord);
					t.nUnits++;
				}
			}
			t.unit[s] = u;
		}
		// The vertex colour carries the first channel a stage reads.
		if (chan <= 1 && t.rasChan < 0) {
			const uint32_t ce = t.colorEnv[s], ae = t.alphaEnv[s];
			bool reads = false;
			for (int i = 0; i < 4; i++) {
				uint32_t c = (ce >> (4 * i)) & 15, a = (ae >> (4 + 3 * i)) & 7;
				if (c == 10 || c == 11 || a == 5) reads = true;
			}
			if (reads) t.rasChan = int(chan);
		}
	}
	if (t.rasChan < 0) t.rasChan = 0;
}

// GX registers PREV, C0-C2 by channel: where the value lives.
enum : uint8_t { LOC_CONST, LOC_PREV, LOC_BUF, LOC_LOST };
struct TevLoc {
	uint8_t kind;
	int16_t v[3];  // LOC_CONST: rgb, or v[0] for alpha
};

static inline int sext11(uint32_t v) { return int(v << 21) >> 21; }

static inline uint8_t clampByte(int v) { return uint8_t(v < 0 ? 0 : v > 255 ? 255 : v); }

// The channel (CH_RGB, CH_A) of PREVIOUS / the buffer a source reads.
static inline int srcChannel(const TevSrc& s) { return s.comp == 3 ? CH_A : CH_RGB; }

// Translates the TEV setup into st's combiner stages. unitWhite[u]: unit u has no texture (reads
// as white). Returns false (and logs) when the setup cannot be expressed.
static bool translateTev(const TevSetup& t, const bool unitWhite[PICA_MAX_TEX], PicaState& st) {
	const uint32_t N = t.numStages;
	if (t.tooManyTex) {
		tevLogOnce("more than 3 texture coordinate/map pairs", 0, N, t.colorEnv[0], t.alphaEnv[0]);
		return false;
	}
	// Liveness: bit 2 * reg + channel, live after each stage.
	uint8_t liveAfter[16];
	{
		uint8_t live = uint8_t(1u << (2 * ((t.colorEnv[N - 1] >> 22) & 3) + CH_RGB) |
		                       1u << (2 * ((t.alphaEnv[N - 1] >> 22) & 3) + CH_A));
		for (int s = int(N) - 1; s >= 0; s--) {
			liveAfter[s] = live;
			const uint32_t ce = t.colorEnv[s], ae = t.alphaEnv[s];
			live &= uint8_t(~(1u << (2 * ((ce >> 22) & 3) + CH_RGB)));
			live &= uint8_t(~(1u << (2 * ((ae >> 22) & 3) + CH_A)));
			for (int i = 0; i < 4; i++) {
				uint32_t c = (ce >> (4 * i)) & 15, a = (ae >> (4 + 3 * i)) & 7;
				if (c < 8) live |= uint8_t(1u << (2 * (c >> 1) + (c & 1)));
				if (a < 4) live |= uint8_t(1u << (2 * a + CH_A));
			}
		}
	}

	TevLoc loc[4][2];
	for (int r = 0; r < 4; r++) {
		const uint32_t ra = g.bp[BP_TEV_REG + 2 * r], bg = g.bp[BP_TEV_REG + 2 * r + 1];
		loc[r][CH_RGB] = TevLoc{LOC_CONST, {int16_t(sext11(ra & 0x7FF)), int16_t(sext11((bg >> 12) & 0x7FF)),
		                                    int16_t(sext11(bg & 0x7FF))}};
		loc[r][CH_A] = TevLoc{LOC_CONST, {int16_t(sext11((ra >> 12) & 0x7FF)), 0, 0}};
	}
	int prevOcc[2] = {-1, -1}, bufOcc[2] = {-1, -1};
	int n = 0;  // PICA stages emitted
	int curTex = -1;  // texture unit `tex` holds (-1: white), as genFS keeps it between stages
	static const int k8[8] = {255, 223, 191, 159, 127, 95, 63, 31};
	static const int nc[2] = {3, 1};

	for (uint32_t s = 0; s < N; s++) {
		const uint32_t ce = t.colorEnv[s], ae = t.alphaEnv[s], ord = t.order[s];
		const uint32_t chan = (ord >> 7) & 7;
		const uint32_t rasSwap = t.swap[ae & 3], texSwap = t.swap[(ae >> 2) & 3];
		if (t.unit[s] >= 0) curTex = unitWhite[t.unit[s]] ? -1 : t.unit[s];
		const char* softFail = nullptr;

		// konst
		const uint32_t kc = t.ksel[s] & 31, ka = (t.ksel[s] >> 5) & 31;
		auto kreg = [](uint32_t i, int comp) {  // comp 0-3: r, g, b, a
			const uint32_t ra = g.kreg[2 * i], bg = g.kreg[2 * i + 1];
			return int(comp == 0 ? ra & 0xFF : comp == 1 ? (bg >> 12) & 0xFF : comp == 2 ? bg & 0xFF : (ra >> 12) & 0xFF);
		};
		TevSrc konstC, konstA;
		if (kc < 8) konstC = tsConst1(k8[kc]);
		else if (kc >= 12 && kc < 16) konstC = tsConst(kreg(kc - 12, 0), kreg(kc - 12, 1), kreg(kc - 12, 2));
		else if (kc >= 16) konstC = tsConst1(kreg(kc & 3, int((kc - 16) >> 2)));
		else konstC = tsConst1(255);
		if (ka < 8) konstA = tsConst1(k8[ka]);
		else if (ka >= 16) konstA = tsConst1(kreg(ka & 3, int((ka - 16) >> 2)));
		else konstA = tsConst1(255);

		// A texture or vertex colour input through a swap table. comp: 4 colour, 3 alpha.
		auto swapped = [&](uint8_t src, uint32_t swap, int comp) -> TevSrc {
			const int sr = swap & 3, sg = (swap >> 2) & 3, sb = (swap >> 4) & 3, sa = (swap >> 6) & 3;
			if (comp == 3) return tsSrc(src, uint8_t(sa));
			if (sr == 0 && sg == 1 && sb == 2) return tsSrc(src, 4);
			if (sr == sg && sg == sb) return tsSrc(src, uint8_t(sr));
			softFail = "swap table permutes colour";
			return tsSrc(src, 4);
		};
		auto texIn = [&](int comp) -> TevSrc {
			if (curTex < 0) return tsConst1(255);
			return swapped(uint8_t(SRC_TEX0 + curTex), texSwap, comp);
		};
		auto rasIn = [&](int comp) -> TevSrc {
			if (chan > 1) {
				if (chan != 7) softFail = "bump alpha (indirect) colour";
				return tsConst1(0);
			}
			if (int(chan) != t.rasChan) softFail = "stages read both colour channels";
			return swapped(SRC_PRIMARY, rasSwap, comp);
		};
		bool lost = false;
		// Register reg's channel ch, read as comp (4: colour, 3: alpha replicated).
		auto regIn = [&](int reg, int ch, int comp) -> TevSrc {
			const TevLoc& l = loc[reg][ch];
			switch (l.kind) {
			case LOC_CONST: return ch == CH_A ? tsConst1(l.v[0]) : tsConst(l.v[0], l.v[1], l.v[2]);
			case LOC_PREV: return tsSrc(SRC_PREVIOUS, uint8_t(comp));
			case LOC_BUF: return tsSrc(SRC_BUFFER, uint8_t(comp));
			default: lost = true; return tsConst1(0);
			}
		};
		auto colorIn = [&](uint32_t c) -> TevSrc {
			if (c < 8) return regIn(int(c >> 1), (c & 1) ? CH_A : CH_RGB, (c & 1) ? 3 : 4);
			switch (c) {
			case 8: return texIn(4);
			case 9: return texIn(3);
			case 10: return rasIn(4);
			case 11: return rasIn(3);
			case 12: return tsConst1(255);
			case 13: return tsConst1(128);
			case 14: return konstC;
			default: return tsConst1(0);
			}
		};
		auto alphaIn = [&](uint32_t a) -> TevSrc {
			if (a < 4) return regIn(int(a), CH_A, 3);
			switch (a) {
			case 4: return texIn(3);
			case 5: return rasIn(3);
			case 6: return konstA;
			default: return tsConst1(0);
			}
		};

		ChanPlan plan[2];
		plan[CH_RGB] = planChannel(colorIn((ce >> 12) & 15), colorIn((ce >> 8) & 15), colorIn((ce >> 4) & 15),
		                           colorIn(ce & 15), 3, (ce >> 16) & 3, (ce >> 18) & 1, (ce >> 20) & 3, (ce >> 19) & 1);
		plan[CH_A] = planChannel(alphaIn((ae >> 13) & 7), alphaIn((ae >> 10) & 7), alphaIn((ae >> 7) & 7),
		                         alphaIn((ae >> 4) & 7), 1, (ae >> 16) & 3, (ae >> 18) & 1, (ae >> 20) & 3, (ae >> 19) & 1);
		if (lost) {
			tevLogOnce("register value lost (no room in the combiner buffer)", s, N, ce, ae);
			return false;
		}
		for (int ch = 0; ch < 2; ch++) {
			if (plan[ch].fail) {
				tevLogOnce(plan[ch].fail, s, N, ce, ae);
				return false;
			}
			if (plan[ch].softScale) tevLogOnce("scale 1/2 (drawn at 1)", s, N, ce, ae);
		}
		if (softFail) tevLogOnce(softFail, s, N, ce, ae);

		// An operation with two different constants loads one into PREVIOUS first: a one-op
		// channel becomes two.
		for (int ch = 0; ch < 2; ch++) {
			ChanPlan& p = plan[ch];
			for (int k = 0; k < p.nops; k++) {
				TevSrc ks[3];
				int nk = 0;
				for (int j = 0; j < p.op[k].n; j++) {
					if (!p.op[k].s[j].k) continue;
					bool dup = false;
					for (int i = 0; i < nk; i++) dup |= tsSame(ks[i], p.op[k].s[j], nc[ch]);
					if (!dup) ks[nk++] = p.op[k].s[j];
				}
				if (nk < 2) continue;
				if (nk > 2 || p.nops == 2) {
					tevLogOnce("too many constants in one stage", s, N, ce, ae);
					return false;
				}
				TevSrc mid = tsSrc(SRC_PREVIOUS, uint8_t(ch == CH_RGB ? 4 : 3));
				mid.mid = true;
				TevOp main = p.op[0];
				for (int j = 0; j < main.n; j++)
					if (main.s[j].k && tsSame(main.s[j], ks[0], nc[ch])) main.s[j] = mid;
				p.op[0] = TevOp{FN_REPLACE, 1, 0, {ks[0]}};
				p.op[1] = main;
				p.nops = 2;
				break;
			}
		}

		const int dest[2] = {int((ce >> 22) & 3), int((ae >> 22) & 3)};
		const bool computes[2] = {!plan[CH_RGB].folded, !plan[CH_A].folded};
		const bool two = plan[CH_RGB].nops == 2 || plan[CH_A].nops == 2;
		const int stagesNeeded = (computes[0] || computes[1] ? 1 : 0) + (two ? 1 : 0);
		if (n + stagesNeeded > PICA_MAX_STAGES) {
			tevLogOnce("more than 6 combiner stages", s, N, ce, ae);
			return false;
		}

		// Values in PREVIOUS that this stage overwrites but later stages read go to the buffer:
		// the stage that produced them (n - 1) updates it, and it shows from stage n + 1 on.
		bool saved[2] = {false, false};
		for (int ch = 0; ch < 2; ch++) {
			if (!computes[ch]) continue;
			const int r = prevOcc[ch];
			if (r < 0) continue;
			if (dest[ch] != r && (liveAfter[s] >> (2 * r + ch) & 1)) {
				const int b = bufOcc[ch];
				const bool bufFree = b < 0 || b == dest[ch] || !(liveAfter[s] >> (2 * b + ch) & 1);
				if (n < 1 || n - 1 > 3 || !bufFree) {
					tevLogOnce("no room to keep a register value", s, N, ce, ae);
					return false;
				}
				if (b >= 0) loc[b][ch].kind = LOC_LOST;
				if (ch == CH_RGB) st.buf_rgb |= uint8_t(1u << (n - 1));
				else st.buf_alpha |= uint8_t(1u << (n - 1));
				loc[r][ch].kind = LOC_BUF;
				bufOcc[ch] = r;
				saved[ch] = true;
			} else {
				loc[r][ch].kind = LOC_LOST;
			}
			prevOcc[ch] = -1;
		}
		// With two PICA stages, the second one's inputs must still be what this GX stage read:
		// PREVIOUS only of a channel the first one passes through, the buffer only of a channel
		// not saved to now.
		if (two) {
			for (int ch = 0; ch < 2; ch++) {
				const ChanPlan& p = plan[ch];
				if (!p.nops) continue;
				const TevOp& o = p.op[p.nops - 1];
				for (int j = 0; j < o.n; j++) {
					const TevSrc& x = o.s[j];
					if (x.k || x.mid) continue;
					const int xc = srcChannel(x);
					if ((x.src == SRC_PREVIOUS && plan[xc].nops == 2) || (x.src == SRC_BUFFER && saved[xc])) {
						tevLogOnce("value overwritten between two combiner stages", s, N, ce, ae);
						return false;
					}
				}
			}
		}

		auto opRgb = [](const TevSrc& x) -> uint32_t {
			static const uint8_t k[5] = {0x4, 0x8, 0xC, 0x2, 0x0};
			return (x.k ? 0u : k[x.comp]) | (x.inv ? 1u : 0u);
		};
		auto opA = [](const TevSrc& x) -> uint32_t {
			static const uint8_t k[4] = {0x2, 0x4, 0x6, 0x0};
			return (x.k ? 0u : k[x.comp > 3 ? 3 : x.comp]) | (x.inv ? 1u : 0u);
		};
		// Emits PICA stage(s): `which` 0 is the first of two, 1 the last (or only) one.
		for (int which = two ? 0 : 1; which < 2 && stagesNeeded; which++) {
			PicaTevStage& ps = st.stage[n++];
			memset(&ps, 0, sizeof ps);
			int16_t kv[4] = {0, 0, 0, 0};
			for (int ch = 0; ch < 2; ch++) {
				const ChanPlan& p = plan[ch];
				const TevOp* o = nullptr;
				if (computes[ch]) o = which == 0 ? (p.nops == 2 ? &p.op[0] : nullptr) : &p.op[p.nops - 1];
				uint32_t srcs = SRC_PREVIOUS, ops = 0;
				uint8_t func = FN_REPLACE, scl = 0;
				if (o) {
					srcs = 0;
					for (int j = 0; j < 3; j++) {
						const TevSrc& x = j < o->n ? o->s[j] : o->s[0];
						srcs |= uint32_t(x.k ? SRC_CONSTANT : x.src) << (4 * j);
						ops |= (ch == CH_RGB ? opRgb(x) : opA(x)) << (4 * j);
						if (x.k) {
							if (ch == CH_RGB) memcpy(kv, x.v, sizeof x.v);
							else kv[3] = x.v[0];
						}
					}
					func = o->func;
					scl = o->scale;
				}
				if (ch == CH_RGB) {
					ps.src_rgb = uint16_t(srcs);
					ps.op_rgb = uint16_t(ops);
					ps.func_rgb = func;
					ps.scale_rgb = scl;
				} else {
					ps.src_alpha = uint16_t(srcs);
					ps.op_alpha = uint16_t(ops);
					ps.func_alpha = func;
					ps.scale_alpha = scl;
				}
			}
			ps.color = uint32_t(clampByte(kv[0])) | uint32_t(clampByte(kv[1])) << 8 |
			           uint32_t(clampByte(kv[2])) << 16 | uint32_t(clampByte(kv[3])) << 24;
		}
		// Where the stage's results live now.
		for (int ch = 0; ch < 2; ch++) {
			const int r = dest[ch];
			if (bufOcc[ch] == r) bufOcc[ch] = -1;
			if (computes[ch]) {
				loc[r][ch].kind = LOC_PREV;
				prevOcc[ch] = r;
			} else {
				if (prevOcc[ch] == r) prevOcc[ch] = -1;
				loc[r][ch] = TevLoc{LOC_CONST, {plan[ch].fv[0], plan[ch].fv[1], plan[ch].fv[2]}};
			}
		}
	}

	// The output: the last stage's colour and alpha registers, moved into PREVIOUS if needed.
	const int outReg[2] = {int((t.colorEnv[N - 1] >> 22) & 3), int((t.alphaEnv[N - 1] >> 22) & 3)};
	if (n == 0 || prevOcc[CH_RGB] != outReg[CH_RGB] || prevOcc[CH_A] != outReg[CH_A]) {
		if (n == PICA_MAX_STAGES) {
			tevLogOnce("more than 6 combiner stages", N - 1, N, t.colorEnv[N - 1], t.alphaEnv[N - 1]);
			return false;
		}
		PicaTevStage& ps = st.stage[n++];
		memset(&ps, 0, sizeof ps);
		int16_t kv[4] = {0, 0, 0, 0};
		for (int ch = 0; ch < 2; ch++) {
			const TevLoc& l = loc[outReg[ch]][ch];
			uint8_t src = SRC_PREVIOUS;
			if (l.kind == LOC_LOST) {
				tevLogOnce("output value lost", N - 1, N, t.colorEnv[N - 1], t.alphaEnv[N - 1]);
				return false;
			}
			if (l.kind == LOC_BUF) src = SRC_BUFFER;
			if (l.kind == LOC_CONST) {
				src = SRC_CONSTANT;
				if (ch == CH_RGB) memcpy(kv, l.v, sizeof l.v);
				else kv[3] = l.v[0];
			}
			if (ch == CH_RGB) ps.src_rgb = src;
			else ps.src_alpha = src;
		}
		ps.color = uint32_t(clampByte(kv[0])) | uint32_t(clampByte(kv[1])) << 8 | uint32_t(clampByte(kv[2])) << 16 |
		           uint32_t(clampByte(kv[3])) << 24;
	}
	st.nstages = uint8_t(n);
	return true;
}

// GX alpha compare (BP 0xF3: ref0, ref1, comp0, comp1, op) reduced to the one PICA test.
static void alphaTest(PicaState& st) {
	const uint32_t r = g.bp[BP_ALPHACOMPARE];
	const uint32_t ref0 = r & 0xFF, ref1 = (r >> 8) & 0xFF, c0 = (r >> 16) & 7, c1 = (r >> 19) & 7, op = (r >> 22) & 3;
	uint32_t comp = c0, ref = ref0;
	if (op == 0 && c0 == 7) comp = c1, ref = ref1;       // AND with ALWAYS: the other one decides
	else if (op == 1 && c0 == 0) comp = c1, ref = ref1;  // OR with NEVER: the other one decides
	st.alpha_test = comp != 7;
	st.alpha_func = kTestFunc[comp];
	st.alpha_ref = uint8_t(ref);
}

// Texture coordinate `coord` of a vertex: texgen (XF 0x1040 + coord) with its texture matrix,
// and the dual-texture post matrix when enabled, as sms-pc-port's vertex shader computes it.
static void texCoord(int coord, const uint8_t* v, const VtxFmtLayout& L, uint32_t fmt, const uint8_t defTex[8],
                     float& u, float& w) {
	const uint32_t tg = g.xfReg[XFR_TEXGEN + coord];
	const uint32_t proj = (tg >> 1) & 1, form = (tg >> 2) & 1, type = (tg >> 4) & 7, row = (tg >> 7) & 31;
	float sv[4] = {0, 0, 1, 1};
	if (row == 0) {
		memcpy(sv, v, 12);
	} else if (row == 1 && (fmt & VF_NRM)) {
		memcpy(sv, v + L.nrm, 12);
	} else if (row >= 5 && row <= 12 && (fmt & (VF_TEX0 << (row - 5)))) {
		memcpy(sv, v + L.tex[row - 5], 8);
	}
	if (form == 0 && row < 5) sv[2] = 1.0f;  // AB11
	if (type != 0) {  // emboss and colour texgens: not supported, pass the source through
		u = sv[0];
		w = sv[1];
		return;
	}
	uint32_t tm = (fmt & VF_MTX) ? v[L.mtx + 1 + coord] : defTex[coord];
	const float* m = reinterpret_cast<const float*>(&g.xfMem[tm * 4]);
	float t0 = m[0] * sv[0] + m[1] * sv[1] + m[2] * sv[2] + m[3] * sv[3];
	float t1 = m[4] * sv[0] + m[5] * sv[1] + m[6] * sv[2] + m[7] * sv[3];
	float t2 = proj ? m[8] * sv[0] + m[9] * sv[1] + m[10] * sv[2] + m[11] * sv[3] : 1.0f;
	if (g.xfReg[XFR_DUALTEX] & 1) {
		const uint32_t post = g.xfReg[XFR_POSTTEX + coord];
		if ((post >> 8) & 1) {
			float len = sqrtf(t0 * t0 + t1 * t1 + t2 * t2);
			if (len > 0.f) t0 /= len, t1 /= len, t2 /= len;
		}
		const float* p = reinterpret_cast<const float*>(&g.xfMem[0x500 + (post & 63) * 4]);
		float a = p[0] * t0 + p[1] * t1 + p[2] * t2 + p[3];
		float b = p[4] * t0 + p[5] * t1 + p[6] * t2 + p[7];
		float c = p[8] * t0 + p[9] * t1 + p[10] * t2 + p[11];
		t0 = a, t1 = b, t2 = c;
	}
	if (t2 != 0.f && t2 != 1.f) {
		t0 /= t2;
		t1 /= t2;
	}
	u = t0;
	w = t1;
}

static void drawTriangles() {
	if (s_afterTexCopy) s_drawsAfterCopy++;
	else s_drawsBeforeCopy++;
	const VtxFmtLayout& L = vtxFmtLayout(s_bfmt);
	uint32_t n = s_bcount, nidx = uint32_t(s_idx.size());

	// Textures the TEV stages sample, one PICA unit per (map, coordinate) pair (none in
	// debug-colour mode).
	const bool debug = s_debugColors;
	TevSetup tev;
	readTev(tev);
	const int nUnits = debug ? 0 : tev.nUnits;
	PicaTex* utex[PICA_MAX_TEX] = {};
	const EfbCopy* ucopy[PICA_MAX_TEX] = {};
	bool uwhite[PICA_MAX_TEX] = {true, true, true};
	for (int u = 0; u < nUnits; u++) {
		utex[u] = textureForMap(tev.unitMap[u], &ucopy[u]);
		uwhite[u] = utex[u] == nullptr;
	}

	PicaVtx* out;
	uint16_t* oidx;
	if (!pica_alloc(n, nidx, &out, &oidx)) return;

	// Projection (GXSetProjection) and viewport (GXSetViewport), as the GX shader reads them.
	const float p0 = xff(XFR_PROJ), p1 = xff(XFR_PROJ + 1), p2 = xff(XFR_PROJ + 2), p3 = xff(XFR_PROJ + 3),
	            p4 = xff(XFR_PROJ + 4), p5 = xff(XFR_PROJ + 5);
	const bool ortho = g.xfReg[XFR_PROJ + 6] & 1;
	const float vsx = xff(XFR_VIEWPORT), vsy = xff(XFR_VIEWPORT + 1), vsz = xff(XFR_VIEWPORT + 2);
	const float vcx = xff(XFR_VIEWPORT + 3) - 342.0f, vcy = xff(XFR_VIEWPORT + 4) - 342.0f, vcz = xff(XFR_VIEWPORT + 5);
	const float kInvZ = 1.0f / 16777215.0f;

	// The colour channel the TEV stages read (0 or 1): material register or vertex colour;
	// lighting reduced to one fixed light.
	const int ch = debug ? 0 : tev.rasChan;
	const uint32_t cctrl = g.xfReg[XFR_COLOR0CTRL + ch], actrl = g.xfReg[XFR_ALPHA0CTRL + ch], mat = g.xfReg[XFR_MAT0 + ch];
	const bool hasClr = (s_bfmt & (VF_CLR0 << ch)) != 0, hasNrm = (s_bfmt & VF_NRM) != 0, hasMtx = (s_bfmt & VF_MTX) != 0;
	const bool clrFromVtx = (cctrl & 1) && hasClr, alphaFromVtx = (actrl & 1) && hasClr;
	const bool lit = ((cctrl >> 1) & 1) && hasNrm;
	const uint8_t matRGBA[4] = {uint8_t(mat >> 24), uint8_t(mat >> 16), uint8_t(mat >> 8), uint8_t(mat)};
	const uint32_t defPm = g.xfReg[XFR_MATIDX_A] & 63;
	const uint32_t matA = g.xfReg[XFR_MATIDX_A], matB = g.xfReg[XFR_MATIDX_B];
	const uint8_t defTex[8] = {uint8_t((matA >> 6) & 63),  uint8_t((matA >> 12) & 63), uint8_t((matA >> 18) & 63),
	                           uint8_t((matA >> 24) & 63), uint8_t(matB & 63),         uint8_t((matB >> 6) & 63),
	                           uint8_t((matB >> 12) & 63), uint8_t((matB >> 18) & 63)};
	uint8_t dbg[3] = {0, 0, 0};
	if (debug) debugColor(hasMtx ? s_vdata[L.mtx] : defPm, s_vdata.data(), s_bstride, nidx, dbg);

	for (uint32_t i = 0; i < n; i++) {
		const uint8_t* v = s_vdata.data() + size_t(i) * s_bstride;
		float pos[3];
		memcpy(pos, v, 12);
		uint32_t pm = hasMtx ? v[L.mtx] : defPm;
		const float* m = xfMtx(pm * 4);
		float vx = m[0] * pos[0] + m[1] * pos[1] + m[2] * pos[2] + m[3];
		float vy = m[4] * pos[0] + m[5] * pos[1] + m[6] * pos[2] + m[7];
		float vz = m[8] * pos[0] + m[9] * pos[1] + m[10] * pos[2] + m[11];
		float cx, cy, cz, cw;
		if (!ortho) {
			cx = p0 * vx + p1 * vz;
			cy = p2 * vy + p3 * vz;
			cz = p4 * vz + p5;
			cw = -vz;
		} else {
			cx = p0 * vx + p1;
			cy = p2 * vy + p3;
			cz = p4 * vz + p5;
			cw = 1.0f;
		}
		PicaVtx& o = out[i];
		o.x = vcx * cw + vsx * cx;
		o.y = vcy * cw + vsy * cy;
		o.z = (vcz * cw + vsz * cz) * kInvZ;
		o.w = cw;

		for (int u = 0; u < PICA_MAX_TEX; u++) {
			float* tc = o.tc[u];
			if (!utex[u]) {
				tc[0] = tc[1] = 0.f;
				continue;
			}
			texCoord(tev.unitCoord[u], v, L, s_bfmt, defTex, tc[0], tc[1]);
			if (const EfbCopy* c = ucopy[u])  // through the copy's source rectangle into the EFB snapshot
				pica_efb_uv(c->x + tc[0] * c->w, c->y + tc[1] * c->h, &tc[0], &tc[1]);
		}

		const uint8_t* src = debug ? dbg : clrFromVtx ? v + L.clr[ch] : matRGBA;
		float shade = 1.0f;
		if (lit || (debug && hasNrm)) {
			float nrm[3];
			memcpy(nrm, v + L.nrm, 12);
			const float* nm = xfMtx(256 + pm * 3);  // normal matrices: 3x3, word 256 + 3 * index
			float nx = nm[0] * nrm[0] + nm[1] * nrm[1] + nm[2] * nrm[2];
			float ny = nm[3] * nrm[0] + nm[4] * nrm[1] + nm[5] * nrm[2];
			float nz = nm[6] * nrm[0] + nm[7] * nrm[1] + nm[8] * nrm[2];
			float len = sqrtf(nx * nx + ny * ny + nz * nz);
			float d = len > 0.f ? (0.3f * nx + 0.6f * ny + 0.74f * nz) / len : 1.0f;  // light from up-front
			shade = 0.35f + 0.65f * (d > 0.f ? d : 0.f);
		}
		o.rgba[0] = clamp255(src[0] * shade);
		o.rgba[1] = clamp255(src[1] * shade);
		o.rgba[2] = clamp255(src[2] * shade);
		o.rgba[3] = alphaFromVtx ? v[L.clr[ch] + 3] : matRGBA[3];
	}
	for (uint32_t i = 0; i < nidx; i++) oidx[i] = uint16_t(s_idx[i]);

	PicaState st;
	memset(&st, 0, sizeof st);  // the PICA side compares states bytewise
	currentPicaState(st);
	alphaTest(st);
	static const uint8_t kWrap[4] = {0, 2, 3, 2};  // clamp, repeat, mirror -> GPU_TEXTURE_WRAP_PARAM
	for (int u = 0; u < nUnits; u++) {
		if (!utex[u]) continue;
		const uint32_t mode0 = g.bp[bpTexReg(BP_TX_MODE0, tev.unitMap[u])];
		st.tex[u] = utex[u];
		st.wrap_s[u] = kWrap[mode0 & 3];
		st.wrap_t[u] = kWrap[(mode0 >> 2) & 3];
		st.linear[u] = (mode0 >> 4) & 1;
	}
	if (!debug && !translateTev(tev, uwhite, st)) {
		// Stage-2 rule: the first texture times the vertex colour (the texture alone when that
		// stage's colour is GX_COLOR_NULL).
		memset(st.stage, 0, sizeof st.stage);
		st.buf_rgb = st.buf_alpha = 0;
		st.nstages = 0;
		for (uint32_t s = 0; s < tev.numStages; s++) {
			const int u = tev.unit[s];
			if (u < 0 || !utex[u]) continue;
			PicaTevStage& ps = st.stage[0];
			const bool rasNull = ((tev.order[s] >> 7) & 7) == 7;
			ps.src_rgb = ps.src_alpha = uint16_t(SRC_TEX0 + u) | (rasNull ? 0 : SRC_PRIMARY << 4);
			ps.func_rgb = ps.func_alpha = rasNull ? FN_REPLACE : FN_MODULATE;
			st.nstages = 1;
			break;
		}
	}
	pica_draw_triangles(out, oidx, nidx, &st);
}

// The PICA200 draw: s_vdata/s_idx hold the batch, g the register state.
static void drawBatch() {
	if (s_bclass == PRIM_TRIS && s_bcount) drawTriangles();  // lines and points: not yet
	switch (s_bclass) {
	case PRIM_TRIS:   s_frame.triangles += uint32_t(s_idx.size() / 3); break;
	case PRIM_LINES:  s_frame.lines += uint32_t(s_idx.size() / 2); break;
	case PRIM_POINTS: s_frame.points += uint32_t(s_idx.size()); break;
	}
	s_frame.vertices += s_bcount;
	s_frame.batches++;
}

void flushBatch() {
	if (!s_ready) return;
	if (batchHasVertices()) drawBatch();
	s_vdata.clear();
	s_idx.clear();
	s_bcount = 0;
}

void onStateChange() {
	if (batchHasVertices()) flushBatch();
}

PrimClass primClass(uint8_t op) { return op >= 0xB8 ? PRIM_POINTS : op >= 0xA8 ? PRIM_LINES : PRIM_TRIS; }

uint8_t* primitiveBegin(uint8_t op, uint32_t n, uint32_t fmt, uint32_t stride) {
	PrimClass cls = primClass(op);
	if (batchHasVertices() && (cls != s_bclass || fmt != s_bfmt)) flushBatch();
	s_bclass = cls;
	s_bfmt = fmt;
	s_bstride = stride;
	size_t at = size_t(s_bcount) * stride;
	s_vdata.resize(at + size_t(n) * stride);
	return s_vdata.data() + at;
}

uint32_t primitiveIndices(uint8_t op, uint32_t n, uint32_t base, uint32_t* w) {
	uint32_t* const w0 = w;
	auto tri = [&](uint32_t a, uint32_t b, uint32_t c) {
		w[0] = base + a;
		w[1] = base + b;
		w[2] = base + c;
		w += 3;
	};
	switch (op & 0xF8) {
	case 0x80:
	case 0x88:  // quads
		for (uint32_t i = 0; i + 3 < n; i += 4) {
			tri(i, i + 1, i + 2);
			tri(i, i + 2, i + 3);
		}
		break;
	case 0x90:  // triangles
		for (uint32_t i = 0; i + 2 < n; i += 3) tri(i, i + 1, i + 2);
		break;
	case 0x98:  // strip
		for (uint32_t i = 2; i < n; i++) {
			if (i & 1) tri(i - 1, i - 2, i);
			else tri(i - 2, i - 1, i);
		}
		break;
	case 0xA0:  // fan
		for (uint32_t i = 2; i < n; i++) tri(0, i - 1, i);
		break;
	case 0xA8:  // lines
		for (uint32_t i = 0; i + 1 < n; i += 2) {
			*w++ = base + i;
			*w++ = base + i + 1;
		}
		break;
	case 0xB0:  // line strip
		for (uint32_t i = 1; i < n; i++) {
			*w++ = base + i - 1;
			*w++ = base + i;
		}
		break;
	default:  // points
		for (uint32_t i = 0; i < n; i++) *w++ = base + i;
		break;
	}
	return uint32_t(w - w0);
}

void primitiveEnd(uint8_t op, uint32_t n) {
	if (!s_ready || n == 0) return;
	uint32_t base = s_bcount;
	s_bcount += n;
	s_vdata.resize(size_t(s_bcount) * s_bstride);
	size_t at = s_idx.size();
	s_idx.resize(at + size_t(n) * 3);  // no primitive makes more than 3 indices per vertex
	uint32_t k = primitiveIndices(op, n, base, s_idx.data() + at);
	s_idx.resize(at + k);
}

void appendDecoded(PrimClass cls, uint32_t fmt, uint32_t stride, const uint8_t* verts, uint32_t n, const uint32_t* idx,
                   uint32_t nidx) {
	if (batchHasVertices() && (cls != s_bclass || fmt != s_bfmt)) flushBatch();
	s_bclass = cls;
	s_bfmt = fmt;
	s_bstride = stride;
	if (!s_ready || n == 0) return;
	s_vdata.resize(size_t(s_bcount) * stride);
	s_vdata.insert(s_vdata.end(), verts, verts + size_t(n) * stride);
	uint32_t base = s_bcount;
	s_bcount += n;
	size_t at = s_idx.size();
	s_idx.resize(at + nidx);
	for (uint32_t i = 0; i < nidx; i++) s_idx[at + i] = idx[i] + base;
}

// ---- display-list arena: not used (the front half checks arenaEnabled first) --------------
bool arenaEnabled() { return false; }
uint32_t arenaGeneration() { return 1; }
bool arenaUpload(const uint8_t*, uint32_t, uint32_t, const uint32_t*, uint32_t, ArenaRef&) { return false; }
void appendArena(PrimClass, uint32_t, uint32_t, const ArenaRef&, uint32_t, uint32_t, const uint8_t*) {}
bool batchUsesArena() { return false; }

// ---- frame / copies ---------------------------------------------------------------------------
// Called when a frame is complete (a display copy). Set by the 3DS side to present it.
typedef void (*FrameHook)(const Stats& frame, uint32_t frameNumber);
static FrameHook s_frameHook;

void executeCopy(uint32_t ctrl) {
	if (!s_ready) return;
	flushBatch();
	s_stats.copies++;
	bool disp = (ctrl >> 14) & 1;  // copy to the external frame buffer: the frame is done
	bool clear = (ctrl >> 11) & 1;
	if (!disp) {
		s_texCopies++;
		s_afterTexCopy = true;
		flushBatch();
		const void* d = g.copyDest;
		EfbCopy* c = d ? findEfbCopy(d) : nullptr;
		if (d && !c) {
			if (s_efbCopies.size() >= kMaxEfbCopies) {  // replace the least recently sampled
				size_t oldest = 0;
				for (size_t i = 1; i < s_efbCopies.size(); i++)
					if (s_efbCopies[i].usedFrame < s_efbCopies[oldest].usedFrame) oldest = i;
				pica_tex_free(s_efbCopies[oldest].tex);
				s_efbCopies.erase(s_efbCopies.begin() + oldest);
			}
			PicaTex* t = pica_copy_tex_create();
			if (t) {
				s_efbCopies.push_back(EfbCopy{d, t, 0, 0, 0, 0, g_displayFrames});
				c = &s_efbCopies.back();
			}
		}
		if (c) {
			// BP 0x49: top-left (x, y), 0x4A: size - 1 (w, h), in EFB pixels.
			const uint32_t tl = g.bp[BP_COPY_SRC_TL], wh = g.bp[BP_COPY_SRC_WH];
			c->x = float(tl & 0x3FF);
			c->y = float((tl >> 10) & 0x3FF);
			c->w = float((wh & 0x3FF) + 1);
			c->h = float(((wh >> 10) & 0x3FF) + 1);
			// Alpha reads as 1 when the EFB format has none (anything but RGBA6_Z24, as sms-pc-port's
			// copy) or the copy's texture format has none (RGB565: a GX texture of it decodes alpha 1).
			const uint32_t pix = g.bp[BP_PE_CONTROL] & 7;
			const uint32_t cfmt = ((ctrl >> 3) & 1) << 3 | ((ctrl >> 4) & 7);
			pica_efb_copy_to(c->tex, pix != 1 || cfmt == 4);
			logCopyKind(c, ctrl, pix);
		}
	}
	if (clear || disp) {
		// BP 0x4F: A<<8|R, 0x50: G<<8|B, 0x51: 24-bit Z.
		uint32_t ar = g.bp[BP_CLEAR_AR], gb = g.bp[BP_CLEAR_GB];
		uint32_t rgba = ((ar & 0xFF) << 24) | (((gb >> 8) & 0xFF) << 16) | ((gb & 0xFF) << 8) | ((ar >> 8) & 0xFF);
		if (disp) pica_present();
		// A texture copy that clears (render-to-texture passes) clears the frame in progress; the
		// display copy's clear applies to the next frame.
		if (clear) pica_clear(rgba, g.bp[BP_CLEAR_Z] & 0xFFFFFF);
	}
	if (disp) {
		g_displayFrames++;
		s_stats.frames++;
		s_stats.batches += s_frame.batches;
		s_stats.vertices += s_frame.vertices;
		s_stats.triangles += s_frame.triangles;
		s_stats.lines += s_frame.lines;
		s_stats.points += s_frame.points;
		if (s_frameHook) s_frameHook(s_frame, g_displayFrames);
		memset(&s_frame, 0, sizeof s_frame);
		s_lastBefore = s_drawsBeforeCopy;
		s_lastAfter = s_drawsAfterCopy;
		s_lastCopies = s_texCopies;
		s_drawsBeforeCopy = s_drawsAfterCopy = s_texCopies = 0;
		s_afterTexCopy = false;
	}
}

void markXfMemDirty() {}
void textureInvalidateAll() {}
void textureInvalidateRange(const void*, uint32_t) {}
void textureCpuWrote(const void*, uint32_t) {}
unsigned efbCopyLookup(const void*, int*, int*) { return 0; }
uint32_t peekColor(int, int) { return 0; }
uint32_t peekZ(int, int) { return 0; }
void pixMetricClear() {}
uint32_t pixMetricRead() { return 0; }
bool rendererReady() { return s_ready; }
float efbScale() { return 1.0f; }

}  // namespace gx

// ---- C interface -------------------------------------------------------------------------------
extern "C" {

// GXInit calls this when the renderer is not ready; the 3DS side normally calls it first.
int GXPC_InitAuto(int) {
	if (!gx::s_ready) {
		gx::resetState();
		gx::s_ready = true;
	}
	return 1;
}

void GX3DS_SetDebugColors(int on) { gx::s_debugColors = on != 0; }
int GX3DS_GetDebugColors(void) { return gx::s_debugColors ? 1 : 0; }

// Last frame: triangle draws before its first EFB-to-texture copy, after it, and the copies.
// Texture uploads so far, cache bytes, draws skipped for sampling an EFB copy.
void GX3DS_GetTexStats(uint32_t out[3]) {
	out[0] = gx::s_texUploads;
	out[1] = gx::s_texBytes;
	out[2] = gx::s_copyTexDraws;
}

void GX3DS_GetCopyStats(uint32_t out[3]) {
	out[0] = gx::s_lastBefore;
	out[1] = gx::s_lastAfter;
	out[2] = gx::s_lastCopies;
}

void GX3DS_SetFrameHook(void (*hook)(const void* frameStats, uint32_t frameNumber)) {
	gx::s_frameHook = reinterpret_cast<gx::FrameHook>(hook);
}

// frames, batches, vertices, triangles, lines, points, copies (all frames so far)
void GX3DS_GetStats(uint32_t out[7]) {
	out[0] = gx::s_stats.frames;
	out[1] = gx::s_stats.batches;
	out[2] = gx::s_stats.vertices;
	out[3] = gx::s_stats.triangles;
	out[4] = gx::s_stats.lines;
	out[5] = gx::s_stats.points;
	out[6] = gx::s_stats.copies;
}

}  // extern "C"
