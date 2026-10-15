// 3DS back end for the sms_gx front half (gx_api.cpp, gx_fifo.cpp, gx_vert.cpp, gx_trace.cpp).
//
// The front half turns every GX/GD call, write-gather-pipe byte and display list into register
// writes (g, the BP/CP/XF file) and decodes each primitive's vertices into a packed batch. It
// reaches the renderer only through the functions defined here (see gx_internal.h). This file
// is written for the PICA200:
//   * drawBatch() transforms each batch on the CPU as sms-pc-port's vertex shader does
//     (position matrix from XF memory, GX projection, GX viewport) into EFB pixels and hands
//     the triangles to platform/3ds/pica (citro3d);
//   * stage 1 is untextured: colour is the channel-0 material or vertex colour, shaded by the
//     normal when lighting is on; depth test and blending follow the BP registers;
//   * display copies present the frame.
// Interface is sms-pc-port's (MIT), the implementation is independent.
#include "gx_internal.h"
#include "../pica/pica_draw.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

// The first TEV stage that samples a texture, and what it reads.
struct TexStage {
	bool textured;
	int map, coord;
	bool rasNull;  // the stage's rasterised colour is GX_COLOR_NULL: the texture stands alone
};
static TexStage findTexStage() {
	TexStage ts{false, 0, 0, false};
	const uint32_t numStages = ((g.bp[BP_GENMODE] >> 10) & 15) + 1;
	for (uint32_t s = 0; s < numStages; s++) {
		uint32_t order = (g.bp[BP_TREF + (s >> 1)] >> ((s & 1) * 12)) & 0x3FF;
		if ((order >> 6) & 1) {
			ts.textured = true;
			ts.map = order & 7;
			ts.coord = (order >> 3) & 7;
			ts.rasNull = ((order >> 7) & 7) == 7;
			return ts;
		}
	}
	return ts;
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

	// Texture of the first texturing TEV stage (none in debug-colour mode).
	const TexStage ts = s_debugColors ? TexStage{false, 0, 0, false} : findTexStage();
	PicaTex* tex = nullptr;
	const EfbCopy* efbCopy = nullptr;
	if (ts.textured) tex = textureForMap(ts.map, &efbCopy);
	const bool textured = tex != nullptr;

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

	// Channel 0 colour: material register or vertex colour; lighting reduced to one fixed light.
	const uint32_t cctrl = g.xfReg[XFR_COLOR0CTRL], actrl = g.xfReg[XFR_ALPHA0CTRL], mat = g.xfReg[XFR_MAT0];
	const bool hasClr = (s_bfmt & VF_CLR0) != 0, hasNrm = (s_bfmt & VF_NRM) != 0, hasMtx = (s_bfmt & VF_MTX) != 0;
	const bool clrFromVtx = (cctrl & 1) && hasClr, alphaFromVtx = (actrl & 1) && hasClr;
	const bool lit = ((cctrl >> 1) & 1) && hasNrm;
	const uint8_t matRGBA[4] = {uint8_t(mat >> 24), uint8_t(mat >> 16), uint8_t(mat >> 8), uint8_t(mat)};
	const uint32_t defPm = g.xfReg[XFR_MATIDX_A] & 63;
	const uint32_t matA = g.xfReg[XFR_MATIDX_A], matB = g.xfReg[XFR_MATIDX_B];
	const uint8_t defTex[8] = {uint8_t((matA >> 6) & 63),  uint8_t((matA >> 12) & 63), uint8_t((matA >> 18) & 63),
	                           uint8_t((matA >> 24) & 63), uint8_t(matB & 63),         uint8_t((matB >> 6) & 63),
	                           uint8_t((matB >> 12) & 63), uint8_t((matB >> 18) & 63)};
	static const uint8_t kWhite[4] = {255, 255, 255, 255};
	const bool rasWhite = textured && ts.rasNull;
	const bool debug = s_debugColors;
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

		if (textured) {
			texCoord(ts.coord, v, L, s_bfmt, defTex, o.u, o.v);
			if (efbCopy)  // through the copy's source rectangle into the EFB snapshot
				pica_efb_uv(efbCopy->x + o.u * efbCopy->w, efbCopy->y + o.v * efbCopy->h, &o.u, &o.v);
		} else {
			o.u = o.v = 0.f;
		}

		const uint8_t* src = debug ? dbg : rasWhite ? kWhite : clrFromVtx ? v + L.clr[0] : matRGBA;
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
		o.rgba[3] = rasWhite ? 255 : alphaFromVtx ? v[L.clr[0] + 3] : matRGBA[3];
	}
	for (uint32_t i = 0; i < nidx; i++) oidx[i] = uint16_t(s_idx[i]);

	PicaState st;
	memset(&st, 0, sizeof st);  // the PICA side compares states bytewise
	currentPicaState(st);
	alphaTest(st);
	if (textured) {
		const uint32_t mode0 = g.bp[bpTexReg(BP_TX_MODE0, ts.map)];
		static const uint8_t kWrap[4] = {0, 2, 3, 2};  // clamp, repeat, mirror -> GPU_TEXTURE_WRAP_PARAM
		st.tex = tex;
		st.wrap_s = kWrap[mode0 & 3];
		st.wrap_t = kWrap[(mode0 >> 2) & 3];
		st.linear = (mode0 >> 4) & 1;
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
