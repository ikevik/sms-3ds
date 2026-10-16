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

// ---- profiler -------------------------------------------------------------------------------
// Per-frame time of each renderer phase, in ARM11 system ticks (268,111,856 Hz), averaged over
// kPerfFrames frames for the status screen and logged as [perf] lines. "other" is the rest of the
// frame: game logic, the GX front half (command parsing, vertex decoding) and waiting.
extern "C" uint64_t svcGetSystemTick(void);
enum PerfPhase { PERF_TEX, PERF_TEV, PERF_XFORM, PERF_SUBMIT, PERF_COPY, PERF_PRESENT, PERF_COUNT };
static uint64_t s_perfAcc[PERF_COUNT], s_perfFrameStart, s_perfWindowStart;
static uint32_t s_perfFrames, s_perfBatches, s_perfVerts;
static float s_perfAvgMs[PERF_COUNT + 2];  // phases, then frame and other
static float s_perfAvgBatches, s_perfAvgVerts;
static const uint32_t kPerfFrames = 60;
static const double kTicksPerMs = 268111.856;
struct PerfScope {
	PerfPhase ph;
	uint64_t t0;
	explicit PerfScope(PerfPhase p) : ph(p), t0(svcGetSystemTick()) {}
	~PerfScope() { s_perfAcc[ph] += svcGetSystemTick() - t0; }
};

static void perfFrameEnd() {
	const uint64_t now = svcGetSystemTick();
	if (!s_perfWindowStart) {
		s_perfWindowStart = now;
		memset(s_perfAcc, 0, sizeof s_perfAcc);
		s_perfFrames = s_perfBatches = s_perfVerts = 0;
		return;
	}
	if (++s_perfFrames < kPerfFrames) return;
	const double frames = s_perfFrames, total = double(now - s_perfWindowStart);
	double ours = 0;
	for (int i = 0; i < PERF_COUNT; i++) {
		s_perfAvgMs[i] = float(double(s_perfAcc[i]) / frames / kTicksPerMs);
		ours += s_perfAcc[i];
	}
	s_perfAvgMs[PERF_COUNT] = float(total / frames / kTicksPerMs);
	s_perfAvgMs[PERF_COUNT + 1] = float((total - ours) / frames / kTicksPerMs);
	s_perfAvgBatches = float(s_perfBatches / frames);
	s_perfAvgVerts = float(s_perfVerts / frames);
	fprintf(stderr,
	        "[perf] frame %.1f ms (%.1f fps) | tex %.2f tev %.2f xform %.2f submit %.2f copy %.2f present %.2f | "
	        "other %.1f | %.0f batches %.0f verts/frame\n",
	        s_perfAvgMs[PERF_COUNT], 1000.0 / s_perfAvgMs[PERF_COUNT], s_perfAvgMs[PERF_TEX], s_perfAvgMs[PERF_TEV],
	        s_perfAvgMs[PERF_XFORM], s_perfAvgMs[PERF_SUBMIT], s_perfAvgMs[PERF_COPY], s_perfAvgMs[PERF_PRESENT],
	        s_perfAvgMs[PERF_COUNT + 1], s_perfAvgBatches, s_perfAvgVerts);
	s_perfWindowStart = now;
	memset(s_perfAcc, 0, sizeof s_perfAcc);
	s_perfFrames = s_perfBatches = s_perfVerts = 0;
}

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

// Depth (BP 0x40), blending and write masks (0x41, and 0x42's constant destination alpha), as
// sms-pc-port's renderer maps them (gx_render.cpp). Logic ops are not mapped (drawn unblended).
static void currentPicaState(PicaState& st) {
	enum { EQ_ADD = 0, EQ_REVSUB = 2, F_ZERO = 0, F_ONE = 1, F_DSTA = 8, F_INVDSTA = 9, F_CONSTA = 12 };
	const uint32_t z = g.bp[BP_ZMODE], c = g.bp[BP_CMODE0], c1 = g.bp[BP_CMODE1];
	st.depth_test = z & 1;
	st.depth_func = kTestFunc[(z >> 1) & 7];
	st.depth_write = (z & 1) && ((z >> 4) & 1);
	// Culling (BP 0x00 bits 14-15, hardware encoding: 1 back, 2 front, 3 all). GX front faces are
	// clockwise on screen: clockwise in the PICA's y-up clip space, where its front is
	// counter-clockwise.
	const uint32_t cull = (g.bp[BP_GENMODE] >> 14) & 3;
	st.cull = cull == 1 ? 1 /* GPU_CULL_FRONT_CCW */ : cull == 2 ? 2 /* GPU_CULL_BACK_CCW */ : 0;
	const bool hasAlpha = (g.bp[BP_PE_CONTROL] & 7) == 1;  // GX_PF_RGBA6_Z24
	const bool colorUpd = (c >> 3) & 1, alphaUpd = ((c >> 4) & 1) && hasAlpha;
	const bool dstAlpha = (c1 >> 8) & 1;  // the EFB alpha written is a constant
	st.write_mask = uint8_t((colorUpd ? 7 : 0) | (alphaUpd || (dstAlpha && hasAlpha) ? 8 : 0));
	// Without EFB alpha, destination alpha reads as 1.
	auto noDstAlpha = [&](uint8_t f) -> uint8_t {
		if (hasAlpha) return f;
		return f == F_DSTA ? F_ONE : f == F_INVDSTA ? F_ZERO : f;
	};
	// Scissor (BP 0x20/0x21: coordinates + 342), off when it covers the EFB.
	{
		const uint32_t tl = g.bp[BP_SCISSOR_TL], br = g.bp[BP_SCISSOR_BR];
		int top = int(tl & 0x7FF) - 342, left = int((tl >> 12) & 0x7FF) - 342;
		int bottom = int(br & 0x7FF) - 342 + 1, right = int((br >> 12) & 0x7FF) - 342 + 1;
		auto clampi = [](int v, int hi) { return v < 0 ? 0 : v > hi ? hi : v; };
		left = clampi(left, PICA_EFB_WIDTH), right = clampi(right, PICA_EFB_WIDTH);
		top = clampi(top, PICA_EFB_HEIGHT), bottom = clampi(bottom, PICA_EFB_HEIGHT);
		if (right < left) right = left;
		if (bottom < top) bottom = top;
		st.scissor = !(left == 0 && top == 0 && right == PICA_EFB_WIDTH && bottom >= PICA_EFB_HEIGHT);
		if (st.scissor) {
			st.sc_x0 = uint16_t(left);
			st.sc_y0 = uint16_t(top);
			st.sc_x1 = uint16_t(right);
			st.sc_y1 = uint16_t(bottom);
		}
	}
	const uint8_t aSrc = dstAlpha ? F_CONSTA : F_ONE, aDst = F_ZERO;
	st.blend_color = uint32_t(c1 & 0xFF) << 24;
	st.blend = 0;
	if (c & 1) {
		st.blend = 1;
		if ((c >> 11) & 1) {  // subtract: dst - src
			st.blend_eq = EQ_REVSUB;
			st.blend_src = st.blend_dst = F_ONE;
			st.blend_aeq = dstAlpha ? EQ_ADD : EQ_REVSUB;
			st.blend_asrc = dstAlpha ? aSrc : F_ONE;
			st.blend_adst = dstAlpha ? aDst : F_ONE;
		} else {
			st.blend_eq = st.blend_aeq = EQ_ADD;
			st.blend_src = noDstAlpha(kBlendSrc[(c >> 8) & 7]);
			st.blend_dst = noDstAlpha(kBlendDst[(c >> 5) & 7]);
			st.blend_asrc = dstAlpha ? aSrc : st.blend_src;
			st.blend_adst = dstAlpha ? aDst : st.blend_dst;
		}
	} else if (dstAlpha) {  // colour as is, alpha the constant
		st.blend = 1;
		st.blend_eq = st.blend_aeq = EQ_ADD;
		st.blend_src = F_ONE;
		st.blend_dst = F_ZERO;
		st.blend_asrc = aSrc;
		st.blend_adst = aDst;
	}
}

static inline uint8_t clamp255(float v) { return v <= 0.f ? 0 : v >= 255.f ? 255 : uint8_t(v + 0.5f); }

// Debug colours (GX3DS_SetDebugColors, SELECT on the 3DS side): each draw gets a colour from a
// hash of what it draws (format, size, first vertex, matrix), so the same model keeps its
// colour from frame to frame, and every surface is shaded by its view-space normal. Alpha and
// blending are left alone so fades stay transparent.
static volatile bool s_debugColors;
// Opaque mode (GX3DS_SetDebugColors 2): blending and the alpha test off, to tell what a draw's
// colour is from how it is blended.
static volatile bool s_opaque;
// One-frame trace (GX3DS_TraceNextFrame, L on the 3DS side): every draw, copy and clear of the
// next frame, in order, to the log.
static volatile bool s_traceArm;
static bool s_tracing;
static unsigned s_traceDraw;
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
	uint32_t hashedFrame = 0, hashedClock = 0;  // when the data was last hashed (write-stamp clock)
};
static std::unordered_map<TexKey, TexEntry, TexKeyHash> s_texCache;
static uint32_t s_texBytes;
static std::vector<uint8_t> s_decodeBuf, s_scaleBuf;
static uint32_t s_texUploads, s_copyTexDraws;
static std::unordered_map<PicaTex*, uint32_t> s_texAvg;  // average RGBA of each uploaded texture

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
	        "[copy] dest %p rect %d,%d %dx%d fmt %u intensity %d half %d clear %d (AR %04x GB %04x) efb-pixfmt %u%s\n",
	        c->dest, int(c->x), int(c->y), int(c->w), int(c->h), fmt, ((ctrl >> 15) & 3) == 3, (ctrl >> 9) & 1,
	        (ctrl >> 11) & 1, g.bp[BP_CLEAR_AR] & 0xFFFF, g.bp[BP_CLEAR_GB] & 0xFFFF, pix, pix == 3 ? " (Z copy: not supported, colour copied)" : "");
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
	// As sms-pc-port's cache: when no write stamp (GXPC_InvalidateRange below) touched the data
	// since it was last hashed, trust it, but still hash every kRecheckFrames frames for writes
	// nothing reported. TLUTs live in TMEM, rewritten by register loads: always hashed (small).
	const uint64_t th = tlut ? hashBytes(tlut, tlutBytes) : 0;
	e.checkedFrame = g_displayFrames;
	if (e.tex && th == e.tlutHash && trustWriteStamps() && g_displayFrames - e.hashedFrame < kRecheckFrames &&
	    unwrittenSince(ptr, bytes, e.hashedClock))
		return e.tex;
	e.hashedClock = writeClock();
	e.hashedFrame = g_displayFrames;
	const uint64_t dh = hashBytes(ptr, bytes);
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
	{  // average colour, for the draw trace
		uint64_t sum[4] = {0, 0, 0, 0};
		const size_t px = size_t(pw) * ph;
		for (size_t i = 0; i < px; i++)
			for (int c = 0; c < 4; c++) sum[c] += src[i * 4 + c];
		s_texAvg[e.tex] = uint32_t(sum[0] / px) << 24 | uint32_t(sum[1] / px) << 16 | uint32_t(sum[2] / px) << 8 |
		                  uint32_t(sum[3] / px);
	}
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
	bool softScale;    // scale 1/2, drawn at 1
	bool softCompare;  // compare drawn as a steep ramp
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

// GX compare stage on constants: d + (a OP b ? c : 0), per genFS. mode: scale << 1 | op. The
// GR8/GR16/GR24 forms compare the colour inputs' r(, g, b); for the alpha channel those are not
// known here (nc == 1): -1.
static int tevCompareFold(const int16_t* a, const int16_t* b, int c, int d, int i, int nc, uint32_t mode,
                          uint32_t clampv) {
	bool cond;
	if (mode >= 6) {
		cond = mode == 6 ? a[i] > b[i] : a[i] == b[i];
	} else {
		if (nc == 1) return -1;
		int x = a[0] & 255, y = b[0] & 255;
		if (mode >= 2) x |= (a[1] & 255) << 8, y |= (b[1] & 255) << 8;
		if (mode >= 4) x |= (a[2] & 255) << 16, y |= (b[2] & 255) << 16;
		cond = (mode & 1) ? x == y : x > y;
	}
	int r = d + (cond ? c : 0);
	if (clampv) return r < 0 ? 0 : r > 255 ? 255 : r;
	return r < -1024 ? -1024 : r > 1023 ? 1023 : r;
}

static ChanPlan planChannel(TevSrc A, TevSrc B, TevSrc C, TevSrc D, int nc, uint32_t bias, uint32_t sub,
                            uint32_t scale, uint32_t clampv) {
	ChanPlan p;
	memset(&p, 0, sizeof p);
	for (TevSrc* s : {&A, &B, &C})
		if (s->k)
			for (int i = 0; i < 3; i++) s->v[i] &= 255;
	auto op = [&](uint8_t fn, std::initializer_list<TevSrc> srcs) {
		TevOp& o = p.op[p.nops++];
		o.func = fn;
		o.n = 0;
		for (const TevSrc& s : srcs) o.s[o.n++] = s;
	};
	auto failed = [&](const char* why) {
		p.fail = why;
		return p;
	};
	if (bias == 3) {  // compare: d + (a OP b ? c : 0), no scale
		const uint32_t mode = scale << 1 | sub;
		if (A.k && B.k && C.k && D.k) {
			p.folded = true;
			for (int i = 0; i < nc; i++) {
				const int r = tevCompareFold(A.v, B.v, C.v[i], D.v[i], i, nc, mode, clampv);
				if (r < 0 && nc == 1) return failed("alpha compare of colour inputs");
				p.fv[i] = int16_t(r);
			}
			return p;
		}
		// Never true (0 > x), or adding nothing: d.
		if ((!sub && tsAll(A, nc, 0)) || tsAll(C, nc, 0)) {
			if (D.k) {
				p.folded = true;
				for (int i = 0; i < nc; i++)
					p.fv[i] = int16_t(clampv ? (D.v[i] < 0 ? 0 : D.v[i] > 255 ? 255 : D.v[i]) : D.v[i]);
			} else {
				op(FN_REPLACE, {D});
			}
			return p;
		}
		// a > b ? 1 : 0 as a steep ramp, clamp(32 * (a - b)): the GR8-GR24 forms compare red
		// (the inputs are usually replicated), RGB8/A8 each component.
		if (!sub && tsAll(C, nc, 255) && tsAll(D, nc, 0)) {
			TevSrc a = A, b = B;
			for (TevSrc* x : {&a, &b}) {
				if (mode >= 6) continue;
				if (nc == 1) return failed("alpha compare of colour inputs");
				if (x->k) x->v[1] = x->v[2] = x->v[0];
				else if (x->comp == 4) x->comp = 0;
			}
			TevSrc mid = tsSrc(SRC_PREVIOUS, uint8_t(nc == 3 ? 4 : 3));
			mid.mid = true;
			op(FN_SUBTRACT, {a, b});
			p.op[0].scale = 2;
			op(FN_ADD, {mid, mid});
			p.op[1].scale = 2;
			p.softCompare = true;
			return p;
		}
		return failed("compare stage");
	}
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

	// A constant added to d +/- lerp, per component: the bias, and parts of the lerp below.
	int off[3];
	for (int i = 0; i < 3; i++) off[i] = i < nc ? (bias == 1 ? 128 : bias == 2 ? -128 : 0) : 0;
	if (D.k && lf == LERP && A.k && B.k) {
		// d +/- lerp(a, b, c) with constant a, b, d: lerp(d +/- a, d +/- b, c), when both fit.
		TevSrc A2 = A, B2 = B;
		bool fits = true;
		for (int i = 0; i < nc; i++) {
			const int dv = D.v[i] + off[i];
			const int a = sub ? dv - A.v[i] : dv + A.v[i], b = sub ? dv - B.v[i] : dv + B.v[i];
			fits &= a >= 0 && a <= 255 && b >= 0 && b <= 255;
			A2.v[i] = int16_t(a);
			B2.v[i] = int16_t(b);
		}
		if (fits) {
			A = A2;
			B = B2;
			D = tsConst1(0);
			memset(off, 0, sizeof off);
			sub = 0;
		}
	}
	if (lf == LERP && C.k && (A.k || B.k)) {
		// A constant c and one constant end: x * y plus a constant.
		for (int i = 0; i < nc; i++) {
			const int k = B.k ? tevLerp(0, B.v[i], C.v[i]) : tevLerp(A.v[i], 0, C.v[i]);
			off[i] += sub ? -k : k;
		}
		if (B.k) {
			X = A;
			Y = C;
			Y.inv = true;
		} else {
			X = B;
			Y = C;
		}
		lf = LXY;
	}
	if (lf == LXY && X.k && Y.k) {  // a constant product
		TevSrc K = tsConst1(0);
		for (int i = 0; i < nc; i++) K.v[i] = int16_t(tevLerp(0, kval(X, i), kval(Y, i)));
		lf = LX;
		X = K;
	}
	if (lf == LX && X.k && !D.k) {  // a constant lerp joins the constant
		for (int i = 0; i < nc; i++) off[i] += sub ? -X.v[i] : X.v[i];
		lf = LZ;
	}

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

	if (D.k) {  // the constant joins a constant d
		bool neg = false, pos = false, over = false;
		for (int i = 0; i < nc; i++) {
			int v = D.v[i] + off[i];
			D.v[i] = int16_t(v > 255 ? 255 : v);
			if (v < 0) neg = true;
			if (v > 0) pos = true;
			if (v > 255) over = true;
			off[i] = 0;
		}
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
	bool offPos = false, offNeg = false, offHalf = true;
	for (int i = 0; i < nc; i++) {
		offPos |= off[i] > 0;
		offNeg |= off[i] < 0;
		offHalf &= off[i] == off[0];
	}
	if (offPos || offNeg) {  // a varying d and a constant
		if (lf == LZ && offPos && offNeg) {  // d + k, k of mixed sign: d + (k + 1/2) - 1/2
			TevSrc K = tsConst1(0);
			for (int i = 0; i < nc; i++) {
				const int v = off[i] + 128;
				if (v < 0 || v > 255) return failed("constant of mixed sign with a varying d");
				K.v[i] = int16_t(v);
			}
			op(FN_ADD_SIGNED, {D, K});
			return finish();
		}
		if (offPos && offNeg) return failed("constant of mixed sign with a varying d");
		offHalf &= off[0] == 128 || off[0] == -128;
		TevSrc K = tsConst1(0);
		for (int i = 0; i < nc; i++) K.v[i] = int16_t(off[i] < 0 ? -off[i] : off[i]);
		const uint8_t addK = offPos ? FN_ADD : FN_SUBTRACT;
		if (lf == LZ) {
			op(addK, {D, K});
		} else if (offHalf && !sub && offNeg) {  // d + lerp - 1/2
			if (lf == LX) op(FN_ADD_SIGNED, {X, D});
			else {
				lerpOp();
				op(FN_ADD_SIGNED, {mid, D});
			}
		} else if (offHalf && sub && offPos) {  // d - lerp + 1/2 ~ d + (1 - lerp) - 1/2
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
		} else if (!sub && lf == LXY) {  // d + x * y, then the constant
			op(FN_MULTIPLY_ADD, {X, Y, D});
			op(addK, {mid, K});
		} else if (lf == LX) {  // d +/- x, then the constant
			op(sub ? FN_SUBTRACT : FN_ADD, sub ? std::initializer_list<TevSrc>{D, X} : std::initializer_list<TevSrc>{X, D});
			op(addK, {mid, K});
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

// The GX TEV state of the draw, decoded. Textures are (map, coordinate) pairs; PICA texture
// units are given to the pairs the translation samples (unitPair).
static const int kMaxPairs = 8;
struct TevSetup {
	uint32_t numStages;
	uint32_t colorEnv[16], alphaEnv[16], order[16], ksel[16], swap[4];
	int stagePair[16];  // texture pair each stage samples (-1: none)
	int nPairs;
	int pairMap[kMaxPairs], pairCoord[kMaxPairs];
	int rasRgbChan, rasAChan;  // colour channels the vertex colour's rgb and alpha carry (0, 1)
	bool rgbVarying[2], aVarying[2];  // channel colour / alpha differs per vertex (vertex colour or lighting)
	int16_t rasConst[2][4];  // otherwise: its colour (material register)
	int nUnits;              // set by the translation
	int unitPair[PICA_MAX_TEX];
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
	// Colour channels: per-vertex, or one material colour for the draw.
	for (int c = 0; c < 2; c++) {
		const uint32_t cctrl = g.xfReg[XFR_COLOR0CTRL + c], actrl = g.xfReg[XFR_ALPHA0CTRL + c], mat = g.xfReg[XFR_MAT0 + c];
		const bool hasClr = (s_bfmt & (VF_CLR0 << c)) != 0, hasNrm = (s_bfmt & VF_NRM) != 0;
		t.rgbVarying[c] = ((cctrl & 1) && hasClr) || (((cctrl >> 1) & 1) && hasNrm);
		t.aVarying[c] = ((actrl & 1) && hasClr) || (((actrl >> 1) & 1) && hasNrm);
		t.rasConst[c][0] = int16_t(mat >> 24);
		t.rasConst[c][1] = int16_t((mat >> 16) & 0xFF);
		t.rasConst[c][2] = int16_t((mat >> 8) & 0xFF);
		t.rasConst[c][3] = int16_t(mat & 0xFF);
	}
	bool needRgb[2] = {false, false}, needA[2] = {false, false};
	for (uint32_t s = 0; s < t.numStages; s++) {
		const uint32_t ord = t.order[s], map = ord & 7, coord = (ord >> 3) & 7, chan = (ord >> 7) & 7;
		t.stagePair[s] = -1;
		if ((ord >> 6) & 1) {
			int p = 0;
			while (p < t.nPairs && (t.pairMap[p] != int(map) || t.pairCoord[p] != int(coord))) p++;
			if (p == t.nPairs && p == kMaxPairs) {
				p = -1;  // more pairs than tracked: the stage reads white
			} else if (p == t.nPairs) {
				t.pairMap[p] = int(map);
				t.pairCoord[p] = int(coord);
				t.nPairs++;
			}
			t.stagePair[s] = p;
		}
		// Which parts (rgb, alpha) of which channel the stage reads, through its swap table.
		if (chan <= 1) {
			const uint32_t ce = t.colorEnv[s], ae = t.alphaEnv[s], sw = t.swap[ae & 3];
			auto part = [&](uint32_t comp) { (comp == 3 ? needA : needRgb)[chan] = true; };
			for (int i = 0; i < 4; i++) {
				uint32_t c = (ce >> (4 * i)) & 15, a = (ae >> (4 + 3 * i)) & 7;
				if (c == 10)
					for (int k = 0; k < 3; k++) part((sw >> (2 * k)) & 3);
				if (c == 11 || a == 5) part((sw >> 6) & 3);
			}
		}
	}
	// The vertex colour carries the rgb of a channel whose rgb is read (one that varies per vertex
	// first), and the alpha of one whose alpha is read.
	auto pick = [](const bool need[2], const bool varying[2]) {
		for (int c = 0; c < 2; c++)
			if (need[c] && varying[c]) return c;
		return need[1] && !need[0] ? 1 : 0;
	};
	t.rasRgbChan = pick(needRgb, t.rgbVarying);
	t.rasAChan = pick(needA, t.aVarying);
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

// A texture pair as a combiner source until it gets a texture unit.
enum : uint8_t { SRC_PAIR = 0x10 };

// Translates the TEV setup into st's combiner stages and gives texture units to the pairs it
// samples (t.nUnits, t.unitPair). pairWhite[p]: pair p has no texture (reads as white).
// Returns false (and logs) when the setup cannot be expressed.
static bool translateTev(TevSetup& t, const bool pairWhite[kMaxPairs], PicaState& st) {
	const uint32_t N = t.numStages;
	t.nUnits = 0;
	int unitOfPair[kMaxPairs];
	for (int i = 0; i < kMaxPairs; i++) unitOfPair[i] = -1;
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
	// The buffer's initial colour as a constant: set, its value, and whether a value was saved
	// over it.
	bool bufColorSet[2] = {false, false}, bufTouched[2] = {false, false};
	TevSrc bufColor[2];
	int n = 0;  // PICA stages emitted
	int curTex = -1;  // texture pair `tex` holds (-1: white), as genFS keeps it between stages
	static const int k8[8] = {255, 223, 191, 159, 127, 95, 63, 31};
	static const int nc[2] = {3, 1};

	for (uint32_t s = 0; s < N; s++) {
		const uint32_t ce = t.colorEnv[s], ae = t.alphaEnv[s], ord = t.order[s];
		const uint32_t chan = (ord >> 7) & 7;
		const uint32_t rasSwap = t.swap[ae & 3], texSwap = t.swap[(ae >> 2) & 3];
		if (t.stagePair[s] >= 0) curTex = pairWhite[t.stagePair[s]] ? -1 : t.stagePair[s];
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
			return swapped(uint8_t(SRC_PAIR + curTex), texSwap, comp);
		};
		auto rasIn = [&](int comp) -> TevSrc {
			if (chan > 1) {
				if (chan != 7) softFail = "bump alpha (indirect) colour";
				return tsConst1(0);
			}
			// The components read; each comes from the vertex colour (when it carries that part of
			// this channel) or is constant for the draw.
			const int sr = rasSwap & 3, sg = (rasSwap >> 2) & 3, sb = (rasSwap >> 4) & 3, sa = (rasSwap >> 6) & 3;
			const int* cs = comp == 3 ? &sa : nullptr;
			const int rgbc[3] = {sr, sg, sb};
			bool fromVtx = false, isConst = true;
			for (int k = 0; k < (comp == 3 ? 1 : 3); k++) {
				const int c = cs ? *cs : rgbc[k];
				const bool alphaPart = c == 3;
				const bool carried = alphaPart ? int(chan) == t.rasAChan : int(chan) == t.rasRgbChan;
				const bool varying = alphaPart ? t.aVarying[chan] : t.rgbVarying[chan];
				if (carried) fromVtx = true;
				if (carried || varying) isConst = false;
			}
			if (isConst) {
				const int16_t* c = t.rasConst[chan];
				return comp == 3 ? tsConst1(c[sa]) : tsConst(c[sr], c[sg], c[sb]);
			}
			if (!fromVtx) softFail = "stages read both colour channels";
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
			if (plan[ch].softCompare) tevLogOnce("compare drawn as a ramp", s, N, ce, ae);
		}
		if (softFail) tevLogOnce(softFail, s, N, ce, ae);

		// An operation with two different constants reads one from the combiner buffer's initial
		// colour while the buffer holds it (from PICA stage 1, until a value is saved there), or
		// else loads one into PREVIOUS first: a one-op channel becomes two.
		bool noSave[2] = {false, false};
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
				if (nk == 2 && n >= 1 && !bufTouched[ch] &&
				    (!bufColorSet[ch] || tsSame(bufColor[ch], ks[0], nc[ch]))) {
					TevSrc b = tsSrc(SRC_BUFFER, uint8_t(ch == CH_RGB ? 4 : 3));
					for (int j = 0; j < p.op[k].n; j++)
						if (p.op[k].s[j].k && tsSame(p.op[k].s[j], ks[0], nc[ch])) p.op[k].s[j] = b;
					bufColor[ch] = ks[0];
					bufColorSet[ch] = true;
					noSave[ch] = true;
					continue;
				}
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
		// A register copied onto itself (it is in PREVIOUS already) needs no PICA stage.
		bool noop[2] = {false, false};
		for (int ch = 0; ch < 2; ch++) {
			const ChanPlan& p = plan[ch];
			const TevSrc& x = p.op[0].s[0];
			noop[ch] = !p.folded && p.nops == 1 && p.op[0].func == FN_REPLACE && p.op[0].scale == 0 && !x.k && !x.mid &&
			           !x.inv && x.src == SRC_PREVIOUS && x.comp == (ch == CH_RGB ? 4 : 3) && prevOcc[ch] == dest[ch];
			if (noop[ch]) plan[ch].nops = 0;
		}
		const bool computes[2] = {!plan[CH_RGB].folded && !noop[CH_RGB], !plan[CH_A].folded && !noop[CH_A]};
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
				if (n < 1 || n - 1 > 3 || !bufFree || noSave[ch]) {
					tevLogOnce("no room to keep a register value", s, N, ce, ae);
					return false;
				}
				if (b >= 0) loc[b][ch].kind = LOC_LOST;
				if (ch == CH_RGB) st.buf_rgb |= uint8_t(1u << (n - 1));
				else st.buf_alpha |= uint8_t(1u << (n - 1));
				loc[r][ch].kind = LOC_BUF;
				bufOcc[ch] = r;
				saved[ch] = true;
				bufTouched[ch] = true;
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
						uint32_t src = x.k ? SRC_CONSTANT : x.src;
						if (src >= SRC_PAIR) {  // a texture: its unit, given on first use
							const int pr = int(src - SRC_PAIR);
							if (unitOfPair[pr] < 0) {
								if (t.nUnits == PICA_MAX_TEX) {
									tevLogOnce("more than 3 textures sampled", s, N, ce, ae);
									return false;
								}
								unitOfPair[pr] = t.nUnits;
								t.unitPair[t.nUnits++] = pr;
							}
							src = SRC_TEX0 + uint32_t(unitOfPair[pr]);
						}
						srcs |= src << (4 * j);
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
			if (noop[ch]) continue;
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
	// A constant output goes into the last stage when that stage passes its channel through.
	if (n > 0) {
		PicaTevStage& last = st.stage[n - 1];
		for (int ch = 0; ch < 2; ch++) {
			const TevLoc& l = loc[outReg[ch]][ch];
			if (prevOcc[ch] == outReg[ch] || l.kind != LOC_CONST) continue;
			uint16_t& src = ch == CH_RGB ? last.src_rgb : last.src_alpha;
			const uint16_t opv = ch == CH_RGB ? last.op_rgb : last.op_alpha;
			const uint8_t fn = ch == CH_RGB ? last.func_rgb : last.func_alpha, sc = ch == CH_RGB ? last.scale_rgb : last.scale_alpha;
			if (src != SRC_PREVIOUS || opv != 0 || fn != FN_REPLACE || sc != 0) continue;
			src = SRC_CONSTANT;
			if (ch == CH_RGB)
				last.color = (last.color & 0xFF000000u) | uint32_t(clampByte(l.v[0])) | uint32_t(clampByte(l.v[1])) << 8 |
				             uint32_t(clampByte(l.v[2])) << 16;
			else
				last.color = (last.color & 0x00FFFFFFu) | uint32_t(clampByte(l.v[0])) << 24;
			prevOcc[ch] = outReg[ch];
		}
	}
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
	if (bufColorSet[CH_RGB])
		st.buf_color |= uint32_t(clampByte(bufColor[CH_RGB].v[0])) | uint32_t(clampByte(bufColor[CH_RGB].v[1])) << 8 |
		                uint32_t(clampByte(bufColor[CH_RGB].v[2])) << 16;
	if (bufColorSet[CH_A]) st.buf_color |= uint32_t(clampByte(bufColor[CH_A].v[0])) << 24;
	st.nstages = uint8_t(n);
	return true;
}

// THP movies (THPPlayer's THPGXYuv2RgbSetup) convert Y, U, V textures to RGB through signed,
// unclamped TEV intermediates, which PICA stages (clamped to 0..1) cannot hold. Per component,
// the setup computes out = y + off + cu * u + cv * v (off, cu, cv from C0 and konst 0-1; green
// through the alpha channel); here d = off + cu * u + cv * v is built as 0.5 + d / 2 from two
// interpolations whose endpoints stay in range, and applied as two halves with ADD_SIGNED
// (same-sign steps, so the clamp in between is harmless).
static bool isThpSetup(const TevSetup& t) {
	return t.numStages == 4 && (t.colorEnv[0] & 0xFFFF) == 0xF8E2 && (t.colorEnv[1] & 0xFFFF) == 0xF8E0 &&
	       (t.colorEnv[2] & 0xFFFF) == 0xF8C0 && (t.colorEnv[3] & 0xFFFF) == 0x10EF && t.stagePair[0] >= 0 &&
	       t.stagePair[1] >= 0 && t.stagePair[2] >= 0 && t.stagePair[0] != t.stagePair[1] &&
	       t.stagePair[1] != t.stagePair[2] && t.stagePair[0] != t.stagePair[2];
}

static void translateThp(TevSetup& t, PicaState& st) {
	// Units 0-2: U, V, Y (the textures of stages 0-2).
	t.nUnits = 3;
	for (int u = 0; u < 3; u++) t.unitPair[u] = t.stagePair[u];
	auto kc = [](uint32_t i, int comp) {  // konst i, comp 0-3: r, g, b, a (0..1)
		const uint32_t ra = g.kreg[2 * i], bg = g.kreg[2 * i + 1];
		return float(comp == 0 ? ra & 0xFF : comp == 1 ? (bg >> 12) & 0xFF : comp == 2 ? bg & 0xFF : (ra >> 12) & 0xFF) /
		       255.f;
	};
	const uint32_t ra = g.bp[BP_TEV_REG + 2], bg = g.bp[BP_TEV_REG + 3];  // C0
	const float c0r = sext11(ra & 0x7FF) / 255.f, c0b = sext11(bg & 0x7FF) / 255.f, a0 = sext11((ra >> 12) & 0x7FF) / 255.f;
	const float off[3] = {2 * c0r, a0, 2 * c0b};
	const float cu[3] = {2 * kc(0, 0), -kc(0, 3), 2 * kc(0, 2)};
	const float cv[3] = {2 * kc(1, 0), -kc(1, 3), 2 * kc(1, 2)};
	uint8_t duLo[3], duHi[3], evLo[3], evHi[3];
	auto byte = [](float x) { return clampByte(int(x * 255.f + 0.5f)); };
	for (int i = 0; i < 3; i++) {
		const float beta = off[i] / 2 + 0.25f * cu[i];
		duLo[i] = byte(0.5f - 0.25f * cu[i]);
		duHi[i] = byte(0.5f + 0.25f * cu[i]);
		evLo[i] = byte(0.5f + beta);
		evHi[i] = byte(0.5f + beta + 0.5f * cv[i]);
	}
	auto rgb = [](const uint8_t c[3]) { return uint32_t(c[0]) | uint32_t(c[1]) << 8 | uint32_t(c[2]) << 16; };
	const uint8_t texU = SRC_TEX0, texV = SRC_TEX0 + 1, texY = SRC_TEX0 + 2;
	auto stage = [&](int i, uint8_t func, uint8_t s0, uint8_t s1, uint8_t s2, uint32_t color) {
		PicaTevStage& ps = st.stage[i];
		memset(&ps, 0, sizeof ps);
		ps.src_rgb = uint16_t(s0 | s1 << 4 | s2 << 8);
		ps.func_rgb = func;
		ps.src_alpha = SRC_CONSTANT;  // the setup's alpha is 0
		ps.func_alpha = FN_REPLACE;
		ps.color = color;
	};
	// 0-1: E = interpolate(evLo, evHi, v), kept in the buffer (from stage 3).
	stage(0, FN_REPLACE, SRC_CONSTANT, 0, 0, rgb(evLo));
	stage(1, FN_INTERPOLATE, SRC_CONSTANT, SRC_PREVIOUS, texV, rgb(evHi));
	// 2: D_u = interpolate(duLo, duHi, u), duLo from the buffer's initial colour.
	stage(2, FN_INTERPOLATE, SRC_CONSTANT, SRC_BUFFER, texU, rgb(duHi));
	// 3: 0.5 + d / 2 = D_u + E - 0.5, kept in the buffer (from stage 5).
	stage(3, FN_ADD_SIGNED, SRC_PREVIOUS, SRC_BUFFER, 0, 0);
	// 4-5: y + d / 2, then + d / 2.
	stage(4, FN_ADD_SIGNED, texY, SRC_PREVIOUS, 0, 0);
	stage(5, FN_ADD_SIGNED, SRC_PREVIOUS, SRC_BUFFER, 0, 0);
	st.nstages = 6;
	st.buf_rgb = uint8_t(1u << 1 | 1u << 3);
	st.buf_alpha = 0;
	st.buf_color = rgb(duLo);
}

// Logs each distinct TEV setup once (sdmc log), with its textures and translation, to find
// what the game draws with.
// Returns the setup's number in the dump (0: not numbered).
static unsigned tevDump(const TevSetup& t, const PicaState& st, bool ok, PicaTex* const ptex[kMaxPairs],
                        const EfbCopy* const pcopy[kMaxPairs], const char* extra) {
	static std::vector<uint64_t> seen;
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](uint64_t x) { h = (h ^ x) * 1099511628211ull; };
	mix(t.numStages);
	for (uint32_t s = 0; s < t.numStages; s++) {
		mix(t.colorEnv[s]);
		mix(t.alphaEnv[s]);
		mix(t.order[s]);
		mix(t.ksel[s]);
	}
	for (int i = 0; i < 4; i++) mix(t.swap[i]);
	const uint32_t numInd = (g.bp[BP_GENMODE] >> 16) & 7;
	mix(numInd);
	mix(ok);
	mix(st.nstages);
	for (int p = 0; p < t.nPairs; p++)
		mix((pcopy[p] ? 2 : ptex[p] ? 1 : 0) | (g.bp[bpTexReg(BP_TX_IMAGE0, t.pairMap[p])] >> 20 & 15) << 2);
	for (size_t i = 0; i < seen.size(); i++)
		if (seen[i] == h) return unsigned(i + 1);
	if (seen.size() >= 400) return 0;
	seen.push_back(h);
	fprintf(stderr, "[tevdump] #%u frame %u: %u stages, %u indirect, %s (%u PICA stages, buf %x/%x %08x)\n",
	        unsigned(seen.size()), g_displayFrames, t.numStages, numInd, ok ? "translated" : "FALLBACK", st.nstages,
	        st.buf_rgb, st.buf_alpha, (unsigned)st.buf_color);
	for (uint32_t s = 0; s < t.numStages; s++)
		fprintf(stderr, "[tevdump]   gx %u: ce %06x ae %06x order %03x ksel %03x\n", s, t.colorEnv[s], t.alphaEnv[s],
		        t.order[s], t.ksel[s]);
	for (int p = 0; p < t.nPairs; p++) {
		const uint32_t img0 = g.bp[bpTexReg(BP_TX_IMAGE0, t.pairMap[p])];
		int unit = -1;
		for (int u = 0; u < t.nUnits; u++)
			if (t.unitPair[u] == p) unit = u;
		fprintf(stderr, "[tevdump]   texture %d: map %d coord %d fmt %u %ux%u%s, unit %d\n", p, t.pairMap[p], t.pairCoord[p],
		        (img0 >> 20) & 15, (img0 & 0x3FF) + 1, ((img0 >> 10) & 0x3FF) + 1,
		        pcopy[p] ? " EFB copy" : ptex[p] ? "" : " MISSING", unit);
	}
	fprintf(stderr, "[tevdump]   regs");
	for (int r = 0; r < 4; r++)
		fprintf(stderr, " %06x:%06x", g.bp[BP_TEV_REG + 2 * r] & 0xFFFFFF, g.bp[BP_TEV_REG + 2 * r + 1] & 0xFFFFFF);
	fprintf(stderr, " konst");
	for (int r = 0; r < 4; r++) fprintf(stderr, " %06x:%06x", g.kreg[2 * r] & 0xFFFFFF, g.kreg[2 * r + 1] & 0xFFFFFF);
	fprintf(stderr, " swap %02x %02x %02x %02x ras rgb/a %02d (varying %d %d)\n", t.swap[0], t.swap[1], t.swap[2], t.swap[3],
	        t.rasRgbChan * 10 + t.rasAChan, t.rgbVarying[0] | t.aVarying[0] << 1, t.rgbVarying[1] | t.aVarying[1] << 1);
	fprintf(stderr, "[tevdump]   %s\n", extra);
	for (int i = 0; i < st.nstages; i++) {
		const PicaTevStage& p = st.stage[i];
		fprintf(stderr, "[tevdump]   pica %d: src %03x/%03x op %03x/%03x fn %u/%u scale %u/%u const %08x\n", i, p.src_rgb,
		        p.src_alpha, p.op_rgb, p.op_alpha, p.func_rgb, p.func_alpha, p.scale_rgb, p.scale_alpha,
		        (unsigned)p.color);
	}
	return unsigned(seen.size());
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

// Clears the combiner stages a translation filled in.
static void resetTev(PicaState& st, TevSetup& t) {
	memset(st.stage, 0, sizeof st.stage);
	st.buf_rgb = st.buf_alpha = 0;
	st.buf_color = 0;
	st.nstages = 0;
	t.nUnits = 0;
}

// TEV translation cache. readTev/translateTev read only the registers hashed here (TEV envs,
// orders, konst selection and swap tables, colour registers and konst colours, channel controls
// and material colours), the vertex format, which texture pairs have a texture, and the PicaState
// built before them (blend, depth, alpha test, scissor). Draws repeat the same few setups every
// frame, so translating each setup once saves most of the per-draw TEV cost.
struct TevCacheEntry {
	TevSetup tev;
	PicaState st;
	bool translated;
	unsigned dumpId;
};
static std::unordered_map<uint64_t, TevCacheEntry> s_tevCache;
static const size_t kTevCacheMax = 1024;

static uint64_t tevCacheKey(const bool pwhite[kMaxPairs], int nPairs, const PicaState& pre, bool debug) {
	uint64_t h = 1469598103934665603ull;
	auto mix = [&](uint64_t x) { h = (h ^ x) * 1099511628211ull; };
	mix(g.bp[BP_GENMODE]);
	for (int i = 0; i < 32; i++) mix(g.bp[BP_TEV_COLOR_ENV + i]);
	for (int i = 0; i < 8; i++) mix(g.bp[BP_TREF + i]);
	for (int i = 0; i < 8; i++) mix(g.bp[BP_TEV_KSEL + i]);
	for (int i = 0; i < 8; i++) mix(g.bp[BP_TEV_REG + i]);
	for (int i = 0; i < 8; i++) mix(g.kreg[i]);
	for (int c = 0; c < 2; c++) {
		mix(g.xfReg[XFR_COLOR0CTRL + c]);
		mix(g.xfReg[XFR_ALPHA0CTRL + c]);
		mix(g.xfReg[XFR_MAT0 + c]);
	}
	mix(s_bfmt);
	uint32_t white = 0;
	for (int p = 0; p < nPairs; p++) white |= uint32_t(pwhite[p]) << p;
	mix(white | uint32_t(nPairs) << 16 | uint32_t(debug) << 24);
	const uint8_t* b = reinterpret_cast<const uint8_t*>(&pre);
	for (size_t i = 0; i < sizeof pre; i += 8) {
		uint64_t w = 0;
		memcpy(&w, b + i, sizeof pre - i < 8 ? sizeof pre - i : 8);
		mix(w);
	}
	return h;
}

static void drawTriangles() {
	if (((g.bp[BP_GENMODE] >> 14) & 3) == 3) return;  // every face culled
	if (s_afterTexCopy) s_drawsAfterCopy++;
	else s_drawsBeforeCopy++;
	const VtxFmtLayout& L = vtxFmtLayout(s_bfmt);
	uint32_t n = s_bcount, nidx = uint32_t(s_idx.size());

	s_perfBatches++;
	s_perfVerts += n;
	// The textures the TEV stages sample, per (map, coordinate) pair, and the combiner stages,
	// which give PICA texture units to the pairs they use (none in debug-colour mode).
	const bool debug = s_debugColors;
	uint64_t perfTev0 = svcGetSystemTick();
	TevSetup tev;
	readTev(tev);
	s_perfAcc[PERF_TEV] += svcGetSystemTick() - perfTev0;
	PicaTex* ptex[kMaxPairs] = {};
	const EfbCopy* pcopy[kMaxPairs] = {};
	bool pwhite[kMaxPairs];
	for (int p = 0; p < kMaxPairs; p++) pwhite[p] = true;
	if (!debug) {
		PerfScope ps(PERF_TEX);
		for (int p = 0; p < tev.nPairs; p++) {
			ptex[p] = textureForMap(tev.pairMap[p], &pcopy[p]);
			pwhite[p] = ptex[p] == nullptr;
		}
	}
	perfTev0 = svcGetSystemTick();

	PicaState st;
	memset(&st, 0, sizeof st);  // the PICA side compares states bytewise
	currentPicaState(st);
	alphaTest(st);
	bool translated = true;
	unsigned dumpId = 0;
	const uint64_t tevKey = tevCacheKey(pwhite, tev.nPairs, st, debug);
	auto tevHit = s_tevCache.find(tevKey);
	const bool cached = tevHit != s_tevCache.end();
	if (cached) {
		tev = tevHit->second.tev;
		st = tevHit->second.st;
		translated = tevHit->second.translated;
		dumpId = tevHit->second.dumpId;
	}
	if (cached) {
	} else if (debug) {
		tev.nUnits = 0;
	} else if (isThpSetup(tev) && ptex[tev.stagePair[0]] && ptex[tev.stagePair[1]] && ptex[tev.stagePair[2]]) {
		translateThp(tev, st);
	} else if (!(translated = translateTev(tev, pwhite, st))) {
		// The first k stages in full, the rest passing the colour through (keeping their alpha,
		// then not), for the largest k that can be expressed.
		const TevSetup full = tev;
		for (int k = int(full.numStages) - 1; k >= 1 && !translated; k--) {
			for (int keepAlpha = 1; keepAlpha >= 0 && !translated; keepAlpha--) {
				tev = full;
				// The registers the k-th stage writes, passed on into PREV.
				const uint32_t keepC = (full.colorEnv[k - 1] >> 22) & 3, keepA = (full.alphaEnv[k - 1] >> 22) & 3;
				for (uint32_t s = uint32_t(k); s < full.numStages; s++) {
					tev.colorEnv[s] = 0x08FFF0 | 2 * keepC;
					if (!keepAlpha) tev.alphaEnv[s] = 0x08FF80 | keepA << 4;
				}
				resetTev(st, tev);
				if ((translated = translateTev(tev, pwhite, st)))
					tevLogOnce(keepAlpha ? "drawn with the colour of its first stages" : "drawn with its first stages only",
					           uint32_t(k), full.numStages, full.colorEnv[0], full.alphaEnv[0]);
			}
		}
		if (translated) {
			memcpy(tev.colorEnv, full.colorEnv, sizeof tev.colorEnv);  // as the game set them, for the dump
			memcpy(tev.alphaEnv, full.alphaEnv, sizeof tev.alphaEnv);
		}
	}
	if (!cached && !debug && !translated) {
		resetTev(st, tev);
		for (uint32_t s = 0; s < tev.numStages; s++) {
			const int pr = tev.stagePair[s];
			if (pr < 0 || !ptex[pr]) continue;
			PicaTevStage& ps = st.stage[0];
			const bool rasNull = ((tev.order[s] >> 7) & 7) == 7;
			ps.src_rgb = ps.src_alpha = uint16_t(SRC_TEX0) | (rasNull ? 0 : SRC_PRIMARY << 4);
			ps.func_rgb = ps.func_alpha = rasNull ? FN_REPLACE : FN_MODULATE;
			st.nstages = 1;
			tev.nUnits = 1;
			tev.unitPair[0] = pr;
			break;
		}
	}
	if (!cached) {
		if (s_tevCache.size() >= kTevCacheMax) s_tevCache.clear();
		TevCacheEntry& e = s_tevCache[tevKey];
		e.tev = tev;
		e.st = st;
		e.translated = translated;
		e.dumpId = 0;
	}
	static const uint8_t kWrap[4] = {0, 2, 3, 2};  // clamp, repeat, mirror -> GPU_TEXTURE_WRAP_PARAM
	PicaTex* utex[PICA_MAX_TEX] = {};
	const EfbCopy* ucopy[PICA_MAX_TEX] = {};
	int ucoord[PICA_MAX_TEX] = {};
	for (int u = 0; u < tev.nUnits; u++) {
		const int pr = tev.unitPair[u];
		utex[u] = ptex[pr];
		ucopy[u] = pcopy[pr];
		ucoord[u] = tev.pairCoord[pr];
		if (!utex[u]) continue;
		const uint32_t mode0 = g.bp[bpTexReg(BP_TX_MODE0, tev.pairMap[pr])];
		st.tex[u] = utex[u];
		st.wrap_s[u] = kWrap[mode0 & 3];
		st.wrap_t[u] = kWrap[(mode0 >> 2) & 3];
		st.linear[u] = (mode0 >> 4) & 1;
	}
	if (!debug && !cached) {
		// Blend (BP 0x41), depth (0x40), alpha compare (0xF3), channel controls and the first
		// vertex's colours, for the dump.
		const VtxFmtLayout& L0 = vtxFmtLayout(s_bfmt);
		uint32_t c0 = 0, c1 = 0;
		if (s_bcount && (s_bfmt & VF_CLR0)) memcpy(&c0, s_vdata.data() + L0.clr[0], 4);
		if (s_bcount && (s_bfmt & (VF_CLR0 << 1))) memcpy(&c1, s_vdata.data() + L0.clr[1], 4);
		char extra[200];
		snprintf(extra, sizeof extra,
		         "clip %s cmode0 %06x zmode %06x alphacmp %06x chans %u ctrl %04x/%04x %04x/%04x mat %08x %08x vtx clr %08x %08x (fmt %x)",
		         g.xfReg[0x05] ? "off" : "on", g.bp[BP_CMODE0] & 0xFFFFFF, g.bp[BP_ZMODE] & 0xFFFFFF, g.bp[BP_ALPHACOMPARE] & 0xFFFFFF,
		         g.xfReg[XFR_NUMCHANS] & 3, g.xfReg[XFR_COLOR0CTRL] & 0x7FFF, g.xfReg[XFR_ALPHA0CTRL] & 0x7FFF,
		         g.xfReg[XFR_COLOR0CTRL + 1] & 0x7FFF, g.xfReg[XFR_ALPHA0CTRL + 1] & 0x7FFF, g.xfReg[XFR_MAT0],
		         g.xfReg[XFR_MAT0 + 1], __builtin_bswap32(c0), __builtin_bswap32(c1), s_bfmt);
		dumpId = tevDump(tev, st, translated, ptex, pcopy, extra);
		s_tevCache[tevKey].dumpId = dumpId;
	}
	if (s_opaque) {
		st.blend = 0;
		st.alpha_test = 0;
		st.alpha_func = 1;  // GPU_ALWAYS
	}

	s_perfAcc[PERF_TEV] += svcGetSystemTick() - perfTev0;
	PicaVtx* out;
	uint16_t* oidx;
	if (!pica_alloc(n, nidx, &out, &oidx)) return;
	const uint64_t perfXform0 = svcGetSystemTick();

	// Projection (GXSetProjection) and viewport (GXSetViewport), as the GX shader reads them.
	const float p0 = xff(XFR_PROJ), p1 = xff(XFR_PROJ + 1), p2 = xff(XFR_PROJ + 2), p3 = xff(XFR_PROJ + 3),
	            p4 = xff(XFR_PROJ + 4), p5 = xff(XFR_PROJ + 5);
	const bool ortho = g.xfReg[XFR_PROJ + 6] & 1;
	const float vsx = xff(XFR_VIEWPORT), vsy = xff(XFR_VIEWPORT + 1), vsz = xff(XFR_VIEWPORT + 2);
	const float vcx = xff(XFR_VIEWPORT + 3) - 342.0f, vcy = xff(XFR_VIEWPORT + 4) - 342.0f, vcz = xff(XFR_VIEWPORT + 5);
	const float kInvZ = 1.0f / 16777215.0f;
	const bool noClip = g.xfReg[0x05] != 0;

	// The colour channels the TEV stages read (0 or 1) for the rgb and the alpha: material
	// register or vertex colour; lighting reduced to one fixed light.
	const int ch = debug ? 0 : tev.rasRgbChan, cha = debug ? 0 : tev.rasAChan;
	const uint32_t cctrl = g.xfReg[XFR_COLOR0CTRL + ch], actrl = g.xfReg[XFR_ALPHA0CTRL + cha], mat = g.xfReg[XFR_MAT0 + ch],
	               amat = g.xfReg[XFR_MAT0 + cha];
	const bool hasClr = (s_bfmt & (VF_CLR0 << ch)) != 0, hasAClr = (s_bfmt & (VF_CLR0 << cha)) != 0,
	           hasNrm = (s_bfmt & VF_NRM) != 0, hasMtx = (s_bfmt & VF_MTX) != 0;
	const bool clrFromVtx = (cctrl & 1) && hasClr, alphaFromVtx = (actrl & 1) && hasAClr;
	const bool lit = ((cctrl >> 1) & 1) && hasNrm;
	const uint8_t matRGBA[4] = {uint8_t(mat >> 24), uint8_t(mat >> 16), uint8_t(mat >> 8), uint8_t(amat)};
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
		// GX clipping off (XF 0x1005): nothing is clipped by depth, it is clamped. The PICA always
		// clips, so keep the depth inside its range (skies sit on or past the far plane).
		if (noClip && cw > 0.f) o.z = o.z < 0.f ? 0.f : o.z > cw ? cw : o.z;
		o.w = cw;

		for (int u = 0; u < PICA_MAX_TEX; u++) {
			float* tc = o.tc[u];
			if (!utex[u]) {
				tc[0] = tc[1] = 0.f;
				continue;
			}
			texCoord(ucoord[u], v, L, s_bfmt, defTex, tc[0], tc[1]);
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
		o.rgba[3] = alphaFromVtx ? v[L.clr[cha] + 3] : matRGBA[3];
	}
	for (uint32_t i = 0; i < nidx; i++) oidx[i] = uint16_t(s_idx[i]);

	if (s_tracing) {
		float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f, z0 = 1e9f, z1 = -1e9f, w0 = 1e9f, w1 = -1e9f;
		unsigned nan = 0;
		for (uint32_t i = 0; i < n; i++) {
			const PicaVtx& o = out[i];
			if (o.w != o.w || o.x != o.x || o.y != o.y) nan++;
			w0 = o.w < w0 ? o.w : w0, w1 = o.w > w1 ? o.w : w1;
			if (o.w <= 0.f) continue;
			const float x = o.x / o.w, y = o.y / o.w, z = o.z / o.w;
			x0 = x < x0 ? x : x0, x1 = x > x1 ? x : x1, y0 = y < y0 ? y : y0, y1 = y > y1 ? y : y1;
			z0 = z < z0 ? z : z0, z1 = z > z1 ? z : z1;
		}
		fprintf(stderr,
		        "[trace] draw %u tev#%u v%u x %.0f..%.0f y %.0f..%.0f z %.4f..%.4f blend %u eq %u/%u f %u,%u/%u,%u bc %08lx "
		        "mask %x depth %u f%u w%u sc %u %u,%u-%u,%u atest %u f%u r%u stages %u tr %d units %d copy %d v0 %02x%02x%02x%02x\n",
		        s_traceDraw++, dumpId, n, x0, x1, y0, y1, z0, z1, st.blend, st.blend_eq, st.blend_aeq, st.blend_src, st.blend_dst,
		        st.blend_asrc, st.blend_adst, (unsigned long)st.blend_color, st.write_mask, st.depth_test, st.depth_func,
		        st.depth_write, st.scissor, st.sc_x0, st.sc_y0, st.sc_x1, st.sc_y1, st.alpha_test, st.alpha_func, st.alpha_ref,
		        st.nstages, translated, tev.nUnits, ucopy[0] != nullptr, n ? out[0].rgba[0] : 0, n ? out[0].rgba[1] : 0,
		        n ? out[0].rgba[2] : 0, n ? out[0].rgba[3] : 0);
		const float* m0 = xfMtx((hasMtx ? s_vdata[L.mtx] : defPm) * 4);
		float p0f[3] = {0, 0, 0};
		if (n) memcpy(p0f, s_vdata.data(), 12);
		for (int u = 0; u < tev.nUnits; u++) {
			auto it = utex[u] ? s_texAvg.find(utex[u]) : s_texAvg.end();
			fprintf(stderr, "[trace]   unit %d: map %d avg %08lx tc0 %g %g%s\n", u, tev.pairMap[tev.unitPair[u]],
			        (unsigned long)(it != s_texAvg.end() ? it->second : 0), n ? out[0].tc[u][0] : 0.f,
			        n ? out[0].tc[u][1] : 0.f, ucopy[u] ? " (EFB copy)" : utex[u] ? "" : " (none)");
		}
		fprintf(stderr, "[trace]   w %g..%g nan %u v0 %g %g %g %g pos %g %g %g pm %u m %g %g %g %g / %g %g %g %g / %g %g %g %g proj %g %g %g %g %g %g %s vp %g %g %g %g %g %g fmt %x\n",
		        w0, w1, nan, n ? out[0].x : 0.f, n ? out[0].y : 0.f, n ? out[0].z : 0.f, n ? out[0].w : 0.f, p0f[0], p0f[1], p0f[2],
		        hasMtx ? s_vdata[L.mtx] : defPm, m0[0], m0[1], m0[2], m0[3], m0[4], m0[5], m0[6], m0[7], m0[8], m0[9], m0[10],
		        m0[11], p0, p1, p2, p3, p4, p5, ortho ? "ortho" : "persp", vsx, vsy, vsz, vcx, vcy, vcz, s_bfmt);
	}
	s_perfAcc[PERF_XFORM] += svcGetSystemTick() - perfXform0;
	{
		PerfScope ps(PERF_SUBMIT);
		pica_draw_triangles(out, oidx, nidx, &st);
	}
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
	if (s_tracing) {
		const uint32_t tl = g.bp[BP_COPY_SRC_TL], wh = g.bp[BP_COPY_SRC_WH];
		fprintf(stderr, "[trace] copy %s rect %u,%u %ux%u fmt %u clear %u (AR %04x GB %04x Z %06x) dest %p\n",
		        (ctrl >> 14) & 1 ? "DISPLAY" : "texture", tl & 0x3FF, (tl >> 10) & 0x3FF, (wh & 0x3FF) + 1,
		        ((wh >> 10) & 0x3FF) + 1, ((ctrl >> 3) & 1) << 3 | ((ctrl >> 4) & 7), (ctrl >> 11) & 1,
		        g.bp[BP_CLEAR_AR] & 0xFFFF, g.bp[BP_CLEAR_GB] & 0xFFFF, g.bp[BP_CLEAR_Z] & 0xFFFFFF, g.copyDest);
	}
	if ((ctrl >> 14) & 1) {  // a display copy ends the traced frame, or starts the armed one
		if (s_tracing) {
			s_tracing = false;
			fprintf(stderr, "[trace] end\n");
		} else if (s_traceArm) {
			s_traceArm = false;
			s_tracing = true;
			s_traceDraw = 0;
			fprintf(stderr, "[trace] begin frame %u\n", g_displayFrames + 1);
		}
	}
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
			{
				PerfScope ps(PERF_COPY);
				pica_efb_copy_to(c->tex, pix != 1 || cfmt == 4);
			}
			logCopyKind(c, ctrl, pix);
		}
	}
	if (clear || disp) {
		// BP 0x4F: A<<8|R, 0x50: G<<8|B, 0x51: 24-bit Z.
		uint32_t ar = g.bp[BP_CLEAR_AR], gb = g.bp[BP_CLEAR_GB];
		uint32_t rgba = ((ar & 0xFF) << 24) | (((gb >> 8) & 0xFF) << 16) | ((gb & 0xFF) << 8) | ((ar >> 8) & 0xFF);
		if (disp) {
			{
				PerfScope ps(PERF_PRESENT);
				pica_present();
			}
			perfFrameEnd();
		}
		// The display copy's clear applies to the next frame.
		if (clear && disp) {
			pica_clear(rgba, g.bp[BP_CLEAR_Z] & 0xFFFFFF);
		} else if (clear) {  // a texture copy clears its source rectangle only
			const uint32_t tl = g.bp[BP_COPY_SRC_TL], wh = g.bp[BP_COPY_SRC_WH];
			pica_clear_rect(float(tl & 0x3FF), float((tl >> 10) & 0x3FF), float((wh & 0x3FF) + 1),
			                float(((wh >> 10) & 0x3FF) + 1), rgba, g.bp[BP_CLEAR_Z] & 0xFFFFFF);
		}
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

// 0: material, 1: debug colours, 2: opaque (material colours, no blending or alpha test).
void GX3DS_TraceNextFrame(void) { gx::s_traceArm = true; }

// os_core.cpp's cache operations (DCFlushRange, DCStoreRange, ...) report CPU writes here (a weak
// reference sms-pc-port's GL renderer defines): stamp the pages, so the texture cache rehashes
// only what the game rewrote.
void GXPC_InvalidateRange(const void* p, uint32_t n) { gx::memoryWritten(p, n); }

// Averages over the last profiler window, in ms: tex, tev, xform, submit, copy, present, frame,
// other; then batches and vertices per frame.
void GX3DS_GetPerf(float out[10]) {
	for (int i = 0; i < gx::PERF_COUNT + 2; i++) out[i] = gx::s_perfAvgMs[i];
	out[gx::PERF_COUNT + 2] = gx::s_perfAvgBatches;
	out[gx::PERF_COUNT + 3] = gx::s_perfAvgVerts;
}

void GX3DS_SetDebugColors(int mode) {
	gx::s_debugColors = mode == 1;
	gx::s_opaque = mode == 2;
}
int GX3DS_GetDebugColors(void) { return gx::s_debugColors ? 1 : gx::s_opaque ? 2 : 0; }

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
