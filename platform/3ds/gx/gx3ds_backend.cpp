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
// In debug mode a frame is drawn only up to its first EFB-to-texture copy, whose clear is
// ignored: post effects (the title's shimmer mesh) redraw that copy over the whole screen, and
// untextured they would hide the scene. Counters for the status screen.
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

static void drawTriangles() {
	if (s_afterTexCopy) s_drawsAfterCopy++;
	else s_drawsBeforeCopy++;
	if (s_debugColors && s_afterTexCopy) return;
	const VtxFmtLayout& L = vtxFmtLayout(s_bfmt);
	uint32_t n = s_bcount, nidx = uint32_t(s_idx.size());
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

		const uint8_t* src = debug ? dbg : clrFromVtx ? v + L.clr[0] : matRGBA;
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
		o.rgba[3] = alphaFromVtx ? v[L.clr[0] + 3] : matRGBA[3];
	}
	for (uint32_t i = 0; i < nidx; i++) oidx[i] = uint16_t(s_idx[i]);

	PicaState st;
	currentPicaState(st);
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
	}
	if (clear || disp) {
		// BP 0x4F: A<<8|R, 0x50: G<<8|B, 0x51: 24-bit Z.
		uint32_t ar = g.bp[BP_CLEAR_AR], gb = g.bp[BP_CLEAR_GB];
		uint32_t rgba = ((ar & 0xFF) << 24) | (((gb >> 8) & 0xFF) << 16) | ((gb & 0xFF) << 8) | ((ar >> 8) & 0xFF);
		if (disp) pica_present();
		// A texture copy that clears (render-to-texture passes) clears the frame in progress; the
		// display copy's clear applies to the next frame.
		if (clear && !(s_debugColors && !disp)) pica_clear(rgba, g.bp[BP_CLEAR_Z] & 0xFFFFFF);
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
