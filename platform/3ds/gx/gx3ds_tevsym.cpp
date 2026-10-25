// TEV setups compiled to PICA combiners through their algebra; see gx3ds_tevsym.h.
#include "gx3ds_tevsym.h"
#include <math.h>
#include <string.h>
#ifdef TEVSYM_HOST_DEBUG
#include <stdio.h>
#endif

namespace {

const float EPS = 1e-4f;

// PICA combiner values (as gx3ds_backend.cpp uses them).
enum : uint8_t { P_PRIMARY = 0x0, P_TEX0 = 0x3, P_BUFFER = 0xD, P_CONSTANT = 0xE, P_PREVIOUS = 0xF };
enum : uint8_t { F_REPLACE = 0, F_MODULATE = 1, F_ADD = 2, F_ADD_SIGNED = 3, F_INTERPOLATE = 4, F_SUBTRACT = 5, F_MADD = 8 };

// ---- Polynomials over atoms -------------------------------------------------------------------
// Atoms, each 0..1: a texture pair's colour or alpha, the vertex colour or alpha, a compare mask
// (0 or 1), a clamped temporary, a lerp of two sources.
enum : uint8_t { AT_TEX = 1, AT_TEXA = 9, AT_V = 17, AT_VA = 18, AT_MASK = 20, AT_TEMP = 24, AT_LERP = 44, AT_END = 52 };
const int kMaxTerms = 24, kMaxTemps = 20, kMaxMasks = 4, kMaxLerps = 8;

bool g_fail;  // a capacity ran out or a form is not handled: the compile gives up
const char* g_why;  // why the last variant failed (TEVSYM_HOST_DEBUG prints it)

inline bool isMask(uint8_t a) { return a >= AT_MASK && a < AT_TEMP; }

struct Term {
	uint8_t a[3], na;
	float c[3];
};
struct Poly {
	uint8_t n, nt;
	Term t[kMaxTerms];
};

Poly pZero(int n) {
	Poly p;
	p.n = uint8_t(n);
	p.nt = 0;
	return p;
}

bool cZero(const float* c, int n) {
	for (int i = 0; i < n; i++)
		if (fabsf(c[i]) > EPS) return false;
	return true;
}

void pAddTerm(Poly& p, const uint8_t* a, int na, const float* c) {
	static const uint8_t none[1] = {0};
	if (!a) a = none;
	if (cZero(c, p.n)) return;
	for (int i = 0; i < p.nt; i++) {
		Term& t = p.t[i];
		if (t.na != na || memcmp(t.a, a, size_t(na))) continue;
		for (int k = 0; k < p.n; k++) t.c[k] += c[k];
		if (cZero(t.c, p.n)) p.t[i] = p.t[--p.nt];
		return;
	}
	if (p.nt == kMaxTerms) {
		g_fail = true;
		return;
	}
	Term& t = p.t[p.nt++];
	memset(&t, 0, sizeof t);
	memcpy(t.a, a, size_t(na));
	t.na = uint8_t(na);
	for (int k = 0; k < 3; k++) t.c[k] = k < p.n ? c[k] : 0;
}

Poly pConst(int n, const float* v) {
	Poly p = pZero(n);
	pAddTerm(p, nullptr, 0, v);
	return p;
}
Poly pConst1(int n, float v) {
	const float c[3] = {v, v, v};
	return pConst(n, c);
}
Poly pAtom(int n, uint8_t a) {
	Poly p = pZero(n);
	const float one[3] = {1, 1, 1};
	pAddTerm(p, &a, 1, one);
	return p;
}

// p + k q
Poly pAdd(const Poly& p, const Poly& q, float k = 1) {
	Poly r = p;
	for (int i = 0; i < q.nt; i++) {
		float c[3];
		for (int j = 0; j < 3; j++) c[j] = k * q.t[i].c[j];
		pAddTerm(r, q.t[i].a, q.t[i].na, c);
	}
	return r;
}

Poly pMul(const Poly& p, const Poly& q) {
	Poly r = pZero(p.n);
	for (int i = 0; i < p.nt; i++)
		for (int j = 0; j < q.nt; j++) {
			uint8_t a[8];
			int na = 0;
			for (int x = 0; x < p.t[i].na; x++) a[na++] = p.t[i].a[x];
			for (int y = 0; y < q.t[j].na; y++) {
				const uint8_t b = q.t[j].a[y];
				bool dup = false;
				if (isMask(b))  // m * m = m
					for (int x = 0; x < na; x++) dup |= a[x] == b;
				if (!dup) a[na++] = b;
			}
			if (na > 3) {
				g_fail = true;
				return r;
			}
			for (int x = 1; x < na; x++)  // sorted
				for (int y = x; y > 0 && a[y - 1] > a[y]; y--) {
					const uint8_t s = a[y];
					a[y] = a[y - 1];
					a[y - 1] = s;
				}
			float c[3];
			for (int k = 0; k < 3; k++) c[k] = p.t[i].c[k] * q.t[j].c[k];
			pAddTerm(r, a, na, c);
		}
	return r;
}

Poly pScale(const Poly& p, float k) {
	Poly r = p;
	for (int i = 0; i < r.nt; i++)
		for (int j = 0; j < 3; j++) r.t[i].c[j] *= k;
	return r;
}

bool pIsConst(const Poly& p) { return p.nt == 0 || (p.nt == 1 && p.t[0].na == 0); }
void pConstVal(const Poly& p, float* v) {
	v[0] = v[1] = v[2] = 0;
	for (int i = 0; i < p.nt; i++)
		if (p.t[i].na == 0)
			for (int k = 0; k < 3; k++) v[k] = p.t[i].c[k];
}
int pFind(const Poly& p, const uint8_t* a, int na) {
	for (int i = 0; i < p.nt; i++)
		if (p.t[i].na == na && !memcmp(p.t[i].a, a, size_t(na))) return i;
	return -1;
}

void pRange(const Poly& p, float* lo, float* hi) {
	for (int k = 0; k < 3; k++) lo[k] = hi[k] = 0;
	for (int i = 0; i < p.nt; i++)
		for (int k = 0; k < p.n; k++) {
			const float c = p.t[i].c[k];
			if (p.t[i].na == 0) lo[k] += c, hi[k] += c;
			else if (c > 0) hi[k] += c;
			else lo[k] += c;
		}
}

// p with mask m set to 0 (p0) and the part it multiplies (p1): p = p0 + m p1.
void pSplit(const Poly& p, uint8_t m, Poly& p0, Poly& p1) {
	p0 = pZero(p.n);
	p1 = pZero(p.n);
	for (int i = 0; i < p.nt; i++) {
		const Term& t = p.t[i];
		uint8_t a[3];
		int na = 0;
		bool has = false;
		for (int x = 0; x < t.na; x++)
			if (t.a[x] == m) has = true;
			else a[na++] = t.a[x];
		pAddTerm(has ? p1 : p0, a, na, t.c);
	}
}

int pMasks(const Poly& p, uint8_t* ms) {
	int n = 0;
	for (int i = 0; i < p.nt; i++)
		for (int x = 0; x < p.t[i].na; x++) {
			const uint8_t a = p.t[i].a[x];
			if (!isMask(a)) continue;
			bool seen = false;
			for (int j = 0; j < n; j++) seen |= ms[j] == a;
			if (!seen && n < kMaxMasks) ms[n++] = a;
		}
	return n;
}

// The range over every setting of the masks (0 or 1).
void pRangeSplit(const Poly& p, float* lo, float* hi) {
	uint8_t ms[kMaxMasks];
	const int nm = pMasks(p, ms);
	for (int k = 0; k < 3; k++) lo[k] = 1e9f, hi[k] = -1e9f;
	for (int bits = 0; bits < (1 << nm); bits++) {
		Poly q = p;
		for (int j = 0; j < nm; j++) {
			Poly q0, q1;
			pSplit(q, ms[j], q0, q1);
			q = (bits >> j) & 1 ? pAdd(q0, q1) : q0;
		}
		float l[3], h[3];
		pRange(q, l, h);
		for (int k = 0; k < p.n; k++) {
			if (l[k] < lo[k]) lo[k] = l[k];
			if (h[k] > hi[k]) hi[k] = h[k];
		}
	}
}

bool pInUnit(const Poly& p) {
	float lo[3], hi[3];
	pRangeSplit(p, lo, hi);
	for (int k = 0; k < p.n; k++)
		if (lo[k] < -EPS || hi[k] > 1 + EPS) return false;
	return true;
}

// ---- Symbolic evaluation of the GX stages -----------------------------------------------------

struct MaskDef {
	uint8_t src;  // AT_TEXA + pair or AT_VA
	float thr;
	bool inv;     // src < thr (else src > thr)
};
struct Sym {
	Poly rgb, a;
	Poly temp[kMaxTemps];
	int nTemp;
	MaskDef mask[kMaxMasks];
	int nMask;
};

const int k8[8] = {255, 223, 191, 159, 127, 95, 63, 31};

float konstC(const TevSymIn& in, uint32_t sel, int comp) {
	if (sel < 8) return k8[sel] / 255.0f;
	if (sel >= 12 && sel < 16) return in.konst[sel - 12][comp] / 255.0f;
	if (sel >= 16) return in.konst[(sel - 16) & 3][(sel - 16) >> 2] / 255.0f;
	return 1;
}
float konstA(const TevSymIn& in, uint32_t sel) {
	if (sel < 8) return k8[sel] / 255.0f;
	if (sel >= 16) return in.konst[(sel - 16) & 3][(sel - 16) >> 2] / 255.0f;
	return 1;
}

// A constant input of a, b or c: its low 8 bits (ONE stays 1).
Poly low8(const Poly& p) {
	if (!pIsConst(p)) return p;
	float v[3];
	pConstVal(p, v);
	for (int k = 0; k < p.n; k++)
		if (fabsf(v[k] - 1) > EPS) v[k] = float(int(lroundf(v[k] * 255)) & 255) / 255.0f;
	return pConst(p.n, v);
}

bool symEval(const TevSymIn& in, int dropPair, Sym& S) {
	g_fail = false;
	S.nTemp = 0;
	S.nMask = 0;
	Poly R[4][2];
	for (int r = 0; r < 4; r++) {
		const float c[3] = {in.reg[r][0] / 255.0f, in.reg[r][1] / 255.0f, in.reg[r][2] / 255.0f};
		R[r][0] = pConst(3, c);
		R[r][1] = pConst1(1, in.reg[r][3] / 255.0f);
	}
	int curPair = -1;  // the texture TEXC reads in a stage without one: the last sampled (none: white)
	for (int s = 0; s < in.numStages; s++) {
		const uint32_t ce = in.ce[s], ae = in.ae[s], ord = in.order[s];
		const uint32_t chan = (ord >> 7) & 7;
		if (in.stagePair[s] >= 0) curPair = in.stagePair[s];
		const uint32_t rasSwap = in.swap[ae & 3], texSwap = in.swap[(ae >> 2) & 3];
		const uint32_t kcSel = in.ksel[s] & 31, kaSel = (in.ksel[s] >> 5) & 31;
		// texture and vertex colour inputs: identity swaps only
		auto texIn = [&](int n, bool alpha) -> Poly {
			if (curPair < 0) return pConst1(n, 1);
			if (in.pairConst[curPair] || curPair == dropPair) {
				const int16_t* v = in.pairValue[curPair];
				if (alpha) return pConst1(n, v[(texSwap >> 6) & 3] / 255.0f);
				const float c[3] = {v[texSwap & 3] / 255.0f, v[(texSwap >> 2) & 3] / 255.0f, v[(texSwap >> 4) & 3] / 255.0f};
				return pConst(n, c);
			}
			if (alpha ? ((texSwap >> 6) & 3) != 3 : (texSwap & 0x3F) != 0x24) g_fail = true;
			return pAtom(n, uint8_t((alpha ? AT_TEXA : AT_TEX) + curPair));
		};
		auto rasIn = [&](int n, bool alpha) -> Poly {
			if (chan > 1) return pConst1(n, 0);
			if (alpha ? ((rasSwap >> 6) & 3) != 3 : (rasSwap & 0x3F) != 0x24) g_fail = true;
			if (!alpha) {
				if (int(chan) == in.rasRgbChan && in.rgbVarying[chan]) return pAtom(n, AT_V);
				if (in.rgbVarying[chan]) {
					if (!in.chanSpec[chan]) g_fail = true;  // both channels vary: not expressible
					return pConst1(n, 0);
				}
				const float c[3] = {in.rasConst[chan][0] / 255.0f, in.rasConst[chan][1] / 255.0f, in.rasConst[chan][2] / 255.0f};
				return pConst(n, c);
			}
			if (int(chan) == in.rasAChan && in.aVarying[chan]) return pAtom(n, AT_VA);
			if (in.aVarying[chan]) g_fail = true;
			return pConst1(n, in.rasConst[chan][3] / 255.0f);
		};
		// a colour register's alpha, replicated over rgb
		auto rep3 = [](const Poly& a1) {
			Poly r = pZero(3);
			for (int i = 0; i < a1.nt; i++) {
				const float c[3] = {a1.t[i].c[0], a1.t[i].c[0], a1.t[i].c[0]};
				pAddTerm(r, a1.t[i].a, a1.t[i].na, c);
			}
			return r;
		};
		auto colorIn = [&](uint32_t c) -> Poly {
			if (c < 8) return (c & 1) ? rep3(R[c >> 1][1]) : R[c >> 1][0];
			switch (c) {
			case 8: return texIn(3, false);
			case 9: return rep3(texIn(1, true));
			case 10: return rasIn(3, false);
			case 11: return rep3(rasIn(1, true));
			case 12: return pConst1(3, 1);
			case 13: return pConst1(3, 128 / 255.0f);
			case 14: {
				const float v[3] = {konstC(in, kcSel, 0), konstC(in, kcSel, 1), konstC(in, kcSel, 2)};
				return pConst(3, v);
			}
			default: return pZero(3);
			}
		};
		auto alphaIn = [&](uint32_t a) -> Poly {
			if (a < 4) return R[a][1];
			switch (a) {
			case 4: return texIn(1, true);
			case 5: return rasIn(1, true);
			case 6: return pConst1(1, konstA(in, kaSel));
			default: return pZero(1);
			}
		};
		Poly out[2];
		int dest[2];
		for (int g = 0; g < 2; g++) {
			const uint32_t env = g ? ae : ce;
			const int n = g ? 1 : 3;
			Poly A, B, C, D;
			if (!g) {
				A = colorIn((env >> 12) & 15), B = colorIn((env >> 8) & 15), C = colorIn((env >> 4) & 15), D = colorIn(env & 15);
			} else {
				A = alphaIn((env >> 13) & 7), B = alphaIn((env >> 10) & 7), C = alphaIn((env >> 7) & 7), D = alphaIn((env >> 4) & 7);
			}
			A = low8(A), B = low8(B), C = low8(C);
			const uint32_t bias = (env >> 16) & 3, sub = (env >> 18) & 1, clampv = (env >> 19) & 1, scale = (env >> 20) & 3;
			Poly v;
			if (bias == 3) {
				// d + (a > b ? c : 0) of a texture's or the vertex's alpha against a constant: a mask.
				// The R8 / GR16 / BGR24 forms compare a and b's red (and green, blue) as one number:
				// the same as the components when they are replicated. Alpha: the A8 form only.
				const uint32_t mode = scale << 1 | sub;
				if ((mode & 1) || (g && mode != 6)) {
					g_fail = true;
					return false;
				}
				auto single = [&](const Poly& x, uint8_t& atom) {
					if (x.nt != 1 || x.t[0].na != 1) return false;
					for (int k = 0; k < n; k++)
						if (fabsf(x.t[0].c[k] - 1) > EPS) return false;
					atom = x.t[0].a[0];
					return (atom >= AT_TEXA && atom < AT_V) || atom == AT_VA;
				};
				auto replicated = [&](const float* v) { return n == 1 || (fabsf(v[1] - v[0]) < EPS && fabsf(v[2] - v[0]) < EPS); };
				uint8_t src;
				float thr[3];
				bool inv;
				if (pIsConst(A) && pIsConst(B)) {  // both constant: folded (as their red for R8-BGR24)
					float x[3], y[3];
					pConstVal(A, x);
					pConstVal(B, y);
					const bool cond = lroundf(x[0] * 255) > lroundf(y[0] * 255);
					if (!replicated(x) || !replicated(y)) {
						g_fail = true;
						return false;
					}
					v = cond ? pAdd(D, C) : D;
					goto clamped;
				}
				if (pIsConst(B) && single(A, src)) {
					pConstVal(B, thr);
					inv = false;
				} else if (pIsConst(A) && single(B, src)) {
					pConstVal(A, thr);
					inv = true;
				} else {
					g_fail = true;
					return false;
				}
				if (!replicated(thr)) {
					g_fail = true;
					return false;
				}
				int m = -1;
				for (int i = 0; i < S.nMask; i++)
					if (S.mask[i].src == src && fabsf(S.mask[i].thr - thr[0]) < EPS && S.mask[i].inv == inv) m = i;
				if (m < 0) {
					if (S.nMask == kMaxMasks) {
						g_fail = true;
						return false;
					}
					m = S.nMask++;
					S.mask[m] = MaskDef{src, thr[0], inv};
				}
				v = pAdd(D, pMul(pAtom(n, uint8_t(AT_MASK + m)), C));
			} else {
				const Poly lerp = pAdd(A, pMul(pAdd(B, A, -1), C));
				v = pAdd(D, lerp, sub ? -1.0f : 1.0f);
				if (bias) v = pAdd(v, pConst1(n, bias == 1 ? 0.5f : -0.5f));
				if (scale) v = pScale(v, scale == 1 ? 2.0f : scale == 2 ? 4.0f : 0.5f);
			}
		clamped:
			if (clampv && !pInUnit(v)) {  // a clamp that can matter: a temporary
				if (S.nTemp == kMaxTemps) {
					g_fail = true;
					return false;
				}
				S.temp[S.nTemp] = v;
				v = pAtom(n, uint8_t(AT_TEMP + S.nTemp));
				S.nTemp++;
			}
			out[g] = v;
			dest[g] = int((env >> 22) & 3);
		}
		R[dest[0]][0] = out[0];
		R[dest[1]][1] = out[1];
		if (g_fail) return false;
	}
	S.rgb = R[(in.ce[in.numStages - 1] >> 22) & 3][0];
	S.a = R[(in.ae[in.numStages - 1] >> 22) & 3][1];
	return !g_fail;
}

// ---- Operations: a DAG of combiner operations -------------------------------------------------

enum : uint8_t { O_SRC, O_K, O_BI, O_NODE, O_MASK };
struct Opnd {
	uint8_t kind;
	uint8_t atom;  // O_SRC: AT_TEX.., AT_TEXA.., AT_V, AT_VA; O_MASK: the mask
	bool inv;      // 1 - x
	int16_t node;
	float k[3];    // O_K, O_BI
};
struct NodeT {
	uint8_t fn, scale, nargs, n;  // n: 3 colour, 1 alpha
	Opnd arg[3];
};
const int kMaxNodes = 24;

Opnd oK(int n, const float* v) {
	Opnd o;
	memset(&o, 0, sizeof o);
	o.kind = O_K;
	for (int k = 0; k < 3; k++) {
		const float x = k < n ? v[k] : v[0];
		o.k[k] = x < 0 ? 0 : x > 1 ? 1 : x;
	}
	return o;
}
Opnd oK1(float v) {
	const float c[3] = {v, v, v};
	return oK(3, c);
}
Opnd oSrc(uint8_t a, bool inv = false) {
	Opnd o;
	memset(&o, 0, sizeof o);
	o.kind = O_SRC;
	o.atom = a;
	o.inv = inv;
	return o;
}
Opnd oInv(Opnd o) {
	o.inv = !o.inv;
	return o;
}

struct Variant {
	int lerpCse;   // share lerps of two sources found in this many polynomials (0: off)
	int maskOps;   // operations a mask takes: 3 exact (x256), 2 a ramp (x32)
	bool scaled;   // single-term values as w * clamp(x - k)
	bool useBi;    // the buffer's initial colour as a second constant
};

struct Val {
	Opnd o;
	float W[3];  // value = W * o
};

struct Compiler {
	const Sym* S;
	int n;
	Variant v;
	NodeT* nodes;
	int* nNodes;
	bool keep[kTevSymPairs];  // pairs with texture units (sources)
	bool tempDone[kMaxTemps];
	Val tempVal[kMaxTemps];
	Poly temps[kMaxTemps];  // the temporaries (with lerp atoms substituted)
	int nLerp;
	uint8_t lerpA[kMaxLerps], lerpB[kMaxLerps];
	float lerpR[kMaxLerps][3];
	int lerpNode[kMaxLerps];
	int biLeft;

	Val val(Opnd o) {
		Val x;
		x.o = o;
		x.W[0] = x.W[1] = x.W[2] = 1;
		return x;
	}
	Opnd node(uint8_t fn, int nargs, const Opnd* args, int scale = 1) {
		Opnd o;
		memset(&o, 0, sizeof o);
		if (*nNodes == kMaxNodes) {
			g_fail = true;
			return o;
		}
		// two different constants in one operation: one from the buffer's initial colour
		NodeT& nd = nodes[*nNodes];
		memset(&nd, 0, sizeof nd);
		nd.fn = fn;
		nd.scale = uint8_t(scale);
		nd.nargs = uint8_t(nargs);
		nd.n = uint8_t(n);
		for (int i = 0; i < nargs; i++) nd.arg[i] = args[i];
		int firstK = -1;
		for (int i = 0; i < nargs; i++) {
			if (nd.arg[i].kind != O_K) continue;
			if (firstK < 0) {
				firstK = i;
				continue;
			}
			if (!memcmp(nd.arg[i].k, nd.arg[firstK].k, sizeof nd.arg[i].k)) continue;
			if (!biLeft) {  // a third constant: not handled (rare)
				g_fail = true;
				return o;
			}
			biLeft--;
			nd.arg[i].kind = O_BI;
		}
		o.kind = O_NODE;
		o.node = int16_t((*nNodes)++);
		return o;
	}
	Opnd node1(uint8_t fn, Opnd a, int scale = 1) { return node(fn, 1, &a, scale); }
	Opnd node2(uint8_t fn, Opnd a, Opnd b, int scale = 1) {
		const Opnd x[2] = {a, b};
		return node(fn, 2, x, scale);
	}
	Opnd node3(uint8_t fn, Opnd a, Opnd b, Opnd c, int scale = 1) {
		const Opnd x[3] = {a, b, c};
		return node(fn, 3, x, scale);
	}
	Opnd bi(const float* y) {
		Opnd o = oK(n, y);
		o.kind = O_BI;
		return o;
	}
	bool isSrc(uint8_t a) const {
		if (a >= AT_TEX && a < AT_TEXA) return keep[a - AT_TEX];
		if (a >= AT_TEXA && a < AT_V) return keep[a - AT_TEXA];
		return a == AT_V || a == AT_VA;
	}

	Val atom(uint8_t a) {
		if (isSrc(a)) return val(oSrc(a));
		if (a >= AT_TEMP && a < AT_LERP) {
			const int t = a - AT_TEMP;
			if (!tempDone[t]) {
				tempVal[t] = value(temps[t]);
				tempDone[t] = true;
			}
			return tempVal[t];
		}
		if (a >= AT_LERP && a < AT_END) {
			const int l = a - AT_LERP;
			if (lerpNode[l] < 0) {
				const Opnd o = node3(F_INTERPOLATE, oSrc(lerpA[l]), oSrc(lerpB[l]), oK(n, lerpR[l]));
				lerpNode[l] = o.node;
			}
			Opnd o;
			memset(&o, 0, sizeof o);
			o.kind = O_NODE;
			o.node = int16_t(lerpNode[l]);
			return val(o);
		}
		g_fail = true;
		return val(oK1(0));
	}

	Opnd exact(const Val& x) {
		bool one = true;
		for (int k = 0; k < n; k++) one &= fabsf(x.W[k] - 1) < EPS;
		if (one) return x.o;
		if (x.o.kind == O_K) {
			float c[3];
			for (int k = 0; k < 3; k++) c[k] = x.o.k[k] * x.W[k];
			return oK(3, c);
		}
		return node2(F_MODULATE, x.o, oK(n, x.W));
	}

	Val value(const Poly& p) {
		if (g_fail) return val(oK1(0));
		if (pIsConst(p)) {
			float c[3];
			pConstVal(p, c);
			return val(oK(n, c));
		}
		if (p.nt == 1 && p.t[0].na == 1) {
			bool one = true;
			for (int k = 0; k < n; k++) one &= fabsf(p.t[0].c[k] - 1) < EPS;
			if (one) return atom(p.t[0].a[0]);
		}
		uint8_t ms[kMaxMasks];
		if (pMasks(p, ms)) return select(p, ms[0]);
		// a source in every term: s * (p / s), exact while p / s stays in 0..1
		float c0[3];
		pConstVal(p, c0);
		if (cZero(c0, n) && p.nt > 1) {
			for (int x = 0; x < p.t[0].na; x++) {
				const uint8_t a = p.t[0].a[x];
				if (!isSrc(a)) continue;
				bool all = true;
				for (int i = 1; i < p.nt && all; i++) {
					bool has = false;
					for (int y = 0; y < p.t[i].na; y++) has |= p.t[i].a[y] == a;
					all = has;
				}
				if (!all) continue;
				Poly q = pZero(p.n);
				for (int i = 0; i < p.nt; i++) {
					uint8_t b[3];
					int nb = 0;
					bool removed = false;
					for (int y = 0; y < p.t[i].na; y++)
						if (p.t[i].a[y] == a && !removed) removed = true;
						else b[nb++] = p.t[i].a[y];
					pAddTerm(q, b, nb, p.t[i].c);
				}
				if (!pInUnit(q)) continue;
				const Opnd qo = exact(value(q));
				return val(node2(F_MODULATE, qo, oSrc(a)));
			}
		}
		return affine(p, nullptr);
	}

	Opnd selNode(uint8_t m, const Poly& A, const Poly& B) {
		const Opnd a = exact(value(A));
		const Opnd b = exact(value(B));
		Opnd mo;
		memset(&mo, 0, sizeof mo);
		mo.kind = O_MASK;
		mo.atom = m;
		return node3(F_INTERPOLATE, a, b, mo);
	}

	Val select(const Poly& p, uint8_t m) {
		Poly p0, p1;
		pSplit(p, m, p0, p1);
		const Poly A = pAdd(p0, p1), &B = p0;
		// what both sides share (and the constant) is added after the select
		Poly common = pZero(p.n);
		for (int i = 0; i < p0.nt; i++)
			if (p0.t[i].na == 0 || pFind(p1, p0.t[i].a, p0.t[i].na) < 0) pAddTerm(common, p0.t[i].a, p0.t[i].na, p0.t[i].c);
		if (common.nt) {
			const Poly A2 = pAdd(A, common, -1), B2 = pAdd(B, common, -1);
			if (pInUnit(A2) && pInUnit(B2)) {
				const Opnd s = selNode(m, A2, B2);
				return affine(common, &s);
			}
		}
		return val(selNode(m, A, B));
	}

	struct T {
		Opnd o, o2;  // o2: the second factor of a product
		bool prod, inv;
		float w[3];
	};

	Val affine(Poly p, const Opnd* extra) {
		if (g_fail) return val(oK1(0));
		// a temporary clamped only at the top inside a sum that only grows: clamp(clamp(x) + y) =
		// clamp(x + y)
		for (bool changed = true; changed;) {
			changed = false;
			bool mono = true;
			for (int i = 0; i < p.nt; i++)
				for (int k = 0; k < n; k++) mono &= p.t[i].c[k] >= -EPS;
			if (!mono) break;
			for (int i = 0; i < p.nt && !changed; i++) {
				const Term& t = p.t[i];
				if (t.na != 1 || t.a[0] < AT_TEMP || t.a[0] >= AT_LERP) continue;
				bool one = true;
				for (int k = 0; k < n; k++) one &= fabsf(t.c[k] - 1) < EPS;
				const Poly& tp = temps[t.a[0] - AT_TEMP];
				uint8_t ms[kMaxMasks];
				bool tmono = !pMasks(tp, ms);
				for (int j = 0; j < tp.nt; j++)
					for (int k = 0; k < n; k++) tmono &= tp.t[j].c[k] >= -EPS;
				if (!one || !tmono) continue;
				Poly q = pZero(p.n);
				for (int j = 0; j < p.nt; j++)
					if (j != i) pAddTerm(q, p.t[j].a, p.t[j].na, p.t[j].c);
				p = pAdd(q, tp);
				changed = true;
			}
		}
		if (extra) {
			// S + Q: ADD_SIGNED(S, Q + 1/2) or ADD(S, Q) when that part fits in 0..1, or S - k
			const Poly q = pAdd(p, pConst1(n, 0.5f));
			if (pInUnit(q)) return val(node2(F_ADD_SIGNED, *extra, exact(value(q))));
			if (pInUnit(p)) return val(node2(F_ADD, *extra, exact(value(p))));
			if (pIsConst(p)) {
				float c[3];
				pConstVal(p, c);
				if (c[0] <= EPS && c[1] <= EPS && c[2] <= EPS) {
					for (int k = 0; k < 3; k++) c[k] = -c[k];
					return val(node2(F_SUBTRACT, *extra, oK(n, c)));
				}
			}
		}
		T terms[kMaxTerms + 1];
		int nt = 0;
		float b[3];
		pConstVal(p, b);
		if (extra) {
			memset(&terms[nt], 0, sizeof terms[nt]);
			terms[nt].o = *extra;
			terms[nt].w[0] = terms[nt].w[1] = terms[nt].w[2] = 1;
			nt++;
		}
		for (int i = 0; i < p.nt; i++) {
			const Term& t = p.t[i];
			if (t.na == 0) continue;
			T& x = terms[nt];
			memset(&x, 0, sizeof x);
			if (t.na == 1) {
				const Val av = atom(t.a[0]);
				if (av.o.kind == O_K) {
					for (int k = 0; k < n; k++) b[k] += t.c[k] * av.W[k] * av.o.k[k];
					continue;
				}
				x.o = av.o;
				for (int k = 0; k < n; k++) x.w[k] = t.c[k] * av.W[k];
			} else if (t.na == 2) {
				const Val v1 = atom(t.a[0]), v2 = atom(t.a[1]);
				x.o = v1.o;
				x.o2 = v2.o;
				x.prod = true;
				for (int k = 0; k < n; k++) x.w[k] = t.c[k] * v1.W[k] * v2.W[k];
			} else {
				g_fail = true;
				return val(oK1(0));
			}
			nt++;
		}
		if (g_fail) return val(oK1(0));
		// one source, any weights: b + w x = lerp(b, b + w, x) when both ends are in range
		if (nt == 1 && !terms[0].prod) {
			float e0[3], e1[3];
			bool fits = true, z0 = true, z1 = true;
			for (int k = 0; k < n; k++) {
				e0[k] = b[k];
				e1[k] = b[k] + terms[0].w[k];
				fits &= e0[k] >= -EPS && e0[k] <= 1 + EPS && e1[k] >= -EPS && e1[k] <= 1 + EPS;
				z0 &= fabsf(e0[k]) < EPS;
				z1 &= fabsf(e1[k]) < EPS;
			}
			if (fits) {
				if (z0) return val(node2(F_MODULATE, terms[0].o, oK(n, e1)));
				if (z1) return val(node2(F_MODULATE, oInv(terms[0].o), oK(n, e0)));
				if (biLeft) {
					biLeft--;
					return val(node3(F_INTERPOLATE, oK(n, e1), bi(e0), terms[0].o));
				}
			}
		}
		// all weights positive: w x = w - w (1 - x) for the negative parts
		T pos[2 * kMaxTerms + 2];
		int np = 0;
		for (int i = 0; i < nt; i++) {
			T wp = terms[i], wn = terms[i];
			bool anyP = false, anyN = false;
			for (int k = 0; k < 3; k++) {
				const float w = k < n ? terms[i].w[k] : 0;
				wp.w[k] = w > 0 ? w : 0;
				wn.w[k] = w < 0 ? -w : 0;
				anyP |= wp.w[k] > EPS;
				anyN |= wn.w[k] > EPS;
			}
			if (anyP) pos[np++] = wp;
			if (anyN) {
				wn.inv = true;
				pos[np++] = wn;
				for (int k = 0; k < n; k++) b[k] -= wn.w[k];
			}
		}
		if (!np) return val(oK(n, b));
		float lo[3], hi[3], wmax = 0;
		bool allNeg = true;
		for (int k = 0; k < n; k++) {
			lo[k] = b[k];
			hi[k] = b[k];
			for (int i = 0; i < np; i++) hi[k] += pos[i].w[k];
			allNeg &= hi[k] <= EPS;
		}
		if (allNeg) return val(oK1(0));
		for (int i = 0; i < np; i++)
			for (int k = 0; k < n; k++)
				if (pos[i].w[k] > wmax) wmax = pos[i].w[k];
		// one term, w x + b with b <= 0 and no clamp at the top: w * clamp(x - b / w), the factor w
		// left to the reader
		if (np == 1 && !pos[0].prod && v.scaled) {
			bool ok = true, bz = true;
			float kk[3];
			for (int k = 0; k < n; k++) {
				const float w = pos[0].w[k];
				ok &= b[k] <= EPS && w + b[k] <= 1 + EPS && (w <= EPS || -b[k] / w <= 1 + EPS);
				bz &= -b[k] < EPS;
				kk[k] = w > EPS ? -b[k] / w : 1;
			}
			if (ok) {
				Opnd o = pos[0].o;
				if (pos[0].inv) o = oInv(o);
				Val r;
				r.o = bz ? o : node2(F_SUBTRACT, o, oK(n, kk));
				for (int k = 0; k < 3; k++) r.W[k] = k < n ? pos[0].w[k] : pos[0].w[0];
				return r;
			}
		}
		float minLo = lo[0];
		for (int k = 1; k < n; k++) minLo = lo[k] < minLo ? lo[k] : minLo;
		float sigma = 1;
		if (minLo < -EPS) sigma = minLo >= -1 - EPS ? 0.5f : 0.25f;
		while (wmax * sigma > 1 + EPS) sigma *= 0.5f;
		if (sigma < 0.25f - EPS) {
			g_fail = true;
			return val(oK1(0));
		}
		float beta[3], y0[3];
		bool betaZ = true, y0Z = true;
		for (int k = 0; k < 3; k++) {
			const int kk = k < n ? k : 0;
			beta[k] = -sigma * lo[kk] > 0 ? -sigma * lo[kk] : 0;
			y0[k] = sigma * lo[kk] + beta[k];
			betaZ &= beta[k] < EPS;
			y0Z &= fabsf(y0[k]) < EPS;
		}
		int nProd = 0;
		for (int i = 0; i < np; i++) nProd += pos[i].prod;
		bool used[2 * kMaxTerms + 2] = {};
		Opnd acc;
		bool have = false;
		auto scaledOne = [&](const T& t) {
			for (int k = 0; k < n; k++)
				if (fabsf(sigma * t.w[k] - 1) > EPS) return false;
			return true;
		};
		if (nProd > 1)
			for (int i = 0; i < np; i++)
				if (pos[i].prod && !scaledOne(pos[i])) {
					g_fail = true;
					return val(oK1(0));
				}
		// a lerp: two plain terms whose scaled weights add up to 1 (y0 = 0)
		if (y0Z && !nProd && np >= 2) {
			for (int i = 0; i < np && !have; i++)
				for (int j = i + 1; j < np && !have; j++) {
					bool sum1 = true;
					for (int k = 0; k < n; k++) sum1 &= fabsf(sigma * (pos[i].w[k] + pos[j].w[k]) - 1) < 1e-3f;
					if (!sum1) continue;
					float kk[3];
					for (int k = 0; k < 3; k++) kk[k] = sigma * pos[i].w[k < n ? k : 0];
					Opnd oi = pos[i].o, oj = pos[j].o;
					if (pos[i].inv) oi = oInv(oi);
					if (pos[j].inv) oj = oInv(oj);
					acc = node3(F_INTERPOLATE, oi, oj, oK(n, kk));
					have = true;
					used[i] = used[j] = true;
				}
		}
		// with nothing to start from, a term of weight 1 starts the sum at no cost (a source first)
		if (!have && y0Z && !nProd)
			for (int pass = 0; pass < 2 && !have; pass++)
				for (int i = 0; i < np && !have; i++) {
					if (used[i] || !scaledOne(pos[i]) || (pass == 0) != (pos[i].o.kind == O_SRC)) continue;
					acc = pos[i].inv ? oInv(pos[i].o) : pos[i].o;
					used[i] = true;
					have = true;
				}
		// products first, then the plain terms
		for (int pass = 0; pass < 2; pass++)
			for (int i = 0; i < np; i++) {
				if (used[i] || pos[i].prod != (pass == 0)) continue;
				const T& t = pos[i];
				float w[3];
				for (int k = 0; k < 3; k++) w[k] = sigma * t.w[k < n ? k : 0];
				const bool one = scaledOne(t);
				if (t.prod) {
					if (t.inv) {
						g_fail = true;
						return val(oK1(0));
					}
					if (!have) {
						const Opnd pn = node2(F_MODULATE, t.o, t.o2);
						if (one && y0Z) acc = pn;
						else if (one) acc = node2(F_ADD, pn, oK(n, y0));
						else if (y0Z) acc = node2(F_MODULATE, pn, oK(n, w));
						else if (biLeft) {
							biLeft--;
							acc = node3(F_MADD, pn, oK(n, w), bi(y0));
						} else acc = node2(F_ADD, node2(F_MODULATE, pn, oK(n, w)), oK(n, y0));
						have = true;
					} else {
						if (!one) {
							g_fail = true;
							return val(oK1(0));
						}
						acc = node3(F_MADD, t.o, t.o2, acc);
					}
					continue;
				}
				Opnd o = t.o;
				if (t.inv) o = oInv(o);
				if (!have) {
					if (y0Z) acc = one ? o : node2(F_MODULATE, o, oK(n, w));
					else if (one) acc = node2(F_ADD, o, oK(n, y0));
					else if (biLeft) {
						biLeft--;
						acc = node3(F_MADD, o, oK(n, w), bi(y0));
					} else acc = node2(F_ADD, node2(F_MODULATE, o, oK(n, w)), oK(n, y0));
					have = true;
				} else {
					acc = one ? node2(F_ADD, o, acc) : node3(F_MADD, o, oK(n, w), acc);
				}
			}
		if (!have) {
			g_fail = true;
			return val(oK1(0));
		}
		const int up = int(lroundf(1 / sigma));
		if (!betaZ) acc = node2(F_SUBTRACT, acc, oK(n, beta), up);
		else if (up > 1) {
			if (acc.kind != O_NODE) acc = node1(F_REPLACE, acc);
			if (!g_fail) nodes[acc.node].scale = uint8_t(up);
		}
		return val(acc);
	}
};

// Lerps of two plain sources in the same proportion: an atom each when found in `minCount`
// polynomials (rgb's and the temporaries'), substituted there.
void substituteLerps(Poly* polys, int np, Compiler& C, int minCount) {
	struct Cand {
		uint8_t a1, a2;
		float r[3];
		int count;
	} cand[32];
	int nc = 0;
	auto ratio = [](const Term& t1, const Term& t2, float* r) {
		for (int k = 0; k < 3; k++) r[k] = t1.c[k] / (t1.c[k] + t2.c[k]);
	};
	for (int pi = 0; pi < np; pi++) {
		const Poly& p = polys[pi];
		for (int i = 0; i < p.nt; i++)
			for (int j = 0; j < p.nt; j++) {
				const Term &t1 = p.t[i], &t2 = p.t[j];
				if (t1.na != 1 || t2.na != 1 || t1.a[0] >= t2.a[0] || !C.isSrc(t1.a[0]) || !C.isSrc(t2.a[0])) continue;
				bool posw = true;
				for (int k = 0; k < p.n; k++) posw &= t1.c[k] > EPS && t2.c[k] > EPS;
				if (!posw) continue;
				float r[3];
				ratio(t1, t2, r);
				int c = 0;
				while (c < nc && !(cand[c].a1 == t1.a[0] && cand[c].a2 == t2.a[0] && fabsf(cand[c].r[0] - r[0]) < 1e-3f &&
				                   fabsf(cand[c].r[1] - r[1]) < 1e-3f && fabsf(cand[c].r[2] - r[2]) < 1e-3f))
					c++;
				if (c == nc) {
					if (nc == 32) continue;
					cand[nc] = Cand{t1.a[0], t2.a[0], {r[0], r[1], r[2]}, 0};
					nc++;
				}
				cand[c].count++;
			}
	}
	for (int c = 0; c < nc && C.nLerp < kMaxLerps; c++) {
		if (cand[c].count < minCount) continue;
		const uint8_t atom = uint8_t(AT_LERP + C.nLerp);
		bool used = false;
		for (int pi = 0; pi < np; pi++) {
			Poly& p = polys[pi];
			const int i = pFind(p, &cand[c].a1, 1), j = pFind(p, &cand[c].a2, 1);
			if (i < 0 || j < 0) continue;
			float r[3];
			ratio(p.t[i], p.t[j], r);
			bool same = true;
			for (int k = 0; k < p.n; k++) same &= fabsf(r[k] - cand[c].r[k]) < 1e-3f;
			if (!same) continue;
			float w[3];
			for (int k = 0; k < 3; k++) w[k] = p.t[i].c[k] + p.t[j].c[k];
			Poly q = pZero(p.n);
			for (int x = 0; x < p.nt; x++)
				if (x != i && x != j) pAddTerm(q, p.t[x].a, p.t[x].na, p.t[x].c);
			pAddTerm(q, &atom, 1, w);
			p = q;
			used = true;
		}
		if (used) {
			C.lerpA[C.nLerp] = cand[c].a1;
			C.lerpB[C.nLerp] = cand[c].a2;
			memcpy(C.lerpR[C.nLerp], cand[c].r, sizeof cand[c].r);
			C.lerpNode[C.nLerp] = -1;
			C.nLerp++;
		}
	}
}

// ---- Scheduling into combiner stages ----------------------------------------------------------

struct Program {
	int nst;
	int rgb[PICA_MAX_STAGES];    // node per stage (-1: PREVIOUS passed through)
	int alpha[PICA_MAX_STAGES];  // -1 pass, -2 mask's first operation, -3 a later one, else a node
	uint8_t bufRgb;
};

struct Sched {
	const NodeT* nodes;
	int nNodes;
	int rnodes[kMaxNodes], nr;  // rgb nodes reachable from the root, children first
	int anodes[kMaxNodes], na;  // alpha nodes, a chain
	int maskOps;
	bool hasMask;
	Program best;
	bool found;
	int order[kMaxNodes];
	int tries;

	bool layout(const int* seq, int total) {
		// seq[t]: rgb node or -1, t < total
		int pos[kMaxNodes];
		for (int i = 0; i < nNodes; i++) pos[i] = -1;
		for (int t = 0; t < total; t++)
			if (seq[t] >= 0) pos[seq[t]] = t;
		int firstReader = 99, lastReader = -1;
		float biv[3];
		bool biSet = false;
		for (int t = 0; t < total; t++) {
			if (seq[t] < 0) continue;
			const NodeT& nd = nodes[seq[t]];
			for (int i = 0; i < nd.nargs; i++) {
				const Opnd& a = nd.arg[i];
				if (a.kind == O_MASK) {
					if (t < firstReader) firstReader = t;
					if (t > lastReader) lastReader = t;
				}
				if (a.kind == O_BI) {
					if (biSet && memcmp(biv, a.k, sizeof biv)) return false;
					memcpy(biv, a.k, sizeof biv);
					biSet = true;
				}
			}
		}
		if (hasMask && firstReader < maskOps) return false;
		int nst = total;
		const int startA = (na ? (nst - na) : 0);
		int sa = startA < 0 ? 0 : startA;
		if (sa < lastReader) sa = lastReader;
		if (hasMask && sa < maskOps) sa = maskOps;
		if (na && sa + na > nst) nst = sa + na;
		if (nst > PICA_MAX_STAGES) return false;
		// buffer writes: a node read other than in the next stage
		int writer[PICA_MAX_STAGES];
		for (int t = 0; t < PICA_MAX_STAGES; t++) writer[t] = -1;
		for (int t = 0; t < total; t++) {
			if (seq[t] < 0) continue;
			const NodeT& nd = nodes[seq[t]];
			for (int i = 0; i < nd.nargs; i++) {
				if (nd.arg[i].kind != O_NODE) continue;
				const int st = pos[nd.arg[i].node];
				if (st < 0) return false;
				if (st == t - 1) continue;
				if (st > 3 || t < st + 2) return false;
				writer[st] = nd.arg[i].node;
			}
		}
		auto bufAt = [&](int t) {  // the node the buffer shows in stage t (-1: its initial colour)
			for (int s = t - 2; s >= 0; s--)
				if (writer[s] >= 0) return writer[s];
			return -1;
		};
		for (int t = 0; t < total; t++) {
			if (seq[t] < 0) continue;
			const NodeT& nd = nodes[seq[t]];
			int bufNode = -2;
			bool bufInit = false;
			for (int i = 0; i < nd.nargs; i++) {
				const Opnd& a = nd.arg[i];
				if (a.kind == O_NODE && pos[a.node] != t - 1) {
					if (bufAt(t) != a.node) return false;
					if (bufNode != -2 && bufNode != a.node) return false;
					bufNode = a.node;
				}
				if (a.kind == O_BI) {
					if (t < 1 || bufAt(t) != -1) return false;
					bufInit = true;
				}
			}
			if (bufInit && bufNode != -2) return false;
		}
		Program pr;
		pr.nst = nst;
		pr.bufRgb = 0;
		for (int t = 0; t < PICA_MAX_STAGES; t++) {
			pr.rgb[t] = t < total ? seq[t] : -1;
			pr.alpha[t] = -1;
			if (writer[t] >= 0) pr.bufRgb |= uint8_t(1u << t);
		}
		if (hasMask)
			for (int t = 0; t < maskOps; t++) pr.alpha[t] = t ? -3 : -2;
		for (int i = 0; i < na; i++) pr.alpha[sa + i] = anodes[i];
		if (!found || pr.nst < best.nst) {
			best = pr;
			found = true;
		}
		return true;
	}

	void pads(int i, int t, int* seq) {
		if (tries > 4000) return;
		if (i == nr) {
			tries++;
			layout(seq, t);
			return;
		}
		for (int p = 0; t + p + (nr - i) <= PICA_MAX_STAGES; p++) {
			if (found && t + p + (nr - i) >= best.nst) break;
			for (int k = 0; k < p; k++) seq[t + k] = -1;
			seq[t + p] = order[i];
			pads(i + 1, t + p + 1, seq);
		}
	}

	void orders(int k, bool* done) {
		if (tries > 4000 || (found && best.nst <= nr)) return;
		if (k == nr) {
			int seq[PICA_MAX_STAGES];
			pads(0, 0, seq);
			return;
		}
		for (int i = 0; i < nr; i++) {
			const int nd = rnodes[i];
			if (done[nd]) continue;
			bool ready = true;
			for (int j = 0; j < nodes[nd].nargs; j++)
				if (nodes[nd].arg[j].kind == O_NODE && !done[nodes[nd].arg[j].node]) ready = false;
			if (!ready) continue;
			done[nd] = true;
			order[k] = nd;
			orders(k + 1, done);
			done[nd] = false;
		}
	}
};

void collect(const NodeT* nodes, int root, bool* seen, int* out, int& n) {
	if (seen[root]) return;
	for (int i = 0; i < nodes[root].nargs; i++)
		if (nodes[root].arg[i].kind == O_NODE) collect(nodes, nodes[root].arg[i].node, seen, out, n);
	seen[root] = true;
	out[n++] = root;
}

// ---- Emission ---------------------------------------------------------------------------------

struct Emit {
	const NodeT* nodes;
	const int* unitOf;  // pair -> unit
	const Program* pr;
	uint32_t bufColor;
	bool biRgb, biA;

	// source and operand of one input (rgb: operand 0 rgb, 1 1-rgb, 2 alpha, 3 1-alpha; alpha: 0 a, 1 1-a)
	void input(const Opnd& a, int t, bool alpha, uint8_t& src, uint8_t& op, uint32_t& kcol) {
		uint8_t o = 0;
		switch (a.kind) {
		case O_SRC:
			if (a.atom == AT_V) src = P_PRIMARY, o = 0;
			else if (a.atom == AT_VA) src = P_PRIMARY, o = alpha ? 0 : 2;
			else if (a.atom >= AT_TEXA) src = uint8_t(P_TEX0 + unitOf[a.atom - AT_TEXA]), o = alpha ? 0 : 2;
			else src = uint8_t(P_TEX0 + unitOf[a.atom - AT_TEX]), o = 0;
			break;
		case O_K:
			src = P_CONSTANT;
			if (alpha) kcol = (kcol & 0x00FFFFFFu) | uint32_t(lroundf(a.k[0] * 255)) << 24;
			else
				kcol = (kcol & 0xFF000000u) | uint32_t(lroundf(a.k[0] * 255)) | uint32_t(lroundf(a.k[1] * 255)) << 8 |
				       uint32_t(lroundf(a.k[2] * 255)) << 16;
			break;
		case O_BI:
			src = P_BUFFER;
			if (alpha) {
				bufColor = (bufColor & 0x00FFFFFFu) | uint32_t(lroundf(a.k[0] * 255)) << 24;
				biA = true;
			} else {
				bufColor = (bufColor & 0xFF000000u) | uint32_t(lroundf(a.k[0] * 255)) | uint32_t(lroundf(a.k[1] * 255)) << 8 |
				           uint32_t(lroundf(a.k[2] * 255)) << 16;
				biRgb = true;
			}
			break;
		case O_NODE: {
			int st = -1;
			for (int s = 0; s < pr->nst; s++)
				if ((alpha ? pr->alpha[s] : pr->rgb[s]) == a.node) st = s;
			src = st == t - 1 ? P_PREVIOUS : P_BUFFER;
			o = 0;
			break;
		}
		case O_MASK: src = P_PREVIOUS, o = 2; break;
		}
		if (a.inv) o |= 1;
		op = o;
	}
};

uint8_t scaleCode(int s) { return uint8_t(s == 4 ? 2 : s == 2 ? 1 : 0); }

void emitProgram(const NodeT* nodes, const Program& pr, const MaskDef* mask, const int* unitOf, TevSymOut& out) {
	Emit e{nodes, unitOf, &pr, 0, false, false};
	memset(out.stage, 0, sizeof out.stage);
	for (int t = 0; t < pr.nst; t++) {
		PicaTevStage& ps = out.stage[t];
		uint32_t kcol = 0;
		uint8_t s[3] = {P_PREVIOUS, P_PREVIOUS, P_PREVIOUS}, o[3] = {0, 0, 0};
		if (pr.rgb[t] >= 0) {
			const NodeT& nd = nodes[pr.rgb[t]];
			for (int i = 0; i < nd.nargs; i++) e.input(nd.arg[i], t, false, s[i], o[i], kcol);
			ps.func_rgb = nd.fn;
			ps.scale_rgb = scaleCode(nd.scale);
		} else {
			ps.func_rgb = F_REPLACE;
		}
		ps.src_rgb = uint16_t(s[0] | s[1] << 4 | s[2] << 8);
		ps.op_rgb = uint16_t(o[0] | o[1] << 4 | o[2] << 8);
		uint8_t sa[3] = {P_PREVIOUS, P_PREVIOUS, P_PREVIOUS}, oa[3] = {0, 0, 0};
		ps.func_alpha = F_REPLACE;
		const int a = pr.alpha[t];
		if (a == -2) {
			// a > k: (a - k) x4, then (x + x) x4 each: an 8-bit step (1/255) becomes 1 after three
			const MaskDef& m = mask[0];
			Opnd src = oSrc(m.src);
			e.input(src, t, true, sa[0], oa[0], kcol);
			kcol = (kcol & 0x00FFFFFFu) | uint32_t(lroundf(m.thr * 255)) << 24;
			if (m.inv) {
				sa[1] = sa[0], oa[1] = oa[0];
				sa[0] = P_CONSTANT, oa[0] = 0;
			} else {
				sa[1] = P_CONSTANT, oa[1] = 0;
			}
			ps.func_alpha = F_SUBTRACT;
			ps.scale_alpha = 2;
		} else if (a == -3) {
			ps.func_alpha = F_ADD;
			ps.scale_alpha = 2;
		} else if (a >= 0) {
			const NodeT& nd = nodes[a];
			for (int i = 0; i < nd.nargs; i++) e.input(nd.arg[i], t, true, sa[i], oa[i], kcol);
			ps.func_alpha = nd.fn;
			ps.scale_alpha = scaleCode(nd.scale);
		}
		ps.src_alpha = uint16_t(sa[0] | sa[1] << 4 | sa[2] << 8);
		ps.op_alpha = uint16_t(oa[0] | oa[1] << 4 | oa[2] << 8);
		ps.color = kcol;
	}
	out.nstages = pr.nst;
	out.buf_rgb = pr.bufRgb;
	out.buf_alpha = 0;
	out.buf_color = e.bufColor;
}

bool compileVariant(const TevSymIn& in, int drop, const Variant& v, TevSymOut& out) {
	static Sym S;
	if (!symEval(in, drop, S)) { g_why = "stages not expressible"; return false; }
	if (S.nMask > 1) { g_why = "more than one mask"; return false; }
	static NodeT nodes[kMaxNodes];
	int nNodes = 0;
	Compiler C;
	memset(&C, 0, sizeof C);
	C.S = &S;
	C.v = v;
	C.nodes = nodes;
	C.nNodes = &nNodes;
	int unitOf[kTevSymPairs], nUnits = 0;
	for (int p = 0; p < in.nPairs; p++) {
		unitOf[p] = -1;
		C.keep[p] = !in.pairConst[p] && p != drop;
		if (C.keep[p]) {
			if (nUnits == PICA_MAX_TEX) { g_why = "more than 3 textures"; return false; }
			out.unitPair[nUnits] = p;
			unitOf[p] = nUnits++;
		}
	}
	for (int i = 0; i < S.nTemp; i++) C.temps[i] = S.temp[i];
	for (int l = 0; l < kMaxLerps; l++) C.lerpNode[l] = -1;
	// lerps shared by rgb and the colour temporaries
	Poly polys[kMaxTemps + 1];
	int map[kMaxTemps + 1], np = 0;
	polys[np] = S.rgb;
	map[np++] = -1;
	for (int i = 0; i < S.nTemp; i++)
		if (S.temp[i].n == 3) {
			polys[np] = S.temp[i];
			map[np++] = i;
		}
	if (v.lerpCse) substituteLerps(polys, np, C, v.lerpCse);
	for (int i = 1; i < np; i++) C.temps[map[i]] = polys[i];
	g_fail = false;
	C.n = 3;
	C.biLeft = v.useBi ? 1 : 0;
	const Opnd rr = C.exact(C.value(polys[0]));
	Opnd rroot = rr;
	if (!g_fail && rroot.kind != O_NODE) rroot = C.node1(F_REPLACE, rroot);
	if (g_fail) { g_why = "colour not compiled"; return false; }
	// alpha: a compiler of its own (own constants), a chain of operations
	Compiler A = C;
	A.n = 1;
	A.biLeft = v.useBi ? 1 : 0;
	A.nLerp = 0;
	for (int i = 0; i < kMaxTemps; i++) A.tempDone[i] = false;
	for (int i = 0; i < S.nTemp; i++) A.temps[i] = S.temp[i];
	const int firstA = nNodes;
	Opnd aroot = A.exact(A.value(S.a));
	if (!g_fail && aroot.kind != O_NODE) aroot = A.node1(F_REPLACE, aroot);
	if (g_fail) { g_why = "alpha not compiled"; return false; }
	Sched sc;
	memset(&sc, 0, sizeof sc);
	sc.nodes = nodes;
	sc.nNodes = nNodes;
	sc.maskOps = v.maskOps;
	sc.hasMask = S.nMask > 0;
	bool seen[kMaxNodes] = {};
	collect(nodes, rroot.node, seen, sc.rnodes, sc.nr);
	bool seenA[kMaxNodes] = {};
	collect(nodes, aroot.node, seenA, sc.anodes, sc.na);
	for (int i = 0; i < sc.nr; i++)
		if (sc.rnodes[i] >= firstA) { g_why = "colour reads alpha operations"; return false; }
	// the alpha nodes: a chain, each reading only the one before
	for (int i = 0; i < sc.na; i++) {
		const NodeT& nd = nodes[sc.anodes[i]];
		if (sc.anodes[i] < firstA) { g_why = "alpha reads colour operations"; return false; }
		for (int j = 0; j < nd.nargs; j++) {
			if (nd.arg[j].kind == O_MASK) { g_why = "alpha reads a mask"; return false; }
			if (nd.arg[j].kind == O_NODE && (i == 0 || nd.arg[j].node != sc.anodes[i - 1])) { g_why = "alpha not a chain"; return false; }
		}
	}
#ifdef TEVSYM_HOST_DEBUG
	for (int i = 0; i < nNodes; i++) {
		printf("     n%d fn %d x%d:", i, nodes[i].fn, nodes[i].scale);
		for (int j = 0; j < nodes[i].nargs; j++) {
			const Opnd& a = nodes[i].arg[j];
			if (a.kind == O_NODE) printf(" n%d%s", a.node, a.inv ? "'" : "");
			else if (a.kind == O_SRC) printf(" atom%d%s", a.atom, a.inv ? "'" : "");
			else if (a.kind == O_MASK) printf(" mask");
			else printf(" %s(%.3f %.3f %.3f)", a.kind == O_K ? "k" : "bi", a.k[0], a.k[1], a.k[2]);
		}
		printf("\n");
	}
	printf("     temps %d, lerps %d\n", S.nTemp, C.nLerp);
#endif
	if (sc.nr > PICA_MAX_STAGES) { g_why = "more than 6 colour operations"; return false; }
	bool done[kMaxNodes] = {};
	sc.orders(0, done);
	if (!sc.found) { g_why = "no schedule"; return false; }
	emitProgram(nodes, sc.best, S.mask, unitOf, out);
	out.nUnits = nUnits;
	out.dropped = drop;
	return true;
}

// ---- Checking a translation against GX -------------------------------------------------------

struct Rng {
	uint32_t s;
	int next() {
		s = s * 1664525u + 1013904223u;
		return int(s >> 24);
	}
};

inline int clampI(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }

// One sample of the inputs: texture pairs' rgba, the vertex colour (rgb, a), the colour channels.
struct Sample {
	int tex[kTevSymPairs][4];
	int prim[4];
	int ras[2][4];
};

// A texel: 0 or 255 a quarter of the time each (masks and ramps hold many of both), else any.
int texel(Rng& r) {
	const int k = r.next();
	return k < 64 ? 0 : k < 128 ? 255 : r.next();
}

void makeSample(const TevSymIn& in, Rng& r, Sample& s) {
	for (int p = 0; p < in.nPairs; p++) {
		const bool grey = (r.next() & 1) != 0;  // intensity textures: all components equal
		for (int c = 0; c < 4; c++) s.tex[p][c] = in.pairConst[p] ? in.pairValue[p][c] : grey && c ? s.tex[p][0] : texel(r);
	}
	for (int c = 0; c < 4; c++) s.prim[c] = r.next();
	const int cr = in.rasRgbChan, ca = in.rasAChan;
	if (!in.rgbVarying[cr])
		for (int c = 0; c < 3; c++) s.prim[c] = in.rasConst[cr][c];
	if (!in.aVarying[ca]) s.prim[3] = in.rasConst[ca][3];
	for (int ch = 0; ch < 2; ch++) {
		for (int c = 0; c < 4; c++) s.ras[ch][c] = in.rasConst[ch][c];
		if (ch == cr)
			for (int c = 0; c < 3; c++) s.ras[ch][c] = s.prim[c];
		else if (in.rgbVarying[ch])
			for (int c = 0; c < 3; c++) s.ras[ch][c] = in.chanSpec[ch] ? 0 : r.next();
		if (ch == ca) s.ras[ch][3] = s.prim[3];
		else if (in.aVarying[ch]) s.ras[ch][3] = r.next();
	}
}

// GX's TEV on one sample (Dolphin's software renderer's arithmetic): rgba out.
void gxEval(const TevSymIn& in, const Sample& smp, int* outc) {
	int R[4][4];
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++) R[r][c] = in.reg[r][c];
	int texv[4] = {255, 255, 255, 255};
	for (int s = 0; s < in.numStages; s++) {
		const uint32_t ce = in.ce[s], ae = in.ae[s], chan = (in.order[s] >> 7) & 7;
		const int pr = in.stagePair[s];
		const uint32_t rs = in.swap[ae & 3], ts = in.swap[(ae >> 2) & 3];
		if (pr >= 0)
			for (int c = 0; c < 4; c++) texv[c] = smp.tex[pr][(ts >> (2 * c)) & 3];
		int ras[4] = {0, 0, 0, 0};
		if (chan <= 1)
			for (int c = 0; c < 4; c++) ras[c] = smp.ras[chan][(rs >> (2 * c)) & 3];
		const uint32_t kc = in.ksel[s] & 31, ka = (in.ksel[s] >> 5) & 31;
		int kcol[3], kal;
		for (int c = 0; c < 3; c++) kcol[c] = int(lroundf(konstC(in, kc, c) * 255));
		kal = int(lroundf(konstA(in, ka) * 255));
		auto cin = [&](uint32_t sel, int c) -> int {
			if (sel < 8) return (sel & 1) ? R[sel >> 1][3] : R[sel >> 1][c];
			switch (sel) {
			case 8: return texv[c];
			case 9: return texv[3];
			case 10: return ras[c];
			case 11: return ras[3];
			case 12: return 255;
			case 13: return 128;
			case 14: return kcol[c];
			default: return 0;
			}
		};
		auto ain = [&](uint32_t sel) -> int {
			if (sel < 4) return R[sel][3];
			switch (sel) {
			case 4: return texv[3];
			case 5: return ras[3];
			case 6: return kal;
			default: return 0;
			}
		};
		int res[4];
		static const int lsh[4] = {0, 1, 2, 0}, rsh[4] = {0, 0, 0, 1}, biasv[4] = {0, 128, -128, 0};
		auto combine = [&](int a, int b, int c, int d, uint32_t env) {
			const uint32_t bias = (env >> 16) & 3, sub = (env >> 18) & 1, scale = (env >> 20) & 3;
			a &= 255, b &= 255, c &= 255;
			const int c2 = c + (c >> 7);
			int lerp = a * (256 - c2) + b * c2;
			lerp <<= lsh[scale];
			lerp += scale == 3 ? 0 : sub ? 127 : 128;
			lerp >>= 8;
			if (sub) lerp = -lerp;
			int r = ((d + biasv[bias]) << lsh[scale]) + lerp;
			return r >> rsh[scale];
		};
		auto finish = [](int v, uint32_t env) {
			return (env >> 19) & 1 ? clampI(v, 0, 255) : clampI(v, -1024, 1023);
		};
		// colour
		{
			const uint32_t a = (ce >> 12) & 15, b = (ce >> 8) & 15, c = (ce >> 4) & 15, d = ce & 15;
			if (((ce >> 16) & 3) == 3) {
				const uint32_t mode = ((ce >> 20) & 3) << 1 | ((ce >> 18) & 1);
				for (int k = 0; k < 3; k++) {
					bool cond;
					if (mode >= 6) {
						const int x = cin(a, k) & 255, y = cin(b, k) & 255;
						cond = mode == 6 ? x > y : x == y;
					} else {
						int x = cin(a, 0) & 255, y = cin(b, 0) & 255;
						if (mode >= 2) x |= (cin(a, 1) & 255) << 8, y |= (cin(b, 1) & 255) << 8;
						if (mode >= 4) x |= (cin(a, 2) & 255) << 16, y |= (cin(b, 2) & 255) << 16;
						cond = mode & 1 ? x == y : x > y;
					}
					res[k] = finish(cin(d, k) + (cond ? (cin(c, k) & 255) : 0), ce);
				}
			} else {
				for (int k = 0; k < 3; k++) res[k] = finish(combine(cin(a, k), cin(b, k), cin(c, k), cin(d, k), ce), ce);
			}
		}
		// alpha
		{
			const uint32_t a = (ae >> 13) & 7, b = (ae >> 10) & 7, c = (ae >> 7) & 7, d = (ae >> 4) & 7;
			if (((ae >> 16) & 3) == 3) {
				const uint32_t mode = ((ae >> 20) & 3) << 1 | ((ae >> 18) & 1);
				bool cond;
				if (mode >= 6) {
					const int x = ain(a) & 255, y = ain(b) & 255;
					cond = mode == 6 ? x > y : x == y;
				} else {  // the colour inputs a and b, as GX compares them
					const uint32_t ca = (ce >> 12) & 15, cb = (ce >> 8) & 15;
					int x = cin(ca, 0) & 255, y = cin(cb, 0) & 255;
					if (mode >= 2) x |= (cin(ca, 1) & 255) << 8, y |= (cin(cb, 1) & 255) << 8;
					if (mode >= 4) x |= (cin(ca, 2) & 255) << 16, y |= (cin(cb, 2) & 255) << 16;
					cond = mode & 1 ? x == y : x > y;
				}
				res[3] = finish(ain(d) + (cond ? (ain(c) & 255) : 0), ae);
			} else {
				res[3] = finish(combine(ain(a), ain(b), ain(c), ain(d), ae), ae);
			}
		}
		const int dc = int((ce >> 22) & 3), da = int((ae >> 22) & 3);
		for (int k = 0; k < 3; k++) R[dc][k] = res[k];
		R[da][3] = res[3];
	}
	const int oc = int((in.ce[in.numStages - 1] >> 22) & 3), oa = int((in.ae[in.numStages - 1] >> 22) & 3);
	for (int k = 0; k < 3; k++) outc[k] = clampI(R[oc][k], 0, 255);
	outc[3] = clampI(R[oa][3], 0, 255);
}

// The PICA's combiners on one sample (8-bit, rounded): rgba out.
void picaEval(const TevSymIn& in, const TevSymOut& p, const Sample& smp, int* outc) {
	int prev[4] = {0, 0, 0, 0}, buf[4] = {0, 0, 0, 0};
	int next[4] = {int(p.buf_color & 0xFF), int((p.buf_color >> 8) & 0xFF), int((p.buf_color >> 16) & 0xFF), int(p.buf_color >> 24)};
	int unit[PICA_MAX_TEX][4];
	for (int u = 0; u < PICA_MAX_TEX; u++)
		for (int c = 0; c < 4; c++) unit[u][c] = u < p.nUnits && p.unitPair[u] >= 0 ? smp.tex[p.unitPair[u]][c] : 255;
	(void)in;
	for (int t = 0; t < p.nstages; t++) {
		const PicaTevStage& s = p.stage[t];
		const int kc[4] = {int(s.color & 0xFF), int((s.color >> 8) & 0xFF), int((s.color >> 16) & 0xFF), int(s.color >> 24)};
		auto source = [&](int code) -> const int* {
			static const int zero[4] = {0, 0, 0, 0};
			switch (code) {
			case 0x0: case 0x1: return smp.prim;
			case 0x3: case 0x4: case 0x5: return unit[code - 3];
			case 0xD: return buf;
			case 0xE: return kc;
			case 0xF: return prev;
			default: return zero;
			}
		};
		int out[4];
		for (int half = 0; half < 2; half++) {
			const uint16_t srcs = half ? s.src_alpha : s.src_rgb, ops = half ? s.op_alpha : s.op_rgb;
			const int fn = half ? s.func_alpha : s.func_rgb, sc = 1 << (half ? s.scale_alpha : s.scale_rgb);
			int v[3][3];
			for (int j = 0; j < 3; j++) {
				const int* x = source((srcs >> (4 * j)) & 15);
				const int o = (ops >> (4 * j)) & 15;
				for (int c = 0; c < 3; c++) {
					int y;
					if (!half) {
						const int sel = o >> 1;
						y = sel == 0 ? x[c] : sel == 1 ? x[3] : sel == 2 ? x[0] : sel == 4 ? x[1] : sel == 6 ? x[2] : x[c];
					} else {
						const int sel = (o >> 1) & 3;
						y = sel == 0 ? x[3] : x[sel - 1];
					}
					v[j][c] = (o & 1) ? 255 - y : y;
				}
			}
			for (int c = 0; c < (half ? 1 : 3); c++) {
				const int a = v[0][c], b = v[1][c], cc = v[2][c];
				int r;
				switch (fn) {
				case F_REPLACE: r = a; break;
				case F_MODULATE: r = (a * b + 127) / 255; break;
				case F_ADD: r = a + b; break;
				case F_ADD_SIGNED: r = a + b - 128; break;
				case F_INTERPOLATE: r = (a * cc + b * (255 - cc) + 127) / 255; break;
				case F_SUBTRACT: r = a - b; break;
				case F_MADD: r = (a * b + 255 * cc + 127) / 255; break;
				case 9: r = clampI(a + b, 0, 255) * cc / 255; break;
				default: r = a; break;
				}
				r = clampI(r, 0, 255) * sc;
				out[half ? 3 : c] = clampI(r, 0, 255);
			}
		}
		for (int c = 0; c < 4; c++) prev[c] = out[c];
		for (int c = 0; c < 4; c++) buf[c] = next[c];
		if (t < 4) {
			if ((p.buf_rgb >> t) & 1)
				for (int c = 0; c < 3; c++) next[c] = out[c];
			if ((p.buf_alpha >> t) & 1) next[3] = out[3];
		}
	}
	for (int c = 0; c < 4; c++) outc[c] = p.nstages ? prev[c] : smp.prim[c];
}

}  // namespace

int tevSymGxAlpha(const TevSymIn& in, const int (*tex)[4], const int* prim) {
	Sample s;
	memset(&s, 0, sizeof s);
	for (int p = 0; p < in.nPairs; p++)
		for (int c = 0; c < 4; c++) s.tex[p][c] = in.pairConst[p] ? in.pairValue[p][c] : tex[p][c];
	for (int c = 0; c < 4; c++) s.prim[c] = prim[c];
	for (int ch = 0; ch < 2; ch++)
		for (int c = 0; c < 4; c++) s.ras[ch][c] = in.rasConst[ch][c];
	for (int c = 0; c < 3; c++) s.ras[in.rasRgbChan][c] = prim[c];
	s.ras[in.rasAChan][3] = prim[3];
	int out[4];
	gxEval(in, s, out);
	return out[3];
}

int tevSymError(const TevSymIn& in, const TevSymOut& out) {
	Rng r{12345u};
	const int N = 64;
	int sum[4] = {0, 0, 0, 0}, worst[4] = {0, 0, 0, 0};
	for (int i = 0; i < N; i++) {
		Sample s;
		makeSample(in, r, s);
		int g[4], p[4];
		gxEval(in, s, g);
		picaEval(in, out, s, p);
		for (int c = 0; c < 4; c++) {
			const int d = g[c] > p[c] ? g[c] - p[c] : p[c] - g[c];
			sum[c] += d;
			if (d > worst[c]) worst[c] = d;
		}
	}
	float e = 0;
	for (int c = 0; c < 4; c++) e += (float(sum[c]) / N + 0.25f * float(worst[c])) / 255.0f;
	return int(lroundf(e * 1000));
}

namespace {
const Variant kVariants[] = {
    {0, 3, true, true}, {1, 3, true, true}, {2, 3, true, true}, {0, 3, false, true},
    {0, 2, true, true}, {1, 2, true, true}, {2, 2, true, true}, {0, 3, true, false},
};
const int kNumVariants = int(sizeof kVariants / sizeof kVariants[0]);
}  // namespace

bool tevSymCompile(const TevSymIn& in, TevSymOut& out, int* errOut, int* choiceOut) {
	int nTex = 0;
	for (int p = 0; p < in.nPairs; p++) nTex += !in.pairConst[p];
	// past three textures, each one in turn read as its average, but for those a compare reads
	// (read as a constant, a mask would pick one side everywhere)
	int drops[kTevSymPairs + 1], nd = 0;
	if (nTex <= PICA_MAX_TEX) {
		drops[nd++] = -1;
	} else {
		static Sym S;
		bool masked[kTevSymPairs] = {};
		if (symEval(in, -1, S))
			for (int m = 0; m < S.nMask; m++)
				if (S.mask[m].src >= AT_TEXA && S.mask[m].src < AT_V) masked[S.mask[m].src - AT_TEXA] = true;
		for (int p = 0; p < in.nPairs; p++)
			if (!in.pairConst[p] && !masked[p]) drops[nd++] = p;
	}
	bool found = false;
	int best = 1 << 30, choice = -1;
	TevSymOut cand;
	for (int d = 0; d < nd; d++)
		for (int vi = 0; vi < kNumVariants; vi++) {
			const Variant& v = kVariants[vi];
			memset(&cand, 0, sizeof cand);
			if (!compileVariant(in, drops[d], v, cand)) {
#ifdef TEVSYM_HOST_DEBUG
				printf("   drop %d variant %d/%d/%d/%d: failed %s\n", drops[d], v.lerpCse, v.maskOps, v.scaled, v.useBi, g_why);
#endif
				continue;
			}
			const int e = tevSymError(in, cand);
#ifdef TEVSYM_HOST_DEBUG
			printf("   drop %d variant %d/%d/%d/%d: %d stages, error %d\n", drops[d], v.lerpCse, v.maskOps, v.scaled, v.useBi, cand.nstages, e);
#endif
			if (!found || e < best || (e == best && cand.nstages < out.nstages)) {
				out = cand;
				best = e;
				choice = (drops[d] + 1) * 16 + vi;
				found = true;
			}
		}
	if (errOut) *errOut = best;
	if (choiceOut) *choiceOut = choice;
	return found;
}

bool tevSymCompileChoice(const TevSymIn& in, int choice, TevSymOut& out) {
	const int drop = choice / 16 - 1, vi = choice % 16;
	if (choice < 0 || vi >= kNumVariants || drop >= in.nPairs) return false;
	memset(&out, 0, sizeof out);
	return compileVariant(in, drop, kVariants[vi], out);
}
