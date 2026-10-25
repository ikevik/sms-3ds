// TEV setups compiled to PICA combiners through their algebra (gx3ds_tevsym.cpp), for those the
// stage-by-stage translator (gx3ds_backend.cpp, translateTev) can only approximate, and a check
// of any translation against GX's own arithmetic on sampled inputs.
//
// The GX stages are evaluated symbolically: each register becomes a polynomial over the
// textures, the vertex colour and compare masks (a texture's alpha above or below a constant),
// with a clamp kept only where the value's range leaves 0..1. The result is split at the mask
// (mask ? A : B), its sums are built in a scaled domain (y = s * x + b, so that no partial sum
// clamps before the result does), and the operations are scheduled into the six combiner stages
// with PREVIOUS and the combiner buffer (a stage's output shows there two stages later).
#ifndef SMS3DS_GX3DS_TEVSYM_H
#define SMS3DS_GX3DS_TEVSYM_H
#include <stdint.h>
#include "../pica/pica_draw.h"

static const int kTevSymPairs = 8;

struct TevSymIn {
	int numStages;
	uint32_t ce[16], ae[16], order[16], ksel[16];
	uint8_t swap[4];             // swap tables: r | g << 2 | b << 4 | a << 6
	int16_t reg[4][4];           // PREV, C0-C2 as the draw starts: r, g, b, a (11-bit signed)
	uint8_t konst[4][4];         // K0-K3: r, g, b, a
	int stagePair[16];           // the texture pair each stage samples (-1: none)
	int nPairs;
	bool pairConst[kTevSymPairs];        // no texture: the pair reads pairValue
	int16_t pairValue[kTevSymPairs][4];  // that colour, else the texture's average (when one is left out)
	int rasRgbChan, rasAChan;            // the colour channels the vertex colour's rgb and alpha carry
	bool rgbVarying[2], aVarying[2], chanSpec[2];
	int16_t rasConst[2][4];              // a channel's colour when it does not vary
};

struct TevSymOut {
	PicaTevStage stage[PICA_MAX_STAGES];
	int nstages;
	uint8_t buf_rgb, buf_alpha;
	uint32_t buf_color;
	int nUnits, unitPair[PICA_MAX_TEX];
	int dropped;  // the pair read as its average colour (-1: none)
};

// The setup compiled: the most accurate of the variants tried, with its error (as tevSymError)
// and which variant it was (for tevSymCompileChoice). False when none can be compiled.
bool tevSymCompile(const TevSymIn& in, TevSymOut& out, int* err, int* choice);

// The setup compiled with the variant an earlier tevSymCompile chose (for another setup with the
// same stages and inputs, as when only the constants change from frame to frame).
bool tevSymCompileChoice(const TevSymIn& in, int choice, TevSymOut& out);

// GX's TEV on one input: tex[pair] rgba and the vertex colour prim (rgba), as the channel model
// gives it; the output alpha.
int tevSymGxAlpha(const TevSymIn& in, const int (*tex)[4], const int* prim);

// A translation's error against GX on sampled textures and vertex colours: per component the mean
// plus a quarter of the largest difference, summed over rgb and alpha, in thousandths of 1.
int tevSymError(const TevSymIn& in, const TevSymOut& out);

#endif
