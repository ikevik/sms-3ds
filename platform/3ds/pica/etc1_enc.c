/* ETC1 and ETC1A4 encoding for PICA textures (pica_draw.c: PICA_TF_ETC1, PICA_TF_ETC1A4).
 *
 * Layout on the PICA: 8x8 tiles in memory order from the image's bottom (as the other formats),
 * each holding four 4x4 ETC1 blocks (left-bottom, right-bottom, left-top, right-top in image
 * terms: the tile's first block row is its lower one), each block's 64-bit word stored
 * little-endian; ETC1A4 puts a 64-bit word of 4-bit alpha before each block. Inside a block,
 * texel (x, y) counts y from the block's first row as stored, which is the lower one in the
 * image, and is texel x * 4 + y of the block's index and alpha bits.
 *
 * Encoding: for each block both splits into two 4x2 or 2x4 halves are tried. A half's base colour
 * is its average, quantised as a 5-bit pair with a 3-bit difference when the two halves are close
 * enough, else as two 4-bit colours. ETC1 adds the same offset to red, green and blue, so for
 * a texel the best offset is the one nearest to the mean of its three differences from the base
 * (exact while no channel clips); each half's table is the one with the least such error, and
 * the split with the smaller error over all three channels (clipped as decoded) is kept. With
 * alpha (ETC1A4), texels under half alpha do not count towards a half's colour. */
#include <stdint.h>
#include <string.h>

static const int kMod[8][2] = {{2, 8}, {5, 17}, {9, 29}, {13, 42}, {18, 60}, {24, 80}, {33, 106}, {47, 183}};

static inline int clamp255(int v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

/* One half: its texels (block index i = x * 4 + y), the base colour (decoded), and the table and
 * texel indices it gets; returns the error over all three channels. */
static uint32_t half_fit(const uint8_t (*px)[4], const uint8_t* use, const int* idx, int n, const int base[3], int* table,
                         uint32_t* msb, uint32_t* lsb)
{
	int e[8];
	const int bsum = base[0] + base[1] + base[2];
	for (int k = 0; k < n; k++) {
		const uint8_t* p = px[idx[k]];
		e[k] = p[0] + p[1] + p[2] - bsum; /* 3 x the best offset for this texel */
	}
	uint32_t best = 0xFFFFFFFFu;
	int bt = 0;
	for (int t = 0; t < 8; t++) {
		const int a = 3 * kMod[t][0], b = 3 * kMod[t][1];
		uint32_t err = 0;
		for (int k = 0; k < n; k++) {
			if (!use[idx[k]]) continue;
			const int v = e[k] < 0 ? -e[k] : e[k];
			const int da = v - a, db = v - b;
			const uint32_t ea = (uint32_t)(da * da), eb = (uint32_t)(db * db);
			err += ea < eb ? ea : eb;
		}
		if (err < best) {
			best = err;
			bt = t;
		}
	}
	*table = bt;
	uint32_t err = 0;
	for (int k = 0; k < n; k++) {
		const int v = e[k] < 0 ? -e[k] : e[k];
		const int a = kMod[bt][0], b = kMod[bt][1];
		const int big = (v - 3 * a) * (v - 3 * a) > (v - 3 * b) * (v - 3 * b);
		const int neg = e[k] < 0;
		const int m = (neg ? -1 : 1) * (big ? b : a);
		/* index: 0 +a, 1 +b, 2 -a, 3 -b (msb: negative, lsb: the larger one) */
		const int i = idx[k];
		if (neg) *msb |= 1u << i;
		if (big) *lsb |= 1u << i;
		if (!use[i]) continue;
		const uint8_t* p = px[i];
		for (int c = 0; c < 3; c++) {
			const int d = p[c] - clamp255(base[c] + m);
			err += (uint32_t)(d * d);
		}
	}
	return err;
}

/* One block: px[x * 4 + y] RGBA (y from the block's first row as stored), use[] the texels that
 * count. */
static uint64_t encode_block(const uint8_t (*px)[4], const uint8_t* use, uint32_t* errOut)
{
	uint64_t bestWord = 0;
	uint32_t bestErr = 0xFFFFFFFFu;
	for (int flip = 0; flip < 2; flip++) {
		int idx[2][8], n[2] = {0, 0};
		for (int x = 0; x < 4; x++)
			for (int y = 0; y < 4; y++) {
				const int h = flip ? (y >= 2) : (x >= 2);
				idx[h][n[h]++] = x * 4 + y;
			}
		int avg[2][3];
		for (int h = 0; h < 2; h++) {
			int sum[3] = {0, 0, 0}, cnt = 0;
			for (int k = 0; k < 8; k++)
				if (use[idx[h][k]]) {
					for (int c = 0; c < 3; c++) sum[c] += px[idx[h][k]][c];
					cnt++;
				}
			for (int c = 0; c < 3; c++) avg[h][c] = cnt ? (sum[c] + cnt / 2) / cnt : 0;
		}
		int q5[2][3], q4[2][3];
		int diffOk = 1;
		for (int h = 0; h < 2; h++)
			for (int c = 0; c < 3; c++) {
				q5[h][c] = (avg[h][c] * 31 + 127) / 255;
				q4[h][c] = (avg[h][c] * 15 + 127) / 255;
			}
		for (int c = 0; c < 3; c++) {
			const int d = q5[1][c] - q5[0][c];
			if (d < -4 || d > 3) diffOk = 0;
		}
		int base[2][3];
		for (int h = 0; h < 2; h++)
			for (int c = 0; c < 3; c++)
				base[h][c] = diffOk ? (q5[h][c] << 3 | q5[h][c] >> 2) : (q4[h][c] << 4 | q4[h][c]);
		uint32_t msb = 0, lsb = 0;
		int table[2];
		const uint32_t err = half_fit(px, use, idx[0], 8, base[0], &table[0], &msb, &lsb) +
		                     half_fit(px, use, idx[1], 8, base[1], &table[1], &msb, &lsb);
		if (err >= bestErr) continue;
		bestErr = err;
		uint32_t hi;
		if (diffOk)
			hi = (uint32_t)q5[0][0] << 27 | (uint32_t)((q5[1][0] - q5[0][0]) & 7) << 24 | (uint32_t)q5[0][1] << 19 |
			     (uint32_t)((q5[1][1] - q5[0][1]) & 7) << 16 | (uint32_t)q5[0][2] << 11 |
			     (uint32_t)((q5[1][2] - q5[0][2]) & 7) << 8 | 1u << 1;
		else
			hi = (uint32_t)q4[0][0] << 28 | (uint32_t)q4[1][0] << 24 | (uint32_t)q4[0][1] << 20 | (uint32_t)q4[1][1] << 16 |
			     (uint32_t)q4[0][2] << 12 | (uint32_t)q4[1][2] << 8;
		hi |= (uint32_t)table[0] << 5 | (uint32_t)table[1] << 2 | (uint32_t)flip;
		bestWord = (uint64_t)hi << 32 | (uint64_t)(msb << 16 | (lsb & 0xFFFF));
	}
	*errOut = bestErr;
	return bestWord;
}

/* dst: the texture's level in the PICA's layout; rgba: w x h RGBA8 from the image's top row (as
 * pica_tex_upload takes it); alpha: ETC1A4 (a 4-bit alpha word before each block). w and h are
 * multiples of 8. Returns the colour error: the sum of squared differences over red, green and
 * blue of the texels that count. */
uint64_t etc1_encode_tex(void* dst, uint32_t w, uint32_t h, const uint8_t* rgba, int alpha)
{
	uint64_t total = 0;
	uint64_t* out = (uint64_t*)dst;
	uint8_t px[16][4], use[16];
	for (uint32_t ty = 0; ty < h; ty += 8)
		for (uint32_t tx = 0; tx < w; tx += 8)
			for (int b = 0; b < 4; b++) {
				const uint32_t bx = tx + (b & 1) * 4, by = ty + (b >> 1) * 4; /* by: rows from the image's bottom */
				uint64_t aw = 0;
				for (int x = 0; x < 4; x++)
					for (int y = 0; y < 4; y++) {
						const uint8_t* p = rgba + ((size_t)(h - 1 - (by + y)) * w + bx + x) * 4;
						const int i = x * 4 + y;
						memcpy(px[i], p, 4);
						use[i] = !alpha || p[3] >= 128;
						if (alpha) aw |= (uint64_t)((p[3] * 15 + 127) / 255) << (4 * i);
					}
				int any = 0;
				for (int i = 0; i < 16; i++) any |= use[i];
				if (!any) memset(use, 1, sizeof use); /* all transparent: any colour */
				if (alpha) *out++ = aw;
				uint32_t err;
				*out++ = encode_block((const uint8_t (*)[4])px, use, &err);
				total += err;
			}
	return total;
}
