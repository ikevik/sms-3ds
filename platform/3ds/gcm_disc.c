#include "gcm_disc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static FILE*    s_fp;
static uint8_t* s_fst;
static uint32_t s_fstSize;
static char     s_id[8];
static uint8_t  s_rev;

static uint32_t be32(const uint8_t* p)
{
	return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

bool gcm_read(void* dst, uint32_t len, uint32_t image_offset)
{
	if (!s_fp) return false;
	if (fseek(s_fp, (long)image_offset, SEEK_SET) != 0) return false;
	return fread(dst, 1, len, s_fp) == len;
}

bool gcm_open(const char* path)
{
	gcm_close();
	s_fp = fopen(path, "rb");
	if (!s_fp) return false;

	uint8_t hdr[0x440];
	if (fread(hdr, 1, sizeof(hdr), s_fp) != sizeof(hdr)) { gcm_close(); return false; }
	if (be32(hdr + GCM_MAGIC_OFFSET) != GCM_MAGIC)       { gcm_close(); return false; }

	memcpy(s_id, hdr, 6);
	s_id[6] = 0;
	s_rev   = hdr[7];

	uint32_t fstOff = be32(hdr + GCM_FST_OFF_PTR);
	s_fstSize       = be32(hdr + GCM_FST_SZ_PTR);
	if (s_fstSize < 12 || s_fstSize > (4u << 20)) { gcm_close(); return false; }

	s_fst = (uint8_t*)malloc(s_fstSize);
	if (!s_fst || !gcm_read(s_fst, s_fstSize, fstOff)) { gcm_close(); return false; }
	return true;
}

void gcm_close(void)
{
	if (s_fp)  { fclose(s_fp); s_fp = NULL; }
	if (s_fst) { free(s_fst);  s_fst = NULL; }
	s_fstSize = 0;
	s_id[0] = 0;
	s_rev = 0;
}

const char* gcm_game_id(void)  { return s_id; }
uint8_t     gcm_revision(void) { return s_rev; }

bool gcm_boot_dol(GcmFile* out)
{
	if (!s_fp) return false;
	uint8_t p[4];
	if (!gcm_read(p, 4, 0x420)) return false;
	uint32_t dolOff = be32(p);
	if (dolOff == 0) return false;

	// DOL header: 18 section offsets at 0x00, 18 section sizes at 0x90.
	uint8_t hdr[0x100];
	if (!gcm_read(hdr, sizeof(hdr), dolOff)) return false;
	uint32_t end = 0x100;
	for (int i = 0; i < 18; i++) {
		uint32_t off = be32(hdr + i * 4);
		uint32_t len = be32(hdr + 0x90 + i * 4);
		if (off && len && off + len > end) end = off + len;
	}
	out->offset = dolOff;
	out->length = end;
	return true;
}

// FST entry (12 bytes): [flags:8 | nameOff:24][offset/parent:32][length/nextIdx:32]
static const char* fst_name(uint32_t idx, uint32_t count)
{
	const uint8_t* e = s_fst + idx * 12;
	uint32_t nameOff = be32(e) & 0x00FFFFFF;
	return (const char*)(s_fst + count * 12 + nameOff);
}

static int ieq(const char* a, size_t alen, const char* b)
{
	if (strlen(b) != alen) return 0;
	for (size_t i = 0; i < alen; i++)
		if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i])) return 0;
	return 1;
}

bool gcm_find(const char* path, GcmFile* out)
{
	if (!s_fst) return false;
	uint32_t count = be32(s_fst + 8);          // root entry's "length" = total entries
	uint32_t cur = 0;                           // start at root dir (index 0)
	uint32_t curEnd = count;

	if (*path == '/') path++;
	while (*path) {
		const char* seg = path;
		while (*path && *path != '/') path++;
		size_t segLen = (size_t)(path - seg);
		bool last = (*path == 0);
		if (*path == '/') path++;

		bool found = false;
		for (uint32_t i = cur + 1; i < curEnd; ) {
			const uint8_t* e = s_fst + i * 12;
			bool isDir = e[0] != 0;
			const char* nm = fst_name(i, count);
			if (ieq(seg, segLen, nm)) {
				if (isDir && !last) { cur = i; curEnd = be32(e + 8); found = true; break; }
				if (!isDir && last) {
					out->offset = be32(e + 4);
					out->length = be32(e + 8);
					return true;
				}
			}
			i = isDir ? be32(e + 8) : i + 1;   // skip over a directory's children
		}
		if (!found) return false;
	}
	return false;
}
