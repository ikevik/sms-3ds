// GameCube disc (GCM/ISO) reader over the 3DS SD card.
// Counterpart of the "disc" module in sms-pc-port's platform/ layer: the game
// reads all its assets from the user's own GMSE01 image at run time.
#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GCM_MAGIC_OFFSET 0x1C
#define GCM_MAGIC        0xC2339F3Du
#define GCM_FST_OFF_PTR  0x424
#define GCM_FST_SZ_PTR   0x428

typedef struct {
	uint32_t offset;   // byte offset inside the image
	uint32_t length;   // file size in bytes
} GcmFile;

// Opens the image, validates the GCM magic and loads the FST into memory.
bool gcm_open(const char* path);
void gcm_close(void);

// Game ID such as "GMSE01" (7 bytes incl. NUL) and disc revision.
const char* gcm_game_id(void);
uint8_t     gcm_revision(void);

// Looks up a path like "/game/mario.dol" or "/scene/dolpic0.szs" (case-insensitive).
bool gcm_find(const char* path, GcmFile* out);

// The boot DOL is NOT in the FST on GameCube discs; its offset is stored in the
// disc header (0x420) and its size is derived from the DOL's own section table.
bool gcm_boot_dol(GcmFile* out);

// Raw reads from the image (what DVDReadAsync would map to).
bool gcm_read(void* dst, uint32_t len, uint32_t image_offset);

#ifdef __cplusplus
}
#endif
