/* citro3d's uniforms.o (libcitro3d.a, devkitPro) in our own build, with a faster dirty scan.
 *
 * citro3d's draw calls run C3D_UpdateUniforms on every draw, and it scanned the 96 float-uniform
 * dirty flags one byte at a time: in Azahar that scan was ~9% of the render thread (the draws set
 * a handful of the 96). This file defines every symbol uniforms.o does, so the linker never takes
 * the library's: the same arrays (sizes and alignment as the library's), C3D_UpdateUniforms
 * sending the same GPU commands in the same order, and the internal functions citro3d's other
 * objects call (C3Di_DirtyUniforms, C3Di_LoadShaderUniforms, C3Di_ClearShaderUniforms), each
 * doing what the library's disassembly does. Only the scan changed: four flags a word, runs of
 * dirty flags found as before. It is linked as a plain object (platform/3ds/pica is a source
 * directory of the app), ahead of -lcitro3d. */
#include <3ds.h>
#include <citro3d.h>
#include <string.h>

C3D_FVec C3D_FVUnif[2][C3D_FVUNIF_COUNT];
C3D_IVec C3D_IVUnif[2][C3D_IVUNIF_COUNT];
u16 C3D_BoolUnifs[2];

/* Word-aligned so the scan can read four flags at once (96 = 24 words a shader type). */
bool C3D_FVUnifDirty[2][C3D_FVUNIF_COUNT] __attribute__((aligned(4)));
bool C3D_IVUnifDirty[2][C3D_IVUNIF_COUNT] __attribute__((aligned(4)));
bool C3D_BoolUnifsDirty[2] __attribute__((aligned(4)));

/* The library's statics: the shader's constant float uniforms, and the uniforms ever uploaded (set
 * dirty again when a program is bound: C3Di_DirtyUniforms). */
static struct {
	bool dirty;
	int count;
	float24Uniform_s* data;
} s_shaderFVec[2];
static bool s_fvEverDirty[2][C3D_FVUNIF_COUNT] __attribute__((aligned(4)));
static bool s_ivEverDirty[2][C3D_IVUNIF_COUNT] __attribute__((aligned(4)));

void C3Di_DirtyUniforms(GPU_SHADER_TYPE type);
void C3Di_LoadShaderUniforms(shaderInstance_s* si);
void C3Di_ClearShaderUniforms(GPU_SHADER_TYPE type);

void C3D_UpdateUniforms(GPU_SHADER_TYPE type)
{
	const int offset = type == GPU_GEOMETRY_SHADER ? (GPUREG_GSH_BOOLUNIFORM - GPUREG_VSH_BOOLUNIFORM) : 0;
	int i;

	/* The float uniforms that come from the shader's constants. */
	if (s_shaderFVec[type].dirty) {
		for (i = 0; i < s_shaderFVec[type].count; i++) {
			float24Uniform_s* u = &s_shaderFVec[type].data[i];
			GPUCMD_AddIncrementalWrites(GPUREG_VSH_FLOATUNIFORM_CONFIG + offset, (u32*)u, 4);
			C3D_FVUnifDirty[type][u->id] = false;
		}
		s_shaderFVec[type].dirty = false;
	}

	/* The float uniforms: each run of consecutive dirty ones in one upload, as the library does. */
	bool* dirty = C3D_FVUnifDirty[type];
	const u32* words = (const u32*)dirty;
	i = 0;
	while (i < C3D_FVUNIF_COUNT) {
		if ((i & 3) == 0 && words[i >> 2] == 0) {
			i += 4;
			continue;
		}
		if (!dirty[i]) {
			i++;
			continue;
		}
		int j = i;
		while (j < C3D_FVUNIF_COUNT && dirty[j]) j++;
		GPUCMD_AddWrite(GPUREG_VSH_FLOATUNIFORM_CONFIG + offset, 0x80000000 | i);
		GPUCMD_AddWrites(GPUREG_VSH_FLOATUNIFORM_DATA + offset, (u32*)&C3D_FVUnif[type][i], (j - i) * 4);
		for (int k = i; k < j; k++) {
			dirty[k] = false;
			s_fvEverDirty[type][k] = true;
		}
		i = j;
	}

	/* The integer uniforms. */
	for (i = 0; i < C3D_IVUNIF_COUNT; i++) {
		if (!C3D_IVUnifDirty[type][i]) continue;
		GPUCMD_AddWrite(GPUREG_VSH_INTUNIFORM_I0 + offset + i, C3D_IVUnif[type][i]);
		C3D_IVUnifDirty[type][i] = false;
		s_ivEverDirty[type][i] = false; /* (as the library: cleared, not set) */
	}

	/* The bool uniforms. */
	if (C3D_BoolUnifsDirty[type]) {
		GPUCMD_AddWrite(GPUREG_VSH_BOOLUNIFORM + offset, 0x7FFF0000 | C3D_BoolUnifs[type]);
		C3D_BoolUnifsDirty[type] = false;
	}
}

void C3Di_DirtyUniforms(GPU_SHADER_TYPE type)
{
	int i;
	C3D_BoolUnifsDirty[type] = true;
	if (s_shaderFVec[type].count) s_shaderFVec[type].dirty = true;
	for (i = 0; i < C3D_FVUNIF_COUNT; i++) C3D_FVUnifDirty[type][i] = C3D_FVUnifDirty[type][i] || s_fvEverDirty[type][i];
	for (i = 0; i < C3D_IVUNIF_COUNT; i++) C3D_IVUnifDirty[type][i] = C3D_IVUnifDirty[type][i] || s_ivEverDirty[type][i];
}

void C3Di_LoadShaderUniforms(shaderInstance_s* si)
{
	/* The DVLE's type is its first byte (read as the library reads it). */
	const GPU_SHADER_TYPE type = (GPU_SHADER_TYPE)(*(const u8*)si->dvle);
	if (si->boolUniformMask) {
		C3D_BoolUnifs[type] &= ~si->boolUniformMask;
		C3D_BoolUnifs[type] |= si->boolUniforms;
	}
	if (type == GPU_GEOMETRY_SHADER) C3D_BoolUnifs[type] &= ~BIT(15);
	C3D_BoolUnifsDirty[type] = true;
	if (si->intUniformMask) {
		for (int i = 0; i < 4; i++)
			if (si->intUniformMask & BIT(i)) {
				C3D_IVUnif[type][i] = si->intUniforms[i];
				C3D_IVUnifDirty[type][i] = true;
			}
	}
	s_shaderFVec[type].count = si->numFloat24Uniforms;
	s_shaderFVec[type].data = si->float24Uniforms;
	s_shaderFVec[type].dirty = true;
}

void C3Di_ClearShaderUniforms(GPU_SHADER_TYPE type)
{
	s_shaderFVec[type].dirty = false;
	s_shaderFVec[type].count = 0;
	s_shaderFVec[type].data = NULL;
}
