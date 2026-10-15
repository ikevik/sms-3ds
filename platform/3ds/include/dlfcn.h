/* dlfcn.h for the 3DS. devkitARM's newlib has no dynamic loading; the PC port only
 * looks up optional SDL entry points by name, so dlsym can simply answer NULL
 * (platform/3ds/posix_shims.cpp). Adapted from chasem-dev/sms-switch (MIT).
 * Uses the toolchain's header when it has one. */
#ifndef SMS_3DS_DLFCN_H
#define SMS_3DS_DLFCN_H
#if !defined(SMS_3DS_FORCE_SHIMS) && defined(__has_include_next)
#if __has_include_next(<dlfcn.h>)
#include_next <dlfcn.h>
#define SMS_3DS_HAVE_REAL_DLFCN 1
#endif
#endif
#ifndef SMS_3DS_HAVE_REAL_DLFCN
#ifdef __cplusplus
extern "C" {
#endif
#define RTLD_LAZY 1
#define RTLD_NOW 2
#define RTLD_GLOBAL 0x100
#define RTLD_LOCAL 0
#define RTLD_DEFAULT ((void*)0)
void* dlopen(const char* name, int flags);
void* dlsym(void* handle, const char* name);
int dlclose(void* handle);
char* dlerror(void);
typedef struct {
	const char* dli_fname;
	void* dli_fbase;
	const char* dli_sname;
	void* dli_saddr;
} Dl_info;
int dladdr(const void* addr, Dl_info* info);
#ifdef __cplusplus
}
#endif
#endif
#endif
