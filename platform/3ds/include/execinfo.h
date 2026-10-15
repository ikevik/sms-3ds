/* execinfo.h for the 3DS (backtrace is a stub in platform/3ds/posix_shims.cpp).
 * Adapted from chasem-dev/sms-switch (MIT). Uses the toolchain's header when present. */
#ifndef SMS_3DS_EXECINFO_H
#define SMS_3DS_EXECINFO_H
#if !defined(SMS_3DS_FORCE_SHIMS) && defined(__has_include_next)
#if __has_include_next(<execinfo.h>)
#include_next <execinfo.h>
#define SMS_3DS_HAVE_REAL_EXECINFO 1
#endif
#endif
#ifndef SMS_3DS_HAVE_REAL_EXECINFO
#ifdef __cplusplus
extern "C" {
#endif
int backtrace(void** frames, int capacity);
char** backtrace_symbols(void* const* frames, int count);
void backtrace_symbols_fd(void* const* frames, int count, int fd);
#ifdef __cplusplus
}
#endif
#endif
#endif
