/* sys/utsname.h for the 3DS (uname is in platform/3ds/posix_shims.cpp).
 * Adapted from chasem-dev/sms-switch (MIT). Uses the toolchain's header when present. */
#ifndef SMS_3DS_SYS_UTSNAME_H
#define SMS_3DS_SYS_UTSNAME_H
#if !defined(SMS_3DS_FORCE_SHIMS) && defined(__has_include_next)
#if __has_include_next(<sys/utsname.h>)
#include_next <sys/utsname.h>
#define SMS_3DS_HAVE_REAL_UTSNAME 1
#endif
#endif
#ifndef SMS_3DS_HAVE_REAL_UTSNAME
#ifdef __cplusplus
extern "C" {
#endif
struct utsname {
	char sysname[65], nodename[65], release[65], version[65], machine[65];
};
int uname(struct utsname* name);
#ifdef __cplusplus
}
#endif
#endif
#endif
