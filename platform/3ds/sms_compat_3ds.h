/* Force-included (-include) into every sms-pc-port platform unit built for the 3DS.
 * Covers POSIX bits newlib on devkitARM lacks, without touching the PC port's sources. */
#ifndef SMS_COMPAT_3DS_H
#define SMS_COMPAT_3DS_H
#ifdef __cplusplus
#include <cerrno>
#include <ctime>
#else
#include <errno.h>
#include <time.h>
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#ifndef TIMER_ABSTIME
#define TIMER_ABSTIME 1
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern volatile int sms3ds_ending;  /* pad_3ds.c: the app is ending (its threads park) */
void sms3ds_park(void);             /* pad_3ds.c: the thread waits until the process ends */
extern volatile int sms3ds_suspended;  /* pad_3ds.c: the HOME Menu is up */
void sms3ds_wait_resumed(void);        /* pad_3ds.c: waits until it is gone */
#ifdef __cplusplus
}
#endif

/* port_host.h sleeps to an absolute deadline with clock_nanosleep (newlib has none).
 * Renamed through a macro so it can never clash with a declaration in the toolchain.
 * Its callers (the VI retrace timer, AI's host clock) wake up on their own: while the HOME Menu is
 * up they wait for it to go, and once the app is ending they park instead (sms3ds_park). */
#define clock_nanosleep sms3ds_clock_nanosleep
static inline int sms3ds_clock_nanosleep(int clock, int flags, const struct timespec* req, struct timespec* rem)
{
	struct timespec now, left = *req;
	(void)rem;
	if (sms3ds_suspended)
		sms3ds_wait_resumed();
	if (sms3ds_ending)
		sms3ds_park();
	if (flags & TIMER_ABSTIME) {
		if (clock_gettime((clockid_t)clock, &now) != 0)
			return -1;
		left.tv_sec = req->tv_sec - now.tv_sec;
		left.tv_nsec = req->tv_nsec - now.tv_nsec;
		if (left.tv_nsec < 0) {
			left.tv_sec -= 1;
			left.tv_nsec += 1000000000L;
		}
		if (left.tv_sec < 0)
			return 0;
	}
	while (nanosleep(&left, &left) != 0) {
		if (errno != EINTR)
			return errno;
	}
	if (sms3ds_ending)
		sms3ds_park();
	return 0;
}
#endif
