/* LD_PRELOAD shim for the Gate 0 CPU prebuild: refuse every open of a GPU device node.
 *
 * Any open/openat (and their 64-bit and fortified variants) of a path beginning
 * with "/dev/nvidia" fails with EACCES before reaching the kernel, so the CUDA
 * driver cannot initialize a device in this process or its children (LD_PRELOAD
 * is inherited). Each refusal is appended to $NOGPU_LOG ("pid path"). Every
 * other path is passed to the real libc function unchanged.
 *
 *   gcc -shared -fPIC -O2 -o nogpu_shim.so nogpu_shim.c -ldl
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int refused(const char *path) {
    if (path == NULL || strncmp(path, "/dev/nvidia", 11) != 0) return 0;
    const char *log = getenv("NOGPU_LOG");
    if (log != NULL) {
        int (*real_open)(const char *, int, ...) = dlsym(RTLD_NEXT, "open");
        int fd = real_open(log, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
        if (fd >= 0) {
            char line[512];
            int n = snprintf(line, sizeof line, "%d %s\n", (int)getpid(), path);
            if (n > 0) (void)!write(fd, line, (size_t)n);
            close(fd);
        }
    }
    errno = EACCES;
    return 1;
}

#define MODE_ARG(flags) \
    mode_t mode = 0; \
    if ((flags) & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); }

#define WRAP_OPEN(name) \
    int name(const char *path, int flags, ...) { \
        MODE_ARG(flags) \
        if (refused(path)) return -1; \
        int (*real)(const char *, int, ...) = dlsym(RTLD_NEXT, #name); \
        return real(path, flags, mode); \
    }

#define WRAP_OPENAT(name) \
    int name(int dirfd, const char *path, int flags, ...) { \
        MODE_ARG(flags) \
        if (refused(path)) return -1; \
        int (*real)(int, const char *, int, ...) = dlsym(RTLD_NEXT, #name); \
        return real(dirfd, path, flags, mode); \
    }

WRAP_OPEN(open)
WRAP_OPEN(open64)
WRAP_OPENAT(openat)
WRAP_OPENAT(openat64)

int __open_2(const char *path, int flags) {
    if (refused(path)) return -1;
    int (*real)(const char *, int) = dlsym(RTLD_NEXT, "__open_2");
    return real(path, flags);
}
int __open64_2(const char *path, int flags) {
    if (refused(path)) return -1;
    int (*real)(const char *, int) = dlsym(RTLD_NEXT, "__open64_2");
    return real(path, flags);
}
int __openat_2(int dirfd, const char *path, int flags) {
    if (refused(path)) return -1;
    int (*real)(int, const char *, int) = dlsym(RTLD_NEXT, "__openat_2");
    return real(dirfd, path, flags);
}
int __openat64_2(int dirfd, const char *path, int flags) {
    if (refused(path)) return -1;
    int (*real)(int, const char *, int) = dlsym(RTLD_NEXT, "__openat64_2");
    return real(dirfd, path, flags);
}
