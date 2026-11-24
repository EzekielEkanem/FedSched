#ifndef FEDSCHED_CONTROLLER_H
#define FEDSCHED_CONTROLLER_H

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "fedsched.h"
#include <signal.h>
#include <sys/syscall.h>

/*
 * Phase 3, Part 2:
 * Threshold-based adaptive cgroup controller.
 *
 * This prototype only controls the dedicated
 * /sys/fs/cgroup/fedsched-worker cgroup.
 */

#define WORKER_CGROUP "/sys/fs/cgroup/fedsched-worker"

#define CPU_PRESSURE_HIGH 20.0
#define CPU_PRESSURE_LOW   5.0

#define MEMORY_PRESSURE_HIGH 10.0
#define MEMORY_PRESSURE_LOW   2.0

#define MEMORY_AVAILABLE_LOW_KB  (1024ULL * 1024)
#define MEMORY_AVAILABLE_SAFE_KB (1536ULL * 1024)

#define HIGH_SAMPLES 3
#define LOW_SAMPLES  5

#define COOLDOWN_SECONDS 5.0

/*
 * Emergency memory circuit breaker.
 * To be adjusted after measuring the VM's
 * normal and stressed memory behavior.
 */

#define EMERGENCY_MEMORY_KB (256ULL * 1024)
#define CRITICAL_MEMORY_KB  (128ULL * 1024)

#define EMERGENCY_MEMORY_PSI 25.0

#define RESUME_MEMORY_KB (512ULL * 1024)
#define RESUME_MEMORY_PSI 5.0

#define EMERGENCY_SAMPLES 2
#define RECOVERY_SAMPLES 5

#define MAX_PAUSE_SECONDS 30.0
#define REPAUSE_COOLDOWN_SECONDS 30.0

/* One logical CPU worth of bandwidth */
#define THROTTLED_CPU_QUOTA 100000ULL
#define CPU_PERIOD         100000ULL

/* Allow 256 MiB above current memory usage */
#define MEMORY_HEADROOM (256ULL * 1024 * 1024)
#define MIN_MEMORY_HIGH (512ULL * 1024 * 1024)

enum control_mode {
    CONTROL_NORMAL,
    CONTROL_THROTTLED
};


struct controller {
    /* Existing threshold controller */
    enum control_mode mode;

    int high_count;
    int low_count;

    bool apply;
    bool originals_saved;

    int cgroup_fd;

    char original_cpu_max[128];
    char original_memory_high[128];

    struct timespec last_change;

    long worker_pid;
    int worker_pidfd;
    bool breaker_paused;
    int emergency_count;
    int recovery_count;
    struct timespec pause_started;
    struct timespec last_resume;
};


/*
 * Read one cgroup control file.
 */
static int cg_read(
    int dirfd,
    const char *name,
    char *output,
    size_t size)
{
    if (size < 2)
        return -1;

    int fd = openat(
        dirfd, name,
        O_RDONLY | O_CLOEXEC | O_NOFOLLOW
    );

    if (fd < 0)
        return -1;

    ssize_t n = read(fd, output, size - 1);
    int saved_errno = errno;
    close(fd);

    if (n <= 0) {
        errno = n < 0 ? saved_errno : EIO;
        return -1;
    }

    output[n] = '\0';

    return 0;
}

/*
 * Write one cgroup control file.
 *
 * Never create files here: kernel cgroup
 * control files must already exist.
 */
static int cg_write(
    int dirfd,
    const char *name,
    const char *value)
{
    int fd = openat(
        dirfd, name,
        O_WRONLY | O_CLOEXEC | O_NOFOLLOW
    );

    if (fd < 0)
        return -1;

    size_t len = strlen(value);
    ssize_t n = write(fd, value, len);
    int saved_errno = errno;

    close(fd);

    if (n != (ssize_t)len) {
        errno = n < 0 ? saved_errno : EIO;
        return -1;
    }

    return 0;
}

static double controller_elapsed(
    const struct timespec *old,
    const struct timespec *now)
{
    return (now->tv_sec - old->tv_sec) +
           (now->tv_nsec - old->tv_nsec) / 1e9;
}

/*
 * Verify the worker PID is in the
 * dedicated cgroup and the daemon is not.
 */
static int verify_cgroup_worker(
    int dirfd,
    long worker_pid)
{
    char contents[4096];

    if (worker_pid <= 1)
        return -1;

    if (worker_pid == (long)getpid())
        return -1;

    if (cg_read(dirfd, "cgroup.procs",
                contents, sizeof(contents)) != 0)
        return -1;

    bool worker_found = false;
    char *saveptr = NULL;

    for (char *line = strtok_r(
            contents, "\n", &saveptr);
         line;
         line = strtok_r(NULL, "\n", &saveptr)) {

        char *end = NULL;
        long pid = strtol(line, &end, 10);

        if (!end || *end != '\0')
            continue;

        if (pid == (long)getpid())
            return -1;

        if (pid == worker_pid)
            worker_found = true;
    }

    return worker_found ? 0 : -1;
}

/*
 * Initialize controller.
 *
 * Default mode is dry run.
 */
static int controller_init(
    struct controller *ctrl,
    bool apply,
    long worker_pid)
{
    memset(ctrl, 0, sizeof(*ctrl));

    ctrl->cgroup_fd = -1;
    ctrl->worker_pidfd = -1;

    ctrl->mode = CONTROL_NORMAL;
    ctrl->apply = apply;
    ctrl->worker_pid = worker_pid;

    ctrl->cgroup_fd = open(
        WORKER_CGROUP,
        O_RDONLY | O_DIRECTORY |
        O_CLOEXEC | O_NOFOLLOW
    );

    if (ctrl->cgroup_fd < 0) {
        perror("Opening worker cgroup");
        return -1;
    }

    if (verify_cgroup_worker(
            ctrl->cgroup_fd, worker_pid) != 0) {
        fprintf(stderr,
            "Worker PID is not in the dedicated "
            "cgroup, or daemon is inside it.\n");
        close(ctrl->cgroup_fd);
        ctrl->cgroup_fd = -1;
        return -1;
    }

    ctrl->worker_pidfd = (int)syscall(
        SYS_pidfd_open,
        (pid_t)worker_pid,
        0
    );

    if (ctrl->worker_pidfd < 0) {
        perror("pidfd_open");

        close(ctrl->cgroup_fd);
        ctrl->cgroup_fd = -1;

        return -1;
    }

    if (verify_cgroup_worker(
            ctrl->cgroup_fd, worker_pid) != 0) {

        fprintf(stderr,
            "Worker membership changed during initialization\n");

        close(ctrl->worker_pidfd);
        ctrl->worker_pidfd = -1;

        close(ctrl->cgroup_fd);
        ctrl->cgroup_fd = -1;

        return -1;
    }

    if (cg_read(
            ctrl->cgroup_fd,
            "cpu.max",
            ctrl->original_cpu_max,
            sizeof(ctrl->original_cpu_max)) != 0 ||
        cg_read(
            ctrl->cgroup_fd,
            "memory.high",
            ctrl->original_memory_high,
            sizeof(ctrl->original_memory_high)) != 0) {
        perror("Reading original cgroup limits");
        close(ctrl->worker_pidfd);
        ctrl->worker_pidfd = -1;
        close(ctrl->cgroup_fd);
        ctrl->cgroup_fd = -1;
        return -1;
    }

    ctrl->original_cpu_max[
        strcspn(ctrl->original_cpu_max, "\n")
    ] = '\0';

    ctrl->original_memory_high[
        strcspn(ctrl->original_memory_high, "\n")
    ] = '\0';

    ctrl->originals_saved = true;

    if (clock_gettime(
            CLOCK_MONOTONIC,
            &ctrl->last_change) != 0)
        close(ctrl->worker_pidfd);
        ctrl->worker_pidfd = -1;
        close(ctrl->cgroup_fd);
        ctrl->cgroup_fd = -1;
        return -1;

    printf(
        "[Controller] Mode: %s\n",
        apply ? "APPLY" : "DRY RUN"
    );

    printf(
        "[Controller] Original cpu.max: %s\n",
        ctrl->original_cpu_max
    );

    printf(
        "[Controller] Original memory.high: %s\n",
        ctrl->original_memory_high
    );

    return 0;
}

/*
 * Enter throttled state.
 *
 * CPU quota is reduced to one logical CPU.
 *
 * memory.high is placed above current usage
 * to restrict future growth conservatively.
 */
static int apply_throttle(struct controller *ctrl)
{
    char memory_current_text[128];
    char memory_high_text[128];

    unsigned long long current = 0;

    if (cg_read(
            ctrl->cgroup_fd,
            "memory.current",
            memory_current_text,
            sizeof(memory_current_text)) != 0)
        return -1;

    char *end = NULL;
    errno = 0;
    current = strtoull(
        memory_current_text, &end, 10
    );

    if (errno || end == memory_current_text ||
        (*end != '\0' && *end != '\n'))
        return -1;

    if (current > ULLONG_MAX - MEMORY_HEADROOM)
        return -1;

    unsigned long long high =
        current + MEMORY_HEADROOM;

    if (high < MIN_MEMORY_HIGH)
        high = MIN_MEMORY_HIGH;

    snprintf(
        memory_high_text,
        sizeof(memory_high_text),
        "%llu",
        high
    );

    if (!ctrl->apply) {
        printf(
            "[DRY RUN] Set cpu.max=%llu %llu "
            "and memory.high=%s\n",
            THROTTLED_CPU_QUOTA,
            CPU_PERIOD,
            memory_high_text
        );
        return 0;
    }

    if (cg_write(
            ctrl->cgroup_fd,
            "cpu.max",
            "100000 100000") != 0) {
        perror("Writing cpu.max");
        return -1;
    }

    if (cg_write(
            ctrl->cgroup_fd,
            "memory.high",
            memory_high_text) != 0) {
        perror("Writing memory.high");

        /* Best-effort rollback of CPU quota */
        if (cg_write(
                ctrl->cgroup_fd,
                "cpu.max",
                ctrl->original_cpu_max) != 0)
            perror("Rolling back cpu.max");

        return -1;
    }

    printf(
        "[Controller] Applied cpu.max=100000 100000 "
        "memory.high=%s\n",
        memory_high_text
    );

    return 0;
}

/*
 * Restore the original cgroup limits.
 */
static int restore_limits(struct controller *ctrl)
{
    if (!ctrl->originals_saved)
        return -1;

    if (!ctrl->apply) {
        printf(
            "[DRY RUN] Restore original "
            "CPU and memory limits\n"
        );
        return 0;
    }

    int cpu_ok = cg_write(
        ctrl->cgroup_fd,
        "cpu.max",
        ctrl->original_cpu_max
    );

    int memory_ok = cg_write(
        ctrl->cgroup_fd,
        "memory.high",
        ctrl->original_memory_high
    );

    if (cpu_ok != 0 || memory_ok != 0) {
        perror("Restoring cgroup limits");
        return -1;
    }

    printf(
        "[Controller] Original limits restored\n"
    );

    return 0;
}

/*
 * Evaluate the system health once per report.
 *
 * Hysteresis:
 *   3 high-pressure samples to throttle.
 *   5 healthy samples to recover.
 *
 * Cooldown:
 *   At least 5 seconds between changes.
 */
static int controller_update(
    struct controller *ctrl,
    const struct system_health *h)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return -1;

    bool pressure_high =
        h->cpu_pressure >= CPU_PRESSURE_HIGH ||
        h->memory_pressure >= MEMORY_PRESSURE_HIGH ||
        h->available_memory_kb <=
            MEMORY_AVAILABLE_LOW_KB;

    bool recovered =
        h->cpu_pressure <= CPU_PRESSURE_LOW &&
        h->memory_pressure <= MEMORY_PRESSURE_LOW &&
        h->available_memory_kb >=
            MEMORY_AVAILABLE_SAFE_KB;

    double since_change = controller_elapsed(
        &ctrl->last_change, &now
    );

    if (ctrl->mode == CONTROL_NORMAL) {
        ctrl->high_count =
            pressure_high ? ctrl->high_count + 1 : 0;

        if (ctrl->high_count >= HIGH_SAMPLES &&
            since_change >= COOLDOWN_SECONDS) {

            if (apply_throttle(ctrl) != 0)
                return -1;

            ctrl->mode = CONTROL_THROTTLED;
            ctrl->last_change = now;
            ctrl->high_count = 0;
            ctrl->low_count = 0;

            printf(
                "[Controller] NORMAL -> THROTTLED\n"
            );
        }

    } else {
        ctrl->low_count =
            recovered ? ctrl->low_count + 1 : 0;

        if (ctrl->low_count >= LOW_SAMPLES &&
            since_change >= COOLDOWN_SECONDS) {

            if (restore_limits(ctrl) != 0)
                return -1;

            ctrl->mode = CONTROL_NORMAL;
            ctrl->last_change = now;
            ctrl->low_count = 0;
            ctrl->high_count = 0;

            printf(
                "[Controller] THROTTLED -> NORMAL\n"
            );
        }
    }

    return 0;
}


/*
 * Send a signal to the original worker using pidfd.
 *
 * This avoids signaling an unrelated process if
 * the numeric PID is reused.
 */
static int breaker_signal_worker(
    struct controller *ctrl,
    int signo)
{
    if (ctrl->worker_pidfd < 0)
        return -1;

    if (!ctrl->apply) {
        printf(
            "[DRY RUN] Would send %s to worker PID %ld\n",
            signo == SIGSTOP ? "SIGSTOP" : "SIGCONT",
            ctrl->worker_pid
        );
        return 0;
    }

    if (syscall(
            SYS_pidfd_send_signal,
            ctrl->worker_pidfd,
            signo,
            NULL,
            0) < 0) {

        perror("pidfd_send_signal");
        return -1;
    }

    return 0;
}

/*
 * Suspend the background PEFT worker.
 */
static int breaker_pause(
    struct controller *ctrl,
    const struct timespec *now)
{
    if (ctrl->breaker_paused)
        return 0;

    /*
     * Verify the worker is still in its
     * dedicated cgroup before stopping it.
     */
    if (verify_cgroup_worker(
            ctrl->cgroup_fd,
            ctrl->worker_pid) != 0) {

        fprintf(stderr,
            "[Breaker] Worker membership invalid; "
            "refusing SIGSTOP\n");

        return -1;
    }

    if (breaker_signal_worker(ctrl, SIGSTOP) != 0)
        return -1;

    ctrl->breaker_paused = true;
    ctrl->pause_started = *now;

    ctrl->emergency_count = 0;
    ctrl->recovery_count = 0;

    printf(
        "[Breaker] EMERGENCY: worker %ld -> PAUSED\n",
        ctrl->worker_pid
    );

    return 0;
}

/*
 * Resume the PEFT worker.
 */
static int breaker_resume(
    struct controller *ctrl,
    const struct timespec *now,
    const char *reason)
{
    if (!ctrl->breaker_paused)
        return 0;

    if (breaker_signal_worker(ctrl, SIGCONT) != 0)
        return -1;

    ctrl->breaker_paused = false;
    ctrl->last_resume = *now;

    ctrl->emergency_count = 0;
    ctrl->recovery_count = 0;

    printf(
        "[Breaker] RESUMED worker %ld: %s\n",
        ctrl->worker_pid,
        reason
    );

    return 0;
}

/*
 * Evaluate emergency memory conditions.
 *
 * Returns:
 *   -1 : error
 *    0 : ordinary controller may run
 *    1 : breaker handled the interval,
 *        so skip ordinary controller update
 */
static int breaker_update(
    struct controller *ctrl,
    const struct system_health *h)
{
    struct timespec now;

    if (clock_gettime(
            CLOCK_MONOTONIC, &now) != 0)
        return -1;

    /*
     * Memory pressure becomes critical
     * if either:
     *
     * 1. Very low RAM plus high PSI.
     * 2. Extremely low RAM.
     */
    bool critical =
        (h->available_memory_kb < EMERGENCY_MEMORY_KB &&
         h->memory_pressure >= EMERGENCY_MEMORY_PSI)
        ||
        h->available_memory_kb < CRITICAL_MEMORY_KB;

    /*
     * Require clear memory recovery before
     * resuming under normal conditions.
     */
    bool recovered =
        h->available_memory_kb >= RESUME_MEMORY_KB &&
        h->memory_pressure <= RESUME_MEMORY_PSI;

    /*
     * CASE 1: Worker is already paused.
     */
    if (ctrl->breaker_paused) {

        double paused_seconds = controller_elapsed(
            &ctrl->pause_started, &now
        );

        if (recovered)
            ctrl->recovery_count++;
        else
            ctrl->recovery_count = 0;

        /*
         * Maximum pause protects against
         * leaving training suspended forever.
         */
        if (paused_seconds >= MAX_PAUSE_SECONDS) {
            if (breaker_resume(
                    ctrl, &now,
                    "maximum pause timeout") != 0)
                return -1;

            return 1;
        }

        if (ctrl->recovery_count >= RECOVERY_SAMPLES) {
            if (breaker_resume(
                    ctrl, &now,
                    "memory recovered") != 0)
                return -1;

            return 1;
        }

        printf(
            "[Breaker] PAUSED for %.1f seconds "
            "(available=%llu MB, PSI=%.1f%%)\n",
            paused_seconds,
            h->available_memory_kb / 1024,
            h->memory_pressure
        );

        return 1;
    }

    /*
     * CASE 2: Worker is running.
     *
     * Prevent immediate re-pausing after
     * a recent recovery or timeout.
     */
    if (ctrl->last_resume.tv_sec != 0 ||
        ctrl->last_resume.tv_nsec != 0) {

        double since_resume = controller_elapsed(
            &ctrl->last_resume, &now
        );

        if (since_resume < REPAUSE_COOLDOWN_SECONDS) {
            ctrl->emergency_count = 0;
            return 0;
        }
    }

    if (critical)
        ctrl->emergency_count++;
    else
        ctrl->emergency_count = 0;

    if (ctrl->emergency_count >= EMERGENCY_SAMPLES) {

        if (breaker_pause(ctrl, &now) != 0)
            return -1;

        return 1;
    }

    return 0;
}

static void controller_destroy(struct controller *ctrl)
{
    struct timespec now;

    if (ctrl->cgroup_fd >= 0) {

        /*
         * Restore original CPU and memory limits.
         * Do this before resuming training.
         */
        if (ctrl->mode == CONTROL_THROTTLED) {

            if (restore_limits(ctrl) != 0) {
                fprintf(stderr,
                    "[Controller] WARNING: "
                    "could not restore cgroup limits\n");
            }
        }

        /*
         * If FedSched stopped the worker,
         * attempt to resume it on shutdown.
         */
        if (ctrl->breaker_paused) {

            if (clock_gettime(
                    CLOCK_MONOTONIC, &now) == 0) {

                if (breaker_resume(
                        ctrl, &now,
                        "controller shutdown") != 0) {

                    fprintf(stderr,
                        "[Breaker] WARNING: "
                        "worker may still be stopped\n");
                }
            } else {
                /*
                 * Still attempt SIGCONT even if
                 * clock_gettime fails.
                 */
                if (breaker_signal_worker(
                        ctrl, SIGCONT) != 0) {

                    fprintf(stderr,
                        "[Breaker] WARNING: "
                        "SIGCONT failed during shutdown\n");
                }
            }
        }

        close(ctrl->cgroup_fd);
        ctrl->cgroup_fd = -1;
    }

    /*
     * Release the worker's PID descriptor.
     */
    if (ctrl->worker_pidfd >= 0) {
        close(ctrl->worker_pidfd);
        ctrl->worker_pidfd = -1;
    }
}

#endif
