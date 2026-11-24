#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include "fedsched.h"
#include "fedsched.skel.h"
#include "controller.h"

#define REPORT_INTERVAL 1.0
#define POLL_TIMEOUT_MS 100

static volatile sig_atomic_t stop = 0;

/* Snapshot of CPU time counters from /proc/stat */
struct cpu_times {
    unsigned long long total;
    unsigned long long idle;
};

/* Snapshot of aggregated BPF counters */
struct counter_snapshot {
    unsigned long long switches;
    unsigned long long wakeups;
    unsigned long long reclaims;
    unsigned long long kswapd;
    unsigned long long drops;
};

/* The daemon's persistent state */
struct daemon_state {
    struct cpu_times previous_cpu;
    struct counter_snapshot previous_counters;

    unsigned long long previous_cpu_psi;
    unsigned long long previous_memory_psi;

    unsigned long long events_received;

    int have_cpu;
    int have_cpu_psi;
    int have_memory_psi;
};

static void handle_signal(int signo)
{
    (void)signo;
    stop = 1;
}

/*
 * Calculate elapsed monotonic time.
 */
static double elapsed_seconds(
    const struct timespec *start,
    const struct timespec *end)
{
    return (end->tv_sec - start->tv_sec) +
           (end->tv_nsec - start->tv_nsec) / 1e9;
}

/*
 * Ring-buffer event callback.
 *
 * Keep it lightweight: record how many
 * structured events arrived.
 */
static int handle_event(void *ctx, void *data, size_t size)
{
    struct daemon_state *state = ctx;
    const struct fedsched_event *event = data;

    if (size < sizeof(*event)) {
        fprintf(stderr, "Invalid BPF event size\n");
        return 0;
    }

    if (event->type >= EVENT_EXEC &&
        event->type <= EVENT_KSWAPD_WAKE) {
        state->events_received++;
    }

    return 0;
}

/*
 * Read aggregate CPU counters from /proc/stat.
 */
static int read_cpu_times(struct cpu_times *result)
{
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp)
        return -1;

    char line[512];

    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return -1;
    }

    fclose(fp);

    unsigned long long user, nice, system, idle;
    unsigned long long iowait, irq, softirq, steal;

    int fields = sscanf(
        line,
        "cpu %llu %llu %llu %llu %llu %llu %llu %llu",
        &user, &nice, &system, &idle,
        &iowait, &irq, &softirq, &steal
    );

    if (fields != 8)
        return -1;

    result->idle = idle + iowait;
    result->total =
        user + nice + system + idle +
        iowait + irq + softirq + steal;

    return 0;
}

/*
 * Estimate CPU utilization between snapshots.
 */
static double calculate_cpu_utilization(
    const struct cpu_times *old,
    const struct cpu_times *current)
{
    if (current->total <= old->total ||
        current->idle < old->idle)
        return 0.0;

    unsigned long long total_delta =
        current->total - old->total;

    unsigned long long idle_delta =
        current->idle - old->idle;

    if (idle_delta > total_delta)
        return 0.0;

    return 100.0 *
           (total_delta - idle_delta) /
           total_delta;
}

/*
 * Read MemAvailable from /proc/meminfo.
 */
static int read_available_memory(
    unsigned long long *available_kb)
{
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp)
        return -1;

    char line[256];
    int found = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemAvailable: %llu kB",
                   available_kb) == 1) {
            found = 1;
            break;
        }
    }

    fclose(fp);
    return found ? 0 : -1;
}

/*
 * Read cumulative PSI "some" stall time.
 *
 * The total field is measured in microseconds.
 */
static int read_psi_total(
    const char *path,
    unsigned long long *total_us)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
        return -1;

    char line[256];
    int found = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "some ", 5) != 0)
            continue;

        char *ptr = strstr(line, "total=");
        if (!ptr)
            break;

        if (sscanf(ptr, "total=%llu", total_us) == 1)
            found = 1;

        break;
    }

    fclose(fp);
    return found ? 0 : -1;
}

/*
 * Convert cumulative PSI stall-time differences
 * to a percentage of the measurement interval.
 */
static double calculate_psi(
    unsigned long long old_total,
    unsigned long long new_total,
    double seconds)
{
    if (seconds <= 0 || new_total < old_total)
        return 0.0;

    double percent =
        100.0 * (new_total - old_total) /
        (seconds * 1000000.0);

    return percent > 100.0 ? 100.0 : percent;
}

/*
 * Read all per-CPU entries from the BPF map,
 * then sum them into one system-wide snapshot.
 *
 * 'values' has one element for every possible CPU.
 */
static int read_bpf_counters(
    int map_fd,
    struct telemetry_stats *values,
    int ncpus,
    struct counter_snapshot *out)
{
    unsigned int key = 0;

    memset(values, 0,
           (size_t)ncpus * sizeof(*values));
    memset(out, 0, sizeof(*out));

    if (bpf_map_lookup_elem(map_fd, &key, values) != 0)
        return -1;

    for (int i = 0; i < ncpus; i++) {
        out->switches += values[i].sched_switches;
        out->wakeups += values[i].sched_wakeups;
        out->reclaims += values[i].direct_reclaims;
        out->kswapd += values[i].kswapd_wakes;
        out->drops += values[i].dropped_events;
    }

    return 0;
}

static unsigned long long counter_delta(
    unsigned long long current,
    unsigned long long previous)
{
    return current >= previous ?
           current - previous : 0;
}

static unsigned long long events_per_second(
    unsigned long long delta,
    double seconds)
{
    if (seconds <= 0)
        return 0;

    return (unsigned long long)(delta / seconds + 0.5);
}

/*
 * Print one periodic system-health report.
 */
static bool report_health(
    struct daemon_state *state,
    int stats_fd,
    struct telemetry_stats *cpu_values,
    int ncpus,
    double seconds,
    struct system_health *out)
{
    struct system_health health = {0};

    struct cpu_times current_cpu;
    struct counter_snapshot current_counters;

    unsigned long long cpu_psi = 0;
    unsigned long long memory_psi = 0;

    int cpu_ok = read_cpu_times(&current_cpu) == 0;
    int mem_ok = read_available_memory(
        &health.available_memory_kb) == 0;

    int cpu_psi_ok = read_psi_total(
        "/proc/pressure/cpu", &cpu_psi) == 0;

    int mem_psi_ok = read_psi_total(
        "/proc/pressure/memory", &memory_psi) == 0;

    int counters_ok = read_bpf_counters(
        stats_fd, cpu_values, ncpus,
        &current_counters) == 0;

    if (cpu_ok) {
        if (state->have_cpu)
            health.cpu_utilization =
                calculate_cpu_utilization(
                    &state->previous_cpu,
                    &current_cpu);

        state->previous_cpu = current_cpu;
        state->have_cpu = 1;
    }

    if (cpu_psi_ok) {
        if (state->have_cpu_psi)
            health.cpu_pressure = calculate_psi(
                state->previous_cpu_psi,
                cpu_psi, seconds);

        state->previous_cpu_psi = cpu_psi;
        state->have_cpu_psi = 1;
    }

    if (mem_psi_ok) {
        if (state->have_memory_psi)
            health.memory_pressure = calculate_psi(
                state->previous_memory_psi,
                memory_psi, seconds);

        state->previous_memory_psi = memory_psi;
        state->have_memory_psi = 1;
    }

    if (counters_ok) {
        struct counter_snapshot *old =
            &state->previous_counters;

        health.context_switches_per_sec =
            events_per_second(
                counter_delta(current_counters.switches,
                              old->switches), seconds);

        health.wakeups_per_sec =
            events_per_second(
                counter_delta(current_counters.wakeups,
                              old->wakeups), seconds);

        health.direct_reclaims_per_sec =
            events_per_second(
                counter_delta(current_counters.reclaims,
                              old->reclaims), seconds);

        health.kswapd_wakes_per_sec =
            events_per_second(
                counter_delta(current_counters.kswapd,
                              old->kswapd), seconds);

        health.ring_drops =
            counter_delta(current_counters.drops,
                          old->drops);

        state->previous_counters = current_counters;
    }

    health.ring_events_received = state->events_received;
    state->events_received = 0;

    printf("\n=========== FedSched System Health ===========\n");

    if (cpu_ok)
        printf("CPU utilization:          %8.1f%%\n",
               health.cpu_utilization);

    if (cpu_psi_ok)
        printf("CPU pressure (some):      %8.1f%%\n",
               health.cpu_pressure);

    if (mem_ok)
        printf("Available memory:         %8llu MB\n",
               health.available_memory_kb / 1024);

    if (mem_psi_ok)
        printf("Memory pressure (some):   %8.1f%%\n",
               health.memory_pressure);

    if (counters_ok) {
        printf("\nContext switches:         %8llu /sec\n",
               health.context_switches_per_sec);
        printf("Scheduler wakeups:        %8llu /sec\n",
               health.wakeups_per_sec);
        printf("Direct reclaims:          %8llu /sec\n",
               health.direct_reclaims_per_sec);
        printf("Kswapd wakes:             %8llu /sec\n",
               health.kswapd_wakes_per_sec);
    }

    printf("\nRing events received:     %8llu\n",
           health.ring_events_received);

    if (counters_ok)
        printf("Ring reserve failures:    %8llu\n",
               health.ring_drops);

    printf("==============================================\n");

    /*
    * Do not allow the controller to act on
    * incomplete or invalid health measurements.
    */
    bool valid =
        cpu_ok &&
        mem_ok &&
        cpu_psi_ok &&
        mem_psi_ok &&
        counters_ok &&
        state->have_cpu &&
        state->have_cpu_psi &&
        state->have_memory_psi;

    if (valid)
        *out = health;

    return valid;
}

int main(int argc, char **argv)
{
    struct fedsched_bpf *skel = NULL;
    struct ring_buffer *rb = NULL;
    struct telemetry_stats *cpu_values = NULL;

    struct daemon_state state = {0};
    struct counter_snapshot initial = {0};

    struct timespec last_report, now;

    int err = 0;
    int stats_fd;
    int events_fd;
    int ncpus;

    /*
     * Controller variables
     */
    struct controller ctrl = {
        .cgroup_fd = -1
    };

    bool apply = false;
    long worker_pid = -1;
    char *end = NULL;

    /*
     * Validate command-line arguments.
     *
     * Expected usage:
     * ./build/fedsched --dry-run WORKER_PID
     * ./build/fedsched --apply WORKER_PID
     */
    if (argc != 3 ||
        (strcmp(argv[1], "--dry-run") != 0 &&
         strcmp(argv[1], "--apply") != 0)) {

        fprintf(stderr,
            "Usage: %s --dry-run|--apply WORKER_PID\n",
            argv[0]);

        return 1;
    }

    /*
     * Determine whether cgroup modifications
     * should actually be applied.
     */
    apply = strcmp(argv[1], "--apply") == 0;

    /*
     * Convert the worker PID argument
     * from a string into a number.
     */
    errno = 0;
    worker_pid = strtol(argv[2], &end, 10);

    if (errno || !end || *end != '\0' ||
        worker_pid <= 1) {

        fprintf(stderr, "Invalid worker PID\n");
        return 1;
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    setvbuf(stdout, NULL, _IONBF, 0);

    skel = fedsched_bpf__open();
    if (!skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = fedsched_bpf__load(skel);
    if (err) {
        fprintf(stderr, "BPF load failed: %d\n", err);
        goto cleanup;
    }

    err = fedsched_bpf__attach(skel);
    if (err) {
        fprintf(stderr, "BPF attach failed: %d\n", err);
        goto cleanup;
    }

    stats_fd = bpf_map__fd(skel->maps.stats);
    events_fd = bpf_map__fd(skel->maps.events);

    if (stats_fd < 0 || events_fd < 0) {
        fprintf(stderr, "Failed to access BPF maps\n");
        err = -1;
        goto cleanup;
    }

    ncpus = libbpf_num_possible_cpus();
    if (ncpus <= 0) {
        fprintf(stderr, "Failed to get CPU count\n");
        err = -1;
        goto cleanup;
    }

    cpu_values = calloc((size_t)ncpus,
                        sizeof(*cpu_values));
    if (!cpu_values) {
        err = -ENOMEM;
        goto cleanup;
    }

    rb = ring_buffer__new(
        events_fd,
        handle_event,
        &state,
        NULL
    );

    if (!rb) {
        fprintf(stderr, "Failed to create ring buffer\n");
        err = -1;
        goto cleanup;
    }

    /*
     * Establish baseline readings before
     * beginning periodic reports.
     */
    if (read_bpf_counters(stats_fd, cpu_values,
                          ncpus, &initial) != 0) {
        fprintf(stderr, "Initial BPF read failed\n");
        err = -1;
        goto cleanup;
    }

    state.previous_counters = initial;

    state.have_cpu =
        read_cpu_times(&state.previous_cpu) == 0;

    state.have_cpu_psi =
        read_psi_total("/proc/pressure/cpu",
                       &state.previous_cpu_psi) == 0;

    state.have_memory_psi =
        read_psi_total("/proc/pressure/memory",
                       &state.previous_memory_psi) == 0;

    if (clock_gettime(CLOCK_MONOTONIC,
                      &last_report) != 0) {
        err = -1;
        goto cleanup;
    }

    printf("FedSched monitoring daemon started.\n");
    printf("Reporting system health every second.\n");
    printf("Press Ctrl+C to stop.\n");

    if (controller_init(
            &ctrl,
            apply,
            worker_pid) != 0) {

        fprintf(stderr,
            "Failed to initialize adaptive controller\n");

        err = -1;
        goto cleanup;
    }

    /*
     * Main daemon loop:
     * 1. Consume ring-buffer events.
     * 2. Periodically generate health reports.
     */
    while (!stop) {
        err = ring_buffer__poll(rb, POLL_TIMEOUT_MS);

        if (err == -EINTR) {
            err = 0;
            if (stop)
                break;
            continue;
        }

        if (err < 0) {
            fprintf(stderr,
                    "Ring-buffer polling failed: %d\n",
                    err);
            break;
        }

        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
            err = -1;
            break;
        }

        double elapsed =
            elapsed_seconds(&last_report, &now);

        if (elapsed >= REPORT_INTERVAL) {

            struct system_health health = {0};

            bool valid = report_health(
                &state,
                stats_fd,
                cpu_values,
                ncpus,
                elapsed,
                &health
            );

            
            if (valid) {
                int breaker_result = breaker_update(
                    &ctrl, &health
                );

                if (breaker_result < 0) {
                    fprintf(stderr,
                        "[Breaker] Emergency controller failed\n");

                    err = -1;
                    break;
                }

                if (breaker_result == 0) {

                    if (controller_update(
                            &ctrl, &health) != 0) {

                        fprintf(stderr,
                            "[Controller] Update failed\n");

                        err = -1;
                        break;
                    }
                }

            } else {

                fprintf(stderr,
                    "[Controller] Invalid telemetry; "
                    "skipping ordinary decisions\n");

                ctrl.high_count = 0;
                ctrl.low_count = 0;
            }
            last_report = now;
        }
    }

cleanup:
    controller_destroy(&ctrl);

    ring_buffer__free(rb);
    free(cpu_values);
    fedsched_bpf__destroy(skel);

    printf("\nFedSched daemon stopped.\n");

    return err < 0 ? 1 : 0;
}
