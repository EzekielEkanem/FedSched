#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "fedsched.h"

char LICENSE[] SEC("license") = "GPL";

/* Each CPU maintains its own counters */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct telemetry_stats);
} stats SEC(".maps");

/*
 * 1 MiB buffer shared by the kernel-side
 * eBPF programs and the user-space consumer.
 */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);
} events SEC(".maps");

/* Retrieve the current CPU's counters */
static __always_inline struct telemetry_stats *get_stats(void)
{
    __u32 key = 0;
    return bpf_map_lookup_elem(&stats, &key);
}

/*
 * Every tracepoint can use this function to
 * send a structured record to user space.
 */
static __always_inline
void submit_event(__u32 type, __u64 count,
                  struct telemetry_stats *s)
{
    struct fedsched_event *event;

    /*
     * Reserve space in the ring buffer.
     */
    event = bpf_ringbuf_reserve(
        &events,
        sizeof(struct fedsched_event),
        0
    );

    if (!event) {
        if (s)
            s->dropped_events++;

        return;
    }

    /*
     * Fill the reserved record.
     */
    event->timestamp_ns = bpf_ktime_get_ns();
    event->count = count;

    event->pid =
        bpf_get_current_pid_tgid() >> 32;

    event->cpu = bpf_get_smp_processor_id();
    event->type = type;

    bpf_get_current_comm(
        event->comm,
        sizeof(event->comm)
    );

    /*
     * Make the completed record available
     * to the user-space application.
     */
    bpf_ringbuf_submit(event, 0);
}

/* Process execution monitoring */
SEC("tracepoint/sched/sched_process_exec")
int handle_exec(void *ctx)
{
    submit_event(EVENT_EXEC, 1, 0);

    return 0;
}

/* CPU context-switch monitoring */
SEC("tracepoint/sched/sched_switch")
int handle_sched_switch(void *ctx)
{
    struct telemetry_stats *s = get_stats();

    if (!s)
        return 0;

    s->sched_switches++;

    /* Print sparingly to avoid flooding trace_pipe */
    if ((s->sched_switches % 10000) == 0) {
         submit_event(
            EVENT_SCHED_SWITCH,
            s->sched_switches,
            s
        );
    }

    return 0;
}

/* CPU wakeup monitoring */
SEC("tracepoint/sched/sched_wakeup")
int handle_sched_wakeup(void *ctx)
{
    struct telemetry_stats *s = get_stats();

    if (!s)
        return 0;

    s->sched_wakeups++;

    if ((s->sched_wakeups % 10000) == 0) {
        submit_event(
            EVENT_SCHED_WAKEUP,
            s->sched_wakeups,
            s
        );
    }

    return 0;
}

/* Direct memory reclaim monitoring */
SEC("tracepoint/vmscan/mm_vmscan_direct_reclaim_begin")
int handle_direct_reclaim(void *ctx)
{
    struct telemetry_stats *s = get_stats();

    if (!s)
        return 0;

    s->direct_reclaims++;

    submit_event(
        EVENT_DIRECT_RECLAIM,
        s->direct_reclaims,
        s
    );

    return 0;
}

/* Background memory reclaim monitoring */
SEC("tracepoint/vmscan/mm_vmscan_kswapd_wake")
int handle_kswapd_wake(void *ctx)
{
    struct telemetry_stats *s = get_stats();

    if (!s)
        return 0;

    s->kswapd_wakes++;

    submit_event(
        EVENT_KSWAPD_WAKE,
        s->kswapd_wakes,
        s
    );

    return 0;
}

