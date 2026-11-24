#ifndef FEDSCHED_H
#define FEDSCHED_H

/* Types of kernel events */
enum event_type {
    EVENT_EXEC = 1,
    EVENT_SCHED_SWITCH = 2,
    EVENT_SCHED_WAKEUP = 3,
    EVENT_DIRECT_RECLAIM = 4,
    EVENT_KSWAPD_WAKE = 5
};

/* Records transmitted through the BPF ring buffer */
struct fedsched_event {
    unsigned long long timestamp_ns;
    unsigned long long count;

    unsigned int pid;
    unsigned int cpu;
    unsigned int type;

    char comm[16];
};

/* Counters maintained separately for each CPU */
struct telemetry_stats {
    unsigned long long sched_switches;
    unsigned long long sched_wakeups;
    unsigned long long direct_reclaims;
    unsigned long long kswapd_wakes;
    unsigned long long dropped_events;
};

/* Aggregated system-health measurements */
struct system_health {
    double cpu_utilization;
    double cpu_pressure;
    double memory_pressure;

    unsigned long long available_memory_kb;

    unsigned long long context_switches_per_sec;
    unsigned long long wakeups_per_sec;
    unsigned long long direct_reclaims_per_sec;
    unsigned long long kswapd_wakes_per_sec;

    unsigned long long ring_events_received;
    unsigned long long ring_drops;
};

#endif
