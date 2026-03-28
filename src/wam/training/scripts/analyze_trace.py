"""
Analyze a Chrome trace JSON exported by PyTorch profiler.
Usage: python analyze_trace.py <path/to/trace.json>
"""
import json
import argparse
from collections import defaultdict


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def complete_events(events: list[dict]) -> list[dict]:
    """Return only complete (X) events with a duration."""
    return [e for e in events if e.get("ph") == "X" and "dur" in e]


def build_stats(events: list[dict]) -> dict[str, dict]:
    stats = defaultdict(lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0})
    for e in events:
        name = e["name"]
        dur = float(e["dur"])
        s = stats[name]
        s["count"] += 1
        s["total_us"] += dur
        s["max_us"] = max(s["max_us"], dur)
    return stats


def top_table(stats: dict, n: int = 30, filter_prefix: str | None = None) -> str:
    rows = sorted(stats.items(), key=lambda x: x[1]["total_us"], reverse=True)
    if filter_prefix:
        rows = [(k, v) for k, v in rows if k.startswith(filter_prefix)]
    rows = rows[:n]

    grand_total = sum(v["total_us"] for v in stats.values())

    header = f"{'Name':<55} {'Count':>6} {'Total(ms)':>12} {'Avg(ms)':>10} {'Max(ms)':>10} {'%Total':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for name, v in rows:
        pct = 100 * v["total_us"] / grand_total if grand_total else 0
        lines.append(
            f"{name[:55]:<55} {v['count']:>6} "
            f"{v['total_us']/1e3:>12.2f} "
            f"{v['total_us']/v['count']/1e3:>10.2f} "
            f"{v['max_us']/1e3:>10.2f} "
            f"{pct:>7.1f}%"
        )
    return "\n".join(lines)


def record_function_summary(stats: dict) -> str:
    """Focus on the record_function spans defined in the training script."""
    targets = ["encode_video", "encode_language", "dit_forward", "backward", "optimizer_step"]
    rows = [(k, stats[k]) for k in targets if k in stats]
    if not rows:
        return "No record_function spans found. Make sure the trace includes CPU activity."

    total_us = sum(v["total_us"] for _, v in rows)
    lines = ["\n=== Training step breakdown (record_function spans) ==="]
    lines.append(f"{'Name':<20} {'Total(ms)':>12} {'Avg(ms)':>10} {'%Step':>8}")
    lines.append("-" * 55)
    for name, v in rows:
        pct = 100 * v["total_us"] / total_us if total_us else 0
        lines.append(
            f"{name:<20} {v['total_us']/1e3:>12.2f} "
            f"{v['total_us']/v['count']/1e3:>10.2f} {pct:>7.1f}%"
        )
    lines.append("-" * 55)
    lines.append(f"{'TOTAL':<20} {total_us/1e3:>12.2f}")
    return "\n".join(lines)


def cuda_kernel_summary(events: list[dict]) -> str:
    cuda_events = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "kernel", "gpu_memcpy")]
    if not cuda_events:
        return "\nNo CUDA kernel events found."

    stats = build_stats(cuda_events)
    lines = ["\n=== Top CUDA kernels / ops ==="]
    lines.append(top_table(stats, n=15))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze PyTorch profiler Chrome trace")
    parser.add_argument("trace", help="Path to trace.json")
    parser.add_argument("--top", type=int, default=20, help="Number of top ops to show")
    parser.add_argument("--filter", default=None, help="Filter op names by prefix")
    args = parser.parse_args()

    events = load_events(args.trace)
    complete = complete_events(events)
    stats = build_stats(complete)

    print(f"\nLoaded {len(events)} events, {len(complete)} complete spans\n")
    print(record_function_summary(stats))
    print(cuda_kernel_summary(complete))
    print(f"\n=== Top {args.top} ops overall ===")
    print(top_table(stats, n=args.top, filter_prefix=args.filter))


if __name__ == "__main__":
    main()
