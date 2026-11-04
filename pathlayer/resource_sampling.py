"""RSS memory sampling.

Isolated from ``pathlayer.runtime``/``pathlayer.detectors.resource`` so the
"gracefully unavailable" contract lives in exactly one place: every caller
gets ``int | None`` and never has to guard a platform-specific read itself.
"""

from __future__ import annotations


def sample_rss_bytes() -> int | None:
    """Return the process's current resident set size in bytes, or ``None``.

    Backed by ``/proc/self/status``'s ``VmRSS`` line (Linux only) -- a live,
    *current* reading, not a peak. Unlike a ``getrusage``-style high-water
    mark (monotonically non-decreasing for the life of the process, so it
    would leak a prior logical run's spike into a later one sharing the same
    long-lived process), ``VmRSS`` goes up and down with the process's actual
    live footprint. That means a caller can compare it directly to a fixed
    threshold with no per-run baseline or growth-delta bookkeeping needed --
    the value read on any given event already *is* this process's current
    memory usage, not "growth since some earlier reading."

    Returns ``None`` -- never raises, catching every exception the read or
    parse could produce -- when ``/proc`` is unavailable (e.g. macOS,
    Windows) or the read/parse fails for any other reason, per RES-3's "no
    hard dependency on platform APIs" requirement.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    # Format: "VmRSS:\t   12345 kB\n".
                    kilobytes = int(line.split()[1])
                    return kilobytes * 1024
    except Exception:
        return None
    return None
