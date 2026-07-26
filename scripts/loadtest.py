"""Load test for the HMDA Dash app (app/index.py).

Exercises the real work: the tab-render callback at /_dash-update-component,
plus the static page and layout endpoints. Reports latency percentiles,
throughput and error counts. Stdlib only, no extra dependencies.

Usage
-----
1. Start the app in one terminal, on a port that is free:

       HMDA_PORT=8051 python app/index.py

2. Run this script in another terminal:

       python scripts/loadtest.py http://127.0.0.1:8051

   The URL argument is optional and defaults to http://127.0.0.1:8051.

The numbers reported in the dashboard's Fase 5 "Laporan uji beban" panel came
from this script. Re-run it after any change that could affect response time,
and update LOADTEST_COLD / LOADTEST_CONC in app/index.py so the panel does not
keep showing stale measurements.

Note: app/index.py serves through the Flask development server, which is not
what Vercel runs. Treat the results as a relative profile between tabs, not as
production capacity.
"""

import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8050"
TABS = ["fase1", "fase2", "fase3", "fase4", "fase5"]


def _post_tab(tab):
    """Trigger the tab-routing callback exactly as the browser does."""
    payload = {
        "output": "tab-content.children",
        "outputs": {"id": "tab-content", "property": "children"},
        "inputs": [{"id": "tabs", "property": "value", "value": tab}],
        "changedPropIds": ["tabs.value"],
    }
    req = urllib.request.Request(
        f"{BASE}/_dash-update-component",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        return (time.perf_counter() - t0) * 1000, r.status, len(body)
    except urllib.error.HTTPError as e:
        return (time.perf_counter() - t0) * 1000, e.code, 0
    except Exception:
        return (time.perf_counter() - t0) * 1000, 0, 0


def _get(path):
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
            body = r.read()
        return (time.perf_counter() - t0) * 1000, r.status, len(body)
    except urllib.error.HTTPError as e:
        return (time.perf_counter() - t0) * 1000, e.code, 0
    except Exception:
        return (time.perf_counter() - t0) * 1000, 0, 0


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


def report(name, results, wall):
    lat = [r[0] for r in results]
    ok = sum(1 for r in results if r[1] == 200)
    err = len(results) - ok
    kb = statistics.mean([r[2] for r in results if r[2]] or [0]) / 1024
    print(f"\n{name}")
    print(
        f"  requests {len(results)}  ok {ok}  errors {err}  "
        f"wall {wall:.2f}s  throughput {len(results)/wall:.1f} req/s"
    )
    print(
        f"  latency ms: min {min(lat):.0f}  p50 {pct(lat,50):.0f}  "
        f"p95 {pct(lat,95):.0f}  p99 {pct(lat,99):.0f}  max {max(lat):.0f}  "
        f"mean {statistics.mean(lat):.0f}"
    )
    if kb:
        print(f"  avg payload {kb:.0f} KB")
    return ok, err


def run(fn, args_list, workers):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(fn, args_list))
    return out, time.perf_counter() - t0


def main():
    print(f"Target: {BASE}")

    # readiness
    for i in range(40):
        if _get("/")[1] == 200:
            print(f"Server ready after {i*0.5:.1f}s")
            break
        time.sleep(0.5)
    else:
        print("SERVER NOT REACHABLE")
        return 1

    total_err = 0

    # 1. COLD: first hit of each tab (lru_cache empty for that tab)
    print("\n=== 1. COLD tab render (sequential, cache empty) ===")
    for tab in TABS:
        ms, status, size = _post_tab(tab)
        flag = "OK" if status == 200 else f"ERR {status}"
        print(f"  {tab}: {ms:8.0f} ms  {size/1024:7.0f} KB  {flag}")
        if status != 200:
            total_err += 1

    # 2. WARM: repeated hits, cache populated
    print("\n=== 2. WARM tab render (cache populated) ===")
    for conc in (1, 5, 10, 25):
        args = [TABS[i % len(TABS)] for i in range(100)]
        res, wall = run(_post_tab, args, conc)
        _, err = report(f"concurrency {conc} - 100 requests across 5 tabs", res, wall)
        total_err += err

    # 3. Static page + layout endpoints
    print("\n=== 3. Static endpoints ===")
    for path in ("/", "/_dash-layout", "/_dash-dependencies"):
        res, wall = run(_get, [path] * 100, 10)
        _, err = report(f"GET {path} - 100 requests @ concurrency 10", res, wall)
        total_err += err

    # 4. Sustained mixed load
    print("\n=== 4. Sustained mixed load (30s, concurrency 20) ===")
    deadline = time.perf_counter() + 30
    results = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = []
        i = 0
        while time.perf_counter() < deadline:
            futures.append(ex.submit(_post_tab, TABS[i % len(TABS)]))
            i += 1
            if len(futures) >= 400:
                results += [f.result() for f in futures]
                futures = []
            time.sleep(0.005)
        results += [f.result() for f in futures]
    _, err = report("sustained mixed tab load", results, time.perf_counter() - t0)
    total_err += err

    print(f"\n{'='*60}\nTOTAL ERRORS: {total_err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
