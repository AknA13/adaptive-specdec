"""Prometheus scraping for speculative-decoding acceptance.

Two ways this has already been wrong, both of which produced numbers rather
than errors:

  * startswith() also matched vllm:spec_decode_num_accepted_tokens_per_pos,
    whose buckets sum to the same total, so acceptance came out exactly 2x and
    reported alpha = 1.56.
  * anchoring the name then rejected everything, because the client exports
    counters with a _total suffix, so acceptance silently became unavailable.

Both are invisible unless something checks the arithmetic, so it gets a test.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.bench_serving import spec_delta
import bench.bench_serving as BS

FAIL = []

BODY = """# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens.
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="target"} 1000.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="target"} 780.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="target",position="0"} 460.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="target",position="1"} 320.0
vllm:spec_decode_num_drafts_total{engine="0",model_name="target"} 500.0
vllm:num_requests_running{engine="0",model_name="target"} 3.0
"""


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail}")
    if not cond:
        FAIL.append(name)


def parse(body, monkey_url="http://x"):
    """Drive the real scraper against a canned body."""
    import io
    import urllib.request

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: _Resp(body.encode())
    try:
        return BS.scrape_spec_metrics(monkey_url)
    finally:
        urllib.request.urlopen = orig


def main():
    print("[metrics] counters carry a _total suffix and must still be found")
    got = parse(BODY)
    check("draft tokens", got.get("draft_tokens") == 1000.0, str(got.get("draft_tokens")))
    check("drafts", got.get("drafts") == 500.0, str(got.get("drafts")))
    print("[metrics] _per_pos buckets must NOT be added to the plain counter")
    check("accepted tokens counted once", got.get("accepted_tokens") == 780.0,
          f"{got.get('accepted_tokens')} (1560 would mean per_pos was summed in)")

    print("[metrics] derived quantities")
    d = spec_delta({}, got)
    check("acceptance rate", abs(d["spec_acceptance_rate"] - 0.78) < 1e-9,
          f"{d['spec_acceptance_rate']}")
    check("acceptance rate is a probability", 0.0 <= d["spec_acceptance_rate"] <= 1.0)
    check("mean accepted length", abs(d["spec_mean_accepted_len"] - 2.56) < 1e-9,
          f"{d['spec_mean_accepted_len']}")

    print("[metrics] deltas, not absolutes (counters only ever increase)")
    before = {"draft_tokens": 400.0, "accepted_tokens": 300.0, "drafts": 200.0}
    d2 = spec_delta(before, got)
    check("subtracts the earlier reading",
          abs(d2["spec_acceptance_rate"] - (780 - 300) / (1000 - 400)) < 1e-9,
          f"{d2['spec_acceptance_rate']:.4f}")

    print("[metrics] a server with speculation off yields nothing, not zeros")
    check("no spec metrics -> empty dict", spec_delta({}, {}) == {})

    print()
    if FAIL:
        print(f"FAILED: {FAIL}")
        return 1
    print("all metric-parsing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
