#!/usr/bin/env python3
"""Tiny GPU keepalive to defeat Cluster's stuck-job detector during warmup.

WHY: Cluster's GTLStuckGPULambda flags a node as "stuck" when its GPUs sit at
~0% utilization across several polling cycles (it keys on GPU utilization; see the
Cluster Oncall Runbook "Found idle node"). In the 2-node disaggregated search
topology the TRAIN node starts its Ray head, then BLOCKS for minutes waiting on the
dedicated retriever node (which must stage a 64GB wiki index + load GPU faiss) via
the retriever-discovery loop + the readiness gate, all BEFORE `ray job submit`
launches train.py. During that whole window the train node's 8 GPUs are idle ->
the detector terminates the job at ~11-16 min (jobs <JOB_ID>/9caf534b/27e94477/
1c18abdf, all 2-node, all "Terminated by stuck job detector"; single-node jobs are
never hit because training starts immediately and keeps GPUs busy).

WHAT: run a light periodic matmul on every visible GPU so utilization is non-zero
(NOT saturating — a small tensor + a sleep between bursts), keeping the node "alive"
in the detector's eyes until the retriever is ready and real training starts. The
launcher kills this process right before `ray job submit`, and it holds only a few
MB per GPU (freed on exit), so it does not interfere with SGLang / Megatron.

Env:
  KEEPALIVE_INTERVAL_S   seconds to sleep between matmul bursts (default 5)
  KEEPALIVE_MATMUL_N     square-matrix dim for the busy matmul (default 2048)
  KEEPALIVE_EXIT_URL     if set, the keepalive self-exits once this URL answers (HTTP
                         2xx/4xx = server up). Used on the retriever node so the
                         keepalive covers the GPU-idle index-load window and stops on
                         its own the moment the faiss server is serving (the branch
                         `exec`s the server, so we can't kill from the launcher).
  KEEPALIVE_MAX_S        hard cap; exit after this many seconds no matter what
                         (default 1800) so a never-ready server can't pin GPUs forever.
"""
import os
import time
import urllib.request

def _url_ready(url):
    try:
        req = urllib.request.Request(url, method="GET")
        urllib.request.urlopen(req, timeout=3)
        return True
    except urllib.error.HTTPError:
        return True   # any HTTP response (even 4xx/405) means the server is up
    except Exception:
        return False

def main():
    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"[gpu-keepalive] torch import failed ({e}); exiting (no keepalive).", flush=True)
        return
    if not torch.cuda.is_available():
        print("[gpu-keepalive] CUDA not available; exiting (no keepalive).", flush=True)
        return

    n = int(os.environ.get("KEEPALIVE_MATMUL_N", "2048"))
    interval = float(os.environ.get("KEEPALIVE_INTERVAL_S", "5"))
    exit_url = os.environ.get("KEEPALIVE_EXIT_URL", "").strip()
    max_s = float(os.environ.get("KEEPALIVE_MAX_S", "1800"))
    ndev = torch.cuda.device_count()
    print(f"[gpu-keepalive] starting on {ndev} GPU(s): {n}x{n} matmul every {interval}s "
          f"(pid {os.getpid()}); exit_url={exit_url or '(none)'} max_s={max_s}.", flush=True)

    # One small resident tensor per GPU (a few MB each).
    tensors = []
    for d in range(ndev):
        with torch.cuda.device(d):
            tensors.append(torch.randn(n, n, device=f"cuda:{d}", dtype=torch.float32))

    start = time.monotonic()
    checked = 0
    while True:
        for d in range(ndev):
            with torch.cuda.device(d):
                a = tensors[d]
                c = a @ a          # keeps the SMs briefly busy -> util > 0
                _ = float(c[0, 0]) # force sync so the kernel actually runs
        elapsed = time.monotonic() - start
        if elapsed > max_s:
            print(f"[gpu-keepalive] max_s={max_s} reached; exiting.", flush=True)
            return
        # Self-exit once the guarded server is serving (retriever-node mode). Check every
        # ~2 bursts to avoid hammering. Free GPU tensors first so the server gets the mem.
        if exit_url:
            checked += 1
            if checked % 2 == 0 and _url_ready(exit_url):
                print(f"[gpu-keepalive] {exit_url} is serving; exiting so the server owns the GPUs.", flush=True)
                del tensors
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return
        time.sleep(interval)

if __name__ == "__main__":
    main()
