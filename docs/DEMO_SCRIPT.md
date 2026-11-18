# Screen-recording script (≈ 8 minutes)

Record the terminal and browser together. Keep Ollama's `ollama ps` / logs visible in a small terminal.

1. **0:00 Intro (20 s)** — "Sellbird extended into a local multi-product sales agent; zero paid AI." Show repo + `ollama list` (qwen3:1.7b).
2. **0:20 Hardware (30 s)** — `python -m sellbird.local_sales hardware`; point at RAM/CPU/GPU and the recommended model + profile.
3. **0:50 Start (20 s)** — `python -m sellbird.local_sales serve`; open http://127.0.0.1:8765. Show the green "Ollama ready" banner.
4. **1:10 Configure (40 s)** — *Load company + prospect files* (two products appear), *Generate business profile*.
5. **1:50 ICP (60 s)** — select VaultGuard Backup → *Generate ICP*; then ComplyTrack → different ICP (product-specific).
6. **2:50 Pipeline (2 min)** — set 4 prospects → *Run pipeline*. While waiting, scroll the **Local inference log** (endpoint 127.0.0.1, tokens/sec, $0).
7. **4:50 Results (1.5 min)** — ranking table; open *Details* for a tier-A prospect: research, signals (grounded flag), problem statement, score breakdown, email + checks. Open the bakery/low-fit prospect: no signal → no email.
8. **6:20 Replies (40 s)** — paste a reply, *Classify*; run the 12 sample conversations and show accuracy.
9. **7:00 Failure handling (40 s)** — stop Ollama; click a workflow: red banner, `queued` / `degraded`. Restart Ollama → *Process queue*.
10. **7:40 Offline test (1 min)** — **turn Wi-Fi off on camera**, run `python scripts/offline_acceptance.py --max-prospects 4` and `python -m sellbird.local_sales audit`; show PASS, 0 blocked attempts, $0, loopback endpoint; open `logs/inference.jsonl`.
11. **8:40 Close** — mention synthetic data + known limitations file.
