# Colab local vLLM server — runbook

Hosts `meta-llama/Llama-3.1-8B-Instruct` on a Colab GPU and exposes it to the
Pnyx harness (running on your Mac) via a `cloudflared` quick tunnel. Backs
condition `B1` (homogeneous local pool) and the local slice of `C` / `D_k1` /
`D_k3` / `D_k10` in `pnyx/configs/main/`.

## 1. Open the notebook

Upload `colab/pnyx_vllm_server.ipynb` to Google Colab (File -> Upload notebook),
or open it directly from GitHub if the repo is pushed there.

## 2. Select a GPU runtime

`Runtime -> Change runtime type -> Hardware accelerator: GPU`. A **T4 (16GB)**
works — the notebook launches vLLM with `--max-model-len 8192` and
`--gpu-memory-utilization 0.92`, which fits Llama-3.1-8B-Instruct fp16 comfortably
on a T4. An **A100** is faster to boot but not required. Colab Pro is recommended
for longer, more reliable sessions (see caveats below) but is not strictly required.

## 3. Have a Hugging Face token ready

`meta-llama/Llama-3.1-8B-Instruct` is gated:
1. Accept the license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Create a read token at https://huggingface.co/settings/tokens
3. Paste it into Cell 3 when prompted (`getpass`, hidden input, kept only in the
   notebook process's environment — never written to disk).

## 4. Run the cells top to bottom

1. Markdown intro — read it.
2. Installs `vllm` + `httpx`, downloads and chmods the `cloudflared` binary.
3. Prompts for and sets the HF token.
4. Pre-downloads the model weights in the notebook process
   (`snapshot_download`) and sets `HF_HUB_OFFLINE=1`. This is required: the
   Hub download inside the `vllm serve` subprocess crashes with an
   ASCII-locale `UnicodeEncodeError` (progress-bar glyphs) regardless of the
   env vars exported to it, so the weights must already be in the local cache
   before the serve cell runs. First run takes several minutes here.
5. Launches `vllm serve meta-llama/Llama-3.1-8B-Instruct` as a background
   subprocess (port 8000, `--api-key pnyx-local`, `--max-model-len 8192`,
   `--gpu-memory-utilization 0.92`, `--enforce-eager`) and polls
   `localhost:8000/v1/models` until it's up — it loads weights purely from the
   local cache filled by Cell 4. Guided decoding /
   `response_format: json_schema` on `/v1/chat/completions` works out of the
   box — no extra flag needed.
6. Launches `cloudflared tunnel --url http://localhost:8000` as a background
   subprocess and **prints the public HTTPS URL**. This is the URL you need —
   see step 5 below.
7. Smoke test: POSTs a tiny `json_schema` structured-output request to the
   *public* URL with `Authorization: Bearer pnyx-local` and prints the parsed
   reply + token usage. Confirms the whole path (tunnel -> vLLM -> guided
   decoding) works before you point the harness at it.
8. Keep-alive loop — **blocking, run it last**. Prints a heartbeat every 60s,
   checks that both the vLLM process and the tunnel are still alive, and
   auto-restarts the tunnel (printing a new URL) if it died. Leave this cell
   running for the duration of your session; stop it (interrupt/stop button,
   not the whole runtime) when you're done.

## 5. Wire the URL into the Pnyx configs

Cell 6 prints something like:

```
PUBLIC TUNNEL URL: https://random-words-1234.trycloudflare.com
```

Take that URL, append `/v1`, and paste it as `base_url` in each of the three
placeholder configs (all currently have `base_url: REPLACE_WITH_COLAB_URL`):

- `pnyx/configs/main/B1.yaml`
- `pnyx/configs/main/C.yaml`
- `pnyx/configs/main/D_k1.yaml`, `D_k3.yaml`, `D_k10.yaml`

```yaml
# before
base_url: REPLACE_WITH_COLAB_URL
# after
base_url: https://random-words-1234.trycloudflare.com/v1
```

## 6. Set the API key on the Mac side

The local model's `ModelSpec.api_key_env` names an environment variable that
must hold the same value passed to `--api-key` in Cell 5: `pnyx-local`. In this
repo's existing config convention (see the commented-out `local:` block in
`pnyx/configs/pilot.yaml`) that variable is named **`PNYX_LOCAL_KEY`** — check the
`api_key_env:` field actually written into `pnyx/configs/main/B1.yaml` and match
it exactly. Before running the harness:

```bash
export PNYX_LOCAL_KEY=pnyx-local
# or add a line to the repo's .env alongside OPENROUTER_KEY, matching however
# your shell/session loads it
```

If `pnyx/providers.py` raises "missing API key: environment variable ... not
set", the env var name doesn't match what's in the config — fix the name, not
the value.

## ModelSpec knobs for the local model (reference)

When authoring/checking `B1.yaml`, `C.yaml`, `D_k*.yaml`, the local-model
`ModelSpec` entry should look like:

```yaml
local:
  base_url: https://<tunnel>.trycloudflare.com/v1   # from Cell 6, +"/v1"
  api_key_env: PNYX_LOCAL_KEY                              # must match export above
  model_id: meta-llama/Llama-3.1-8B-Instruct
  price_in: 0.0
  price_out: 0.0
  rpm_limit: 120                                      # local, generous
  supports_json_schema: true                          # vLLM guided-decoding
```

`rpm_limit: 120` is generous because there's no external rate limit on your own
GPU (only the harness's own client-side throttle). `price_in`/`price_out: 0.0`
because there's no per-token billing for a self-hosted model. `supports_json_schema:
true` because vLLM's OpenAI-compatible server enforces `response_format:
json_schema` via guided decoding.

## Colab session-length caveats

- **Colab Pro**: sessions last up to ~24h before a forced disconnect; **free
  tier** sessions are much shorter and disconnect on idle far more aggressively.
  Either tier will disconnect on prolonged idle / inactive browser tab — the
  keep-alive cell's heartbeat output helps keep the tab "active" but is not a
  guarantee against Colab's own idle timeout.
- **The Pnyx harness resumes runs safely** (event-log based, kill/resume-safe
  by design — see `pnyx/runner.py`). If Colab disconnects mid-run:
  1. Re-run this notebook from the top (new runtime = new weights download,
     new tunnel).
  2. Cell 6 gives you a **new** public URL — update the same config files
     (`B1.yaml`/`C.yaml`/`D_k*.yaml`) with the new `base_url`.
  3. Re-run the **same** harness command you were running before — it will
     pick up where the event log left off; you do not need to restart the
     experiment from question 1.
- The tunnel URL also changes any time `cloudflared` itself is restarted (e.g.
  by the keep-alive cell after a transient network blip) — watch its printed
  "New public URL" line and update configs again if that happens mid-session.

## Troubleshooting

- **OOM / CUDA out of memory on startup**: lower `--max-model-len` in Cell 5
  (e.g. `4096`) and/or lower `--gpu-memory-utilization` (e.g. `0.85`); rerun
  Cell 5. A T4's 16GB is tight — leaving less KV-cache headroom (shorter
  `max-model-len`) is the primary lever.
- **`ImportError: libcudart.so.13`**: PyPI `vllm >= 0.20` wheels are compiled
  against CUDA 13 only, and Colab's runtime is CUDA 12.4 — so the install cell
  pins `vllm==0.19.1`, the last PyPI release built against CUDA 12.x. If you
  need a newer vLLM, use the commented-out `+cu129` GitHub-release-wheel block
  in the install cell instead of the pin. Note the serve cell also passes
  `--enforce-eager` (avoids T4 kernel-detection issues; harmless on A100).
- **`UnicodeEncodeError: 'ascii' codec ...` in `/content/vllm_server.log`**:
  the Hub download inside the `vllm serve` subprocess crashes under an ASCII
  locale (progress-bar glyphs) no matter what env vars are exported to it.
  That's why Cell 4 pre-downloads the weights in the notebook process and sets
  `HF_HUB_OFFLINE=1` — make sure you ran Cell 4 (successfully) before Cell 5;
  if you skipped it, run it and re-run Cell 5.
- **403 / gated repo error downloading weights**: you haven't accepted the
  license for `meta-llama/Llama-3.1-8B-Instruct` on the Hub with the account
  whose token you pasted in Cell 3, or the token is wrong/expired. Fix at
  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and re-run Cell 3,
  then Cell 4, then Cell 5.
- **Cell 5 never reports "up"**: check `/content/vllm_server.log` (printed in
  the polling output) — a full traceback lands there.
- **Cell 6 never finds a URL / smoke test can't connect**: check
  `/content/cloudflared.log`; cloudflare's quick-tunnel service is occasionally
  slow to hand out a hostname — Cell 6 retries for up to 120s, rerun the cell
  if it times out.
- **Smoke test 401 Unauthorized**: the `Authorization: Bearer pnyx-local`
  header must match `--api-key` in Cell 5 exactly (both are `pnyx-local` by
  default in this notebook — don't change one without the other).
- **Harness gets "missing API key" from `pnyx/providers.py`**: the
  `api_key_env` name in the YAML config and the exported shell variable name
  don't match — see step 6 above.
