# DOOM RL Arena — Architecture & Product Plan

> A microscope for an RL agent's mind, in the browser, that 30 students can share at once.

**Status:** working draft, v0.1 (2026-05-02)
**Owner:** ignaneswara
**Source repo:** [indugapallignaneswara/DOOMgame-using-RLagent](https://github.com/indugapallignaneswara/DOOMgame-using-RLagent)

---

## 1. Executive summary

Today the project is a Flask + SocketIO web wrapper around Stable-Baselines3 PPO + ViZDoom. It works as a demo, but it bakes in five hard ceilings that block it from being a product:

1. Threading-mode SocketIO single-process server (`app.py:19`)
2. One global `TrainingManager` shared across all users (`doom/trainer.py:23`)
3. JSON registry behind a Python lock (`app.py:41–58`)
4. Frame stream via base64-JPEG over SocketIO (`doom/player.py:144`) — ~10× the bytes of WebRTC, encoder pinned to env-step thread
5. `model_path` resolved from a user-controlled `model_id` with no sandboxing (`app.py:349`)

The deeper miss is **pedagogical**: the UI shows that *training is happening* (loss curves, FPS) but never shows *what the agent has learned*. Students watch numbers go down. They don't watch the agent's eye, its mind, or its hesitation.

This document proposes (a) a five-feature "moonshot" set that turns the project into an interpretability microscope, (b) a three-plane architecture that scales to a classroom of 30+ concurrent students per cohort with public sharable run links, and (c) a four-phase roadmap from current state to product launch.

---

## 2. Current state — what we're working with

| Layer | Today | Code reference |
|---|---|---|
| Web server | Flask + SocketIO, threading async_mode | `app.py:19` |
| Trainer | One global `TrainingManager`, daemon threads | `doom/trainer.py:23–62` |
| Algorithm | PPO only, CnnPolicy, fixed obs (100×160×1) | `doom/trainer.py:88` |
| Env | ViZDoom via `VizDoomGym`, frame_skip=4 | `doom/environment.py:24` |
| Reward shaping | Per-step deltas (kill, damage, ammo) | `doom/rewards.py:7–80` |
| Live metrics | `WebTrainingCallback`, emit_freq=50 | `doom/callbacks.py:11` |
| Live frame | RGB → BGR → JPEG (q=75) → base64 | `doom/player.py:144` |
| Charts | Chart.js, 200-point cap | `static/js/charts.js:188` |
| Persistence | JSON files under `models/` | `app.py:41–96` |
| Eval | `Evaluator.evaluate()`, deterministic policy | `doom/evaluator.py:14` |
| Auth | Username-only, no password | `app.py:215` |
| GPU | AMD MI300X (~192 GB VRAM), single device | runtime |

**Strengths:** clean separation between trainer / player / evaluator / scenarios. Reward shaping is configurable. Pause/resume/stop already wired. SB3 abstractions mean swapping algorithms is mostly a one-line change. Notebook-style readability.

**Pedagogical gaps:** no policy/value visualisation, no advantage/return scrubbing, no policy-vs-policy comparison, no concept-driven lesson templates, no "why did the loss go down at step 47k?" tooling.

---

## 3. Vision — re-framing the product

We are not building "watch a curve and a sprite." We are building **a microscope for an RL agent's mind, with a multiplayer classroom around it.**

Three sentences for the deck:

> An RL classroom where students don't just train agents — they *see* them think. Saliency, value, hesitation, and counterfactuals are first-class citizens of the UI, not optional plots. The arena is multiplayer: agents fight each other, ELO leaderboards drive engagement, and every saved model auto-generates a printable card.

### Who's it for?

- **Primary:** undergrad / Masters CS courses on RL, autonomous systems, deep learning. Instructor sets up a cohort, students each get a quota, weekly assignments, a tournament at the end of term.
- **Secondary:** self-learners on YouTube/Twitch — public sharable run links, "watch my agent train live" social loop.
- **Tertiary:** RL researchers who need a fast UI for sanity-checking new algorithms (the saliency + belief tools are worth the install on their own).

### What does success look like?

- A 50-person CS class can run 50 concurrent training jobs without metrics dropping.
- A student can answer "why did the agent suddenly start hugging the wall at step 80k?" by scrubbing the rollout buffer in the Belief Inspector — without ever running a notebook locally.
- A finished model produces a shareable card with training curve, saliency thumbnail, ELO, and personality scores — students actually screenshot it.
- Three independent classrooms adopt it without anyone from our team in the room.

---

## 4. The five moonshots (priority order)

These are the features that turn the project from "school project demo" into "people film themselves using it."

### 4.1 Brain-cam — live saliency overlay

**The pitch:** every K frames, run **Grad-CAM** (or integrated gradients) on the policy's last convolutional layer; tint the heatmap, overlay it on the live game frame in `/play`. Slider for opacity; toggle for raw / saliency / blended.

**Why it matters:** students see the agent literally fixating on the enemy's torso, then the door, then the medkit. One screen, one "aha." Today's `/play` is a JPEG. Tomorrow's `/play` is a brain scan.

**Feasibility:** Grad-CAM on SB3's `CnnPolicy` is ~30 lines (hook into `policy.features_extractor.cnn[-1]`). Cost: one extra forward pass per overlay frame; for a 100×160 input on MI300X this is sub-millisecond.

**Caveats** (research lit): Grad-CAM on RL CNNs is noisier than on supervised classifiers — value head and policy head produce different maps; should let users toggle which head's saliency is shown. SmoothGrad on top reduces flicker.

**Risk:** the heatmaps may look noisy for under-trained agents. Mitigation: gate the feature behind a "training reward > X" threshold OR display a "this agent is still learning" disclaimer.

### 4.2 Belief Inspector — DevTools for an agent

**The pitch:** pause training at step T. Scrub the rollout buffer like a video timeline. For each frame show:

- **Action probabilities** as horizontal bars (which actions did the agent consider, not just the one it took?)
- **V(s)** — critic prediction, plus realised return, plus residual = "surprise"
- **Â(s,a)** — generalised advantage (already computed by SB3 during PPO update)
- **Saliency** for that frame

Below the timeline: gradient norm per parameter group, KL-divergence to old policy, entropy — at the point in the rollout being scrubbed.

**Why it matters:** RL is famously opaque. With this UI, "the loss went down at step 47k" becomes *"at step 47k the agent finally noticed it had ammo and started shooting on first contact instead of dodging — here's the exact transition where its V(s) jumped from 0.2 to 0.8."*

**Feasibility:** all numbers above are already computed by SB3 internally — we just need to (a) persist the rollout buffer per update step (~1 MB/step for our obs size, manageable), (b) ship the timeline component (uPlot is the right tool — Chart.js cannot scrub 1M points), (c) recompute saliency on demand in a worker.

**Storage:** rollout buffer is ephemeral in SB3; we'd add a hook in `OnPolicyAlgorithm` to dump it to MinIO before each `train()` call. ~100 KB/step compressed × 100k steps × 9 scenarios = ~100 GB per cohort term. Cold-storage policy moves old buffers to glacier after 30 days.

### 4.3 Counterfactual replays — "what if?"

**The pitch:** inside the Belief Inspector, click "what if". Replay the same episode, but force a different action at frame T. A small learned dynamics model predicts the divergent trajectory; the browser shows two paths side-by-side.

**Why it matters:** value functions are hard to teach abstractly. Counterfactuals make them tangible. *"You went left and got hit. If you'd gone right, here's the predicted outcome — the value function picked the lower-risk branch, but barely."*

**Feasibility:** harder. Three options, in increasing fidelity / cost:

1. **Cheap baseline** — no dynamics model; just compare V(s, a_chosen) vs V(s, a_alt). Shows the value function's *belief* about the alternative. Truthful but less visceral.
2. **Dreamer-lite** — train a tiny world model (RSSM with ~5M params) alongside the agent on stored rollouts. Predicts next-frame and reward. Full counterfactual rollouts of ~30 frames. ~1 day GPU per scenario.
3. **Frame-replay with branch detection** — when the env has determinism flags (ViZDoom does), we can re-execute the env from a saved seed + state, swapping one action. Ground truth, no model needed, but only works for *one-step* counterfactuals (after that, randomness diverges).

Recommend (3) for MVP — ground truth, immediate. Add (2) later for multi-step branches.

### 4.4 Reward Sculptor — "RLHF for kids"

**The pitch:** student paints regions of a frame: green = "you want this," red = "you don't." A small auxiliary network learns from these brush strokes; its output is added to the env reward as a shaping term. Two minutes later they're watching an agent they personally taught to prefer cover.

**Why it matters:** reward design is the hardest and most under-taught part of RL. The current UI lets you set five scalars (`kill_reward`, `damage_penalty`, …). This lets you *paint* what you want, then watch it work. Inverse-RL becomes finger painting.

**Feasibility:** mid. The aux network is a tiny ConvNet (~100k params) that maps `obs → r_aux ∈ ℝ`. Trained online with cross-entropy against student labels (positive samples = green strokes, negative = red). The shaping reward is added to the env reward inside `RewardShapingWrapper`. Latency: students paint, see effect within a few thousand env steps (~1 minute on MI300X).

**Risk:** students can paint adversarial rewards that hack the agent. That's actually a feature for a class on reward hacking. Add a "broken agent" hall-of-shame.

### 4.5 Tournament Arena + ELO + model cards

**The pitch:** every saved model auto-generates a **model card**: training curve thumbnail, hyperparams, scenarios beaten, "personality profile" (aggression, exploration, efficiency — derived from action histograms over eval episodes). Multi-agent ViZDoom maps + ELO + weekly tournaments turn homework into sport.

**Why it matters:** without a leaderboard, training tops out at "the loss looks fine, I'll move on." With one, students iterate. The model card is the screenshot that gets shared on Discord and brings new students in.

**Feasibility:** straightforward. ViZDoom has multi-agent support (`Mode.MULTIPLAYER`); two pre-trained policies + a tournament scheduler + an ELO update rule (k=32, standard) is a few hundred lines. Model cards are HTML templates rendered to PNG via Playwright.

### 4.6 Bonus moonshot — LLM commentator

A small Claude / Llama loop watches the live demo, takes a screenshot every 5s, and generates color-commentary in natural language: *"The agent has hesitated at the corner three times. Its V(s) dropped from 0.7 to 0.3 — classic exploration anxiety. It's about to fire."*

Feasibility: trivial (one Claude API call per chunk). Impact on classroom feel: huge. Pair with TTS for full Twitch-streamer effect. Feature-flag for cost control.

---

## 5. Architecture — three planes, scale-out by default

```
┌──────────────────── Browser (Next.js + React + TS) ─────────────────────────┐
│ uPlot  ← charts, rollout-buffer timeline (1M points @ 60fps)                │
│ <video> ← WebRTC game stream (sub-100ms latency)                            │
│ Three.js ← 3D policy embedding gallery (t-SNE / UMAP of features)           │
│ Pyodide cell ← students write a 30-line custom policy in the browser        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTPS · WSS · WebRTC
┌─────────────────── API / Realtime gateway ──────────────────────────────────┐
│ FastAPI + uvicorn ← REST                                                    │
│ NATS or Redis Streams ← live metric pub-sub                                 │
│ LiveKit (or self-hosted Pion) ← WebRTC SFU for game video                   │
│ OAuth2 / SSO ← per-cohort auth, per-user GPU quota                          │
└──────────────────────┬──────────────────────────────┬───────────────────────┘
                       │                              │
┌── Job plane ────────┐│        ┌──── Data plane ────┴────────────────────────┐
│ Ray cluster         ││        │ Postgres ← registry, runs, users, ELO       │
│ ├ Train workers     ││        │ Redis ← session state, live metrics pub-sub │
│ ├ Eval workers      ││        │ MinIO/S3 ← weights, rollouts, saliency PNG  │
│ ├ Tournament pool   ││        │ pgvector ← embedding search ("show me agents│
│ └ cgroup quotas     ││        │            with attention like this")       │
│                     ││        │ W&B ← cross-run analytics, sweeps           │
└─────────────────────┘│        └─────────────────────────────────────────────┘
                       │
              GPU pool: MI300X / H100, scheduled by Ray
```

### Tech-pick rationale

| Choice | Alternative considered | Why this |
|---|---|---|
| Ray | k8s Jobs, raw subprocess | GPU-aware scheduling, fault tolerance, Ray Tune for sweeps for free. `train_all.py` is the toy version of a Ray job. |
| FastAPI | Flask, Django | First-class async, OpenAPI, websocket support, type-safe. |
| WebRTC | Base64-JPEG (current), MJPEG | ~10× smaller, native `<video>` decode, sub-100ms latency. **Required** for saliency overlay to feel live. |
| uPlot | Chart.js, recharts | The current 200-point cap exists because Chart.js falls over at 10k points. uPlot does 1M without breaking. **Required** for rollout-buffer scrubbing. |
| Postgres + pgvector | Mongo, MySQL | Saliency maps and policy embeddings become searchable. |
| MinIO | S3 directly | Self-hosted for data-residency in academic settings; same API. |
| Pyodide | Jupyter kernel per user | Sandboxed in browser; no per-user kernel cost. |
| LiveKit | Naive WebRTC peer-to-peer | SFU scales fan-out to a class of 30. |
| Next.js | SvelteKit, plain Vite | SSR for share-links, mature ecosystem, easy auth integrations. |

### Storage budget (per cohort term, 50 students × 2 trainings/wk × 12 wks)

- Weights (~10 MB/model × 1200) → ~12 GB
- Rollout buffers (compressed) → ~120 GB (90-day retention, then glacier)
- Saliency PNGs (1 per 1k steps) → ~5 GB
- W&B artefacts → ~20 GB
- Postgres → <1 GB

Total ≈ 160 GB hot, manageable on a single MinIO node with 1 TB.

---

## 6. Roadmap — four phases

### Phase 0 — Hygiene (this week, ~2 days)

While the existing 9 trainings finish — fix the leaks before any new feature.

- [ ] Sandbox `model_path` resolution: registry stores S3-style keys, not filesystem paths. Translate at lookup time. Block traversal at the API layer.
- [ ] Server-side input validation on hyperparams (n_steps ∈ [128, 16384], lr ∈ [1e-6, 1e-2], etc.). Currently passed straight through.
- [ ] Move JPEG encoding off the env-step thread (small worker pool with bounded queue).
- [ ] Backport wandb integration from `train_one.py` into `doom/trainer.py` so the existing UI also logs there.
- [ ] Fix the SocketIO `start_demo` event to validate `scenario_key` against the registry instead of trusting the client (`app.py:354`).

### Phase 1 — Interpretability MVP (2 weeks)

Stay on Flask. Pile on features. This is what makes the demo emotionally compelling.

- [ ] **Brain-cam saliency overlay** — `doom/saliency.py` (Grad-CAM on `features_extractor`), new SocketIO channel `game_saliency`, canvas layer in `static/js/player.js`.
- [ ] **Belief Inspector** (read-only first pass) — show action probabilities as bars on the live frame, V(s) as a sparkline above the canvas. No timeline scrubbing yet.
- [ ] **Lesson templates** — preset experiments rendered as "Start" buttons: "PPO vs A2C side-by-side", "with vs without reward shaping", "effect of γ", "sparse vs dense".
- [ ] **Action histogram** in the model detail page — the seed of "personality profile."

### Phase 2 — Foundation rebuild (4 weeks)

No new user-facing features. Better bones.

- [ ] FastAPI rewrite of API surface, OpenAPI generated.
- [ ] Postgres + Redis + MinIO; SQLAlchemy + Alembic migrations.
- [ ] Ray cluster (single-node initially, Ray Train for jobs).
- [ ] Next.js frontend served at `/v2`; old UI stays at `/` for soft cutover.
- [ ] WebRTC streaming (LiveKit, single-server topology to start).
- [ ] OAuth2 (GitHub + Google) + per-user GPU quota.
- [ ] Auto-detect PPO vs DQN vs A2C in registry (algo abstraction).

### Phase 3 — Product (8 weeks)

- [ ] Tournament Arena + ELO + model cards.
- [ ] Reward Sculptor (paint-to-shape).
- [ ] Counterfactual replays (one-step, env-determinism path).
- [ ] Multi-algo support: DQN, A2C, SAC, Dreamer-V3-lite. Algorithm cards explain trade-offs in plain English.
- [ ] Multi-game: MiniGrid (text-rendered, easy to label), Procgen, Atari, Crafter. ViZDoom stays the headliner.
- [ ] Class management: cohort creation, assignment templates, grading rubrics, CSV export of student progress.
- [ ] Public sharable run links (`/run/<id>`, no login required).

### Phase 4 — Moonshots (parallelizable, optional)

- [ ] LLM commentator + TTS.
- [ ] Pyodide policy editor (custom 30-line policies, sandboxed).
- [ ] Policy-embedding gallery + similarity search.
- [ ] Counterfactual replays with full Dreamer-style world model.
- [ ] Mobile companion app — push notification when your training finishes, live-watch on the train.

---

## 7. Success metrics — how we know it's working

| Metric | Target by | Source |
|---|---|---|
| Concurrent training jobs without metric drops | 50 | end of Phase 2 |
| Median game-stream latency | < 100ms | Phase 2 |
| Time-to-first-saliency for a new student | < 2 min from signup | Phase 1 |
| Models with public share links | 1000 | Phase 3 |
| Cohorts using it (instructor signup) | 10 | end of Phase 3 |
| Daily active students | 200 | end of Phase 3 |
| GitHub stars | 2000 | end of Phase 3 |
| Median session length | > 12 min (vs the ~2 min "I clicked start and watched a curve" baseline) | Phase 1 |

---

## 8. Open questions / risks

1. **AMD ROCm tooling is shakier than CUDA.** The MIOpen race we hit today (`gfx942.ukdb` collision) is one example. If we go multi-cluster, do we standardise on H100? Cost trade-off.
2. **Multi-agent ViZDoom for tournament is finicky.** `Mode.MULTIPLAYER` has a known quirk around port allocation under load. May need a custom runner.
3. **Rollout-buffer storage at scale** (Phase 1's Belief Inspector) — ~120 GB / cohort is fine for one institution, but a public-facing version would need aggressive sampling or 30-day TTLs.
4. **Pedagogical research collaboration** — should we partner with a Learning Sciences lab to A/B-test saliency-overlay vs no-overlay on student understanding? Could be the paper that anchors the credibility argument.
5. **Pricing model** — free for individual self-learners, paid per-seat for institutions? Or grant-funded open source forever? Affects every architecture decision (multi-tenant isolation, billing, SSO providers).
6. **Llava / CLIP-style vision model for "what is the agent looking at" semantically** — could replace Grad-CAM for natural-language descriptions ("the agent is looking at the enemy's left flank"). More ambitious; bookmark for Phase 4.

---

## 9. Reviews and contributions

The following sections are critiques and contributions from domain experts who reviewed v0.1 of this plan.

<!-- AGENT_SECTIONS_START -->

### Learning Sciences review — Dr. Riya Kapoor (Vanderbilt Peabody)

**Conditional recommend.** The plan is unusually thoughtful for an engineering document — the framing in §3 ("students don't just train agents; they *see* them think") is exactly the move the field has been waiting for — but two of the moonshots will, as currently scoped, *increase* extraneous cognitive load rather than reduce it, and the lesson templates as listed (§6, Phase 1) will produce confident students who hold systematically wrong mental models. So: ship it, but not without the changes below.

The moonshot that earns its keep is **§4.4, the Reward Sculptor**. This is a rare case where a UI affordance maps cleanly onto a well-evidenced learning mechanism. Painting a reward and watching the agent's behaviour diverge over the next two minutes is a textbook *productive failure* loop in the Kapur (2008, 2014) sense — students generate a flawed solution, encounter the consequence as a perceptual surprise, and only *then* are primed to receive the formal definition of reward hacking. The fact that students can produce an adversarial reward is not a risk to mitigate; it is the entire pedagogical point. I would push the team to lean harder into this and pre-stage at least three "broken-agent" gallery exemplars so students have an anchor for comparison (Schwartz & Bransford's "time for telling," 1998).

The pedagogically suspicious one is **§4.3, counterfactual replays with a Dreamer-lite world model**. Embodied/grounded cognition work (Barsalou; Goldstone & Son's *concreteness fading*) tells us that learners need to trust the substrate before they can reason over it. A learned dynamics model produces a plausible-but-wrong second timeline and presents it with the same visual fidelity as the real one — students cannot tell which branch is ground truth. This is a recipe for the *illusion of explanatory depth* (Rozenblit & Keil, 2002). Option (3), env-determinism replay, is the right MVP and arguably the *only* defensible version for novices; option (2) should be gated behind explicit epistemic markers in the UI ("imagined") before it ever reaches an undergrad.

The concept the lesson templates (§6 Phase 1) silently omit is **the bias-variance / on-policy-vs-off-policy distinction**, and more broadly *why PPO updates discard old data*. Students will leave the course able to tune γ but unable to explain why DQN has a replay buffer and PPO does not. Add a template called "stale data" that visualises the importance-sampling ratio blowing up.

**Concrete UI proposal the author missed:** a *prediction-before-reveal* widget. Before the live frame advances, freeze it for 800 ms and ask the student to click which of three action bars they think the agent will choose; reveal, score, log. This implements the "generation effect" (Slamecka & Graf, 1978) and Vygotskian ZPD scaffolding inside the existing Belief Inspector with no new backend — it's a 100-line React overlay on §4.2.

**Take-away:** an interpretability microscope is only pedagogical if the UI forces the student to commit to a prediction before the system reveals the answer; otherwise you've built a beautiful aquarium.

---

### ML / RL research review — Dr. Anand Iyer

**Verdict:** the product instincts are sharp — RL is opaque, and a microscope UI is genuinely undersupplied — but §4.1 and §4.3 currently sit a half-rung below the rigour bar a research-credible interpretability tool needs. Fix those, and there is a real publishable artefact hiding here.

**§4.1 — Grad-CAM is interpretability theatre, by default.** Selvaraju et al. 2017 designed Grad-CAM for class-discriminative gradients in supervised CNNs. Two assumptions break in RL: (a) policy logits are not class scores — they are sampled from a distribution under a non-stationary objective, so gradients shift across PPO updates and produce visually plausible-but-unfaithful maps; (b) the value head and policy head induce *different* causal saliencies, and blending them is meaningless. Worse, sanity checks (Adebayo et al. 2018, "Sanity Checks for Saliency Maps") routinely fail for Grad-CAM under model/parameter randomisation — students will be looking at edge detectors, not policy reasoning. **Recommendations:** (1) ship **perturbation-based saliency** (Greydanus et al. 2018, "Visualizing and Understanding Atari Agents") as the *default* — it is causal, model-agnostic, and the gold standard for RL pixel attribution; (2) keep Grad-CAM/IG only as a fast preview, with an explicit "approximate" badge; (3) add **SARFA** (Puri et al. 2020) for policy-specific specificity-relevance trade-offs; (4) for the eventual transformer-policy path, attention rollout (Abnar & Zuidema 2020) is the right tool, not Grad-CAM. SHAP/LIME are too slow per-frame; integrated gradients (Sundararajan et al. 2017) is a reasonable middle path. Run Adebayo's randomisation test in CI — do not ship a saliency feature without it.

**§4.3 — Dreamer-lite at 9 scenarios, honestly.** The plan estimate of "1 day GPU per scenario" is optimistic by ~3–5×. RSSM tuning across 9 ViZDoom scenarios with reward sparsity variance is realistically a 3–4 week eng-month, mostly on KL-balancing, free-bits, and reward-head calibration (Hafner et al. DreamerV2/V3). Failure modes you will hit: **model exploitation** (the agent finds high-reward fictions in latent rollouts — Janner et al. 2019), **value drift** under off-policy correction, and **OOD generalisation** beyond ~15 imagined steps. Stick with option (3) for MVP; (2) is a research project, not a feature.

**Publishable angle.** The dataset that falls out of §4.2 — per-step rollout buffers + saliency + value residuals across 9 scenarios and 50+ student-trained policies — is a **NeurIPS Datasets & Benchmarks** submission: *"DOOM-Scope: a multi-policy interpretability benchmark with paired student annotations."* Pair it with the Reward Sculptor strokes (§4.4) and you have human reward-shaping signals that ICML's IMOL workshop would take tomorrow.

**Algorithm zoo.** Drop A2C (PPO dominates it on every axis). Add **IMPALA** for the multi-student distributed-actor story, **Decision Transformers** for the offline-RL/return-conditioning lesson, and a **MuZero-mini** variant that doubles as the dynamics model for §4.3. SAC is dead weight on a discrete-action ViZDoom — keep it only if you add a continuous-control env.

**Take-away:** ship perturbation saliency first, demote Grad-CAM, defer Dreamer-lite, and start curating the rollout dataset on day one — that is the artefact that outlives the demo.

---

### Frontend / UX review — Marco Velez

Headline verdict: the moonshots are right, the frontend stack is half-right, and §5 is going to bleed you out in week three of Phase 2 if you don't make two corrections now. The plan reads like a backend engineer's idea of a frontend — the boxes are labelled correctly, but the seams between them are where the UX lives or dies, and the doc currently waves at those seams.

The fight I want to pick: **LiveKit for game video is overkill and Pyodide for student policies is under-kill.** LiveKit is a 30-MB client SDK, a TURN/STUN dance, and a vendor lock-in tax for what is fundamentally one-way, low-resolution (100×160 upscaled), unidirectional video to ≤30 viewers per cohort. Use **WHEP (WebRTC-HTTP Egress Protocol)** with a thin Pion/Janus/MediaMTX origin and let the `<video>` element do the work — same sub-100ms latency, no SFU billing, no SDK, and you can fall back to a Media Source Extensions H.264 stream behind a CDN for "watch my training live" share-links without spinning up a session. Conversely, Pyodide for student-authored policies is going to die the moment somebody imports `torch`. Pyodide's PyTorch story is incomplete and a 200-MB wheel download per cold load is not a "killer first interaction." Run student code server-side in a **gVisor- or Firecracker-sandboxed worker** with a 5-second wall clock and a Monaco editor in the browser — students get real torch, real GPU, and you get one execution model instead of two.

The moonshot whose UX is *much* harder than the doc admits is **§4.1 Brain-cam**. Synchronising a Grad-CAM heatmap to a 60fps WebRTC frame is a coordinate-transform nightmare: the saliency tensor is 100×160 in policy-space, the `<video>` element is letterboxed at whatever the user's window is, and the heatmap arrives on a *different* WebSocket clock than the video frames. You will get drift. You need PTS-stamped saliency frames, a `requestVideoFrameCallback` loop, and a canvas overlay that resamples on every resize. Then color-blindness: viridis is not safe for deuteranopes against red enemies — you need a perceptually-uniform diverging palette *and* a contour-line fallback toggle. And on a phone the whole thing is unreadable below 360px wide; budget two weeks just for this one overlay.

Killer first interaction, present-tense: I land on doom-rl-arena.app with no account. The page is already showing somebody else's agent playing the `defend_the_center` scenario in a 16:9 hero canvas — saliency is on, dimmed to 30%, and a soft yellow halo tracks where the agent is looking. A typewriter caption in the corner reads *"It just noticed the imp on the left — V(s) jumped from 0.4 to 0.7."* Below the canvas, three buttons: **Watch another**, **Train your own (90 seconds)**, **Fork this agent**. I click Fork, GitHub OAuth pops, and within fifteen seconds I'm watching *my* copy train, with a confetti pop the first time my reward crosses zero.

Missing feature: **shareable replay GIFs with embedded saliency**, auto-cut to the highest-surprise 6-second window per training run. Twitter/Discord embeds are how this project goes viral, and a static model-card PNG won't carry it — motion does.

Take-away: the visualisations are the product. Hire a frontend who has shipped real-time canvas before Phase 2 starts, not after.

---

### Backend / Infrastructure review — Priya Sundaresan

**Verdict:** the architecture is directionally sane and ambitious in the right places, but §5 is an org chart drawn as a system diagram. The hard parts — GPU multi-tenancy, queueing, blast-radius — are hand-waved, and the back-of-envelope math doesn't survive contact with a real classroom.

**The math.** 50 students × 2 trainings/wk = 100 jobs/wk, ~14 active during peak office hours (Mon evening before due date — this is always when it breaks). At 2 GB VRAM/trainer, peak is ~28 GB on a 192 GB MI300X. VRAM is not the bottleneck. The bottleneck is what you saw today: **MIOpen kernel-cache contention on `gfx942.ukdb`**, plus PCIe/HBM bandwidth and SMI-level scheduling. ROCm doesn't have MPS or MIG. You will get tail-latency cliffs above ~8 concurrent processes per device long before VRAM saturates. Plan for ~10 concurrent trainers/MI300X with a per-user `MIOPEN_USER_DB_PATH` and a serialized warm-up phase, not 14+. The "single MI300X handles a cohort" claim is true on paper, false under load. Budget two MI300X with affinity pinning, or accept queueing.

**Picking a fight.** Ray over k8s for a 1-node GPU cluster is fine; Ray over k8s at three classrooms is going to hurt — Ray's autoscaler is not a substitute for a real control plane, and you'll reinvent half of k8s in `serve.py`. Keep Ray for the actor model inside a pod, run Ray-on-k8s (KubeRay). On storage: pgvector for ~50k saliency embeddings is correct — don't pay the Pinecone tax until you cross 10M vectors. W&B, however, is load-bearing in §6 with no fallback; if their pricing changes or their auth is down during a tournament, you're cooked. Mirror runs to local MinIO + a thin Grafana view from day one.

**The missing piece.** There is no job queue between FastAPI and Ray. When 30 students hit "train" at 7pm Sunday, FastAPI will happily accept all 30, Ray will OOM the device, and one student's bad hyperparams will take down nine peers. Put NATS JetStream or Redis Streams in front of Ray with a per-user concurrency limit (1 active, 2 queued) and a hard cgroup memory ceiling per trainer. Also missing: replay-buffer IO. 100 KB/step × 100k steps × 50 students = a write storm of ~500 GB hot data churning through MinIO; you need a single-writer-per-run pattern with batched 4 MB PUTs, not naive per-step writes.

**Cost, 50 students × 12 weeks.** Cloud: 2× MI300X on Hot Aisle / TensorWave at ~$2/hr × 24 × 84 days = ~$8k, plus ~$500 storage/egress. H100 spot equivalent: ~$15k. On-prem amortised on a single owned MI300X box: ~$1.5k power + ops. Verdict: own one box, burst to cloud for tournament week.

**Take-away:** the architecture will demo beautifully and fall over the second week of term. Add a queue, pin MIOpen caches per-user, mirror W&B locally, and budget for two GPUs — then it's a product.

---

### Product / GTM review — Lena Park

Verdict up front: this is a research demo with course-portfolio DNA dressed up as a product plan. It's a genuinely interesting microscope — but the doc reads like an ambitious capstone that hasn't yet picked a buyer, and that's the gap between a 2,000-star repo and a $200k ARR pilot.

**Where it actually sits.** Gymnasium and PettingZoo are the substrate, not competitors — you sit on top. Spinning Up and CleanRL own "minimal, readable code"; you'll never out-minimal them. Hugging Face's Deep RL course owns the free-curriculum funnel; Coursera/edX own the credentialed seat. Atari leaderboards (and AIcrowd/Kaggle) own competitive RL. Distill.pub set the bar for interpretability prose, but nothing alive today does *live, multiplayer, interpretable* RL in a browser. That's the unfilled gap — not "another RL learning platform," but "the only place a class of 30 can scrub a rollout buffer together." Defend that sentence or the project drowns.

**Audience pyramid is inverted.** Undergrad CS departments are the slowest, most procurement-poisoned buyer in EdTech — 9-month sales cycles, IT security review, no champion willing to risk a syllabus. Put self-learners and YouTube/Twitch creators at the top: they create the model cards, the screenshots, the GitHub stars, and *that* is what gets an instructor to adopt you in year two. Researchers stay tertiary; they'll use it but never pay. Willingness to pay lives with institutions; the viral loop lives with individuals. Don't confuse the two.

**Pricing — pick one and commit.** Free open-source core, paid hosted tier per-seat for institutions ($8–15/student/term, ~$3–5k for a 50-person class), grant/sponsorship (Mozilla, NSF, AMD developer relations) underwriting the public infra. An institutional pilot looks like: one school, one term, one professor, free hosted cohort + 4 hours of your founder time, in exchange for a co-authored case study and a session-recording corpus. Cost to you: ~$400 of GPU + a week of support. Cost to them: zero dollars, two hours of IT paperwork. That's the only shape that closes.

**The shareable feature.** Model cards (§4.5). Not Brain-cam — saliency overlays look like a research artefact to non-RL people. The model card with personality scores, ELO, and a training-curve thumbnail is the *Spotify Wrapped* of RL: legible to a recruiter, postable on LinkedIn, screenshottable on X. Build that polished before Belief Inspector. It is the acquisition channel.

**Non-obvious wedge.** Skip universities for year one. Go straight at AI YouTubers (Yannic, Two Minute Papers, sentdex) with a "remote-control my training" embed — they get live audience interaction, you get 50k impressions per video. Parallel track: become the default substrate for AIcrowd's next RL competition. One sponsored tournament gets you more qualified users than ten course adoptions, and the leaderboard is already built (§4.5).

**Take-away:** The interpretability microscope is real and defensible. The product plan around it is currently solving for the wrong buyer. Lead with creators and competitions; let universities pull you in.

---

### Synthesis — what the reviews collectively change

Reading the five together, four convergent corrections jump out that are stronger than any single review:

1. **Demote Grad-CAM, lead with perturbation saliency** (Iyer + Velez agree from different angles — research integrity *and* UX clock-drift are both easier with per-frame causal occlusion than gradient hooks).
2. **Promote model cards above Brain-cam in the Phase 1 priority list** (Park's acquisition argument — they screenshot, saliency doesn't).
3. **Add a job queue, MIOpen-per-user, and W&B mirroring before Phase 2 starts** (Sundaresan's load-day-one trio — none are speculative, all are observed-from-today's-MIOpen-race).
4. **Add the prediction-before-reveal widget to §4.2 spec** (Kapoor's generation effect — converts the Belief Inspector from aquarium to assessment).

Open disagreements between reviewers worth resolving: Velez wants WHEP over LiveKit (frontend simplicity), Sundaresan implicitly assumes a more managed stack — the team needs to decide whether sub-100ms streaming is a Phase-2 requirement or a Phase-3 polish. Iyer and Kapoor agree to defer Dreamer-lite indefinitely; Velez and Park don't comment on it but neither flagged it as a marketing asset, which is its own answer.

<!-- AGENT_SECTIONS_END -->

---

## Appendix A — proof-of-concept that exists today

While drafting this doc, all 9 ViZDoom scenarios are training in parallel on the MI300X (~14 GB VRAM total, ~2 GB per process). Driver: `train_all.py` (subprocess-per-scenario, shared MIOpen kernel cache, staggered launch). Per-run wandb logging via `train_one.py`. As of writing, 5/9 scenarios complete (`basic`, `defend_the_line`, `take_cover`, `defend_the_center`, `health_gathering` — all 100K-step). Final weights in `/mnt/orbit_war_disk/doom-checkpoints/<scenario>/<scenario>_final.zip` plus W&B artefacts.

This is the toy version of the Ray-based job plane in §5. The wandb dashboard is the toy version of the analytics layer.

W&B project: https://wandb.ai/ignaneswara-srm-institute-of-science-and-technology/doom-rl
