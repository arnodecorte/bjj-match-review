# Roadmap

Five phases, each with a shippable artifact. Nothing is wasted — every phase feeds the next.

---

## Phase 0 — Data Foundation *(Weeks 1–3)*
*Goal: A working position classifier trained on real BJJ footage*

### 1. Ingest existing datasets

See [DATA_SOURCES.md](./DATA_SOURCES.md) for full details, download links, and integration notes on all three sources.

```
ViCoS Lab (120k images, 18 classes)           ← anchor training set
  + carlosj934/BJJ_Positions_Submissions       ← adds video clips for Phase 1
  + tk1475/BJJ-Pose-Estimation (97% baseline) ← reference pipeline to fork
       ↓
Unified format: COCO keypoints + position label + athlete bounding boxes
```

### 2. Label taxonomy

This decision is a multiplier — getting it wrong means relabelling everything later.

```
POSITIONS (mutually exclusive, per-athlete)
├── Standing (takedown/grip phase)
├── Guard (closed, open, half, butterfly, De La Riva, spider, lasso)
├── Mount (high, low, S-mount)
├── Side Control
├── Back (seatbelt, turtle)
├── Turtle (defensive)
├── North-South
└── Submission Attempt (arm, choke, leg lock)

EVENTS (point-in-time transitions)
├── Guard Pull
├── Takedown
├── Guard Pass
├── Sweep
├── Reversal / Scramble
├── Submission Attempt Start/End
└── Tap / Match End

CONTROL STATE (per-athlete, per-frame)
├── dominant
├── neutral
└── defensive
```

This schema directly maps to the chess.com-style output layer. Every label collected from here forward should conform to it.

### 3. Fine-tune base model

- YOLOv8-pose or RTMPose for two-person keypoint extraction
- Simple MLP or small transformer head on top of keypoints → position classification
- Target: >85% position classification accuracy before moving on

**Shippable artifact:** A model that watches a video and outputs a timestamped position log.

---

## Phase 1 — Technique Detection Layer *(Weeks 4–7)*
*Goal: Detect named techniques (berimbolo, kimura, heel hook etc.) not just positions*

This is the hard, novel problem the existing datasets don't solve.

### Caption mining pipeline

```python
sources = [
    "FloGrappling breakdowns",       # Commentator narrates in real-time
    "BJJ Fanatics instructionals",   # Technique named constantly
    "YouTube: ADCC / IBJJF matches with commentary"
]

for video in sources:
    transcript = youtube_transcript_api.get(video_id)
    technique_mentions = extract_techniques(transcript,
        keywords=TECHNIQUE_VOCABULARY)  # kimura, berimbolo, etc.

    for mention in technique_mentions:
        clip = extract_clip(video, mention.timestamp, window=±4s)
        frames = sample_frames(clip, n=16)
        queue_for_review(frames, weak_label=mention.technique)
```

### Human-in-the-loop review

Label Studio (self-hosted, free):
- Reviewers see 16 frames + weak label e.g. "kimura?" → confirm / reject / correct
- Target: 500 verified clips per major technique class to start

### Model

Video classification on top of pose data — a small 3D CNN or TimeSformer over keypoint sequences. Classifying *movement patterns of skeletons*, not raw pixels. Far more robust to lighting, gi colour, and camera angle.

**Shippable artifact:** Model outputs `{"technique": "berimbolo", "confidence": 0.87, "timestamp": 142.3}` per clip.

---

## Phase 2 — The Review Engine *(Weeks 8–12)*
*Goal: Convert raw model output into a chess.com-style structured report*

### 2a. Positional control scoring

```
Control Score (t) = Σ position_weight[position] × control_multiplier[control_state]

Where:
  back_mount dominant   = 1.0
  mount dominant        = 0.9
  side control dominant = 0.7
  guard neutral         = 0.5
  guard defensive       = 0.3
  ...

Integrate over time → "Positional Efficiency %" for the match
```

### 2b. Event classification → Brilliant / Mistake / Blunder

```
Brilliant:  low_probability_transition  → dominant position (unexpected + successful)
Best move:  high_probability_transition → dominant position (textbook + successful)
Mistake:    dominant → neutral without submission attempt by opponent
Blunder:    dominant → submission or tap
```

Probability is learned from the dataset — how often does X transition to Y? Unexpected successful transitions are "brilliant."

### 2c. LLM narrative layer

```
Input (structured JSON):
{
  "position_log": [...],
  "events": [...],
  "control_scores": {...},
  "athlete_weaknesses": ["guard retention", "back defence"]
}

Prompt: "You are a BJJ analyst. Generate a match review in the style of
chess.com's game review. Use the structured data below. Be specific,
use BJJ terminology, and end with 3 drilling recommendations."
```

The LLM is a fluent narrator of structured data — it is not doing the analysis. That separation of concerns is intentional.

**Shippable artifact:** End-to-end pipeline — video in → JSON report out. No UI yet. Validate output quality with 10 real matches.

---

## Phase 3 — Product & Feedback Loop *(Weeks 13–20)*
*Goal: Put it in front of real BJJ athletes and close the data flywheel*

### Frontend

- Next.js or SvelteKit
- Video player with timeline scrubber + event annotations (like YouTube chapters, but richer)
- Collapsible sections: Overview score → Position breakdown → Events → Drilling recommendations
- Shareable report link (organic growth mechanic — people will post these)

### The feedback flywheel

Every user interaction becomes a training signal:

```
User clicks "this is wrong" on an event annotation
  → correction UI → feeds back into Label Studio queue

User rates drilling recommendation (thumbs up/down)
  → feeds back into LLM prompt refinement

User watches "retry this mistake" clip
  → implicit signal that mistake detection was meaningful
```

**Active learning trigger:** Any frame where model confidence < 0.6 gets automatically queued for human review. By month 5 you'll have thousands of hard-case labels that couldn't be manufactured.

**Shippable artifact:** Public beta — 2 free reviews/month, waitlist for coach tier.

---

## Phase 4 — Scale & Moat *(Month 6+)*
*Goal: Make the dataset and model defensible*

The moat is **data network effects**:

| Action | Moat it builds |
|---|---|
| Every processed match adds to training data | Model improves with scale |
| User corrections improve label quality | Competitors can't buy this |
| Coach tier bulk uploads | Rapid expansion across gi/nogi, weight classes, belt levels |
| Partner with tournament organisers | Multi-angle professional footage |
| IBJJF / FloGrappling data deal | Potentially exclusive high-quality source |

At ~50k processed matches the position classifier will be better than anything a competitor could train from scratch. That's the defensible position.

---

## Timeline at a Glance

```
Wk 1–3   │ Phase 0 │ Data foundation + base position classifier
Wk 4–7   │ Phase 1 │ Technique detection + caption mining pipeline
Wk 8–12  │ Phase 2 │ Review engine + LLM narrative layer
Wk 13–20 │ Phase 3 │ Product beta + feedback flywheel
Month 6+ │ Phase 4 │ Scale, moat, commercial tiers
```

---

## The One Thing Most Builders Get Wrong

They try to solve computer vision perfectly before building the product. Don't.

An 85% accurate position classifier with a great UX will teach you more in 2 weeks of beta users than 3 months of model iteration in isolation. Ship Phase 2 scrappy, let Phase 3 tell you where the model breaks, then fix it with real-world signal.

The chess.com framing is the strongest asset here — it's not a technical feature, it's a *mental model* users immediately understand. Protect that UX clarity at every phase.
