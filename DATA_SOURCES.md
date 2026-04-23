# Data Sources

The three anchor sources for Phase 0 of the data strategy.

---

## 1. ViCoS Lab — Brazilian Jiu-Jitsu Positions Dataset

**URL:** https://vicos.si/resources/jiujitsu/

The primary anchor dataset. 120,279 labeled images of two athletes sparring, captured across 6 sparring sequences with 3 smartphone cameras.

| Property | Detail |
|---|---|
| Images | 120,279 |
| Positions | 10 combat positions, 18 classes |
| Keypoint format | MS-COCO 17-point (`[x, y, confidence]` per joint) |
| Athletes per image | 2 |
| License | CC BY-NC-SA 4.0 |

**Classes:**
- Standing
- Takedown (×2 — attacker perspective)
- Open Guard (×2)
- Half Guard (×2)
- Closed Guard (×2)
- 50-50 Guard
- Side Control (×2 — includes north-south and knee-on-belly)
- Mount (×2)
- Back (×2)
- Turtle (×2)

**Downloads:**
- Images: http://data.vicos.si/datasets/JuiJuitsu/images.zip
- Annotations: http://data.vicos.si/datasets/JuiJuitsu/annotations.json

**Citation:**
```bibtex
@inproceedings{hudovernik2023MMW,
  author    = {Hudovernik, Valter and Skočaj, Danijel},
  title     = {Video-Based Detection of Combat Positions and Automatic Scoring in Jiu-jitsu},
  booktitle = {Proceedings of the ACM Multimedia Workshop MMSports'22},
  year      = {2022},
  month     = {October},
  location  = {Lisboa, Portugal}
}
```

---

## 2. carlosj934/BJJ_Positions_Submissions — HuggingFace Dataset

**URL:** https://huggingface.co/datasets/carlosj934/BJJ_Positions_Submissions

A growing dataset pairing COCO keypoint annotations with compressed video clips, designed specifically for video transformer models (ViViT etc.). Still early-stage but the schema aligns well with this project.

| Property | Detail |
|---|---|
| Keypoint format | MS-COCO 17-point per athlete |
| Athletes per sample | Up to 2 |
| Video format | MP4, H.264, 360p/480p, 15 FPS |
| Labels | Position + submission attempts |
| Target size | 900+ samples (50+ per position) |
| Status | Actively growing (v1.2.0) |

**Annotation fields:**
- `pose1_keypoints` / `pose2_keypoints` — 17 keypoints as `[x, y, confidence]`
- `position` — BJJ position label
- `video_path` — associated video clip for video model training
- `num_people` — 1 or 2 athletes detected

**Usage:**
```python
from datasets import load_dataset

dataset = load_dataset("carlosj934/BJJ_Positions_Submissions")
sample = dataset['train'][0]
print(sample['position'], sample['video_path'])
```

---

## 3. tk1475/BJJ-Pose-Estimation — Reference Pipeline

**URL:** https://github.com/tk1475/BJJ-Pose-Estimation

An open-source end-to-end pipeline built on the ViCoS dataset. Useful as a reference implementation and baseline to beat. Achieved **97% test accuracy** on position classification.

| Property | Detail |
|---|---|
| Dataset used | ViCoS Lab (120,279 images, 18 classes) |
| Pose extraction | MediaPipe (34 keypoints per frame, 17 per person) |
| Classifier | Vision Transformer (ViT) — Grouped Query Attention, RoPE embeddings |
| Real-time inference | Lightweight MLP on top of MediaPipe keypoints |
| Best accuracy | 97% on 18-class position classification |

**Architecture progression:**

| Experiment | Description | Accuracy |
|---|---|---|
| Baseline | Shallow Transformer, default init | 94% |
| Iteration 2 | Better preprocessing + init | 95% |
| Final | GQA + deeper layers + RoPE | **97%** |

**Key insight for this project:** The 97% result is on the ViCoS 18-class set which doesn't include technique-level labels. This pipeline establishes the performance ceiling for position classification — Phase 1 technique detection is the novel layer on top.

---

## Integration Notes

All three sources use MS-COCO 17-point keypoint format — no conversion needed between them. The unified ingestion target for Phase 0 is:

```
COCO keypoints (17pt, per athlete) + position label + athlete bounding boxes
```

The ViCoS dataset is the anchor for model training. The HuggingFace dataset adds video clip pairs useful for Phase 1 (technique detection over temporal sequences). The tk1475 pipeline provides a working baseline to fork and extend.
