# Spike notes (2026-10-30)

## SAM 3.1

- **Model ids confirmed:** `muse-image-1.0`, `muse-spark-1.1`, `muse-spark-1.2`, `muse-spark-1.2-contributor`, `muse-spark-1.3`, `muse-spark-1.3-contributor`, `muse-voice-transcribe-1.0`, `muse-wonder-1.0`, `sam-3.1`.
- **Mask raster equals box size:** yes. There were 0 resized masks across 24 calls.
- **Timing:** API calls took 0.5–3.6 s each. Decoding took at most 1.8 s (a full-sky mask) and was usually under 0.3 s.
- **Empty result:** "no match" comes back as a normal `message` with one `output_text` part whose text is `""`. It is not an error.
- **Metadata:** `include_confidence` and `mask_encoding` are honored. Confidences came back, and every mask was one-bit (`!`).

### Concept (text) prompts: 6 photos, `scripts/spike_sam.py`

| Photo | Crop prompt | "plant" | Extras |
|---|---|---|---|
| IMG_4214 (corn row, dense) | "corn plant" 0 | 0 | "grass" 0, "sky" 0 |
| IMG_4215 (corn row) | "corn plant" 0 | 7 small pieces (conf 0.50–0.57) | "grass" 0, "sky" 0 |
| IMG_4219 (soybean rows) | "soybean plant" 0 | 2 (0.56, 0.63) | "grass" 0, "sky" 1 (0.95) |
| IMG_4220 (soybean, wide) | "soybean plant" 0 | 0 | "grass" 1 (0.79), "sky" 1 (0.98) |
| IMG_3922 (alfalfa, wide) | "alfalfa plant" 0 | 0 | "dandelion" 36, "sky" 1 (0.98) |
| IMG_3926 (alfalfa, close) | "alfalfa plant" 0 | 0 | "dandelion" 18, "sky" 0 |

A follow-up probe (`scratchpad/sam_probe.py`) tried more phrases:

- **IMG_4214:**
  - "corn", "maize", "corn leaf", "leaf", "weed" and "grass blade" all returned 0.
  - "vegetation" returned 1 mask covering 99.9% of the frame.
- **IMG_4219:**
  - "soybean", "soybean leaf", "leaf", "vegetation" and "weed" all returned 0.
  - "crop row" returned 5 rows covering 61% (conf 0.64–0.70).
  - "seedling" returned 1 (0.51).
- **IMG_3926:**
  - "alfalfa", "clover", "leaf" and "weed" all returned 0.
  - "vegetation" returned the whole frame.
  - "flower" returned 15 (conf 0.92).

**"Plant" minus crop is a no-go.** In crop canopies SAM 3.1 does not pick out individual crop plants, leaves or weeds from a text prompt. Text prompts are still useful for:

- "sky", to find the ground area.
- "crop row", for the crop in row views.
- Visually distinct weeds such as "dandelion" or "flower" in alfalfa.

### Point and box prompts: IMG_4219 (`spike_out/point_prompts_IMG_4219.jpg`)

| Prompt | Result |
|---|---|
| Point on a weed in open soil (285, 640) | That weed only, 2,812 px, conf 0.87 |
| Box around the same weed | Same mask, 3,073 px, conf 0.87 |
| Point on a weed at the row edge (240, 745) | The weed plant, 9,516 px, conf 0.14 (low, but the mask is right) |
| Point on a soybean (430, 490) | The whole connected right-hand row canopy, conf 0.79 |
| Point on a soybean (100, 650) | The whole left-hand row canopy, conf 0.75 |

**Decision (user, 2026-10-30): go with pointing** (spec §8.2.3 fallback, now the primary path):

- Muse Spark looks at the photo and gives a point on each weed.
- SAM segments each point as its own object; one request can carry up to 16 objects.
- Points on the crop (or "crop row") give the crop mask.
- Muse Spark verifies each weed by zooming in.
- Task 11 checks whether Muse Spark's points actually land on weeds.

## Muse Spark (Task 11)

- Tool call round trip: (pending)
- Image in follow-up user message: (pending)
- Structured output: (pending)
- Pointing accuracy: (pending)
- Typical latency per turn: (pending)
