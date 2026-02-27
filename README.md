# lerobotso101_command
# SEVO: Semantic-Enhanced Virtual Observation

## Technical README — Architecture Analysis, Command Reference & Parameter Impact

---

## Table of Contents

1. [Overview](#1-overview)
2. [What ACT and SmolVLA Actually Are](#2-what-act-and-smolvla-actually-are)
3. [What Your Commands Actually Train](#3-what-your-commands-actually-train)
4. [How SEVO Modifies What Each Policy Learns](#4-how-sevo-modifies-what-each-policy-learns)
5. [Why Wrist Cameras Fail — Architecture-Level Explanation](#5-why-wrist-cameras-fail)
6. [Why SmolVLA Needs More Data Than ACT](#6-why-smolvla-needs-more-data-than-act)
7. [Complete Command Reference](#7-complete-command-reference)
8. [Reproducibility Notes](#8-reproducibility-notes)

---

## 1. Overview

SEVO is a **data-centric observation design** method that improves the robustness of imitation-learning policies (ACT, SmolVLA) on low-cost SO-101 robot arms. SEVO does **not** modify any model architecture or training algorithm. Instead, it transforms the RGB camera input before it enters the standard LeRobot training and inference pipeline. This transformation happens entirely in the observation space — the pixel values that the policy sees during both data collection and deployment.

The three SEVO components are:

| Component | What It Physically Does | What It Changes in the Data |
|---|---|---|
| **Body-fixed cameras** | Cameras mounted on the robot body, not the wrist | Stable spatial geometry in every frame; no viewpoint shift as arm moves |
| **Red LED illumination** | 5W red LED (620–630nm) near the arm/gripper | Consistent specular highlights on transparent objects across all lighting conditions |
| **YOLO segmentation overlay** | YOLOv8n-seg detects "bottle" → yellow mask blended onto RGB at α=0.45 | Target object pixels replaced with fixed color C; background pixels unchanged |

Additionally, a **diversified data collection protocol** (varying backgrounds, lighting, distractors) is applied during teleoperation. Our ablation shows this protocol is the single most important factor for generalization, outranking both red light and YOLO overlay.

---

## 2. What ACT and SmolVLA Actually Are

### 2.1 ACT — Action Chunking with Transformers

**Paper:** Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" (RSS 2023)

**Architecture (~80M parameters, all trained from scratch):**

```
                    ┌─────────────────────────────────────────┐
                    │           CVAE Encoder (training only)   │
                    │  Transformer that compresses ground-truth │
                    │  actions + joint state → latent z         │
                    │  (discarded at inference; z = 0)         │
                    └──────────────────┬──────────────────────┘
                                       │ z (style variable)
                                       ▼
  Camera 1 ──► ResNet-18 ──► 300×512 visual tokens ─┐
  Camera 2 ──► ResNet-18 ──► 300×512 visual tokens ──┤
  Camera N ──► ResNet-18 ──► 300×512 visual tokens ──┼──► Transformer Encoder
  Joint state q_t ──► Linear(d_joint → 512) ─────────┤      (self-attention)
  Latent z ──► Linear(d_z → 512) ─────────────────────┘          │
                                                                  │ encoded features
                                                                  ▼
                                                    Transformer Decoder
                                                    (cross-attention on encoder output)
                                                    Queries = k learned position embeddings
                                                          │
                                                          ▼
                                                    k × d_action joint angles
                                                    (chunk of future actions)
```

**Key properties:**
- **Vision backbone**: One ResNet-18 per camera view. Each 640×480 RGB image → 15×20×512 feature map → flattened to 300 vectors of dim 512. 2D sinusoidal positional embeddings preserve spatial structure.
- **No pretrained weights**: `--policy.type=act` creates a fresh model with random initialization. There is no `--policy.path` pointing to a checkpoint. Every single parameter — ResNet-18 conv filters, transformer attention heads, projection layers — starts from random values.
- **Action representation**: **Continuous L1 regression**. The model directly predicts absolute joint angles for the next k timesteps. Loss = L1 distance between predicted and ground-truth joint positions. No discretization, no diffusion, no flow matching.
- **CVAE**: During training, a separate encoder compresses the ground-truth action sequence into a latent variable z that captures "style" (multiple valid trajectories for the same goal). At inference, z is set to zero (prior mean), producing deterministic behavior.
- **Action chunking**: Predicts k=100 future actions at once (chunk), reducing the effective task horizon. Combined with **temporal ensembling** (exponentially weighted blending of overlapping chunks) for smooth execution.
- **Language**: ACT has **no language understanding**. The `--dataset.single_task="Pick the bottle (yolo)"` string is only a dataset metadata label. It is never tokenized or fed into the model. ACT is a pure vision-to-action mapper.

### 2.2 SmolVLA — Small Vision-Language-Action Model

**Paper:** Shukor et al., "SmolVLA: A vision-language-action model for affordable and efficient robotics" (arXiv 2506.01844, 2025)

**Architecture (~450M parameters total; only ~100M trainable during fine-tuning):**

```
  Camera 1 ──►┐                         "Pick the bottle"
  Camera 2 ──►├─► SigLIP Vision Encoder    ──► Text Tokenizer
  Camera N ──►┘   (FROZEN, ~400M params)       │
                    │                           │
                    ▼                           ▼
              64 image tokens            language tokens
              per camera (PixelShuffle)        │
                    │                           │
                    └───────┬───────────────────┘
                            │
                            ▼
                    SmolLM2 Language Decoder
                    (FROZEN, first N=L/2 layers only)
                    Processes: image tokens + language tokens + state token
                            │
                            ▼ VLM features (from intermediate layer)
                            │
                    ┌───────┴────────────────────────────────┐
                    │        Action Expert (~100M params)     │
                    │        (TRAINABLE via fine-tuning)      │
                    │                                         │
                    │  Interleaved attention blocks:           │
                    │    Cross-Attn (attend to VLM features)  │
                    │    Self-Attn (model action dependencies) │
                    │    Cross-Attn → Self-Attn → ...         │
                    │                                         │
                    │  Action generation: FLOW MATCHING        │
                    │    Training: add noise τ to true action  │
                    │    → learn to predict velocity field v_θ │
                    │    Inference: start from noise           │
                    │    → 10 denoising steps → action chunk   │
                    │                                         │
                    │  Hidden dim = 75% of VLM hidden dim     │
                    └─────────────────────────────────────────┘
                            │
                            ▼
                    Continuous action chunk
                    (joint angles for next 50 timesteps)

  Joint state q_t ──► Linear projection ──► single state token
                       (fed into SmolLM2 decoder alongside image/language tokens)
```

**Key properties:**
- **VLM backbone is FROZEN**: SigLIP vision encoder + SmolLM2 language decoder are loaded from `lerobot/smolvla_base` and **never updated** during fine-tuning. Their weights remain exactly as pretrained on internet images/text.
- **Only Action Expert is trained**: When you run `--policy.path=lerobot/smolvla_base`, LeRobot loads the pretrained checkpoint. During fine-tuning, gradients flow only through the Action Expert (~100M params) and linear projectors. The VLM backbone parameters have `requires_grad=False`.
- **Action representation**: **Flow matching** (a continuous-time generative model similar to diffusion). During training, a random noise level τ∈[0,1] is sampled, noise is added to the true action, and the model learns to predict the velocity field that "pushes" noisy actions toward clean ones. At inference, the model starts from pure Gaussian noise and iteratively denoises over ~10 steps to produce the final action chunk. This is fundamentally harder than L1 regression.
- **Language IS used**: The task string `"Pick the bottle (yolo)"` is tokenized by SmolLM2's tokenizer and processed through the frozen language decoder. The resulting language features condition the Action Expert via cross-attention. This means the text instruction genuinely influences the generated actions — unlike ACT where it's ignored.
- **Visual tokens**: SigLIP encodes each 512×512 image into 64 tokens (compressed via PixelShuffle). Only the global image is used (no tiling at runtime).
- **Layer skipping**: The Action Expert attends to VLM features from only the first L/2 layers of SmolLM2, not the final layer. This halves VLM compute during both training and inference.

---

## 3. What Your Commands Actually Train

### 3.1 ACT Training — From Scratch

```bash
lerobot-train \
  --dataset.repo_id="seeedstudio123/${DS}" \
  --policy.type=act \                          # ← creates NEW model, random init
  --output_dir="$OUT" \
  --batch_size=24 \
  --steps=240000 \
  --policy.device=cuda \
  --policy.use_amp=true
```

**What this does, step by step:**

1. **Model creation**: `--policy.type=act` instantiates a fresh ACTPolicy with random weights. There is no pretrained checkpoint. Every parameter is initialized randomly (Kaiming/Xavier initialization for conv layers, standard init for transformers).

2. **Dataset loading**: LeRobot reads the HDF5/Parquet dataset you recorded. Each sample contains:
   - `observation.images.front`: 640×360 RGB tensor (values 0–255, normalized to [0,1])
   - `observation.images.side`: 640×360 RGB tensor
   - `observation.images.wrist` (if present): 640×360 RGB tensor
   - `observation.state`: joint angles (6 DoF for SO-101: 5 revolute + 1 gripper)
   - `action`: target joint angles for the next k timesteps

3. **Forward pass (each training step)**:
   - Each camera image passes through its own ResNet-18: `I_front → ResNet18_front → F_front ∈ R^{300×512}`
   - CVAE encoder compresses ground-truth actions + state → latent z
   - Transformer encoder processes: [F_front; F_side; F_wrist(?); Linear(q_t); Linear(z)]
   - Transformer decoder outputs: predicted action chunk â_{t:t+k}
   - **Loss = Σ |â_i - a_i| (L1)** + β · KL(posterior || prior)

4. **Backward pass**: Gradients flow through **every parameter**: all ResNet-18 weights (conv filters, batch norm), all transformer layers, all projection layers, the CVAE encoder. **Everything** is updated.

5. **What the ResNet-18 learns**: The conv filters in ResNet-18 learn to extract features **specifically from your SEVO-processed images**. If your images have yellow YOLO overlay regions, the early conv layers learn edge detectors and color detectors that respond to yellow-on-arbitrary-background patterns. If your images have red illumination, the filters learn to detect the consistent red specular highlight pattern. **This is why SEVO works so well with ACT — the entire vision backbone adapts to your observation design.**

### 3.2 SmolVLA Fine-Tuning — From Pretrained Base

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \          # ← loads PRETRAINED checkpoint
  --policy.repo_id="local/finetune_${TAG}" \
  --dataset.repo_id="local/${NAME}" \
  --dataset.root="$ROOT" \
  --rename_map='{"observation.images.front":"observation.images.camera1",...}' \
  --batch_size=64 \
  --steps=20000
```

**What this does, step by step:**

1. **Model loading**: `--policy.path=lerobot/smolvla_base` downloads and loads the pretrained SmolVLA checkpoint from HuggingFace. This checkpoint was pretrained on ~487 community datasets (SO-100/SO-101 platforms) for 200K steps with batch size 256.

2. **Freezing**: The VLM backbone (SigLIP + SmolLM2) is set to `requires_grad=False`. Its weights **will not change** during your fine-tuning. Only the Action Expert and linear projectors have `requires_grad=True`.

3. **Camera renaming**: `--rename_map` maps your dataset keys (`front`, `side`, `wrist`) to SmolVLA's expected keys (`camera1`, `camera2`, `camera3`). This is purely a data routing step — it tells SmolVLA which image tensor goes to which input slot.

4. **Forward pass (each training step)**:
   - SigLIP (frozen) processes each camera image → 64 visual tokens per camera
   - SmolLM2 (frozen, first L/2 layers) processes: [image tokens + language tokens + state token] → VLM features
   - **Flow matching training**: sample noise level τ ~ Uniform(0,1), add noise to ground-truth action → noisy_action = (1-τ)·a_true + τ·ε
   - Action Expert (trainable) predicts velocity field: v_θ(noisy_action, τ, VLM_features)
   - **Loss = ||v_θ - (a_true - ε)||²** (flow matching objective)

5. **What gets updated**: ONLY the Action Expert transformer blocks (~100M params) and linear projection layers. The Action Expert learns: "given these (frozen, pretrained) visual features from SigLIP + SmolLM2, what joint angles should I output?"

6. **The critical implication**: SigLIP was pretrained on internet images. It has **never seen** yellow YOLO overlay, red LED illumination, or your specific camera angles. The visual features it extracts for your SEVO-processed images are its "best guess" using representations learned from natural photographs. The Action Expert must learn to interpret these potentially suboptimal features — which is why SmolVLA needs more data (120 episodes) and still trails ACT by ~11 points.

---

## 4. How SEVO Modifies What Each Policy Learns

SEVO operates **entirely in the input pixel space**. It does not modify any model weights, hyperparameters, loss functions, or training algorithms. Here is exactly what changes at the parameter level:

### 4.1 The Pixel-Level Transformation

For every frame at time t, from every body-fixed camera:

```
Raw input: I_t ∈ R^{H×W×3}     (standard RGB, values 0-255)
YOLO mask: M_t ∈ {0,1}^{H×W}   (1 = target object pixel, 0 = background)
Highlight: C = [0, 255, 255]    (yellow in RGB; constant across all frames)
Alpha:     α = 0.45             (blend strength)

SEVO output: Ĩ_t = (1 - α·M_t) ⊙ I_t + α·M_t ⊙ C

For each pixel (i,j):
  If M_t(i,j) = 0 (background):  Ĩ_t(i,j) = I_t(i,j)          [unchanged]
  If M_t(i,j) = 1 (target):      Ĩ_t(i,j) = 0.55·I_t(i,j) + 0.45·C  [yellow-tinted]
```

The red LED adds a physical component: transparent bottle surfaces reflect red light, creating consistent specular patterns in the raw RGB *before* YOLO processing. This makes YOLO detection more reliable and adds a second environment-invariant visual feature.

### 4.2 Impact on ACT's Learned Parameters

Since ACT trains from scratch, SEVO changes **what the entire model learns**:

**ResNet-18 convolutional filters (layers conv1 through layer4):**
- Without SEVO: Early filters learn general edge/texture detectors. Many filters respond to background textures (floor patterns, wall colors, table surfaces). The model learns spurious correlations between background appearance and correct actions.
- With SEVO: Early filters learn two distinct feature types:
  1. **Yellow-region detectors**: Filters that activate on the YOLO overlay color, regardless of what's behind it. These encode target object location and shape.
  2. **Red-highlight detectors**: Filters that activate on the red specular patterns from the LED. These encode 3D surface geometry of the transparent bottle.
- Background-responsive filters still form, but the diversified data collection protocol ensures they see such varied backgrounds that they cannot form stable associations with any particular action.

**Transformer encoder attention patterns:**
- Without SEVO: Attention distributes across all visual tokens (background + object), learning environment-specific spatial correlations.
- With SEVO: Attention concentrates on tokens corresponding to yellow overlay regions and red highlight regions — the only spatially consistent features across diverse training backgrounds.

**Transformer decoder (action prediction):**
- Without SEVO: Decoder associates specific background configurations with specific action trajectories. Changing the background breaks these associations.
- With SEVO: Decoder associates overlay position/shape + red highlight intensity with action trajectories. These features are background-invariant by construction.

**Quantitative effect**: The full SEVO pipeline shifts ACT from 69% (background-dependent features) to 97% (background-invariant features) success rate.

### 4.3 Impact on SmolVLA's Learned Parameters

Since SmolVLA's VLM is frozen, SEVO only changes **what the Action Expert learns** from the fixed visual features:

**SigLIP visual tokens (FROZEN — not changed by SEVO):**
- SigLIP processes your SEVO-enhanced images using its pretrained conv filters. It was trained on internet images, so:
  - It recognizes "yellow region" as a generic color patch, not specifically as "YOLO overlay"
  - It recognizes "red highlight" as a light source reflection
  - It does NOT understand the semantic meaning you've assigned to these visual cues
- However, SigLIP's representation is still useful because: the yellow overlay creates a **consistent, high-contrast visual feature** that SigLIP reliably encodes as the same set of token activations regardless of background. The consistency of the encoding is what matters, not semantic understanding.

**SmolLM2 language processing (FROZEN):**
- Your task string `"Pick the bottle (yolo)"` is tokenized and processed by the frozen language model
- The resulting language features condition the Action Expert via cross-attention
- The word "bottle" activates semantic associations from pretraining; "(yolo)" is treated as additional context

**Action Expert (TRAINABLE — this is what SEVO actually changes):**
- **Cross-attention layers**: Learn which VLM visual tokens to attend to. With SEVO, the expert learns to attend to tokens that encode yellow overlay + red highlights, ignoring background tokens. Without SEVO, attention is diffuse and background-dependent.
- **Self-attention layers**: Model temporal dependencies between action timesteps. With SEVO, these dependencies are anchored to the overlay's position trajectory. Without SEVO, they're anchored to arbitrary background features.
- **Flow matching denoising**: The velocity field v_θ learns to denoise toward actions that correspond to overlay-indicated object positions. The denoising process converges more reliably because the visual conditioning signal (overlay + red light) is consistent.

**Why SmolVLA underperforms ACT by ~11 points**: SigLIP's frozen filters were never optimized for SEVO's visual vocabulary. The yellow overlay activates SigLIP features in a way that's "close enough" but not optimal. ACT's trainable ResNet-18 develops filters precisely tuned to the overlay color/shape, achieving tighter feature-action coupling.

---

## 5. Why Wrist Cameras Fail

### 5.1 Architecture-Level Analysis

**In ACT (oscillatory failure, ~30% degradation):**

The wrist camera adds a third ResNet-18 branch. This branch processes rapidly changing images as the arm moves. During training, ACT's ResNet-18_wrist learns features from these dynamic views. The problem:

1. **Feature instability**: Wrist camera features change drastically frame-to-frame as the arm moves. The transformer encoder receives highly variable tokens from this branch.
2. **Action chunking partially compensates**: ACT predicts k=100 future actions at once. Even if the wrist features are noisy, the temporal ensemble of overlapping chunks provides inertial stability. The arm oscillates but mostly recovers.
3. **With YOLO on wrist (destructive)**: The YOLO mask fills the entire wrist frame when close to the object. The ResNet-18 learns "full-frame yellow = object is right here," but this signal is the same whether the gripper is 5cm or 0.5cm from the object. The policy cannot distinguish approaching from grasping, leading to premature closure.

**In SmolVLA (catastrophic failure, near-zero success):**

SmolVLA's frozen SigLIP processes the wrist camera images. The problem is far worse:

1. **SigLIP compresses to 64 tokens**: The rapidly changing wrist view is compressed into only 64 tokens. The compression discards fine-grained spatial information that might encode gripper-to-object distance.
2. **Frozen features cannot adapt**: Unlike ACT's trainable ResNet-18, SigLIP cannot learn to ignore the wrist camera's noise. It faithfully encodes every frame, producing highly variable token sequences.
3. **Motion-induced false triggering**: During mobile driving (before the robot stops to grasp), the wrist camera sees rapid visual flow. SigLIP encodes this as "dynamic scene change." The Action Expert, having learned from training data that "dynamic visual change = arm is executing grasp," triggers grasping motions in open air.
4. **Depth misjudgment**: SigLIP's visual tokens for the wrist view cannot distinguish "bottle 5cm away" from "bottle touching gripper" — both show a large object filling the frame. The Action Expert closes the gripper prematurely because the frozen visual features provide no depth gradient.
5. **No action-chunking rescue**: SmolVLA's flow matching denoising is more sensitive to conditioning quality than ACT's L1 regression. Bad visual features cause the denoising process to diverge to incorrect action modes, with no temporal ensembling to recover.

### 5.2 Why Most Community Setups Default to Wrist Cameras

Community tutorials and reference designs mount cameras on the wrist because it's physically convenient and provides an intuitive "eye-in-hand" view. For high-precision tabletop tasks with static backgrounds, this can work. But for mobile manipulation with varying environments, the wrist camera's viewpoint dependency becomes a fatal source of distributional shift.

---

## 6. Why SmolVLA Needs More Data Than ACT

| Factor | ACT (80 episodes sufficient) | SmolVLA (120 episodes minimum) |
|---|---|---|
| **Vision backbone** | ResNet-18, trained from scratch — learns SEVO-specific features directly | SigLIP, frozen — pretrained on internet images, never saw YOLO overlay |
| **Action generation** | L1 regression — simple supervised mapping from features to joint angles | Flow matching — must learn velocity field over noise×action joint space |
| **Noise coverage** | N/A — deterministic regression | Each sample seen at multiple noise levels τ∈[0,1]; needs more data to cover this space |
| **Pretraining gap** | No gap — trained from scratch on your data | Action Expert pretrained on SO-100 community data; must overcome distribution shift to SO-101 + SEVO |
| **40 episodes result** | ~97% success | Model fails to converge; near-zero success |
| **Language grounding** | N/A — text ignored | Language tokens condition the Action Expert; additional complexity in the conditioning signal |

The fundamental reason is: **flow matching is a harder learning objective than L1 regression**. L1 regression only needs to learn a point estimate (one action per observation). Flow matching needs to learn an entire vector field that can denoise from any noise level — requiring exponentially more coverage of the data manifold.

---

## 7. Complete Command Reference

### 7.1 Environment Setup

```bash
# Activate environment
conda activate lerobot_lab
cd ~/lerobot_lab
```

### 7.2 YOLO Virtual Camera Setup (Required for YOLO Variants)

**Create v4l2 loopback devices (once per reboot):**
```bash
sudo modprobe v4l2loopback devices=3 video_nr=10,11,12 \
  card_label="YOLO-front","YOLO-side","YOLO-wrist" exclusive_caps=0
sudo chmod 666 /dev/video10 /dev/video11 /dev/video12
```

**Force loopback to 260fps (once per reboot):**
```bash
for v in 10 11 12; do
  echo '@260' | sudo tee "/sys/devices/virtual/video4linux/video${v}/format" >/dev/null
done
```

**Start YOLO segmentation streams (one terminal per camera, keep running):**

```bash
# Terminal A — Front camera (physical /dev/video0 → virtual /dev/video10)
python tools/yolo/yolo_seg_highlight_to_v4l2.py \
  --src /dev/video0 --out /dev/video10 \
  --w 640 --h 360 --fps 260 \
  --model yolov8n-seg.pt --infer_fps 15 \
  --conf 0.20 --iou 0.5 --mask_th 0.3 \
  --target bottle:0,255,255 --alpha 0.45

# Terminal B — Side camera (physical /dev/video2 → virtual /dev/video11)
python tools/yolo/yolo_seg_highlight_to_v4l2.py \
  --src /dev/video2 --out /dev/video11 \
  --w 640 --h 360 --fps 260 \
  --model yolov8n-seg.pt --infer_fps 15 \
  --conf 0.20 --iou 0.5 --mask_th 0.3 \
  --target bottle:0,255,255 --alpha 0.45

# Terminal C — Wrist camera (physical /dev/video4 → virtual /dev/video12)
python tools/yolo/yolo_seg_highlight_to_v4l2.py \
  --src /dev/video4 --out /dev/video12 \
  --w 640 --h 360 --fps 260 \
  --model yolov8n-seg.pt --infer_fps 15 \
  --conf 0.20 --iou 0.5 --mask_th 0.3 \
  --target bottle:0,255,255 --alpha 0.45
```

**YOLO parameter explanation:**

| Parameter | Value | Meaning |
|---|---|---|
| `--model yolov8n-seg.pt` | YOLOv8 nano segmentation | Lightweight model (~6M params), runs at ~30fps on CPU |
| `--conf 0.20` | Confidence threshold | Only detections with confidence ≥ 20% are kept |
| `--iou 0.5` | NMS IoU threshold | Suppresses overlapping detections with IoU ≥ 50% |
| `--mask_th 0.3` | Segmentation mask threshold | Lower = thicker/coarser mask boundary |
| `--target bottle:0,255,255` | Target class + highlight color | Only "bottle" class; highlighted in yellow (BGR: 0,255,255) |
| `--alpha 0.45` | Overlay blend strength | 45% highlight color + 55% original RGB for detected pixels |
| `--infer_fps 15` | YOLO inference rate | Runs detection at 15fps; frames between detections use last mask |

### 7.3 Data Collection (Record)

**YOLO + body-fixed cameras (2 cameras, no wrist):**
```bash
DS="lerobot_lab_yolo_redlight_negative"
rm -rf ~/.cache/huggingface/lerobot/seeedstudio123/${DS}

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    "front":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":360,"fps":260},
    "side" :{"type":"opencv","index_or_path":"/dev/video11","width":640,"height":360,"fps":260}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true \
  --dataset.repo_id=seeedstudio123/${DS} \
  --dataset.num_episodes=80 \
  --dataset.single_task="Pick the bottle (yolo)" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=18 \
  --dataset.reset_time_s=30
```

**NO-YOLO + body-fixed cameras (physical cameras, no overlay):**
```bash
DS="lerobot_lab_noyolo_redlight_negative"
rm -rf ~/.cache/huggingface/lerobot/seeedstudio123/${DS}

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    "front":{"type":"opencv","index_or_path":0,"width":640,"height":360,"fps":260,"fourcc":"MJPG"},
    "side" :{"type":"opencv","index_or_path":2,"width":640,"height":360,"fps":260,"fourcc":"MJPG"}
  }' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true \
  --dataset.repo_id=seeedstudio123/${DS} \
  --dataset.num_episodes=80 \
  --dataset.single_task="Pick the bottle (raw)" \
  --dataset.push_to_hub=false \
  --dataset.episode_time_s=18 \
  --dataset.reset_time_s=30
```

**What `lerobot-record` stores:**
- Per frame: RGB images from each camera (encoded as MP4 video segments), joint angles as Parquet columns
- Per episode: synchronized timestamps, task description string, episode index
- The camera `index_or_path` determines whether the policy sees raw RGB (physical `/dev/video0`) or SEVO-processed RGB (virtual `/dev/video10` from YOLO pipeline). **This single routing choice is the only difference between baseline and SEVO data.**

### 7.4 ACT Training

```bash
DS="lerobot_lab_yolo_redlight_negative"
TAG="act_b24_lrdefault_s240k"
OUT="outputs/train/${TAG}_${DS}"
rm -rf "$OUT"

lerobot-train \
  --dataset.repo_id="seeedstudio123/${DS}" \
  --policy.type=act \
  --output_dir="$OUT" \
  --job_name="${TAG}_${DS}" \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=240000 \
  --batch_size=24 \
  --num_workers=8 \
  --policy.use_amp=true \
  --eval.batch_size=0 --eval.n_episodes=0 --eval_freq=0
```

**What each parameter controls:**

| Parameter | Value | What It Does |
|---|---|---|
| `--policy.type=act` | ACT | **Creates new model from scratch** (random init, ~80M params) |
| `--steps=240000` | 240K | Total gradient update steps. ACT converges around step 26K but benefits from continued training |
| `--batch_size=24` | 24 | Number of (image, action) pairs per gradient step |
| `--policy.use_amp=true` | Mixed precision | FP16 forward pass + FP32 gradients for faster training |
| `--eval.batch_size=0` | Skip eval | No simulation evaluation during training (real-robot eval only) |

**What is being optimized**: L_total = L1(predicted_actions, true_actions) + β·KL(posterior, prior). All ~80M parameters updated.

### 7.5 SmolVLA Training

```bash
ROOT="/home/felixfang/.cache/huggingface/lerobot/seeedstudio123/lerobot_lab_yolo_redlight_negative"
NAME="lerobot_lab_yolo_redlight_negative"
TAG="smolvla_b64_s20k"
OUT="outputs/train/${TAG}_${NAME}"

RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.side":"observation.images.camera2","observation.images.wrist":"observation.images.camera3"}'

rm -rf "$OUT"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="local/finetune_${TAG}_${NAME}" \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --dataset.repo_id="local/${NAME}" \
  --dataset.root="$ROOT" \
  --rename_map="$RENAME_MAP" \
  --batch_size=64 \
  --steps=20000 \
  --output_dir="$OUT" \
  --job_name="${TAG}_${NAME}" \
  --policy.device=cuda \
  --num_workers=8 \
  --policy.use_amp=true \
  --eval.batch_size=0 --eval.n_episodes=0 --eval_freq=0
```

**What each parameter controls:**

| Parameter | Value | What It Does |
|---|---|---|
| `--policy.path=lerobot/smolvla_base` | Pretrained checkpoint | **Loads pretrained SmolVLA** (450M params); VLM frozen, Action Expert trainable |
| `--rename_map='{...}'` | Key mapping | Routes `front`→`camera1`, `side`→`camera2`, `wrist`→`camera3` |
| `--steps=20000` | 20K | Fine-tuning steps. SmolVLA converges faster than ACT because of pretrained base |
| `--batch_size=64` | 64 | Larger batch than ACT because SmolVLA's gradient signal is noisier (flow matching) |

**What is being optimized**: L_flow = ||v_θ(noisy_action, τ, VLM_features) - (true_action - noise)||². Only Action Expert params (~100M) updated.

### 7.6 Evaluation (Deployment)

**ACT + YOLO evaluation:**
```bash
MODEL="act_b24_lrdefault_s240k_lerobot_lab_yolo_redlight_negative"
CKPT="last"
POLICY="outputs/train/${MODEL}/checkpoints/${CKPT}/pretrained_model"
EVAL_ID="eval_${MODEL}_ckpt${CKPT}"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    "front":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":360,"fps":260},
    "side" :{"type":"opencv","index_or_path":"/dev/video11","width":640,"height":360,"fps":260}
  }' \
  --display_data=true \
  --dataset.repo_id=seeedstudio123/${EVAL_ID} \
  --dataset.single_task="Pick the bottle (eval)" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=1800 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --policy.path=${POLICY}
```

**Critical**: During evaluation, the same YOLO pipeline must be running. The policy was trained on SEVO-enhanced images; feeding it raw RGB at test time would create a train-test distribution mismatch.

**SmolVLA + YOLO evaluation (note: camera keys must be camera1/camera2/camera3):**
```bash
MODEL="smolvla_b64_s20k_lerobot_lab_yolo_redlight_negative"
CKPT="last"
POLICY="outputs/train/${MODEL}/checkpoints/${CKPT}/pretrained_model"
EVAL_ID="eval_${MODEL}_ckpt${CKPT}"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{
    "camera1":{"type":"opencv","index_or_path":"/dev/video10","width":640,"height":360,"fps":260},
    "camera2":{"type":"opencv","index_or_path":"/dev/video11","width":640,"height":360,"fps":260}
  }' \
  --display_data=true \
  --dataset.repo_id=seeedstudio123/${EVAL_ID} \
  --dataset.single_task="Pick the bottle (eval)" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=1800 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --policy.path=${POLICY}
```

### 7.7 Troubleshooting

```bash
# Check loopback fps (must be 260, not 30)
for v in 10 11 12; do
  echo -n "/dev/video${v}: "
  v4l2-ctl -d "/dev/video${v}" --get-parm | sed -n 's/.*Frames per second: //p'
done

# Check what's using the video devices
sudo fuser -v /dev/video10 /dev/video11 /dev/video12 || true

# Kill stuck processes
pkill -f yolo_seg_highlight_to_v4l2.py || true
pkill -f lerobot-record || true

# Verify loopback devices have frames
python - <<'PY'
import cv2, time
for dev in ["/dev/video10","/dev/video11","/dev/video12"]:
    cap=cv2.VideoCapture(dev, cv2.CAP_V4L2)
    ok, f = cap.read()
    print(dev, "opened", cap.isOpened(), "read", ok, None if f is None else f.shape)
    cap.release()
    time.sleep(0.1)
PY
```

---

## 8. Reproducibility Notes

### Hardware

| Component | Robot A | Robot B |
|---|---|---|
| Arm | SO-101 (5 DoF + gripper) | SO-101 (5 DoF + gripper) |
| Compute | Jetson Orin NX 16GB | Raspberry Pi 5 8GB |
| Cameras | 2 body-fixed USB RGB (640×360) + 1 wrist (control group) | 3 body-fixed USB RGB (640×360) |
| Red LED | 5W panel, 620–630nm, at arm base | 5W LED, above arm (gradient illumination) |
| YOLO | YOLOv8n-seg (~6M params), ~30fps on CPU | Same |

### Training Compute

- ACT: ~5 hours on single GPU for 240K steps (converges ~26K steps)
- SmolVLA: ~4 hours on single GPU for 20K steps
- Training GPU: RTX 4090 (for i9 workstation) or A100 equivalent

### Dataset Sizes

| Policy | Episodes | Frames per Episode (~50fps × 18s) | Total Frames |
|---|---|---|---|
| ACT | 80 | ~900 | ~72,000 |
| SmolVLA | 120 | ~900 | ~108,000 |

### Software Versions

- LeRobot: latest main branch (2025–2026)
- YOLOv8: ultralytics package, yolov8n-seg.pt
- Python 3.10+, PyTorch 2.0+, CUDA 12.x

---

## License

This work is submitted to IROS 2026 under double-blind review. Code and datasets will be released upon acceptance.
