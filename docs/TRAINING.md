# Training the voice model on your own recordings

A step-by-step note for fine-tuning PocketTTS on **your** speech.

> **Status (26 Nov 2026): not yet run for this project.** Commands and settings come from the [PocketTTS training guide and code](https://github.com/kyutai-labs/pocket-tts/tree/main/training). Values marked **(suggestion)** are starting points that have not been tested. For Telugu or slang, also read [Next phases](NEXT_PHASES.md).

**Contents:** [0. Do you need it?](#0-do-you-need-training) · [1. Requirements](#1-what-you-need) · [2. Record](#2-record-your-data) · [3. Prepare](#3-prepare-the-data) · [4. Set up the GPU machine](#4-set-up-the-gpu-machine) · [5. Align](#5-add-word-timings) · [6. Train](#6-fine-tune) · [7. Compare](#7-compare-checkpoints) · [8. Shrink](#8-shrink-the-model-for-cpu) · [9. Use it](#9-use-your-model-in-pocket-voice) · [Problems](#problems-and-fixes)

---

## 0. Do you need training?

| What you have today | What training does |
|---|---|
| A **voice profile**: the model reads a 10–20 s sample and imitates it. No weights change. | **Changes the model's weights** using hours of your speech. |

Try these first. They take minutes:

1. A cleaner, quieter sample, plus **Reduce background noise** on Voice Setup.
2. Several voice profiles in different styles; pick the best.
3. Steadier settings: `POCKET_TEMP=0.3`, `POCKET_STEPS=2`.
4. Slang in the answers: change `SYSTEM_PROMPT`.

**Train only if** your accent, dialect, slang words or speaking style are still wrong after these.

**What cannot be used for training**
- ❌ Audio the app generated. It only repeats what the model already does and reinforces its mistakes.
- ❌ Other people's recordings without their explicit consent.
- ❌ The chat answers. They come from Groq's model, which this project does not train.

---

## 1. What you need

| Item | Requirement |
|---|---|
| Computer | **Linux with one NVIDIA GPU** (a rented cloud GPU is fine). Not a Mac or a standard Codespace. |
| GPU memory | About 16 GB with `batch_size: 16` and `grad_accum_steps: 4` (the default batch needs about 56 GB) |
| Software | Python 3.10+, [uv](https://docs.astral.sh/uv/), git |
| Disk | Model downloads plus your audio and its precomputed features (about 60 GB per 1,000 h of audio) |
| Hugging Face | Terms accepted at [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) and a **read** token (training starts from the gated weights) |
| Data | Your recordings **with exact transcripts** |

**How much speech?** Kyutai gives no minimum for adapting to one person. Their figures (100 h minimum, 1,000+ h for a strong model) are for training a new language.

| Amount (suggestion) | Expectation |
|---|---|
| Under 30 minutes | Likely too little; the model may get worse |
| 1–5 hours | A first experiment |
| 10+ hours | A more serious attempt |

---

## 2. Record your data

**Setup**
- Same quiet room and same microphone for every session.
- No fan, TV, music or echo. About a hand's width from the mic.
- **WAV, mono, 24 kHz or higher, 16-bit.** Don't apply heavy noise removal or effects.

**How to speak**
- Naturally, in the style you want the assistant to use.
- Include questions, short replies, long sentences, excitement and calm.
- Say numbers, dates, names and the slang words you care about.
- Pause briefly between sentences.

**Clips**
- One sentence or two per file, **4–15 seconds** (suggestion). Hard limits: at least **2 s**, at most **30 s**, at least **2 words**.
- One speaker only. Delete clips with coughs, restarts or interruptions.

**Long recording?** Split it in [Audacity](https://www.audacityteam.org/): select each sentence, press `Cmd/Ctrl + B` to add a label, then *File → Export → Export Multiple*, split by labels, as WAV.

**Folder layout**
```
mydata/
├── clips/
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
└── transcripts.tsv        # one line per clip: file name <TAB> exact words
```
```
0001.wav	Good morning, how did you sleep last night?
0002.wav	Let's meet at five thirty near the main gate.
```

---

## 3. Prepare the data

### 3.1 Draft the transcripts (optional)
This uses Groq Whisper to write a first draft. **It sends your audio to Groq.** Run it in the Pocket Voice project folder, where `GROQ_API_KEY` is set. Save as `draft_transcripts.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, "app")
import asr  # uses your GROQ_API_KEY

folder = Path(sys.argv[1])
with open(folder / "transcripts.tsv", "x", encoding="utf-8") as out:  # "x": never overwrite your corrections
    for clip in sorted((folder / "clips").glob("*.wav")):
        text = asr.transcribe(clip.read_bytes()).replace("\t", " ")
        out.write(f"{clip.name}\t{text}\n")
        out.flush()
        print(clip.name, text)
```
```bash
uv run python draft_transcripts.py mydata
```
Speech recognition is multilingual, but the app asks for English. For another language, change `language="en"` in [app/asr.py](../app/asr.py).

### 3.2 Correct every line by hand
This is the most important step. Wrong text teaches the model to ignore the script.
- Write **exactly** what was said, including slang, fillers and repeats.
- Spell each slang word the same way every time.
- Write numbers and symbols as words: `5:30` → `five thirty`, `%` → `percent`.
- Keep normal punctuation.

### 3.3 Build the training lists
Save as `make_dataset.py` (needs `soundfile`, which both projects include):
```python
import json
import random
import sys
import unicodedata
from pathlib import Path

import soundfile as sf

folder = Path(sys.argv[1]).resolve()
rows, skipped = [], []
for line in (folder / "transcripts.tsv").read_text(encoding="utf-8").splitlines():
    name, _, text = line.partition("\t")
    text = unicodedata.normalize("NFC", text).strip()
    path = folder / "clips" / name.strip()
    if not path.is_file():
        skipped.append((name, "file not found"))
        continue
    info = sf.info(str(path))
    seconds = info.frames / info.samplerate
    if not 2.0 <= seconds <= 30.0:
        skipped.append((name, f"length {seconds:.1f}s (need 2-30 s)"))
    elif len(text.split()) < 2:
        skipped.append((name, "fewer than 2 words"))
    else:
        if any(c.isdigit() for c in text):
            print("WARNING digits in", name, "- write numbers as words")
        rows.append({"path": str(path), "duration": round(seconds, 2), "transcript": text})

random.Random(42).shuffle(rows)
n_valid = max(1, len(rows) // 10)  # 10 % kept aside to check progress
for split, part in (("valid", rows[:n_valid]), ("train", rows[n_valid:])):
    with open(folder / f"{split}.jsonl", "w", encoding="utf-8") as out:
        out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
    print(split, len(part), "clips,", round(sum(r["duration"] for r in part) / 60, 1), "minutes")
for name, why in skipped:
    print("skipped", name, "-", why)
```
**Run this on the GPU machine** (after step 4), because the lists store full file paths.

---

## 4. Set up the GPU machine

```bash
git clone https://github.com/kyutai-labs/pocket-tts.git
cd pocket-tts
uv sync                                   # includes the training tools
uv run hf auth login                      # paste your read token when asked
uv run python -c "import torch; print(torch.cuda.is_available())"   # must print True
```
If it prints `False`, see "Using older cuda versions" in the training guide.

Copy your data into `pocket-tts/data/mydata/` over a secure connection, for example:
```bash
scp -r mydata user@gpu-machine:pocket-tts/data/
```
Your voice recordings are personal data: use a machine you trust and delete them afterwards. Then run step 3.3:
```bash
uv run python make_dataset.py data/mydata
```

---

## 5. Add word timings

The trainer cuts each clip between two words: the first part is the voice sample, the rest is what to speak. It therefore needs the time of every word.
```bash
for s in train valid; do
  uv run python -m training.scripts.align_data data/mydata/$s.jsonl data/mydata/${s}_aligned.jsonl --device cuda
done
wc -l data/mydata/*.jsonl                 # aligned files should keep nearly every line
```
The default aligner is English. For Telugu add `--model anuragshas/wav2vec2-large-xlsr-53-telugu` (see [Next phases](NEXT_PHASES.md)). Many skipped clips usually mean wrong transcripts or a wrong aligner language.

---

## 6. Fine-tune

This continues from Kyutai's pretrained **24-layer English** model ([`finetune.yaml`](https://github.com/kyutai-labs/pocket-tts/blob/main/training/configs/finetune.yaml)).

Create `training/configs/my_voice.yaml`. Settings marked ◆ differ from the official file and are **suggestions** for a small dataset:
```yaml
model_config: pocket_tts/config/english_2026-04_24l.yaml
start_from_pretrained: true

data:
  train_jsonl: data/mydata/train_aligned.jsonl   # ◆
  valid_jsonl: data/mydata/valid_aligned.jsonl   # ◆
  max_duration_sec: 30.0
  max_voice_prompt_sec: 5.0

flow:
  type: lsd
  kwargs: {}

flow_batch_multiplier: 4
eos_loss_weight: 0.1
text_dropout: 0.2
voice_dropout: 0.2

optim:
  lr: 2e-5           # low on purpose, so the model adapts without forgetting
  muon_lr: 5e-4
  schedule: constant
  warmup_steps: 200  # ◆ (official 1000)

run_dir: runs/my_voice     # ◆
batch_size: 16             # ◆ for a 16 GB GPU (official 64)
grad_accum_steps: 4        # ◆
max_steps: 4000            # ◆ (official 20000)
stats_ema_steps: 1000
ema_decay: 0.99            # ◆ short run (official 0.999)
log_freq: 50
valid_freq: 500            # ◆
ckpt_freq: 500             # ◆
num_ckpt_keep: 8           # ◆ keep several to compare
sample_cfg_coef: 2.0
```
Start:
```bash
uv run training/train.py training/configs/my_voice.yaml
```

| Task | How |
|---|---|
| Watch progress | `tail -f runs/my_voice/progress.jsonl` |
| Hear samples made during training | files in `runs/my_voice/samples/` |
| Stop safely | `touch runs/my_voice/STOP` (saves a checkpoint, then exits) |
| Continue later | run the same train command again; it resumes from the latest checkpoint |
| Start over | use a new `run_dir` |

The first run spends extra time converting your audio into features once. Training time depends on your GPU and is not documented for small datasets.

---

## 7. Compare checkpoints

Training too long on little data **overfits**: your phrases sound right but new sentences get worse. Pick by ear.

Write 10 sentences that are **not** in your data (everyday, slang, numbers as words, a long one). Generate them with the original model and with several checkpoints:
```bash
CFG=pocket_tts/config/english_2026-04_24l.yaml
TEXT="Hey, are we still on for tonight, or should we push it to the weekend?"
uv run pocket-tts generate --config $CFG --voice voice.wav --text "$TEXT" --output-path base.wav
for step in 00000500 00001000 00002000 00004000; do
  uv run pocket-tts generate --config $CFG --checkpoint runs/my_voice/checkpoint_$step.pt \
      --voice voice.wav --text "$TEXT" --output-path step_$step.wav
done
```
`voice.wav` is your usual 10–20 s sample. Only the last few checkpoints are kept, as set by `num_ckpt_keep`.

| Score each 1–5 | Question |
|---|---|
| Like me | Does it sound more like you than `base.wav`? |
| Slang / accent | Are your words said your way? |
| Clarity | Is every word understandable? |
| Endings | Does it stop cleanly, without cutting off or rambling? |

Choose the **earliest** checkpoint that is clearly better. If none beats `base.wav`, add more and cleaner data.

---

## 8. Shrink the model for CPU

The fine-tuned model has 24 layers (the released 24-layer English weights are about 1.3 GB) and is slow on a Codespace. **Distillation** teaches a 6-layer model to copy it.

Copy [`depth_distill.yaml`](https://github.com/kyutai-labs/pocket-tts/blob/main/training/configs/depth_distill.yaml) to `training/configs/my_voice_distill.yaml` and change these lines. The official file assumes a model trained from scratch, so **this adaptation is untested**:
```yaml
model_config: pocket_tts/config/english_2026-04.yaml              # 6-layer student, same text tokenizer family
distill_teacher_config: pocket_tts/config/english_2026-04_24l.yaml
distill_teacher_overrides: {}
distill_teacher_weights: runs/my_voice/checkpoint_00002000.pt     # the checkpoint you chose
run_dir: runs/my_voice_distill
batch_size: 16
grad_accum_steps: 4
num_ckpt_keep: 5
data:
  train_jsonl: data/mydata/train_aligned.jsonl
  valid_jsonl: data/mydata/valid_aligned.jsonl
```
Keep the other lines. Then:
```bash
uv run training/train.py training/configs/my_voice_distill.yaml
```
The guide's reference run uses 200,000 steps (word accuracy levels off near 50,000) on far more data. With a small dataset, compare early checkpoints as in step 7. Distilling on only your clips may lose general ability; adding other licensed speech helps.

---

## 9. Use your model in Pocket Voice

**1. Copy to the Codespace** the file `runs/my_voice_distill/model.safetensors` into `models/mine/` (git-ignored).

**2. Create `models/mine/config.yaml`**: copy `pocket_tts/config/english_2026-04.yaml` from the training repo and replace **both** `weights_path` and `weights_path_without_voice_cloning` with the full path of your file:
```yaml
weights_path: /workspaces/pocket-voice/models/mine/model.safetensors
weights_path_without_voice_cloning: /workspaces/pocket-voice/models/mine/model.safetensors
```
**3. Tell the app to load it.** This needs a small code change that is **not built yet**. In `_get_model()` in [app/tts.py](../app/tts.py):
```python
config = os.environ.get("POCKET_CONFIG")
extra = {"config": config} if config else {}
_model = TTSModel.load_model(quantize=get_settings().quantize, **extra)
```
then in `.env`:
```ini
POCKET_CONFIG=/workspaces/pocket-voice/models/mine/config.yaml
```
**4. Re-create your voice profiles.** A saved voice belongs to the model that made it. Use new names so you can switch back.

**To go back**, remove `POCKET_CONFIG`, restart, and use your original profiles (or create them again).

---

## Problems and fixes

| Problem | Try |
|---|---|
| `torch.cuda.is_available()` is `False` | Match the PyTorch build to your GPU driver (training guide) |
| Download refused (401/403) | Accept the model terms and log in with your token |
| Out of GPU memory | `batch_size: 8`, `grad_accum_steps: 8` |
| Many clips skipped in alignment | Fix transcripts; check the aligner language |
| Speaks the wrong words | Transcripts are inaccurate: correct them |
| Noisy output | Your recordings or voice sample are noisy |
| Cuts off or rambles | Check alignment; try `--eos-threshold -3` or `-5` |
| New sentences worse than before | Overfitting: use an earlier checkpoint or more data |

## Rights and safety
- Train only on voices you own or have written permission to use. Follow Kyutai's [prohibited use](https://github.com/kyutai-labs/pocket-tts#prohibited-use) rules.
- Model weights have their own license: read the [model card](https://huggingface.co/kyutai/pocket-tts) before sharing anything derived from them.
- Keep recordings, transcripts and trained weights private. Never commit them to git.
