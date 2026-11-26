# Next phases: Telugu (and other languages)

Goal: ask in Telugu and hear the answer in **your own voice**, in Telugu.

> **Status (26 Nov 2026): plan only.** Nothing on this page has been run for this project. Figures are quoted from the linked sources.

## Summary

| Phase | What | Needs | Result |
|---|---|---|---|
| [1](#phase-1--try-the-existing-community-model) | Try an existing Telugu-capable PocketTTS model | Codespace (CPU) | Know in an afternoon whether it is good enough |
| [2](#phase-2--use-it-in-pocket-voice) | Wire it into the app | Small code changes | Telugu voice assistant |
| [3](#phase-3--measure-quality) | Measure quality | A Telugu listener | Go / no-go decision |
| [4](#phase-4--train-your-own-telugu-model) | Train your own model, **only if Phase 3 fails** | NVIDIA GPU, 100+ hours of speech | Your own Telugu model |

Want **slang or a regional dialect**? See [Slang and dialect](#slang-and-dialect), including how to fine-tune an existing model.

**You do not need to train anything to get started.** You also do not need to train a model on your own voice: after the model speaks Telugu, your voice is cloned from a short recording, as today.

### What already works in Telugu

| Part | Telugu support |
|---|---|
| Speech recognition (Groq Whisper) | Whisper is multilingual. The app currently forces English (Phase 2 fixes this). |
| Answers (Groq chat model) | Can be asked to reply in Telugu. Quality is untested. |
| Speech (official PocketTTS) | **No Telugu.** Official languages: English, French, German, Portuguese, Italian, Spanish, Dutch. |

---

## Phase 1 · Try the existing community model

[**Thanushttk/pocket-tts-multilingual**](https://huggingface.co/Thanushttk/pocket-tts-multilingual) is listed among PocketTTS's [community-trained models](https://github.com/kyutai-labs/pocket-tts#list-of-community-trained-models).

| | From the model card |
|---|---|
| Languages | English, Hindi, Marathi, Kannada, Tamil, Malayalam, **Telugu**, including code-mixing |
| Size | About 418 MB, 6 layers, runs on CPU |
| Cloning | From a short WAV of your voice |
| License | CC-BY-4.0 (credit the author) |
| Telugu results (author's own test, 64 clips) | Word error **13.8%**, voice similarity **0.897**, quality score (UTMOS) **3.70** of 5 |

**Before you rely on it**
- It is **not an official Kyutai model**. The author calls it a first try, and it had no recorded downloads when checked.
- The results are self-reported.
- Its English word error (6.2%) comes from a different test set than the official model's figures, so keep the official model for English until you have compared them yourself.

### Steps (in your Codespace)

**1. Download the model**, pinned to the version reviewed here. `models/` is git-ignored.
```bash
uv run hf download Thanushttk/pocket-tts-multilingual \
    --revision 80c7f2835138c62750d76a93d4333c5c5bb6a754 \
    --local-dir models/te
```

**2. Record yourself**: 10–20 s of clear speech, saved as `my_voice.wav` in the project folder. Try one recording **in Telugu** and one in English, and compare. The author suggests an Indian-accented voice.

**3. Generate Telugu speech.** The paths inside the model's `config.yaml` are relative, so run from its folder:
```bash
cd models/te
uv run pocket-tts generate --config config.yaml \
    --voice ../../my_voice.wav \
    --text "నమస్కారం, మీరు ఎలా ఉన్నారు? ఈ రోజు వాతావరణం చాలా బాగుంది." \
    --temperature 0.3 \
    --output-path ../../telugu_test.wav
cd ../..
```

**4. Listen** to `telugu_test.wav`. If it sounds usable, continue. If the command fails, the model may not match your installed `pocket-tts` version; note the error.

---

## Phase 2 · Use it in Pocket Voice

These are **code changes still to be made**.

**1. Config with absolute paths.** The app runs from the project root, so copy the config and point it at the files:
```bash
D="$PWD/models/te"
sed -e "s#: model.safetensors#: $D/model.safetensors#" \
    -e "s#: tokenizer.model#: $D/tokenizer.model#" \
    "$D/config.yaml" > "$D/config.local.yaml"
grep -n "weights_path\|tokenizer_path" "$D/config.local.yaml"
```

**2. Load it** — in `_get_model()` in [app/tts.py](../app/tts.py):
```python
config = os.environ.get("POCKET_CONFIG")  # unset = official English model
extra = {"config": config} if config else {}
_model = TTSModel.load_model(quantize=get_settings().quantize, **extra)
```

**3. Telugu speech recognition** — in [app/asr.py](../app/asr.py), replace both `language="en"` values:
```python
language=os.environ.get("APP_LANGUAGE", "en")   # add "import os" at the top
```

**4. Settings** in `.env` (no code needed for the prompt):
```ini
POCKET_CONFIG=/workspaces/pocket-voice/models/te/config.local.yaml
APP_LANGUAGE=te
SYSTEM_PROMPT=మీరు స్నేహపూర్వక వాయిస్ సహాయకుడు. ఎప్పుడూ తెలుగు లిపిలో, ఒకటి నుండి మూడు చిన్న వాక్యాల్లో సమాధానం ఇవ్వండి. అంకెలను పదాలుగా రాయండి. మార్క్‌డౌన్ లేదా ఎమోజీలు వాడకండి.
```

**5. Telugu reading passage** — add to `PASSAGES` in [app/pages/1_Voice_Setup.py](../app/pages/1_Voice_Setup.py) (ask a native speaker to check it):
```python
"తెలుగు": "నమస్కారం! నేను నా సొంత గొంతును రికార్డ్ చేస్తున్నాను. నేను స్పష్టంగా, సహజమైన వేగంతో మాట్లాడతాను. "
          "వాతావరణం గురించి అయినా, వంట గురించి అయినా, సమాధానం నా గొంతులాగే వినిపించాలని కోరుకుంటున్నాను.",
```

**6. Create a new voice profile.** A saved voice belongs to the model that made it. Delete the old `default` profile and create it again. Do the same if you switch back to English.

---

## Phase 3 · Measure quality

Have a Telugu speaker rate at least 20 sentences: greetings, long sentences, numbers, dates, names and Telugu–English mixing.

| Check | How | Suggested target (adjust to taste) |
|---|---|---|
| Understandable | Listener writes what they hear | Nearly all words correct |
| Sounds like you | Listener score 1–5 | 4 or more |
| Natural | Listener score 1–5 | 3.5 or more |
| Words right (automatic) | Transcribe the output with Groq Whisper (`language="te"`) and compare | Use to compare versions |
| Speed and memory | `POCKET_CONFIG=... uv run python scripts/benchmark_tts.py --ref my_voice.wav` | Faster than real time |

The benchmark's built-in sentences are English: change `SENTENCES` in [scripts/benchmark_tts.py](../scripts/benchmark_tts.py) to Telugu first.

**Decide:** good enough → stop here. Not good enough → first try a better recording and simpler text. Only then consider Phase 4 or another model (see [Alternatives](#alternatives)).

---

## Phase 4 · Train your own Telugu model

Follows Kyutai's [training guide](https://github.com/kyutai-labs/pocket-tts/blob/main/training/README.md) ("Bring your own data") and [`finetune_language.yaml`](https://github.com/kyutai-labs/pocket-tts/blob/main/training/configs/finetune_language.yaml), which starts from the pretrained English model and learns a new text vocabulary.

### 4.0 Requirements
| Item | Requirement |
|---|---|
| Machine | Linux with an NVIDIA GPU (~16 GB with `batch_size: 16`, `grad_accum_steps: 4`). Not Codespaces or a Mac. |
| Data | Telugu speech **with accurate transcripts**: **100 h minimum, 1,000+ h for a strong model** |
| Disk | About 60 GB per 1,000 h of prepared audio, plus the raw download |
| Time | Not known for your GPU. Kyutai's reference: on 976 h of Czech, fine-tuning reached 12% word error by 10k steps; the config runs 250k steps. |

### 4.1 Get data

| Dataset | Telugu content | License | Notes |
|---|---|---|---|
| [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | ~136 h, 681 speakers, ~48k clips ([paper](https://arxiv.org/html/2409.05356v2)); ~67 GB | CC-BY-4.0 | Gated (accept terms). Mostly spontaneous speech, derived from a speech-recognition corpus. Best starting point; only just above the 100 h minimum. |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) | Larger recognition corpus (Telugu hours not checked) | CC-BY-4.0 | Gated. Noisier audio. |
| [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) | Check Telugu hours | CC-BY-4.0 | Gated. |
| [OpenSLR 66](https://openslr.org/66) | Multi-speaker, about 1 GB | CC BY-SA 4.0 | Share-alike terms. |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) (`te_in`) | Small | CC-BY-4.0 | Better kept for evaluation. |

Read each license yourself before training or sharing a model. Record only people who have agreed.

### 4.2 Prepare the text
- Use Telugu script, normalised to Unicode **NFC**.
- Write numbers, dates, symbols and abbreviations as **Telugu words**. The aligner ignores characters it doesn't know, which breaks the match between text and audio.
- Decide how to treat English words. Keep them only if your aligner and data cover them.
- Remove clips with wrong transcripts, noise or several speakers.
- Keep validation **speakers separate** from training speakers.

### 4.3 Build the manifest
One JSON line per clip. Example converter for a tab-separated file of `audio_path<TAB>transcript` (save as `make_manifest.py` inside your `pocket-tts` clone):
```python
import json, sys, unicodedata
import soundfile as sf

src, dst = sys.argv[1], sys.argv[2]
kept = 0
with open(src, encoding="utf-8") as rows, open(dst, "w", encoding="utf-8") as out:
    for row in rows:
        path, _, text = row.rstrip("\n").partition("\t")
        text = unicodedata.normalize("NFC", text).strip()
        if not text:
            continue
        info = sf.info(path)
        seconds = info.frames / info.samplerate
        if 1.0 <= seconds <= 30.0:  # the training config uses clips up to 30 s
            out.write(json.dumps({"path": path, "duration": round(seconds, 2), "transcript": text},
                                 ensure_ascii=False) + "\n")
            kept += 1
print("kept", kept, "clips")
```
```bash
git clone https://github.com/kyutai-labs/pocket-tts.git && cd pocket-tts && uv sync
uv run python make_manifest.py train.tsv data/train.jsonl
uv run python make_manifest.py valid.tsv data/valid.jsonl
```

### 4.4 Add word timings with a Telugu aligner
[anuragshas/wav2vec2-large-xlsr-53-telugu](https://huggingface.co/anuragshas/wav2vec2-large-xlsr-53-telugu) (Apache-2.0) has a Telugu character vocabulary, which the aligner needs.
```bash
for s in train valid; do
  uv run python -m training.scripts.align_data data/$s.jsonl data/${s}_aligned.jsonl \
      --model anuragshas/wav2vec2-large-xlsr-53-telugu --device cuda
done
```
Check how many clips were skipped, and spot-check some word timings by ear.

### 4.5 Train a Telugu tokenizer
```bash
uv run python -m training.scripts.train_tokenizer data/tokenizer data/train_aligned.jsonl
```
This writes `data/tokenizer.model` with 4,000 pieces, which matches the model's default. If your text is too small for 4,000, pass a smaller `--vocab-size` **and** set `flow_lm.lookup_table.n_bins` to the same number.

### 4.6 Configure and train the 24-layer model
```bash
cp training/configs/finetune_language.yaml training/configs/telugu.yaml
```
Edit:
```yaml
run_dir: runs/telugu
batch_size: 16          # for a ~16 GB GPU
grad_accum_steps: 4
```
Keep `lr: 2e-4` and `reset_text_embedding: true`. Then:
```bash
uv run training/train.py training/configs/telugu.yaml
```
Listen to `runs/telugu/samples/` as it trains. Word accuracy settles early; later steps mostly improve sound quality.

### 4.7 Distil to 6 layers (for fast CPU use)
Copy `training/configs/depth_distill.yaml` and change:
- `distill_teacher_weights` → your final checkpoint in `runs/telugu/`
- `data` → your Telugu manifests
- the tokenizer path (and `n_bins` if changed) for **both** student and teacher, as in your Telugu config
- `batch_size` / `grad_accum_steps` to fit your GPU

```bash
uv run training/train.py training/configs/telugu_distill.yaml
```

### 4.8 Evaluate and use
- Test on speakers and sentences not used in training: word error with a Telugu speech recogniser, voice similarity, and listeners. The bundled `librispeech.py` evaluation is English-only.
- Try it: `uv run pocket-tts generate --config <your_config.yaml> --checkpoint <your_checkpoint> --voice my_voice.wav --text "నమస్కారం"`
- Use it in the app as in [Phase 2](#phase-2--use-it-in-pocket-voice).
- Sharing it: upload weights, tokenizer and config to Hugging Face with the licenses and credits of every dataset used.

---

## Slang and dialect

Try these **in order**. Each step is much cheaper than the next.

| Level | Changes | Training? | Effort |
|---|---|---|---|
| 1. The words | The chat model **writes** in slang | No | Minutes |
| 2. The delivery | Your sample is **spoken** in that style | No | Minutes |
| 3. Pronunciation | Spell tricky words as they **sound** | No | Small code change |
| 4. Fine-tune | The speech model **learns** the dialect | Yes, GPU | Days |

### Level 1 · Make the answers use slang
Set `SYSTEM_PROMPT` in `.env` and restart. Casual English:
```ini
SYSTEM_PROMPT=You are my friendly buddy. Reply in one to three short sentences of casual spoken English with common slang. Plain text only, no emojis.
```
Telugu, everyday spoken style (adjust the region, and have a native speaker check the output):
```ini
SYSTEM_PROMPT=మీరు నా స్నేహితుడు. పుస్తక భాష వద్దు, తెలంగాణలో రోజూ మాట్లాడే యాసలో ఒకటి రెండు చిన్న వాక్యాల్లో తెలుగు లిపిలో జవాబు చెప్పండి. ఎమోజీలు వద్దు.
```
Adding two or three example sentences in your own slang to the prompt usually helps a chat model copy the style. Check the results: chat models can mix dialects.

### Level 2 · Record your sample in that style
Create a new voice profile while **speaking the way you want the assistant to sound**: your slang, rhythm and energy. The sample carries speaking style as well as voice. Make several profiles and compare. This changes delivery, but it does not teach the model new words.

### Level 3 · Fix individual words
If a slang word is said wrongly, write it the way it sounds (for example `gonna`, or an alternative Telugu spelling). A small replacement list applied just before speech generation can do this automatically. **Not built yet.**

### Level 4 · Fine-tune the existing model
Do this only if the model still mispronounces your slang or misses the accent after levels 1–3.

> **Experimental.** The training code can warm-start from any PocketTTS config's weights (`start_from_pretrained: true`, see [`training/args.py`](https://github.com/kyutai-labs/pocket-tts/blob/main/training/args.py)). Kyutai's documented recipe fine-tunes the 24-layer **English** teacher and then distils it. Fine-tuning a 6-layer model directly, as below, is **not a documented recipe and has not been tried**.

| Starting point | Route |
|---|---|
| English slang | Documented: `training/configs/finetune.yaml` (24-layer teacher), then [distil](#47-distil-to-6-layers-for-fast-cpu-use) |
| Telugu slang | Experimental: the community model publishes only its 6-layer version, so fine-tune that directly |

**1. Collect data.** Recordings of the slang, with transcripts written **exactly as spoken**.
- Spell each slang word the same way every time.
- Clean audio, one speaker per clip, 1–30 s each.
- Only voices that gave consent.
- No amount is documented for this. A first experiment could use 1–5 hours (a suggestion, not a tested figure). Mixing in some ordinary speech lowers the risk of the model forgetting it.

**2. Prepare and align** with the [manifest script](#43-build-the-manifest) and the [Telugu aligner](#44-add-word-timings-with-a-telugu-aligner), writing `data/slang_train_aligned.jsonl` and `data/slang_valid_aligned.jsonl`.

**3. Get the model on the GPU machine**: repeat Phase 1 step 1 and Phase 2 step 1 there to create `models/te/config.local.yaml`.

**4. Create `training/configs/slang.yaml`** (a starting point to tune):
```yaml
model_config: /absolute/path/to/models/te/config.local.yaml  # the existing model
start_from_pretrained: true     # continue from its weights
reset_text_embedding: false     # same tokenizer: keep what it already knows
freeze_head: true               # keeps a distilled model's sampler head unchanged

data:
  train_jsonl: data/slang_train_aligned.jsonl
  valid_jsonl: data/slang_valid_aligned.jsonl
  max_duration_sec: 30.0
  max_voice_prompt_sec: 5.0

flow:
  type: lsd
  kwargs: {}

flow_batch_multiplier: 4
eos_loss_weight: 0.1
text_dropout: 0.0               # distilled model already has guidance built in
voice_dropout: 0.0

optim:
  lr: 2e-5                      # low, so it adapts without forgetting
  muon_lr: 5e-4
  schedule: constant
  warmup_steps: 100

run_dir: runs/slang
batch_size: 16                  # for a ~16 GB GPU
grad_accum_steps: 4
max_steps: 2000
ema_decay: 0.99                 # short run: let the averaged weights keep up
log_freq: 50
valid_freq: 250
ckpt_freq: 250
num_ckpt_keep: 8                # keep several to compare by ear
sample_cfg_coef: 1.0
```

**5. Train**
```bash
uv run training/train.py training/configs/slang.yaml
```

**6. Compare checkpoints by ear.** Use the same slang **and** ordinary sentences each time. Stop at the checkpoint where slang improved and normal speech has not got worse.
```bash
uv run pocket-tts generate --config /absolute/path/to/models/te/config.local.yaml \
    --checkpoint runs/slang/checkpoint_00000500.pt \
    --voice my_voice.wav --text "మీ యాస వాక్యం ఇక్కడ" --output-path slang_500.wav
```

**7. Use it in the app** (after the Phase 2 changes): pass both `config=` and `checkpoint=` to `TTSModel.load_model`, then re-create your voice profile.

If results are poor, the documented route is to train a 24-layer Telugu model ([Phase 4](#phase-4--train-your-own-telugu-model)), fine-tune that on slang, and distil it.

The community model is CC-BY-4.0: credit its author if you share a model derived from it.

---

## Alternatives

Not tested, and probably heavier than PocketTTS.

| Option | Note |
|---|---|
| [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | Telugu voice cloning; typically needs a transcript of your sample. Listed as MIT, but a [third party](https://huggingface.co/dheeyantra/dhee-indic-f5) argues non-commercial terms carry over from its base data. Verify first. |
| [shankarpandala/chatterbox-telugu](https://huggingface.co/shankarpandala/chatterbox-telugu) | Community Telugu cloning model. Check its license and hardware needs. |

## Other languages
The same phases apply. First check PocketTTS's official languages (a `--language` option, no training) and its community list. Train only if neither covers your language.

## Risks

| Risk | Reduce it by |
|---|---|
| Community model quality or availability | Pin the revision, keep a local copy, test before relying on it |
| Poor Telugu answers from the chat model | Try `openai/gpt-oss-120b`; have a speaker review |
| Not enough clean data | Combine licensed datasets; filter hard |
| License problems | Record every dataset and model license before sharing |
| Misuse of cloned voices | Own voice or explicit consent only; label speech as AI-generated |

## Checklist
- [ ] Phase 1: community model generates Telugu in my voice
- [ ] Phase 2: app changes, new voice profile
- [ ] Phase 3: listener scores recorded, decision made
- [ ] Phase 4 (only if needed): data licensed → manifests → alignment → tokenizer → training → distillation → evaluation
