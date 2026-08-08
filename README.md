---
tags:
  - keras
  - tensorflow
  - sign-language
  - fingerspelling
  - transformer
  - asl
license: mit
---

# HandTalk (Transformer)

A Transformer-based model that reads a sequence of hand/pose landmarks
(extracted with [MediaPipe](https://developers.google.com/mediapipe)) from a
video of American Sign Language fingerspelling and predicts the spelled-out
phrase, character by character.

Trained on the [ASL Fingerspelling](https://www.kaggle.com/competitions/asl-fingerspelling)
Kaggle dataset. Architecture follows the Keras
["Automatic Speech Recognition with Transformer"](https://keras.io/examples/audio/transformer_asr/)
example, adapted for landmark-sequence input instead of audio spectrograms.

## How it works

1. **Input**: per-frame (x, y, z) coordinates for the dominant hand (21
   points) and 10 pose points related to hand movement (`FEATURE_COLUMNS`
   in `modeling.py`).
2. **Encoder**: 3 strided 1D conv layers downsample the sequence, followed
   by 4 Transformer encoder blocks.
3. **Decoder**: 1 Transformer decoder block, autoregressively predicting
   one character at a time (greedy decoding) until the end token.

## Config

Reverse-engineered from the actual weight shapes (see `config.json`):
`num_hid=200`, `num_head=4`, `num_feed_forward=400`, `num_layers_enc=2`,
`num_layers_dec=1`, `num_classes=62`, `target_maxlen=64`.

## Files in this repo

| File | Purpose |
|---|---|
| `modeling.py` | Model architecture (must be imported to rebuild the model before loading weights) |
| `config.json` | Hyperparameters used to build the architecture |
| `transformer_weights.h5` | Trained weights (`model.save_weights(...)` output — **not** a full SavedModel) |
| `inference.py` | End-to-end example: landmarks in, predicted text out |
| `requirements.txt` | Python dependencies |

> ⚠️ This model was saved with `save_weights()`, not `model.save()`, so the
> weights file alone is not enough — you need `modeling.py` to reconstruct
> the exact architecture first, then load the weights into it.

## Usage

```python
from modeling import build_model, pre_process
import tensorflow as tf

model = build_model()
model.load_weights("transformer_weights.h5")

# landmarks: np.ndarray of shape (num_frames, num_feature_columns)
x = pre_process(tf.constant(landmarks, dtype=tf.float32))[None, ...]
token_ids = model.generate(x, target_start_token_idx=60)
```

See `inference.py` for the full pipeline including turning token ids back
into characters.

## Vocabulary

You'll need your `character_to_prediction_index.json` (character ↔ id
mapping) from training — it is not included in this repo. Place it next to
`inference.py` before running predictions.

## Limitations

- Requires the dominant hand to be visible in frame; heavy occlusion or
  motion blur will degrade predictions.
- Trained specifically on fingerspelled English phrases, not full ASL
  grammar/vocabulary.

## Citation / Acknowledgements

- Dataset: Kaggle ASL Fingerspelling competition
- Model design references: Keras Transformer ASR example, and community
  notebooks by `irohith` and `shlomoron` on Kaggle.
