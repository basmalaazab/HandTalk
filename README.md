# HandTalk — ASL Fingerspelling Recognition 🤟

A Transformer-based model that reads a sequence of hand and pose landmarks
(extracted with [MediaPipe](https://developers.google.com/mediapipe)) from a
video of American Sign Language fingerspelling and predicts the spelled-out
phrase, character by character.

Trained on the [Google - ASL Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling)
Kaggle competition dataset (3M+ fingerspelled characters from 100+ Deaf
signers). Architecture follows the Keras
["Automatic Speech Recognition with Transformer"](https://keras.io/examples/audio/transformer_asr/)
example, adapted to take landmark sequences instead of audio spectrograms.

---

## 🔗 Links

| What | Where |
|---|---|
| Trained model + weights | [huggingface.co/basmalaazab/asl-fingerspelling-transformer](https://huggingface.co/basmalaazab/asl-fingerspelling-transformer) |
| Training notebook | [`Fingerspelling_Recognition.ipynb`](./Fingerspelling_Recognition.ipynb) |
| Live camera demo | Gradio app, runs locally — see [Running the live demo](#-running-the-live-demo-locally) below |

---

## How it works

1. **Input**: per-frame (x, y, z) coordinates for the dominant hand (21
   points) and 10 pose points related to hand/arm movement.
2. **Encoder**: 3 strided 1D convolution layers downsample the landmark
   sequence, followed by 2 Transformer encoder blocks.
3. **Decoder**: 1 Transformer decoder block, autoregressively predicting one
   character at a time (greedy decoding) until an end-of-sequence token.

**Actual trained config** (reverse-engineered from the saved weights):

| Param | Value |
|---|---|
| `num_hid` | 200 |
| `num_head` | 4 |
| `num_feed_forward` | 400 |
| `num_layers_enc` | 2 |
| `num_layers_dec` | 1 |
| `num_classes` | 62 |
| `target_maxlen` | 64 |

---

## Repo structure

```
.
├── Fingerspelling_Recognition.ipynb   # full training notebook (Kaggle)
├── modeling.py                        # model architecture (rebuild before loading weights)
├── config.json                        # hyperparameters matching the trained weights
├── inference.py                       # end-to-end example: landmarks -> predicted text
├── transformer_weights.h5             # trained weights (save_weights output, not a full SavedModel)
├── app.py                             # Gradio live-camera demo
├── requirements.txt                   # Python dependencies
├── upload_to_hub.py                   # pushes the model files to the HF model repo
└── upload_space.py                    # pushes app.py etc. to an HF Space
```

> ⚠️ The model was saved with `model.save_weights(...)`, not `model.save(...)`,
> so `transformer_weights.h5` alone isn't enough to use the model — you need
> `modeling.py` to rebuild the exact same architecture first, then load the
> weights into it. `build_model()` in `modeling.py` does this for you.

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

Main dependencies: `tensorflow`, `mediapipe`, `gradio`, `huggingface_hub`, `numpy`.

---

## 🚀 Using the model

```python
from modeling import build_model, pre_process
import tensorflow as tf

model = build_model()
model.load_weights("transformer_weights.h5")

# landmarks: np.ndarray of shape (num_frames, num_feature_columns)
x = pre_process(tf.constant(landmarks, dtype=tf.float32))[None, ...]
token_ids = model.generate(x, target_start_token_idx=60)
```

See `inference.py` for the full pipeline, including turning predicted token
ids back into readable characters using `character_to_prediction_index.json`
(included in this repo, sourced from the
[competition dataset](https://www.kaggle.com/competitions/asl-fingerspelling/data)).

---

## 🎥 Running the live demo (locally)

The live webcam demo (`app.py`) reads your camera, extracts landmarks with
MediaPipe in real time, buffers ~60 frames, and runs the model to predict
the spelled phrase.

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`) in
your browser and allow camera access.

> This currently runs **locally only**. Hosting a live Gradio app on
> Hugging Face Spaces' free CPU tier now requires a **PRO** subscription
> (static Spaces are still free); it isn't deployed as a public Space for
> that reason.

`character_to_prediction_index.json` is included next to `app.py`, so the
demo shows readable letters instead of raw token ids.

> **Known limitation:** prediction accuracy is noticeably lower in this live
> demo than during training/evaluation. The model was trained on the
> competition's pre-processed landmark sequences (fixed lighting, camera
> angle, and offline MediaPipe extraction); the live demo extracts landmarks
> from a webcam in real time under different conditions, and the fixed
> 60-frame buffering window doesn't always line up with how fast someone
> spells. Improving this — e.g. fine-tuning on real webcam data or making
> the buffering window adaptive — is listed under future improvements.

---

## 📤 Publishing updates

```bash
# push model files (modeling.py, config.json, weights) to the model repo
python upload_to_hub.py --repo-id basmalaazab/asl-fingerspelling-transformer

# push the Gradio app to a Hugging Face Space (requires PRO for CPU hosting)
python upload_space.py --repo-id basmalaazab/asl-fingerspelling-live
```

Both scripts require being logged in once via:
```bash
python -c "from huggingface_hub import login; login()"
```

---

## Limitations

- Requires the dominant hand to be clearly visible; heavy occlusion or
  motion blur degrades predictions.
- Trained specifically on fingerspelled English phrases, not full ASL
  grammar/vocabulary.
- Best results with good, even lighting.

---

## Acknowledgements

- Dataset: [Google - ASL Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling), Kaggle
- Model design references: Keras Transformer ASR example, and community
  notebooks on Kaggle
- Landmark extraction: [MediaPipe Holistic](https://developers.google.com/mediapipe)

## License

MIT
