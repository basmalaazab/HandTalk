"""
app.py
------
Live webcam demo for HandTalk (basmalaazab/asl-fingerspelling-transformer).

Pipeline:
  webcam frame -> MediaPipe Holistic (hand + pose landmarks)
  -> buffer of frames -> Transformer.generate() -> predicted text

Runs on Hugging Face Spaces (Gradio SDK).
"""

import json
import os

import gradio as gr
import mediapipe as mp
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# 1. Download model files from the model repo (not this Space's repo)
# ---------------------------------------------------------------------------
MODEL_REPO = "basmalaazab/asl-fingerspelling-transformer"

modeling_path = hf_hub_download(MODEL_REPO, "modeling.py")
config_path = hf_hub_download(MODEL_REPO, "config.json")
weights_path = hf_hub_download(MODEL_REPO, "transformer_weights.h5")

# modeling.py must be importable -> copy next to this file under a fixed name
import shutil
shutil.copy(modeling_path, os.path.join(os.path.dirname(__file__), "_modeling.py"))

from _modeling import (  # noqa: E402
    build_model,
    pre_process,
    FEATURE_COLUMNS,
    LPOSE,
    RPOSE,
    POSE,
)

model = build_model(config_path=config_path)
model.load_weights(weights_path)

with open(config_path) as f:
    CFG = json.load(f)

START_TOKEN_IDX = CFG["start_token_idx"]
END_TOKEN_IDX = CFG["end_token_idx"]

# ---------------------------------------------------------------------------
# 2. Vocabulary (character <-> id). Optional: upload your own to this Space
#    as character_to_prediction_index.json for readable text output.
# ---------------------------------------------------------------------------
NUM_TO_CHAR = None
vocab_local = os.path.join(os.path.dirname(__file__), "character_to_prediction_index.json")
if os.path.exists(vocab_local):
    with open(vocab_local) as f:
        char_to_num = json.load(f)
    char_to_num["P"] = CFG["pad_token_idx"]
    char_to_num["<"] = CFG["start_token_idx"]
    char_to_num[">"] = CFG["end_token_idx"]
    NUM_TO_CHAR = {v: k for k, v in char_to_num.items()}

# ---------------------------------------------------------------------------
# 3. MediaPipe Holistic setup — one instance shared across frames
# ---------------------------------------------------------------------------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

RHAND_COLS = [f"x_right_hand_{i}" for i in range(21)]
LHAND_COLS = [f"x_left_hand_{i}" for i in range(21)]


def frame_to_row(frame_rgb):
    """Runs MediaPipe on one frame and returns a single row of length
    len(FEATURE_COLUMNS), matching modeling.py's X + Y + Z column order.
    Missing landmarks are filled with NaN (pre_process handles that)."""
    results = holistic.process(frame_rgb)

    def hand_xyz(landmarks):
        if landmarks is None:
            return [np.nan] * 21, [np.nan] * 21, [np.nan] * 21
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        zs = [lm.z for lm in landmarks.landmark]
        return xs, ys, zs

    rx, ry, rz = hand_xyz(results.right_hand_landmarks)
    lx, ly, lz = hand_xyz(results.left_hand_landmarks)

    pose_idx_order = LPOSE + RPOSE  # matches modeling.py's POSE order
    if results.pose_landmarks is not None:
        plm = results.pose_landmarks.landmark
        px = [plm[i].x for i in pose_idx_order]
        py = [plm[i].y for i in pose_idx_order]
        pz = [plm[i].z for i in pose_idx_order]
    else:
        px = py = pz = [np.nan] * len(pose_idx_order)

    x_row = rx + lx + px
    y_row = ry + ly + py
    z_row = rz + lz + pz
    return x_row + y_row + z_row


# ---------------------------------------------------------------------------
# 4. Streaming state: accumulate frames, run inference every WINDOW frames
# ---------------------------------------------------------------------------
WINDOW = 60  # ~2-4 seconds depending on webcam fps; tune as needed
buffer = []
last_prediction = ""


def decode(token_ids):
    chars = []
    for idx in token_ids[1:]:  # skip start token
        idx = int(idx)
        if idx == END_TOKEN_IDX:
            break
        chars.append(NUM_TO_CHAR[idx] if NUM_TO_CHAR else f"[{idx}]")
    return "".join(chars) if NUM_TO_CHAR else " ".join(chars)


def process_stream(frame):
    global buffer, last_prediction

    if frame is None:
        return last_prediction

    row = frame_to_row(frame)
    buffer.append(row)

    if len(buffer) >= WINDOW:
        landmarks = np.array(buffer, dtype=np.float32)
        buffer = []  # reset for the next phrase

        x = pre_process(tf.constant(landmarks))[tf.newaxis, ...]
        token_ids = model.generate(x, START_TOKEN_IDX)[0].numpy()
        last_prediction = decode(token_ids)

    return last_prediction


# ---------------------------------------------------------------------------
# 5. Gradio UI
# ---------------------------------------------------------------------------
vocab_note = (
    "" if NUM_TO_CHAR else
    "\n\n⚠️ No `character_to_prediction_index.json` found in this Space — "
    "showing raw token ids instead of letters. Upload that file to this "
    "Space's Files tab for readable text output."
)

with gr.Blocks(title="HandTalk — Live Demo") as demo:
    gr.Markdown(
        f"# HandTalk — Live Camera Demo\n"
        f"Spell a word with your dominant hand in front of the camera. "
        f"Every ~{WINDOW} frames the model reads the buffered motion and "
        f"predicts the phrase.{vocab_note}"
    )
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, label="Camera")
        out = gr.Textbox(label="Predicted text", interactive=False)

    cam.stream(fn=process_stream, inputs=cam, outputs=out, time_limit=600)

if __name__ == "__main__":
    demo.launch()