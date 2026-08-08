"""
modeling.py
-----------
Model architecture for the HandTalk Transformer.

This file only defines the architecture (layers + Transformer model class).
It does NOT contain trained weights — those live in `transformer_weights.h5`
in this same repo. Because the model was saved with `model.save_weights(...)`
(weights-only, not `SavedModel`/full `.h5`), you MUST rebuild the exact same
architecture with this file before you can load the weights back in.

Usage
-----
    from modeling import build_model, FEATURE_COLUMNS

    model = build_model()                      # builds architecture w/ config.json defaults
    model.load_weights("transformer_weights.h5")

See inference.py for a full end-to-end example (landmarks -> predicted text).
"""

import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Feature columns (hand + pose landmark coordinates used as model input)
# ---------------------------------------------------------------------------
LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

FEATURE_COLUMNS = X + Y + Z

X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "x_" in col]
Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "y_" in col]
Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "right" in col]
LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "left" in col]
RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS) if "pose" in col and int(col[-2:]) in LPOSE]

FRAME_LEN = 128


# ---------------------------------------------------------------------------
# Preprocessing: raw landmark sequence -> fixed-length, normalized tensor
# ---------------------------------------------------------------------------
def resize_pad(x):
    if tf.shape(x)[0] < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN - tf.shape(x)[0]], [0, 0], [0, 0]]))
    else:
        x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
    return x


def pre_process(x):
    """Detects the dominant hand, normalizes coordinates, and pads/resizes
    the sequence to a fixed FRAME_LEN so it can be batched."""
    rhand = tf.gather(x, RHAND_IDX, axis=1)
    lhand = tf.gather(x, LHAND_IDX, axis=1)
    rpose = tf.gather(x, RPOSE_IDX, axis=1)
    lpose = tf.gather(x, LPOSE_IDX, axis=1)

    rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
    lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)

    rnans = tf.math.count_nonzero(rnan_idx)
    lnans = tf.math.count_nonzero(lnan_idx)

    if rnans > lnans:
        hand = lhand
        pose = lpose

        hand_x = hand[:, 0 * (len(LHAND_IDX) // 3): 1 * (len(LHAND_IDX) // 3)]
        hand_y = hand[:, 1 * (len(LHAND_IDX) // 3): 2 * (len(LHAND_IDX) // 3)]
        hand_z = hand[:, 2 * (len(LHAND_IDX) // 3): 3 * (len(LHAND_IDX) // 3)]
        hand = tf.concat([1 - hand_x, hand_y, hand_z], axis=1)

        pose_x = pose[:, 0 * (len(LPOSE_IDX) // 3): 1 * (len(LPOSE_IDX) // 3)]
        pose_y = pose[:, 1 * (len(LPOSE_IDX) // 3): 2 * (len(LPOSE_IDX) // 3)]
        pose_z = pose[:, 2 * (len(LPOSE_IDX) // 3): 3 * (len(LPOSE_IDX) // 3)]
        pose = tf.concat([1 - pose_x, pose_y, pose_z], axis=1)
    else:
        hand = rhand
        pose = rpose

    hand_x = hand[:, 0 * (len(LHAND_IDX) // 3): 1 * (len(LHAND_IDX) // 3)]
    hand_y = hand[:, 1 * (len(LHAND_IDX) // 3): 2 * (len(LHAND_IDX) // 3)]
    hand_z = hand[:, 2 * (len(LHAND_IDX) // 3): 3 * (len(LHAND_IDX) // 3)]
    hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)

    mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
    std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
    hand = (hand - mean) / std

    pose_x = pose[:, 0 * (len(LPOSE_IDX) // 3): 1 * (len(LPOSE_IDX) // 3)]
    pose_y = pose[:, 1 * (len(LPOSE_IDX) // 3): 2 * (len(LPOSE_IDX) // 3)]
    pose_z = pose[:, 2 * (len(LPOSE_IDX) // 3): 3 * (len(LPOSE_IDX) // 3)]
    pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)

    x = tf.concat([hand, pose], axis=1)
    x = resize_pad(x)

    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    return x


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class LandmarkEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.conv3 = tf.keras.layers.Conv1D(num_hid, 11, strides=2, padding="same", activation="relu")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential([
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat([batch_size[..., tf.newaxis], tf.constant([1, 1], dtype=tf.int32)], 0)
        return tf.tile(mask, mult)

    def call(self, enc_out, target, training):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att, training=training))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out, training=training) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out, training=training))
        return ffn_out_norm


class Transformer(keras.Model):
    def __init__(
        self,
        num_hid=64,
        num_head=2,
        num_feed_forward=128,
        source_maxlen=100,
        target_maxlen=100,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=60,
        pad_token_idx=59,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.acc_metric = keras.metrics.Mean(name="edit_dist")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        self.pad_token_idx = pad_token_idx

        self.enc_input = LandmarkEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid)

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [TransformerEncoder(num_hid, num_head, num_feed_forward) for _ in range(num_layers_enc)]
        )

        for i in range(num_layers_dec):
            setattr(self, f"dec_layer_{i}", TransformerDecoder(num_hid, num_head, num_feed_forward))

        self.classifier = layers.Dense(num_classes)

    def decode(self, enc_out, target, training=False):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y, training=training)
        return y

    def call(self, inputs, training=False):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source, training=training)
        y = self.decode(x, target, training=training)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def generate(self, source, target_start_token_idx):
        """Greedy decoding: landmarks -> sequence of predicted token ids."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source, training=False)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        for _ in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input, training=False)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = logits[:, -1][..., tf.newaxis]
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_model(config_path=None):
    """Builds a Transformer with the same config used for training.
    Loads hyperparameters from config.json next to this file unless a
    different path is given."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model = Transformer(
        num_hid=cfg["num_hid"],
        num_head=cfg["num_head"],
        num_feed_forward=cfg["num_feed_forward"],
        source_maxlen=cfg["source_maxlen"],
        target_maxlen=cfg["target_maxlen"],
        num_layers_enc=cfg["num_layers_enc"],
        num_layers_dec=cfg["num_layers_dec"],
        num_classes=cfg["num_classes"],
        pad_token_idx=cfg["pad_token_idx"],
    )

    # Build the model's variables by running one dummy forward pass before
    # load_weights() — Keras subclassed models need this.
    dummy_source = tf.zeros((1, FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
    dummy_target = tf.zeros((1, 1), dtype=tf.int32)
    model([dummy_source, dummy_target], training=False)

    return model
