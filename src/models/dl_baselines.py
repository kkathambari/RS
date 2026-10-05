import os
import time
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Dense, GRU, LSTM, Layer, Dropout,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class StandardAdditiveAttention(Layer):
    """
    Standard Bahdanau-style Additive Attention:
    Scores directly from GRU hidden states h_t:
        e_t = v^T tanh(W h_t + b)
        \alpha_t = softmax(e_t)
        c = \sum \alpha_t h_t
    (Does NOT have the secondary GRU filter present in GRU_Enhanced_Attention).
    """
    def __init__(self, att_units=32, **kwargs):
        super(StandardAdditiveAttention, self).__init__(**kwargs)
        self.att_units = att_units
        self.dense_score = None
        self.last_attention = None

    def build(self, input_shape):
        self.dense_score = Dense(1, activation='tanh', name=f"{self.name}_score")
        super(StandardAdditiveAttention, self).build(input_shape)

    def call(self, inputs, training=None):
        scores = self.dense_score(inputs)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        self.last_attention = attention_weights
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return context_vector

    def get_config(self):
        config = super(StandardAdditiveAttention, self).get_config()
        config.update({'att_units': self.att_units})
        return config


def build_vanilla_gru(time_step: int, num_features: int, units: int = 64,
                      dense_units: int = 32, dropout: float = 0.2) -> Model:
    """Vanilla GRU: Sequences processed and final state feeds Dense head."""
    inputs = Input(shape=(time_step, num_features), name="gru_input")
    gru_out = GRU(units, return_sequences=False, name="vanilla_gru")(inputs)
    drop_1 = Dropout(dropout)(gru_out)
    dense = Dense(dense_units, activation='relu')(drop_1)
    drop_2 = Dropout(dropout)(dense)
    output = Dense(1, name="prediction")(drop_2)
    model = Model(inputs=inputs, outputs=output, name="Baseline_Vanilla_GRU")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_vanilla_lstm(time_step: int, num_features: int, units: int = 64,
                       dense_units: int = 32, dropout: float = 0.2) -> Model:
    """Vanilla LSTM: Standard recurrent LSTM baseline."""
    inputs = Input(shape=(time_step, num_features), name="lstm_input")
    lstm_out = LSTM(units, return_sequences=False, name="vanilla_lstm")(inputs)
    drop_1 = Dropout(dropout)(lstm_out)
    dense = Dense(dense_units, activation='relu')(drop_1)
    drop_2 = Dropout(dropout)(dense)
    output = Dense(1, name="prediction")(drop_2)
    model = Model(inputs=inputs, outputs=output, name="Baseline_Vanilla_LSTM")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_standard_attention_gru(time_step: int, num_features: int, units: int = 64,
                                 att_units: int = 32, dense_units: int = 32,
                                 dropout: float = 0.2) -> Model:
    """GRU + Standard Additive Attention (direct ablation against GRU-Enhanced)."""
    inputs = Input(shape=(time_step, num_features), name="att_gru_input")
    gru_seq = GRU(units, return_sequences=True, name="feature_gru")(inputs)
    drop_1 = Dropout(dropout)(gru_seq)
    att_layer = StandardAdditiveAttention(att_units=att_units, name="std_attention")
    context_vec = att_layer(drop_1)
    dense = Dense(dense_units, activation='relu')(context_vec)
    drop_2 = Dropout(dropout)(dense)
    output = Dense(1, name="prediction")(drop_2)
    model = Model(inputs=inputs, outputs=output, name="Baseline_Standard_Attention_GRU")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_transformer_model(time_step: int, num_features: int, d_model: int = 64,
                            num_heads: int = 4, ff_dim: int = 64, dropout: float = 0.2) -> Model:
    """Transformer Encoder Architecture for Financial Sequences."""
    inputs = Input(shape=(time_step, num_features), name="transformer_input")
    # Linear projection to d_model
    x = Dense(d_model)(inputs)

    # Multi-Head Self-Attention Block
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x1 = LayerNormalization(epsilon=1e-6)(x + attn_out)

    # Feed-Forward Network Block
    ffn = Dense(ff_dim, activation='relu')(x1)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout)(ffn)
    x2 = LayerNormalization(epsilon=1e-6)(x1 + ffn)

    # Aggregation & Output
    pooled = GlobalAveragePooling1D()(x2)
    dense = Dense(32, activation='relu')(pooled)
    drop_final = Dropout(dropout)(dense)
    output = Dense(1, name="prediction")(drop_final)

    model = Model(inputs=inputs, outputs=output, name="Baseline_Transformer")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_dl_baseline(model: Model, X_train, y_train, X_val, y_val,
                      epochs: int = 30, batch_size: int = 64,
                      ckpt_dir: str = "artifacts/checkpoints", ckpt_name: str = "baseline"):
    """Standardized training routine with early stopping on X_val."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}_best.weights.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )
    elapsed = time.time() - t0

    if os.path.exists(ckpt_path):
        try:
            model.load_weights(ckpt_path)
        except Exception:
            pass

    return model, elapsed, history
