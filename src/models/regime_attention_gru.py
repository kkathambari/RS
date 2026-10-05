import os
import time
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GRU, Layer, Dropout, Concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class RegimeAdaptive_GRU_Enhanced_Attention(Layer):
    """
    REGIME-ADAPTIVE TEMPORAL ATTENTION LAYER (CORE RESEARCH CONTRIBUTION)
    
    Conditioning Mechanism:
      h'_t = GRU_att(h_t)                  [Refined temporal hidden representation]
      r_proj = Dense(r)                    [Projected 4-regime probability vector]
      e_t = v^T tanh(h'_t + r_proj + b)    [Regime-modulated temporal score]
      alpha_t = Softmax(e_t)               [Regime-adaptive attention distribution]
      c = sum_t (alpha_t * h_t)            [Dynamically aggregated context vector]
    """
    def __init__(self, att_units: int = 32, **kwargs):
        super(RegimeAdaptive_GRU_Enhanced_Attention, self).__init__(**kwargs)
        self.att_units = att_units
        self.attention_gru = None
        self.regime_projector = None
        self.attention_scorer = None
        self.last_attention = None

    def build(self, input_shape):
        # input_shape is a list of two shapes: [seq_shape, regime_shape]
        self.attention_gru = GRU(units=self.att_units, return_sequences=True, name=f"{self.name}_gru")
        self.regime_projector = Dense(units=self.att_units, activation=None, name=f"{self.name}_regime_proj")
        self.attention_scorer = Dense(1, activation='tanh', name=f"{self.name}_scorer")
        super(RegimeAdaptive_GRU_Enhanced_Attention, self).build(input_shape)

    def call(self, inputs, training=None):
        seq_inputs, regime_inputs = inputs
        # seq_inputs: (batch_size, time_step, d_model)
        # regime_inputs: (batch_size, 4)

        # 1. Temporal sequence refinement
        refined_states = self.attention_gru(seq_inputs, training=training)  # (batch_size, time_step, att_units)

        # 2. Market regime projection & broadcast along time axis
        r_proj = self.regime_projector(regime_inputs)                        # (batch_size, att_units)
        r_broadcast = tf.expand_dims(r_proj, axis=1)                         # (batch_size, 1, att_units)

        # 3. Joint scoring modulated by current market regime
        joint_state = refined_states + r_broadcast
        scores = self.attention_scorer(joint_state)                          # (batch_size, time_step, 1)
        scores = tf.squeeze(scores, axis=-1)                                 # (batch_size, time_step)

        # 4. Softmax normalization over lookback horizon
        attention_weights = tf.nn.softmax(scores, axis=-1)                   # (batch_size, time_step)
        self.last_attention = attention_weights

        # 5. Weighted context vector
        context_vector = tf.reduce_sum(seq_inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return context_vector

    def get_config(self):
        config = super(RegimeAdaptive_GRU_Enhanced_Attention, self).get_config()
        config.update({'att_units': self.att_units})
        return config


class RegimeAdaptive_GRU_Enhanced_Attention_Inspect(Layer):
    """Diagnostic inspector returning [context_vector, refined_states, attention_weights]."""
    def __init__(self, att_units: int = 32, **kwargs):
        super(RegimeAdaptive_GRU_Enhanced_Attention_Inspect, self).__init__(**kwargs)
        self.att_units = att_units
        self.attention_gru = None
        self.regime_projector = None
        self.attention_scorer = None

    def build(self, input_shape):
        self.attention_gru = GRU(units=self.att_units, return_sequences=True, name=f"{self.name}_gru")
        self.regime_projector = Dense(units=self.att_units, activation=None, name=f"{self.name}_regime_proj")
        self.attention_scorer = Dense(1, activation='tanh', name=f"{self.name}_scorer")
        super(RegimeAdaptive_GRU_Enhanced_Attention_Inspect, self).build(input_shape)

    def call(self, inputs, training=None):
        seq_inputs, regime_inputs = inputs
        refined_states = self.attention_gru(seq_inputs, training=training)
        r_proj = self.regime_projector(regime_inputs)
        r_broadcast = tf.expand_dims(r_proj, axis=1)
        joint_state = refined_states + r_broadcast
        scores = self.attention_scorer(joint_state)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.reduce_sum(seq_inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return [context_vector, refined_states, attention_weights]

    def get_config(self):
        config = super(RegimeAdaptive_GRU_Enhanced_Attention_Inspect, self).get_config()
        config.update({'att_units': self.att_units})
        return config


def build_regime_adaptive_model(time_step: int, num_features: int, num_regimes: int = 4,
                                gru_units: int = 64, att_units: int = 32,
                                dense_units: int = 32, dropout_rate: float = 0.2) -> Model:
    """Constructs the proposed Regime-Adaptive Attention GRU network."""
    seq_input = Input(shape=(time_step, num_features), name="seq_input")
    regime_input = Input(shape=(num_regimes,), name="regime_input")

    # Temporal feature encoder
    gru_seq = GRU(gru_units, return_sequences=True, name="gru_encoder")(seq_input)
    drop_1 = Dropout(dropout_rate, name="dropout_1")(gru_seq)

    # Regime-Adaptive Attention Layer
    att_layer = RegimeAdaptive_GRU_Enhanced_Attention(att_units=att_units, name="regime_adaptive_attention")
    context_vector = att_layer([drop_1, regime_input])

    # Prediction Head
    dense_1 = Dense(dense_units, activation='relu', name="dense_1")(context_vector)
    drop_2 = Dropout(dropout_rate, name="dropout_2")(dense_1)
    output = Dense(1, name="price_prediction")(drop_2)

    model = Model(inputs=[seq_input, regime_input], outputs=output, name="Proposed_Regime_Adaptive_GRU")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_regime_adaptive_model(X_train_seq, X_train_reg, y_train,
                                X_val_seq, X_val_reg, y_val,
                                time_step: int = 60, epochs: int = 30, batch_size: int = 64,
                                gru_units: int = 64, att_units: int = 32, dense_units: int = 32,
                                ckpt_dir: str = "artifacts/checkpoints"):
    """Trains the proposed architecture with early stopping on validation split."""
    os.makedirs(ckpt_dir, exist_ok=True)
    num_features = X_train_seq.shape[2]
    num_regimes = X_train_reg.shape[1]

    model = build_regime_adaptive_model(
        time_step=time_step,
        num_features=num_features,
        num_regimes=num_regimes,
        gru_units=gru_units,
        att_units=att_units,
        dense_units=dense_units
    )

    ckpt_path = os.path.join(ckpt_dir, "regime_adaptive_best.weights.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)
    ]

    t0 = time.time()
    history = model.fit(
        [X_train_seq, X_train_reg], y_train,
        validation_data=([X_val_seq, X_val_reg], y_val),
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
