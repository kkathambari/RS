import os
import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GRU, Layer, Dropout, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


class GRU_Enhanced_Attention(Layer):
    """
    Primary temporal attention layer.
    Stage 1: Secondary GRU filters & contextualizes sequence hidden states.
    Stage 2: Dense tanh projection scores temporal significance, normalized via Softmax.
    Output: Dynamically aggregated context vector.
    """
    def __init__(self, att_units=32, **kwargs):
        super(GRU_Enhanced_Attention, self).__init__(**kwargs)
        self.att_units = att_units
        self.attention_gru = None
        self.attention_scorer = None
        self.last_attention = None

    def build(self, input_shape):
        self.attention_gru = GRU(units=self.att_units, return_sequences=True, name=f"{self.name}_gru")
        self.attention_scorer = Dense(1, activation='tanh', name=f"{self.name}_scorer")
        super(GRU_Enhanced_Attention, self).build(input_shape)

    def call(self, inputs, training=None):
        refined_states = self.attention_gru(inputs, training=training)
        scores = self.attention_scorer(refined_states)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        self.last_attention = attention_weights
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return context_vector

    def get_config(self):
        config = super(GRU_Enhanced_Attention, self).get_config()
        config.update({'att_units': self.att_units})
        return config


class GRU_Enhanced_Attention_Inspect(Layer):
    """
    Diagnostic layer returning [context_vector, refined_states, attention_weights]
    for interpretability and visualization.
    """
    def __init__(self, att_units=32, **kwargs):
        super(GRU_Enhanced_Attention_Inspect, self).__init__(**kwargs)
        self.att_units = att_units
        self.attention_gru = None
        self.attention_scorer = None

    def build(self, input_shape):
        self.attention_gru = GRU(units=self.att_units, return_sequences=True, name=f"{self.name}_gru")
        self.attention_scorer = Dense(1, activation='tanh', name=f"{self.name}_scorer")
        super(GRU_Enhanced_Attention_Inspect, self).build(input_shape)

    def call(self, inputs, training=None):
        refined_states = self.attention_gru(inputs, training=training)
        scores = self.attention_scorer(refined_states)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return [context_vector, refined_states, attention_weights]

    def get_config(self):
        config = super(GRU_Enhanced_Attention_Inspect, self).get_config()
        config.update({'att_units': self.att_units})
        return config


def build_gru_model(time_step: int, num_features: int, use_attention: bool = True,
                    gru_units: int = 64, att_units: int = 32, dense_units: int = 32,
                    dropout_rate: float = 0.2) -> Model:
    """Constructs the research architecture: GRU + [Custom Attention | GAP] + Regressor."""
    inputs = Input(shape=(time_step, num_features), name="sequence_input")
    gru_seq = GRU(gru_units, return_sequences=True, name="gru_encoder")(inputs)
    drop_1 = Dropout(dropout_rate, name="dropout_1")(gru_seq)

    if use_attention:
        att_layer = GRU_Enhanced_Attention(att_units=att_units, name="custom_attention_layer")
        context_vec = att_layer(drop_1)
    else:
        context_vec = GlobalAveragePooling1D(name="gap_pooling")(drop_1)

    dense = Dense(dense_units, activation='relu', name="dense_1")(context_vec)
    drop_2 = Dropout(dropout_rate, name="dropout_2")(dense)
    output = Dense(1, name="prediction_head")(drop_2)

    model = Model(inputs=inputs, outputs=output, name="proposed_attention_gru" if use_attention else "ablation_gap_gru")
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_gru_model(X_train, y_train, X_val, y_val, use_attention: bool = True,
                    time_step: int = 60, epochs: int = 30, batch_size: int = 64,
                    gru_units: int = 64, att_units: int = 32, dense_units: int = 32,
                    ckpt_dir: str = "artifacts/checkpoints", ckpt_name: str = None):
    """Trains model strictly using X_val for validation; X_test is held out."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model = build_gru_model(
        time_step=time_step,
        num_features=X_train.shape[2],
        use_attention=use_attention,
        gru_units=gru_units,
        att_units=att_units,
        dense_units=dense_units
    )

    prefix = ckpt_name if ckpt_name is not None else ('att' if use_attention else 'noatt')
    ckpt_path = os.path.join(ckpt_dir, f"{prefix}_best.weights.h5")
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
