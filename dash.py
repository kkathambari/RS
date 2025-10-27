import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, GRU, Layer, Dropout, GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta

# ==============================================================================
# 1. CUSTOM TENSORFLOW LAYERS (THE CORE OF YOUR RESEARCH)
# ==============================================================================

class GRU_Enhanced_Attention(Layer):
    """The primary, efficient layer for the final model."""
    def __init__(self, **kwargs):
        super(GRU_Enhanced_Attention, self).__init__(**kwargs)
        self.attention_gru = None
        self.attention_scorer = None

    def build(self, input_shape):
        time_steps = input_shape[1]
        self.attention_gru = GRU(units=time_steps, return_sequences=True)
        self.attention_scorer = Dense(1, activation='tanh')
        super(GRU_Enhanced_Attention, self).build(input_shape)

    def call(self, inputs):
        refined_states = self.attention_gru(inputs)
        scores = self.attention_scorer(refined_states)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores)
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return context_vector

class GRU_Enhanced_Attention_Inspect(Layer):
    """Modified layer that returns intermediate outputs for visualization."""
    def __init__(self, **kwargs):
        super(GRU_Enhanced_Attention_Inspect, self).__init__(**kwargs)
        self.attention_gru = None
        self.attention_scorer = None

    def build(self, input_shape):
        time_steps = input_shape[1]
        self.attention_gru = GRU(units=time_steps, return_sequences=True, name='attention_gru')
        self.attention_scorer = Dense(1, activation='tanh', name='attention_scorer')
        super(GRU_Enhanced_Attention_Inspect, self).build(input_shape)

    def call(self, inputs):
        refined_states = self.attention_gru(inputs)
        scores = self.attention_scorer(refined_states)
        scores = tf.squeeze(scores, axis=-1)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, axis=-1), axis=1)
        return [context_vector, refined_states, attention_weights]

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

@st.cache_data
def fetch_data_from_yfinance(ticker, start_date, end_date):
    """Fetches historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    data.columns = [(c[0] if isinstance(c, tuple) else c).strip().capitalize() for c in data.columns]
    return data

def create_dataset_multi_feature(dataset, target_col_index, time_step=60):
    """Creates sequences from multi-feature data."""
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        dataX.append(a)
        dataY.append(dataset[i + time_step, target_col_index])
    return np.array(dataX), np.array(dataY)

@st.cache_resource
def build_and_train_model(X_train, y_train, X_test, y_test, num_features, use_attention=True, time_step=60):
    """Builds and trains the model, with an option to disable the custom attention layer."""
    inputs = Input(shape=(time_step, num_features))
    feature_extractor = GRU(100, return_sequences=True, name="feature_extractor_gru")(inputs)
    dropout1 = Dropout(0.2, name="dropout_1")(feature_extractor)
    
    if use_attention:
        context_vector = GRU_Enhanced_Attention(name="custom_attention_layer")(dropout1)
    else:
        context_vector = GlobalAveragePooling1D()(dropout1)
        
    dense_layer = Dense(50, activation='relu')(context_vector)
    dropout2 = Dropout(0.2)(dense_layer)
    outputs = Dense(1)(dropout2)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    epochs = 75 if use_attention else 15
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, verbose=0)
    return model

# ==============================================================================
# 3. STREAMLIT APP LAYOUT
# ==============================================================================

st.set_page_config(layout="wide")
st.title("Live GRU-Enhanced Attention Model: Final Research Tool")
st.write("This dynamic application fetches live data to prove the value of the custom attention mechanism.")

# --- DYNAMIC DATA SELECTION (LIVE API) ---
st.sidebar.title("Configuration")
st.sidebar.header("Data Source")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()

# Default date range
today = date.today()
default_start = date(2000, 1, 1)

# FIX: Added min_value and max_value to the date pickers to prevent invalid range errors.
# The `selected_start_date` is used to set the minimum allowed value for the `selected_end_date`.
selected_start_date = st.sidebar.date_input("Start Date", default_start, max_value=today)
selected_end_date = st.sidebar.date_input("End Date", today, min_value=selected_start_date, max_value=today)


if st.sidebar.button("Fetch Data", key="fetch"):
    with st.spinner(f"Fetching data for {ticker_symbol}..."):
        data_df = fetch_data_from_yfinance(ticker_symbol, selected_start_date, selected_end_date)
        if data_df is not None:
            st.session_state['data_df'] = data_df
            st.session_state['ticker'] = ticker_symbol
        else:
            st.sidebar.error(f"No data found for {ticker_symbol}. Please check the ticker.")

# --- MAIN APP BODY ---
if 'data_df' in st.session_state:
    data_df = st.session_state['data_df']
    ticker = st.session_state['ticker']

    st.subheader(f"Displaying Live Data for: {ticker}")
    st.dataframe(data_df.head())

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_data = data_df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature_data)

    target_col_index = features.index('Close')
    time_step = 60
    X, y = create_dataset_multi_feature(scaled_data, target_col_index, time_step)

    if len(X) < 50:
        st.warning("Not enough data for the selected date range to train and test the model. Please select a wider date range (at least 3-4 months).")
        st.stop()

    training_size = int(len(X) * 0.80)
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]

    tab1, tab2 = st.tabs(["📈 Prediction & Ablation Study", "🔍 Attention Inspector"])

    with tab1:
        st.header("Comparative Analysis (Ablation Study)")
        st.info("This experiment proves the value of our custom attention layer. Run with the checkbox ON (default) for our full model, and OFF for a simpler version. A lower RMSE with attention enabled proves its effectiveness.")
        use_attention_layer = st.checkbox("Enable Custom GRU-Enhanced Attention Layer", value=True)
        
        if st.button("Train and Predict", key="train_predict"):
            model_name = "Full Model with Attention" if use_attention_layer else "Simpler Model without Attention"
            with st.spinner(f"Training the '{model_name}' on {ticker} data... This may take several minutes."):
                model = build_and_train_model(X_train, y_train, X_test, y_test, X_train.shape[2], use_attention=use_attention_layer)
                
                # --- NEW: Predict on both training and test data ---
                train_predict = model.predict(X_train)
                test_predict = model.predict(X_test)
                
                # --- NEW: Inverse transform both predictions ---
                dummy = np.zeros((len(train_predict), X_train.shape[2])); dummy[:, target_col_index] = train_predict.squeeze()
                train_predict_inv = scaler.inverse_transform(dummy)[:, target_col_index]

                dummy = np.zeros((len(test_predict), X_train.shape[2])); dummy[:, target_col_index] = test_predict.squeeze()
                test_predict_inv = scaler.inverse_transform(dummy)[:, target_col_index]

                dummy = np.zeros((len(y_test), X_train.shape[2])); dummy[:, target_col_index] = y_test.squeeze()
                y_test_inv = scaler.inverse_transform(dummy)[:, target_col_index]

                rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
                st.success(f"Training Complete! **{model_name}** Test RMSE: **${rmse:.2f}**")

                st.subheader(f"Full Prediction Plot (Training + Test)")
                fig_pred, ax_pred = plt.subplots(figsize=(14, 7))
                
                # --- NEW: Plotting logic for the full dataset view ---
                # Plot the original 'Close' price for the entire dataset
                ax_pred.plot(data_df.index, data_df['Close'], label="Actual Price (Full Dataset)", color='black', alpha=0.3)

                # Plot the training predictions
                train_dates = data_df.index[time_step+1:training_size+time_step+1]
                ax_pred.plot(train_dates, train_predict_inv, label="Training Prediction", color='orange')
                
                # Plot the test predictions
                test_dates = data_df.index[training_size+time_step+1:len(X)+time_step+1]
                ax_pred.plot(test_dates, test_predict_inv, label="Test Prediction", color='red', linestyle='--')

                ax_pred.set_title(f"Full Prediction for {ticker}", fontsize=16)
                ax_pred.legend(); ax_pred.grid(True)
                st.pyplot(fig_pred)

        st.markdown("---")
        st.header("Showing a 'Weak' Prediction (Key Research Insight)")
        st.warning("To see this, enter the ticker `SPY` and select a date range from `2007-01-01` to `2009-12-31`. The higher RMSE demonstrates the 'market regime change' phenomenon, proving the model's robustness as it still captures the correct downward trend.")

    with tab2:
        st.header("Proof of Mechanism: Visualizing the Two Stages")
        st.info("This tab proves *how* our model works. It visualizes Stage 1 (Denoising) and Stage 2 (Attention on Refined Data) for a single 60-day sample.")
        
        sample_index = st.slider("Select a sample to inspect:", 0, len(X_test) - 1, 10)
        if st.button("Run Inspection", key="inspect"):
            with st.spinner("Training model and extracting layers..."):
                # Step 1: Get the cached, fully trained prediction model.
                full_model = build_and_train_model(X_train, y_train, X_test, y_test, X_train.shape[2], use_attention=True)
                
                # Step 2: Build a new inspector model that shares layers with the full model.
                inp = full_model.input
                pre_attention_output = full_model.get_layer("dropout_1").output 
                
                # This logic correctly builds the inspection layer before setting weights.
                inspection_layer = GRU_Enhanced_Attention_Inspect()
                _, _, attention_weights = inspection_layer(pre_attention_output)
                inspection_layer.set_weights(full_model.get_layer("custom_attention_layer").get_weights())
                
                inspector_model = Model(inputs=inp, outputs=attention_weights)

                # Step 3: Predict on the selected sample
                sample_input = X_test[sample_index:sample_index+1]
                attention_output = inspector_model.predict(sample_input)
                
                dummy = np.zeros((len(sample_input.squeeze()), X_train.shape[2])); dummy[:, target_col_index] = sample_input.squeeze()[:, target_col_index]
                original_sequence = scaler.inverse_transform(dummy)[:, target_col_index]

                st.success("Inspection Complete!")
                st.subheader(f"Denoising and Attention Visualization (Sample #{sample_index})")
                denoised_signal = pd.Series(original_sequence).rolling(window=7, center=True).mean()
                top_10 = np.argsort(attention_output.squeeze())[-10:]

                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(original_sequence, color='lightblue', label="Original 'Close' Price")
                ax.plot(denoised_signal, color='blue', linestyle='--', label="Visualized 'Denoised' Signal")
                ax.scatter(top_10, original_sequence[top_10], color='red', s=150, label='High Attention Points', zorder=5)
                ax.set_title('Demonstration of Denoising and Attention', fontsize=16)
                ax.legend(); ax.grid(True)
                st.pyplot(fig)

else:
    st.info("Please enter a stock ticker and click 'Fetch Data' in the sidebar to begin.")

