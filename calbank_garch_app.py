import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set page configuration
st.set_page_config(page_title="Calbank Volatility Dashboard", layout="wide")

# Custom CSS for white background and Calbank styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: white;
        }
        .css-18e3th9 {
            background-color: white !important;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: #333;
        }
        .stButton>button {
            background-color: #F9B000;
            color: white;
        }
        .stButton>button:hover {
            background-color: #F26722;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Calbank brand colors
PRIMARY_COLOR = "#F9B000"  # Calbank yellow
SECONDARY_COLOR = "#F26722"  # Calbank orange
PLOT_BG = "ivory"
FONT_COLOR = "#000000"
GRID_COLOR = "lightgray"

# Logo and title
col_logo, col_title = st.columns([1, 13])
with col_logo:
    st.image(r"C:\Users\user\Python_Stuff\fin_projs\calbank_volatility_dashboard\images\calbank_logo.jpg", width=60)
with col_title:
    st.markdown(f"<h2 style='color:{PRIMARY_COLOR}; margin-bottom: 0;'>Calbank Volatility Dashboard</h2>", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    path = r"C:\Users\user\Python_Stuff\fin_projs\calbank_volatility_dashboard\data\Daily Shares  ETFs 2023.xlsx"
    data = pd.read_excel(path)
    data['Daily Date'] = pd.to_datetime(data['Daily Date'], dayfirst=True)
    data.set_index('Daily Date', inplace=True)
    data.sort_index(inplace=True)
    data = data[['Closing Price - VWAP (GH¢)']].rename(columns={'Closing Price - VWAP (GH¢)': 'Close'})
    data['returns'] = np.log(data['Close']).diff()
    data.dropna(inplace=True)
    return data

data = load_data()

# Sidebar for model settings
with st.sidebar:
    st.header("🔧 GARCH Model Parameters")
    st.markdown("""
    <style>
    .stSidebar {
        background-color: #ffa952;
    }
    .stSidebar label {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    p = st.number_input("ARCH term (p)", min_value=1, max_value=3, value=1, step=1)
    q = st.number_input("GARCH term (q)", min_value=1, max_value=3, value=1, step=1)

# Function to standardize all plots
def style_plot(fig, y_title):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(
            title="Date",
            title_font=dict(color=FONT_COLOR),
            tickfont=dict(color=FONT_COLOR),
            showgrid=True,
            gridcolor=GRID_COLOR
        ),
        yaxis=dict(
            title=y_title,
            title_font=dict(color=FONT_COLOR),
            tickfont=dict(color=FONT_COLOR),
            showgrid=True,
            gridcolor=GRID_COLOR
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300
    )
    return fig

# Price and returns plots
col1, col2 = st.columns(2)

with col1:
    col_icon1, col_title1 = st.columns([1, 8])
    with col_icon1:
        st.image(r"C:\Users\user\Python_Stuff\fin_projs\calbank_volatility_dashboard\images\trend.png", width=60)
    with col_title1:
        st.markdown("<div style='font-weight: bold; color: grey; font-size: 23px'>Closing Price</div>", unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', mode='lines',
                                   line=dict(color=PRIMARY_COLOR)))
    fig_price = style_plot(fig_price, "Price")
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    col_icon2, col_title2 = st.columns([1, 8])
    with col_icon2:
        st.image(r"C:\Users\user\Python_Stuff\fin_projs\calbank_volatility_dashboard\images\growth.png", width=60)
    with col_title2:
        st.markdown("<div style='font-weight: bold; color: grey; font-size: 23px'>Log Returns</div>", unsafe_allow_html=True)

    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(x=data.index, y=data['returns'], name='Returns', mode='lines',
                                 line=dict(color=PRIMARY_COLOR)))
    fig_ret = style_plot(fig_ret, "Returns")
    st.plotly_chart(fig_ret, use_container_width=True)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)


# Fit GARCH model
model = arch_model(data.returns, vol='GARCH', p=p, q=q)
result = model.fit(disp='off')

# Volatility Over Time (from the GARCH model)
col_vol_icon, col_vol_title = st.columns([1, 15])
with col_vol_icon:
    st.image(r"C:\Users\user\Python_Stuff\fin_projs\calbank_volatility_dashboard\images\volatility.png", width=50)
with col_vol_title:
    st.markdown("<div style='font-weight: bold; color: grey; font-size: 30px'>Volatility Over Time</div>", unsafe_allow_html=True)

fig_vol_time = go.Figure()
fig_vol_time.add_trace(go.Scatter(x=data.index, y=np.sqrt(result.conditional_volatility), mode='lines',
                                  name='Volatility', line=dict(color=PRIMARY_COLOR)))
fig_vol_time = style_plot(fig_vol_time, "Volatility")
st.plotly_chart(fig_vol_time, use_container_width=True)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)


# ACF and PACF of Squared Returns (Styled)
col_icon, col_title = st.columns([1, 16.5])
with col_icon:
    st.image(r"C:\Users\user\Python_Stuff\fin_projs\calbank_volatility_dashboard\images\return-on-investment.png", width=42)
with col_title:
    st.markdown(
        "<div style='font-weight: bold; color: grey; font-size: 25px; padding-top: 5px'>ACF & PACF of Squared Returns</div>",
        unsafe_allow_html=True
    )

# Create squared returns
data['squared_returns'] = data['returns'] ** 2

# Plot with styled figure
fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=100)

# Ivory background and orange lines
fig.patch.set_facecolor('ivory')
ax[0].set_facecolor('ivory')
ax[1].set_facecolor('ivory')

plot_acf(data['squared_returns'].dropna(), lags=20, ax=ax[0], color=PRIMARY_COLOR)
plot_pacf(data['squared_returns'].dropna(), lags=20, ax=ax[1], color=PRIMARY_COLOR)

# Titles and grid
ax[0].set_title("ACF of Squared Returns", fontsize=12, color='black', weight='bold')
ax[1].set_title("PACF of Squared Returns", fontsize=12, color='black', weight='bold')

for axis in ax:
    axis.grid(True, linestyle='--', color='lightgray', alpha=0.7)
    axis.tick_params(axis='x', colors='black')
    axis.tick_params(axis='y', colors='black')

plt.tight_layout()
st.pyplot(fig)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)

# AIC and BIC in styled boxes
col_aic, col_bic = st.columns(2)

with col_aic:
    st.markdown(f"""
    <div style="
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 10px 15px;
        border-radius: 5px;
        text-align: center;
    ">
        <div style="font-weight: bold; color: black;"> AIC (Akaike Info Criterion)</div>
        <div style="font-size: 24px; color: black;">{result.aic:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_bic:
    st.markdown(f"""
    <div style="
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 10px 15px;
        border-radius: 5px;
        text-align: center;
    ">
        <div style="font-weight: bold; color: black;"> BIC (Bayesian Info Criterion)</div>
        <div style="font-size: 24px; color: black;">{result.bic:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)

# 30-Day Volatility Forecast
col_forecast_icon, col_forecast_title = st.columns([1, 16])
with col_forecast_icon:
    st.image(r"C:\Users\user\Python_Stuff\fin_projs\calbank_volatility_dashboard\images\business-forcasting.png", width=45)  # <-- use your own image path here
with col_forecast_title:
    st.markdown("<div style='font-weight: bold; color: grey; font-size: 25px;'>30-Day Volatility Forecast</div>", unsafe_allow_html=True)

forecast = result.forecast(horizon=30)
vol_forecast = forecast.variance.iloc[-1]
vol_forecast_sqrt = np.sqrt(vol_forecast)

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(y=vol_forecast_sqrt.values, mode='lines+markers', name='Volatility',
                             line=dict(color=PRIMARY_COLOR)))
fig_vol = style_plot(fig_vol, "Forecasted Volatility")
st.plotly_chart(fig_vol, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set page configuration
st.set_page_config(page_title="Calbank Volatility Dashboard", layout="wide")

# Custom CSS for white background and Calbank styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: white;
        }
        .css-18e3th9 {
            background-color: white !important;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: #333;
        }
        .stButton>button {
            background-color: #F9B000;
            color: white;
        }
        .stButton>button:hover {
            background-color: #F26722;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Calbank brand colors
PRIMARY_COLOR = "#F9B000"  # Calbank yellow
SECONDARY_COLOR = "#F26722"  # Calbank orange
PLOT_BG = "ivory"
FONT_COLOR = "#000000"
GRID_COLOR = "lightgray"

# Logo and title
col_logo, col_title = st.columns([1, 13])
with col_logo:
    st.image(r"C:\Users\user\Pictures\Saved Pictures\calbank_logo.jpg", width=60)
with col_title:
    st.markdown(f"<h2 style='color:{PRIMARY_COLOR}; margin-bottom: 0;'>Calbank Volatility Dashboard</h2>", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    path = r"C:\Users\user\Downloads\Daily Shares  ETFs 2023.xlsx"
    data = pd.read_excel(path)
    data['Daily Date'] = pd.to_datetime(data['Daily Date'], dayfirst=True)
    data.set_index('Daily Date', inplace=True)
    data.sort_index(inplace=True)
    data = data[['Closing Price - VWAP (GH¢)']].rename(columns={'Closing Price - VWAP (GH¢)': 'Close'})
    data['returns'] = np.log(data['Close']).diff()
    data.dropna(inplace=True)
    return data

data = load_data()

# Sidebar for model settings
with st.sidebar:
    st.header("🔧 GARCH Model Parameters")
    st.markdown("""
    <style>
    .stSidebar {
        background-color: #ffa952;
    }
    .stSidebar label {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    p = st.number_input("ARCH term (p)", min_value=1, max_value=3, value=1, step=1)
    q = st.number_input("GARCH term (q)", min_value=1, max_value=3, value=1, step=1)

# Function to standardize all plots
def style_plot(fig, y_title):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font=dict(color=FONT_COLOR),
        xaxis=dict(
            title="Date",
            title_font=dict(color=FONT_COLOR),
            tickfont=dict(color=FONT_COLOR),
            showgrid=True,
            gridcolor=GRID_COLOR
        ),
        yaxis=dict(
            title=y_title,
            title_font=dict(color=FONT_COLOR),
            tickfont=dict(color=FONT_COLOR),
            showgrid=True,
            gridcolor=GRID_COLOR
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300
    )
    return fig

# Price and returns plots
col1, col2 = st.columns(2)

with col1:
    col_icon1, col_title1 = st.columns([1, 8])
    with col_icon1:
        st.image(r"C:\Users\user\Downloads\trend.png", width=60)
    with col_title1:
        st.markdown("<div style='font-weight: bold; color: grey; font-size: 23px'>Closing Price</div>", unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', mode='lines',
                                   line=dict(color=PRIMARY_COLOR)))
    fig_price = style_plot(fig_price, "Price")
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    col_icon2, col_title2 = st.columns([1, 8])
    with col_icon2:
        st.image(r"C:\Users\user\Downloads\growth.png", width=60)
    with col_title2:
        st.markdown("<div style='font-weight: bold; color: grey; font-size: 23px'>Log Returns</div>", unsafe_allow_html=True)

    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(x=data.index, y=data['returns'], name='Returns', mode='lines',
                                 line=dict(color=PRIMARY_COLOR)))
    fig_ret = style_plot(fig_ret, "Returns")
    st.plotly_chart(fig_ret, use_container_width=True)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)


# Fit GARCH model
model = arch_model(data.returns, vol='GARCH', p=p, q=q)
result = model.fit(disp='off')

# Volatility Over Time (from the GARCH model)
col_vol_icon, col_vol_title = st.columns([1, 15])
with col_vol_icon:
    st.image(r"C:\Users\user\Downloads\volatility.png", width=50)
with col_vol_title:
    st.markdown("<div style='font-weight: bold; color: grey; font-size: 30px'>Volatility Over Time</div>", unsafe_allow_html=True)

fig_vol_time = go.Figure()
fig_vol_time.add_trace(go.Scatter(x=data.index, y=np.sqrt(result.conditional_volatility), mode='lines',
                                  name='Volatility', line=dict(color=PRIMARY_COLOR)))
fig_vol_time = style_plot(fig_vol_time, "Volatility")
st.plotly_chart(fig_vol_time, use_container_width=True)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)


# ACF and PACF of Squared Returns (Styled)
col_icon, col_title = st.columns([1, 16.5])
with col_icon:
    st.image(r"C:\Users\user\Downloads\return-on-investment (1).png", width=42)
with col_title:
    st.markdown(
        "<div style='font-weight: bold; color: grey; font-size: 25px; padding-top: 5px'>ACF & PACF of Squared Returns</div>",
        unsafe_allow_html=True
    )

# Create squared returns
data['squared_returns'] = data['returns'] ** 2

# Plot with styled figure
fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=100)

# Ivory background and orange lines
fig.patch.set_facecolor('ivory')
ax[0].set_facecolor('ivory')
ax[1].set_facecolor('ivory')

plot_acf(data['squared_returns'].dropna(), lags=20, ax=ax[0], color=PRIMARY_COLOR)
plot_pacf(data['squared_returns'].dropna(), lags=20, ax=ax[1], color=PRIMARY_COLOR)

# Titles and grid
ax[0].set_title("ACF of Squared Returns", fontsize=12, color='black', weight='bold')
ax[1].set_title("PACF of Squared Returns", fontsize=12, color='black', weight='bold')

for axis in ax:
    axis.grid(True, linestyle='--', color='lightgray', alpha=0.7)
    axis.tick_params(axis='x', colors='black')
    axis.tick_params(axis='y', colors='black')

plt.tight_layout()
st.pyplot(fig)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)

# AIC and BIC in styled boxes
col_aic, col_bic = st.columns(2)

with col_aic:
    st.markdown(f"""
    <div style="
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 10px 15px;
        border-radius: 5px;
        text-align: center;
    ">
        <div style="font-weight: bold; color: black;"> AIC (Akaike Info Criterion)</div>
        <div style="font-size: 24px; color: black;">{result.aic:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_bic:
    st.markdown(f"""
    <div style="
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 10px 15px;
        border-radius: 5px;
        text-align: center;
    ">
        <div style="font-weight: bold; color: black;"> BIC (Bayesian Info Criterion)</div>
        <div style="font-size: 24px; color: black;">{result.bic:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# Add a nice horizontal line here
st.markdown("<hr style='margin-top: 20px; margin-bottom: 20px; border: 1px solid lightgrey;'>", unsafe_allow_html=True)

# 30-Day Volatility Forecast
col_forecast_icon, col_forecast_title = st.columns([1, 16])
with col_forecast_icon:
    st.image(r"C:\Users\user\Downloads\business-forcasting.png", width=45)  # <-- use your own image path here
with col_forecast_title:
    st.markdown("<div style='font-weight: bold; color: grey; font-size: 25px;'>30-Day Volatility Forecast</div>", unsafe_allow_html=True)

forecast = result.forecast(horizon=30)
vol_forecast = forecast.variance.iloc[-1]
vol_forecast_sqrt = np.sqrt(vol_forecast)

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(y=vol_forecast_sqrt.values, mode='lines+markers', name='Volatility',
                             line=dict(color=PRIMARY_COLOR)))
fig_vol = style_plot(fig_vol, "Forecasted Volatility")
st.plotly_chart(fig_vol, use_container_width=True)
