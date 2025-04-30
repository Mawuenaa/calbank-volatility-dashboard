# Calbank Volatility Dashboard

This project presents a **Streamlit-based dashboard** that models and forecasts stock price volatility for **Calbank PLC** using time series models such as **GARCH**. It visualizes trends, model diagnostics, and forecast results to provide financial insights.

## 🔍 Features

- Historical return analysis of Calbank stock (2015–2024)
- Volatility modeling with ARCH/GARCH models
- Interactive Streamlit dashboard with visual plots
- Forecasting future volatility
- Custom branding with Calbank colors and logo

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Arch (for GARCH modeling)
- Matplotlib, Seaborn
- Streamlit

## 📁 Project Structure

```
calbank-volatility-dashboard/
│
├── data/                       # Contains stock data CSVs
├── images/                     # App icon & charts (e.g., trend.png)
├── calbank_garch_app.py        # Main Streamlit app
├── README.md                   # Project overview
├── requirements.txt            # Required Python packages
```

## ▶️ How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/Mawuenaa/calbank-volatility-dashboard.git
   cd calbank-volatility-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the dashboard**
   ```bash
   streamlit run calbank_garch_app.py
   ```

## 📈 Example Forecast Plot

Below is a preview of the CalBank Volatility Forecast Dashboard created using Streamlit and GARCH modeling:

![Dashboard Screenshot](images/dashboard.png)

## 💡 Motivation

Volatility modeling is essential for **risk management and financial modeling**. This project helps showcase practical applications of time-series forecasting in the Ghanaian equity market, using real data.

## 📬 Contact

For questions or collaboration, contact **Mawuenaa** at gabrielapomah03@gmail.com.
