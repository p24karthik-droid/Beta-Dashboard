# Beta Analysis Dashboard

An interactive web application for calculating and visualizing stock beta using CAPM (Capital Asset Pricing Model) or Rolling Beta methods.

## 🎯 Features

- **CAPM Beta Analysis**: Calculate beta with and without alpha intercept
- **Rolling Beta Analysis**: Track how beta changes over time with customizable windows
- **Interactive Charts**: Plotly-powered visualizations with dark lavender theme
- **750+ NSE Stocks**: Pre-loaded list of Nifty 750 companies
- **Statistical Metrics**: Comprehensive regression statistics including p-values, confidence intervals, R², F-statistics
- **Normalized Price Charts**: Compare stock vs market performance on a base-100 scale
- **Export Options**: Download charts as PNG and data as CSV
- **Responsive Design**: Works on desktop and mobile browsers

## 📊 Analysis Methods

### CAPM Beta
- Beta with Alpha: Full regression analysis with intercept
- Beta without Alpha: Forced-through-origin regression
- Includes 95% confidence intervals and hypothesis testing

### Rolling Beta
- Customizable rolling window size
- Historical beta trends visualization
- Statistical summary (mean, median, std dev, min, max)

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.30.0
- **Data**: yfinance 0.2.33 (Yahoo Finance API)
- **Visualization**: Plotly 5.18.0, Matplotlib 3.8.2
- **Statistics**: statsmodels 0.14.1, scikit-learn 1.3.2
- **Data Processing**: pandas 2.1.4, numpy 1.26.2

## 🚀 Local Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/beta-analysis-dashboard.git
cd beta-analysis-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run beta_dashboard.py
```

4. Open your browser to `http://localhost:8501`

## 📱 Online Access

Visit the live app: **[Your Streamlit URL will be here]**

## 📈 How to Use

1. **Select Market Index**: Choose benchmark (S&P 500, NSEI, etc.)
2. **Choose Company**: Pick from 750+ NSE stocks or major US stocks
3. **Set Date Range**: Define analysis period
4. **Configure Parameters**:
   - Data frequency (Daily/Weekly/Monthly)
   - Beta method (CAPM/Rolling)
   - Risk-free rate
   - Chart color (for PNG exports)
5. **Click "Calculate Beta"**: View results and download data

## 📊 Interpreting Results

- **Beta > 1**: Stock is more volatile than the market
- **Beta < 1**: Stock is less volatile than the market
- **Beta < 0**: Stock moves inversely to the market
- **P-value < 0.05**: Statistically significant beta
- **High R²**: Strong correlation with market

## 📂 Project Structure

```
beta-analysis-dashboard/
├── beta_dashboard.py       # Main application
├── Nifty 750.csv          # NSE stock list
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit theme config
└── README.md              # This file
```

## 🎨 Theme

The dashboard uses a VSCode-inspired dark lavender theme:
- Primary: Lavender (#c792ea)
- Accent: Cyan (#89ddff)
- Background: Dark (#1e1e1e)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available for personal and educational use.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: Data is sourced from Yahoo Finance via yfinance. Market data may have delays or limitations based on the data provider.
