"""
Beta Analysis Dashboard
Simple Streamlit app for calculating stock beta using CAPM and Rolling Beta methods
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression  # Keep for rolling beta
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Beta Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for VSCode Dark Lavender theme
st.markdown("""
    <style>
    /* VSCode Dark Theme Colors */
    :root {
        --vscode-bg: #1e1e1e;
        --vscode-sidebar: #252526;
        --vscode-hover: #2a2d2e;
        --vscode-lavender: #c792ea;
        --vscode-purple: #b392f0;
        --vscode-blue: #82aaff;
        --vscode-cyan: #89ddff;
        --vscode-pink: #f07178;
        --vscode-green: #c3e88d;
        --vscode-yellow: #ffcb6b;
        --vscode-text: #d4d4d4;
        --vscode-text-dim: #858585;
    }
    
    /* Main background */
    .stApp {
        background-color: var(--vscode-bg);
    }
    
    /* Sidebar styling with lavender outline theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(37, 37, 38, 0.95), rgba(30, 30, 30, 0.95));
        border-right: 2px solid rgba(199, 146, 234, 0.3);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--vscode-text);
    }
    
    /* Sidebar header with lavender accent */
    [data-testid="stSidebar"] h2 {
        color: var(--vscode-lavender);
        border-bottom: 2px solid rgba(199, 146, 234, 0.3);
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Main layout */
    .main {
        padding: 2rem;
        color: var(--vscode-text);
    }
    
    /* Buttons - outlined lavender style (matching download buttons) */
    .stButton>button {
        width: 100%;
        background: rgba(199, 146, 234, 0.1);
        border: 1px solid var(--vscode-lavender);
        color: var(--vscode-lavender);
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: rgba(199, 146, 234, 0.2);
        box-shadow: 0 4px 12px rgba(199, 146, 234, 0.4);
        transform: translateY(-2px);
    }
    
    /* Download buttons - outlined lavender style */
    .stDownloadButton>button {
        background: rgba(199, 146, 234, 0.1);
        border: 1px solid var(--vscode-lavender);
        color: var(--vscode-lavender);
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        background: rgba(199, 146, 234, 0.2);
        box-shadow: 0 0 15px rgba(199, 146, 234, 0.3);
    }
    
    /* Professional typography */
    h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, var(--vscode-lavender), var(--vscode-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 3px solid var(--vscode-lavender);
        padding-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        color: var(--vscode-lavender);
    }
    
    h3 {
        font-size: 1.3rem !important;
        font-weight: 500 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        color: var(--vscode-cyan);
    }
    
    h4 {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: var(--vscode-purple);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: var(--vscode-lavender);
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--vscode-text-dim);
    }
    
    /* Compact stats card */
    .stats-card {
        background: linear-gradient(135deg, rgba(199, 146, 234, 0.05), rgba(137, 221, 255, 0.05));
        border-left: 3px solid var(--vscode-lavender);
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 1rem;
        min-height: 200px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        box-shadow: 0 6px 12px rgba(199, 146, 234, 0.2);
        transform: translateY(-2px);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem 1rem;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .stat-label {
        color: var(--vscode-text-dim);
        font-weight: 400;
    }
    
    .stat-value {
        color: var(--vscode-lavender);
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(199, 146, 234, 0.1);
        border-left: 4px solid var(--vscode-lavender);
        border-radius: 6px;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(199, 146, 234, 0.3);
    }
    
    /* Input fields with lavender outline theme (matching download buttons) */
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input,
    .stDateInput>div>div>input {
        background: rgba(199, 146, 234, 0.05);
        color: var(--vscode-text);
        border: 1px solid rgba(199, 146, 234, 0.3);
        border-radius: 6px;
    }
    
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus,
    .stDateInput>div>div>input:focus {
        border-color: var(--vscode-lavender);
        box-shadow: 0 0 0 1px var(--vscode-lavender);
        background: rgba(199, 146, 234, 0.1);
    }
    
    /* Input labels with lavender color */
    .stTextInput>label,
    .stSelectbox>label,
    .stNumberInput>label,
    .stDateInput>label {
        color: var(--vscode-lavender) !important;
        font-weight: 500;
    }
    
    /* Radio buttons with lavender outline */
    .stRadio>div {
        background: rgba(199, 146, 234, 0.05);
        border: 1px solid rgba(199, 146, 234, 0.3);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stRadio>label {
        color: var(--vscode-lavender) !important;
        font-weight: 500;
    }
    
    /* Color picker with lavender outline */
    input[type="color"] {
        border: 2px solid rgba(199, 146, 234, 0.5);
        border-radius: 6px;
    }
    
    /* Markdown content */
    .stMarkdown {
        color: var(--vscode-text);
    }
    
    /* Code blocks */
    code {
        background-color: var(--vscode-hover);
        color: var(--vscode-cyan);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner="Fetching data...")
def fetch_data(stock_ticker, market_ticker, start_date, frequency='Daily'):
    """
    Fetch stock and market data using yfinance
    Returns Close prices for both tickers
    """
    try:
        # Add headers to avoid being blocked by Yahoo Finance on cloud servers
        import yfinance as yf
        from datetime import datetime
        
        # Set up session with proper headers
        session = yf.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        
        # Convert start_date to datetime if it's a string
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        # Download separately to avoid yfinance issues with multiple tickers
        stock_data = yf.download(
            stock_ticker,
            start=start_date,
            auto_adjust=False,
            progress=False,
            session=session,
            timeout=10
        )
        
        market_data = yf.download(
            market_ticker,
            start=start_date,
            auto_adjust=False,
            progress=False,
            session=session,
            timeout=10
        )
        
        if stock_data.empty:
            st.warning(f"No data returned for {stock_ticker}")
            return None, None
            
        if market_data.empty:
            st.warning(f"No data returned for {market_ticker}")
            return None, None
        
        # Extract Close prices
        stock_prices = stock_data['Close'] if 'Close' in stock_data.columns else stock_data.iloc[:, 0]
        market_prices = market_data['Close'] if 'Close' in market_data.columns else market_data.iloc[:, 0]
        
        stock_prices = stock_prices.dropna()
        market_prices = market_prices.dropna()
        
        if len(stock_prices) == 0 or len(market_prices) == 0:
            st.warning(f"No data after dropna. Stock: {len(stock_prices)}, Market: {len(market_prices)}")
            return None, None
        
        # Resample if needed
        if frequency == 'Weekly':
            stock_prices = stock_prices.resample('W').last().dropna()
            market_prices = market_prices.resample('W').last().dropna()
        elif frequency == 'Monthly':
            try:
                stock_prices = stock_prices.resample('ME').last().dropna()
                market_prices = market_prices.resample('ME').last().dropna()
            except:
                # Fallback for older pandas versions
                stock_prices = stock_prices.resample('M').last().dropna()
                market_prices = market_prices.resample('M').last().dropna()
        
        return stock_prices, market_prices
        
    except Exception as e:
        st.error(f"Error in fetch_data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None


def calculate_returns(prices):
    """Calculate log returns"""
    returns = np.log(prices / prices.shift(1)).dropna()
    # Ensure it's a Series, not DataFrame
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    return returns


def calculate_capm_beta(stock_returns, market_returns, with_alpha=True):
    """Calculate CAPM beta using statsmodels OLS regression - provides all statistics naturally"""
    # Ensure both are Series and properly aligned
    if isinstance(stock_returns, pd.Series) and isinstance(market_returns, pd.Series):
        # Use concat to align indices
        data = pd.concat([stock_returns.rename('stock'), market_returns.rename('market')], axis=1)
    else:
        # Fallback if not Series
        data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        })
    
    data = data.dropna()
    
    if len(data) < 2:
        return None
    
    # Prepare X and y for OLS
    X = data['market']
    y = data['stock']
    
    # Add constant (intercept) if with_alpha=True
    if with_alpha:
        X = sm.add_constant(X)
    
    # Fit OLS model - this automatically calculates ALL statistics!
    model = sm.OLS(y, X).fit()
    
    # Extract parameters based on model specification
    if with_alpha:
        alpha = model.params['const']
        beta = model.params['market']
        se_alpha = model.bse['const']
        se_beta = model.bse['market']
        t_alpha = model.tvalues['const']
        t_beta = model.tvalues['market']
        p_value_alpha = model.pvalues['const']
        p_value_beta = model.pvalues['market']
        # 95% confidence intervals
        conf_int = model.conf_int(alpha=0.05)
        alpha_ci_lower = conf_int.loc['const', 0]
        alpha_ci_upper = conf_int.loc['const', 1]
        beta_ci_lower = conf_int.loc['market', 0]
        beta_ci_upper = conf_int.loc['market', 1]
    else:
        alpha = 0
        beta = model.params['market']
        se_alpha = 0
        se_beta = model.bse['market']
        t_alpha = 0
        t_beta = model.tvalues['market']
        p_value_alpha = 1.0
        p_value_beta = model.pvalues['market']
        # 95% confidence intervals
        conf_int = model.conf_int(alpha=0.05)
        alpha_ci_lower = alpha_ci_upper = 0
        beta_ci_lower = conf_int.loc['market', 0]
        beta_ci_upper = conf_int.loc['market', 1]
    
    return {
        'beta': beta,
        'alpha': alpha,
        'r_squared': model.rsquared,
        'model': model,  # Full statsmodels results object
        'data': data,
        # Statistical inference (all from OLS output)
        'se_beta': se_beta,
        'se_alpha': se_alpha,
        't_beta': t_beta,
        't_alpha': t_alpha,
        'p_value_beta': p_value_beta,
        'p_value_alpha': p_value_alpha,
        'beta_ci_lower': beta_ci_lower,
        'beta_ci_upper': beta_ci_upper,
        'alpha_ci_lower': alpha_ci_lower,
        'alpha_ci_upper': alpha_ci_upper,
        'f_statistic': model.fvalue,
        'p_value_f': model.f_pvalue,
        'n_obs': int(model.nobs)
    }


def calculate_rolling_beta(stock_returns, market_returns, window):
    """Calculate rolling beta"""
    data = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    if len(data) < window:
        return None
    
    rolling_betas = []
    dates = []
    
    for i in range(window, len(data) + 1):
        window_data = data.iloc[i-window:i]
        X = window_data['market'].values.reshape(-1, 1)
        y = window_data['stock'].values
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        rolling_betas.append(model.coef_[0])
        dates.append(window_data.index[-1])
    
    return pd.Series(rolling_betas, index=dates)


def plot_capm(result, color, with_alpha=True):
    """Create CAPM regression plot for PNG export - uses chosen color for stock data, black for axes"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.set_facecolor('white')
    
    data = result['data']
    X = data['market'].values
    y = data['stock'].values
    
    # Stock data points in chosen color
    ax.scatter(X, y, alpha=0.6, color=color, s=30, edgecolors='black', linewidths=0.5)
    
    # Generate prediction line
    X_line = np.linspace(X.min(), X.max(), 100)
    if with_alpha:
        X_pred = sm.add_constant(X_line)
        y_line = result['model'].predict(X_pred)
    else:
        y_line = result['model'].predict(X_line)
    
    # Regression line in black
    ax.plot(X_line, y_line, color='black', linewidth=2, label='Regression Line', linestyle='--')
    
    title = f"CAPM Regression {'with' if with_alpha else 'without'} Alpha\n"
    title += f"Beta = {result['beta']:.4f}, "
    if with_alpha:
        title += f"Alpha = {result['alpha']:.6f}, "
    title += f"R² = {result['r_squared']:.4f}"
    
    # All text in black
    ax.set_title(title, fontsize=14, fontweight='bold', color='black')
    ax.set_xlabel('Market Returns', fontsize=12, color='black')
    ax.set_ylabel('Stock Returns', fontsize=12, color='black')
    ax.legend(fontsize=10, facecolor='white', edgecolor='black')
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    
    # Black axes and ticks
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(colors='black', which='both')
    
    plt.tight_layout()
    
    return fig


def plot_rolling(rolling_betas, color, window):
    """Create rolling beta plot for PNG export - uses chosen color for beta line, black for axes"""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    
    # Beta line in chosen color
    ax.plot(rolling_betas.index, rolling_betas.values, color=color, linewidth=2.5, label='Rolling Beta')
    
    # Mean line in black
    ax.axhline(y=rolling_betas.mean(), color='black', linestyle='--', linewidth=1.5,
               label=f'Mean Beta = {rolling_betas.mean():.4f}')
    
    # All text in black
    ax.set_title(f'Rolling Beta ({window}-period window)', fontsize=14, fontweight='bold', color='black')
    ax.set_xlabel('Date', fontsize=12, color='black')
    ax.set_ylabel('Beta', fontsize=12, color='black')
    ax.legend(fontsize=10, facecolor='white', edgecolor='black')
    ax.grid(True, alpha=0.3, color='gray', linestyle=':')
    
    # Black axes and ticks
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.tick_params(colors='black', which='both')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def save_figure(fig):
    """Save figure to bytes"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf


def plot_capm_plotly(result, with_alpha=True):
    """Create interactive Plotly CAPM regression plot"""
    data = result['data']
    X = data['market'].values
    y = data['stock'].values
    
    # Create figure
    fig = go.Figure()
    
    # Scatter plot with lavender/purple theme
    fig.add_trace(go.Scatter(
        x=X,
        y=y,
        mode='markers',
        name='Returns',
        marker=dict(
            color='rgba(199, 146, 234, 0.6)',  # Lavender
            size=8,
            line=dict(color='rgba(179, 146, 240, 1)', width=1)
        )
    ))
    
    # Regression line
    X_line = np.linspace(X.min(), X.max(), 100)
    if with_alpha:
        X_pred = sm.add_constant(X_line)
        y_line = result['model'].predict(X_pred)
    else:
        y_line = result['model'].predict(X_line)
    
    fig.add_trace(go.Scatter(
        x=X_line,
        y=y_line,
        mode='lines',
        name='Regression Line',
        line=dict(color='#f07178', width=3)  # Pink accent
    ))
    
    # Layout (no title - header shown above)
    fig.update_layout(
        title="",
        xaxis_title='Market Returns',
        yaxis_title='Stock Returns',
        template='plotly_dark',
        hovermode='closest',
        showlegend=True,
        height=500,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def plot_rolling_plotly(rolling_betas, window):
    """Create interactive Plotly rolling beta chart with lavender theme"""
    fig = go.Figure()
    
    # Rolling beta line with lavender gradient
    fig.add_trace(go.Scatter(
        x=rolling_betas.index,
        y=rolling_betas.values,
        mode='lines',
        name=f'{window}-period Rolling Beta',
        line=dict(color='#c792ea', width=2.5),  # Lavender
        fill='tozeroy',
        fillcolor='rgba(199, 146, 234, 0.1)'
    ))
    
    # Mean line
    mean_beta = rolling_betas.mean()
    fig.add_trace(go.Scatter(
        x=rolling_betas.index,
        y=[mean_beta] * len(rolling_betas),
        mode='lines',
        name=f'Mean Beta = {mean_beta:.4f}',
        line=dict(color='#89ddff', width=2, dash='dash')  # Cyan accent
    ))
    
    # Layout with lavender theme
    fig.update_layout(
        title=f'Rolling Beta ({window}-period window)',
        xaxis_title='Date',
        yaxis_title='Beta',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        height=500,
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color='#d4d4d4'),
        xaxis=dict(
            gridcolor='rgba(199, 146, 234, 0.1)',
            zerolinecolor='rgba(199, 146, 234, 0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(199, 146, 234, 0.1)',
            zerolinecolor='rgba(199, 146, 234, 0.3)'
        )
    )
    
    return fig


def plot_normalized_prices(stock_prices, market_prices, stock_name, market_name):
    """Create normalized price comparison chart with lavender theme"""
    # Align dates first - only use overlapping dates
    common_dates = stock_prices.index.intersection(market_prices.index)
    stock_aligned = stock_prices.loc[common_dates]
    market_aligned = market_prices.loc[common_dates]
    
    # Normalize to 100 - handle both Series and DataFrame
    stock_norm = (stock_aligned / stock_aligned.iloc[0]) * 100
    market_norm = (market_aligned / market_aligned.iloc[0]) * 100
    
    # Ensure 1D arrays
    if isinstance(stock_norm, pd.DataFrame):
        stock_norm = stock_norm.squeeze()
    if isinstance(market_norm, pd.DataFrame):
        market_norm = market_norm.squeeze()
    
    fig = go.Figure()
    
    # Stock line with lavender
    fig.add_trace(go.Scatter(
        x=stock_norm.index,
        y=stock_norm.values,
        mode='lines',
        name=stock_name,
        line=dict(color='#c792ea', width=2.5)  # Lavender
    ))
    
    # Market line with cyan
    fig.add_trace(go.Scatter(
        x=market_norm.index,
        y=market_norm.values,
        mode='lines',
        name=market_name,
        line=dict(color='#89ddff', width=2.5)  # Cyan
    ))
    
    # Base line at 100
    fig.add_hline(y=100, line_dash='dot', line_color='rgba(255, 203, 107, 0.5)', opacity=0.5)
    
    # Layout with lavender theme
    fig.update_layout(
        title='Normalized Price Performance (Base = 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True,
        height=400,
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(30, 30, 30, 0.8)',
        font=dict(color='#d4d4d4'),
        xaxis=dict(
            gridcolor='rgba(199, 146, 234, 0.1)',
            zerolinecolor='rgba(199, 146, 234, 0.3)'
        ),
        yaxis=dict(
            gridcolor='rgba(199, 146, 234, 0.1)',
            zerolinecolor='rgba(199, 146, 234, 0.3)'
        )
    )
    
    return fig


def create_csv_download(data, filename):
    """Create CSV download button data"""
    csv = data.to_csv(index=True)
    return csv.encode('utf-8')


def load_nifty_750_stocks():
    """Load NSE stocks from Nifty 750 CSV file"""
    try:
        df = pd.read_csv('Nifty 750.csv')
        # Create dictionary with "Symbol - Company Name" as key and "Symbol.NS" as value
        stocks = {}
        for _, row in df.iterrows():
            symbol = row['Symbol']
            company = row['Company Name']
            display_name = f"{symbol}.NS - {company}"
            ticker = f"{symbol}.NS"
            stocks[display_name] = ticker
        return stocks
    except Exception as e:
        st.error(f"Error loading Nifty 750 stocks: {str(e)}")
        # Return fallback list
        return {
            "RELIANCE.NS - Reliance Industries": "RELIANCE.NS",
            "TCS.NS - Tata Consultancy Services": "TCS.NS",
            "INFY.NS - Infosys": "INFY.NS"
        }


def main():
    st.title("Beta Analysis Dashboard")
    st.caption("Calculate and visualize stock beta using CAPM or Rolling Beta methods")
    
    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    
    # Market indices
    market_indices = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "NSE Nifty 50": "^NSEI",
        "BSE Sensex": "^BSESN",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI"
    }
    
    # Load NSE stocks from CSV
    nse_stocks = load_nifty_750_stocks()
    
    # US stocks
    us_stocks = {
        "AAPL - Apple": "AAPL",
        "MSFT - Microsoft": "MSFT",
        "GOOGL - Alphabet": "GOOGL",
        "AMZN - Amazon": "AMZN",
        "TSLA - Tesla": "TSLA",
        "META - Meta": "META",
        "NVDA - NVIDIA": "NVDA",
        "JPM - JPMorgan Chase": "JPM",
        "V - Visa": "V",
        "WMT - Walmart": "WMT"
    }
    
    # Market index selection
    market_display = st.sidebar.selectbox("Market Index", list(market_indices.keys()), index=4)
    market_ticker = market_indices[market_display]
    
    # Company selection based on index
    if "NSE" in market_display or "BSE" in market_display:
        stock_list = nse_stocks
    else:
        stock_list = us_stocks
    
    company_display = st.sidebar.selectbox("Company", list(stock_list.keys()))
    company_ticker = stock_list[company_display]
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            min_value=datetime(2000, 1, 1),
            max_value=datetime.now()
        )
    
    # Frequency
    frequency = st.sidebar.selectbox("Data Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    
    # Beta method
    beta_method = st.sidebar.radio("Beta Method", ["CAPM Beta", "Rolling Beta"])
    
    # Rolling window (conditional)
    rolling_period = None
    if beta_method == "Rolling Beta":
        rolling_period = st.sidebar.number_input(
            "Rolling Window Size",
            min_value=10,
            max_value=500,
            value=60,
            step=5
        )
    
    # Risk-free rate
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.5,
        step=0.1
    ) / 100
    
    # Color picker
    chart_color = st.sidebar.color_picker("Chart Color", value="#0083B8")
    
    # Calculate button
    if st.sidebar.button("Calculate Beta", type="primary"):
        if start_date >= end_date:
            st.error("Start date must be before end date!")
            st.stop()
        
        with st.spinner("Fetching data and calculating..."):
            # Fetch data
            stock_prices, market_prices = fetch_data(company_ticker, market_ticker, start_date, frequency)
            
            if stock_prices is None or market_prices is None or len(stock_prices) == 0 or len(market_prices) == 0:
                st.error(f"Could not fetch data for {company_ticker}")
                st.info(f"""
                **Troubleshooting:**
                - Ticker: {company_ticker}
                - Date range: {start_date} to {end_date}
                - Try a different date range or stock
                """)
                st.stop()
            
            # Calculate returns
            stock_returns = calculate_returns(stock_prices)
            market_returns = calculate_returns(market_prices)
            
            # Adjust for risk-free rate
            periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}
            rf_period = risk_free_rate / periods_per_year[frequency]
            
            stock_excess = stock_returns - rf_period
            market_excess = market_returns - rf_period
            
            # Store in session state to persist across reruns
            st.session_state['data_calculated'] = True
            st.session_state['stock_prices'] = stock_prices
            st.session_state['market_prices'] = market_prices
            st.session_state['stock_returns'] = stock_returns
            st.session_state['market_returns'] = market_returns
            st.session_state['stock_excess'] = stock_excess
            st.session_state['market_excess'] = market_excess
            st.session_state['company_ticker'] = company_ticker
            st.session_state['market_ticker'] = market_ticker
            st.session_state['beta_method'] = beta_method
            st.session_state['rolling_period'] = rolling_period
            st.session_state['chart_color'] = chart_color
    
    # Display results if data has been calculated
    if st.session_state.get('data_calculated', False):
        # Retrieve from session state
        stock_prices = st.session_state['stock_prices']
        market_prices = st.session_state['market_prices']
        stock_returns = st.session_state['stock_returns']
        market_returns = st.session_state['market_returns']
        stock_excess = st.session_state['stock_excess']
        market_excess = st.session_state['market_excess']
        company_ticker = st.session_state['company_ticker']
        market_ticker = st.session_state['market_ticker']
        beta_method = st.session_state['beta_method']
        rolling_period = st.session_state['rolling_period']
        chart_color = st.session_state['chart_color']
        
        # Display data availability info
        stock_start = stock_prices.index[0].strftime('%Y-%m-%d')
        stock_end = stock_prices.index[-1].strftime('%Y-%m-%d')
        market_start = market_prices.index[0].strftime('%Y-%m-%d')
        market_end = market_prices.index[-1].strftime('%Y-%m-%d')
        
        # Convert dates for comparison
        stock_start_date = stock_prices.index[0].date()
        market_start_date = market_prices.index[0].date()
        
        if stock_start_date > start_date or market_start_date > start_date:
            st.info(f"""
            📅 **Data Availability:**
            - **Requested:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
            - **{company_ticker}:** {stock_start} to {stock_end}
            - **{market_ticker}:** {market_start} to {market_end}
            
            ℹ️ Stock may have been listed after your requested start date.
            """)
        
        # Display info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stock", company_ticker)
        with col2:
            st.metric("Market Index", market_ticker)
        with col3:
            st.metric("Data Points", len(stock_returns))
        
        # CAPM Beta
        if beta_method == "CAPM Beta":
            st.markdown("#### CAPM Beta Analysis")
            
            result_with = calculate_capm_beta(stock_excess, market_excess, with_alpha=True)
            result_without = calculate_capm_beta(stock_excess, market_excess, with_alpha=False)
            
            if result_with is None or result_without is None:
                st.error("Failed to calculate beta")
                st.stop()
            
            # Normalized Price Chart (before regression charts)
            st.markdown("**Normalized Price Performance**")
            fig_norm = plot_normalized_prices(stock_prices, market_prices, company_ticker, market_ticker)
            st.plotly_chart(fig_norm, use_container_width=True)
            
            # Regression statistics and charts in aligned columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### CAPM Regression with Alpha")
                # Enhanced stats card with 2-column grid layout
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-grid">
                        <div class="stat-row">
                            <span class="stat-label">Beta</span>
                            <span class="stat-value">{result_with['beta']:.4f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Alpha</span>
                            <span class="stat-value">{result_with['alpha']:.6f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Beta 95% CI</span>
                            <span class="stat-value">[{result_with['beta_ci_lower']:.4f}, {result_with['beta_ci_upper']:.4f}]</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">R-Squared</span>
                            <span class="stat-value">{result_with['r_squared']:.4f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Beta p-value</span>
                            <span class="stat-value">{result_with['p_value_beta']:.4f}{'***' if result_with['p_value_beta'] < 0.01 else '**' if result_with['p_value_beta'] < 0.05 else '*' if result_with['p_value_beta'] < 0.1 else ''}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">F-statistic</span>
                            <span class="stat-value">{result_with['f_statistic']:.2f} (p={result_with['p_value_f']:.4f})</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Observations</span>
                            <span class="stat-value">{result_with['n_obs']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Interactive Plotly chart (no title)
                fig1_plotly = plot_capm_plotly(result_with, with_alpha=True)
                st.plotly_chart(fig1_plotly, use_container_width=True)
            
            with col2:
                st.markdown("#### CAPM Regression without Alpha")
                # Enhanced stats card with 2-column grid layout
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-grid">
                        <div class="stat-row">
                            <span class="stat-label">Beta</span>
                            <span class="stat-value">{result_without['beta']:.4f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">R-Squared</span>
                            <span class="stat-value">{result_without['r_squared']:.4f}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Beta 95% CI</span>
                            <span class="stat-value">[{result_without['beta_ci_lower']:.4f}, {result_without['beta_ci_upper']:.4f}]</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">F-statistic</span>
                            <span class="stat-value">{result_without['f_statistic']:.2f} (p={result_without['p_value_f']:.4f})</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Beta p-value</span>
                            <span class="stat-value">{result_without['p_value_beta']:.4f}{'***' if result_without['p_value_beta'] < 0.01 else '**' if result_without['p_value_beta'] < 0.05 else '*' if result_without['p_value_beta'] < 0.1 else ''}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Observations</span>
                            <span class="stat-value">{result_without['n_obs']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Interactive Plotly chart (no title)
                fig2_plotly = plot_capm_plotly(result_without, with_alpha=False)
                st.plotly_chart(fig2_plotly, use_container_width=True)
            
            # Download section at bottom
            st.markdown("**📥 Download Options**")
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                # Normalized prices CSV
                stock_norm = (stock_prices / stock_prices.iloc[0]) * 100
                market_norm = (market_prices / market_prices.iloc[0]) * 100
                if isinstance(stock_norm, pd.DataFrame):
                    stock_norm = stock_norm.squeeze()
                if isinstance(market_norm, pd.DataFrame):
                    market_norm = market_norm.squeeze()
                norm_data = pd.DataFrame({
                    'Date': stock_prices.index,
                    f'{company_ticker}_Normalized': stock_norm,
                    f'{market_ticker}_Normalized': market_norm
                })
                norm_csv = create_csv_download(norm_data, f"{company_ticker}_normalized_prices.csv")
                st.download_button("📥 Normalized Prices (CSV)", norm_csv, f"{company_ticker}_normalized_prices.csv", "text/csv", use_container_width=True, key="dl_capm_norm")
                
            with col_dl2:
                # With Alpha downloads
                st.markdown("**With Alpha**")
                fig1_mpl = plot_capm(result_with, chart_color, with_alpha=True)
                buf1 = save_figure(fig1_mpl)
                st.download_button("📥 Chart (PNG)", buf1, f"{company_ticker}_capm_with_alpha.png", "image/png", use_container_width=True, key="dl_capm_with_png")
                plt.close(fig1_mpl)
                
                reg_data_with = result_with['data'].copy()
                # For statsmodels with alpha, need to add constant
                X_pred_with = sm.add_constant(reg_data_with[['market']])
                reg_data_with['fitted'] = result_with['model'].predict(X_pred_with)
                reg_csv_with = create_csv_download(reg_data_with, f"{company_ticker}_regression_with_alpha.csv")
                st.download_button("📥 Data (CSV)", reg_csv_with, f"{company_ticker}_regression_with_alpha.csv", "text/csv", use_container_width=True, key="dl_capm_with_csv")
                
            with col_dl3:
                # Without Alpha downloads
                st.markdown("**Without Alpha**")
                fig2_mpl = plot_capm(result_without, chart_color, with_alpha=False)
                buf2 = save_figure(fig2_mpl)
                st.download_button("📥 Chart (PNG)", buf2, f"{company_ticker}_capm_without_alpha.png", "image/png", use_container_width=True, key="dl_capm_without_png")
                plt.close(fig2_mpl)
                
                reg_data_without = result_without['data'].copy()
                # For statsmodels without alpha, predict directly
                reg_data_without['fitted'] = result_without['model'].predict(reg_data_without[['market']])
                reg_csv_without = create_csv_download(reg_data_without, f"{company_ticker}_regression_without_alpha.csv")
                st.download_button("📥 Data (CSV)", reg_csv_without, f"{company_ticker}_regression_without_alpha.csv", "text/csv", use_container_width=True, key="dl_capm_without_csv")
            
            # Interpretation
            st.markdown("**Interpretation**")
            beta = result_with['beta']
            if beta > 1:
                st.info(f"Beta = {beta:.4f} → Stock is **more volatile** than the market")
            elif beta < 1 and beta > 0:
                st.info(f"Beta = {beta:.4f} → Stock is **less volatile** than the market")
            elif beta < 0:
                st.info(f"Beta = {beta:.4f} → Stock moves **inversely** to the market")
            else:
                st.info(f"Beta = {beta:.4f} → No correlation with the market")
        
        # Rolling Beta
        elif beta_method == "Rolling Beta":
            st.markdown("#### Rolling Beta Analysis")
            
            rolling_betas = calculate_rolling_beta(stock_excess, market_excess, rolling_period)
            
            if rolling_betas is None or len(rolling_betas) < 2:
                st.error(f"Insufficient data for rolling beta with window size {rolling_period}")
                st.stop()
            
            # Normalized Price Chart (before rolling beta chart)
            st.markdown("**Normalized Price Performance**")
            fig_norm = plot_normalized_prices(stock_prices, market_prices, company_ticker, market_ticker)
            st.plotly_chart(fig_norm, use_container_width=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Beta", f"{rolling_betas.iloc[-1]:.4f}")
            with col2:
                st.metric("Mean Beta", f"{rolling_betas.mean():.4f}")
            with col3:
                st.metric("Min Beta", f"{rolling_betas.min():.4f}")
            with col4:
                st.metric("Max Beta", f"{rolling_betas.max():.4f}")
            
            # Interactive Plotly chart
            st.markdown("**Historical Beta Over Time**")
            fig_plotly = plot_rolling_plotly(rolling_betas, rolling_period)
            st.plotly_chart(fig_plotly, use_container_width=True)
            
            # Download section
            st.markdown("**📥 Download Options**")
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                # Normalized prices CSV
                stock_norm = (stock_prices / stock_prices.iloc[0]) * 100
                market_norm = (market_prices / market_prices.iloc[0]) * 100
                if isinstance(stock_norm, pd.DataFrame):
                    stock_norm = stock_norm.squeeze()
                if isinstance(market_norm, pd.DataFrame):
                    market_norm = market_norm.squeeze()
                norm_data = pd.DataFrame({
                    'Date': stock_prices.index,
                    f'{company_ticker}_Normalized': stock_norm,
                    f'{market_ticker}_Normalized': market_norm
                })
                norm_csv = create_csv_download(norm_data, f"{company_ticker}_normalized_prices.csv")
                st.download_button("📥 Normalized Prices (CSV)", norm_csv, f"{company_ticker}_normalized_prices.csv", "text/csv", use_container_width=True, key="dl_rolling_norm")
                
            with col_dl2:
                # PNG download (matplotlib)
                fig_mpl = plot_rolling(rolling_betas, chart_color, rolling_period)
                buf = save_figure(fig_mpl)
                st.download_button("📥 Rolling Beta Chart (PNG)", buf, f"{company_ticker}_rolling_beta.png", "image/png", use_container_width=True, key="dl_rolling_png")
                plt.close(fig_mpl)
                
            with col_dl3:
                # CSV data download
                rolling_data = pd.DataFrame({
                    'Date': rolling_betas.index,
                    'Beta': rolling_betas.values
                })
                rolling_csv = create_csv_download(rolling_data, f"{company_ticker}_rolling_beta.csv")
                st.download_button("📥 Rolling Beta Data (CSV)", rolling_csv, f"{company_ticker}_rolling_beta.csv", "text/csv", use_container_width=True, key="dl_rolling_csv")
            
            # Statistics
            st.markdown("**Statistics**")
            stats_df = pd.DataFrame({
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Current'],
                'Value': [
                    f"{rolling_betas.mean():.4f}",
                    f"{rolling_betas.median():.4f}",
                    f"{rolling_betas.std():.4f}",
                    f"{rolling_betas.min():.4f}",
                    f"{rolling_betas.max():.4f}",
                    f"{rolling_betas.iloc[-1]:.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Calculate Beta' to begin")
        
        st.markdown("### 📖 About")
        st.markdown("""
        **CAPM Beta:**
        - Beta > 1: More volatile than market
        - Beta < 1: Less volatile than market
        - Beta < 0: Moves inversely to market
        
        **Rolling Beta:**
        - Shows how beta changes over time
        - Useful for trend analysis
        
        **Data Source:** Yahoo Finance
        """)


if __name__ == "__main__":
    main()
