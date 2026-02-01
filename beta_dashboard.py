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
import os

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


@st.cache_data(ttl=3600)
def fetch_data(stock_ticker, market_ticker, start_date, frequency='Daily'):
    """
    Fetch stock and market data using yfinance with listing date detection
    Returns Close prices for both tickers
    """
    try:
        # First, try to detect listing date
        listing_date = None
        try:
            stock_info = yf.Ticker(stock_ticker)
            full_history = stock_info.history(period="max", auto_adjust=False)
            if not full_history.empty:
                listing_date = full_history.index[0].date()
                
                # If requested start date is before listing, adjust it silently
                if start_date < listing_date:
                    start_date = listing_date
        except:
            pass  # If we can't detect listing date, proceed with user's date
        
        # Download separately to avoid yfinance issues with multiple tickers
        stock_data = yf.download(
            stock_ticker,
            start=start_date,
            auto_adjust=False,
            progress=False
        )
        
        market_data = yf.download(
            market_ticker,
            start=start_date,
            auto_adjust=False,
            progress=False
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
            return None, None, None
        
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
        
        return stock_prices, market_prices, listing_date
        
    except Exception as e:
        st.error(f"Error in fetch_data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None


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


def calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year):
    """
    Calculate annualized Sharpe ratio
    returns: Series of period returns
    risk_free_rate: Annual risk-free rate (as decimal)
    periods_per_year: Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)
    """
    # Calculate excess returns
    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period
    
    # Annualize mean and std
    mean_excess = excess_returns.mean() * periods_per_year
    std_excess = excess_returns.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    if std_excess == 0:
        return 0
    
    sharpe = mean_excess / std_excess
    return sharpe


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


def run_sector_analysis():
    """Sector-wise beta analysis"""
    st.header("📊 Sector Analysis")
    st.markdown("Analyze average beta and risk across different sectors")
    
    # Load NSE stocks
    try:
        df = pd.read_csv("Nifty 750.csv")
    except:
        st.error("Could not load Nifty 750.csv file")
        return
    
    # Sidebar settings for sector analysis
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sector Analysis Settings")
    
    market_ticker = st.sidebar.selectbox(
        "Market Index",
        ["^NSEI", "^BSESN"],
        format_func=lambda x: "NSE Nifty 50" if x == "^NSEI" else "BSE Sensex"
    )
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365*2),
        min_value=datetime(2000, 1, 1),
        max_value=datetime.now(),
        key="sector_start_date"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        min_value=datetime(2000, 1, 1),
        max_value=datetime.now(),
        key="sector_end_date"
    )
    
    min_stocks = st.sidebar.slider(
        "Min stocks per sector",
        min_value=1,
        max_value=10,
        value=3,
        help="Only show sectors with at least this many stocks"
    )
    
    if st.sidebar.button("🔍 Analyze Sectors", type="primary"):
        with st.spinner("Calculating sector betas..."):
            # Calculate beta for each stock
            sector_betas = {}
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                progress_bar.progress((idx + 1) / len(df))
                
                ticker = row['Symbol'] + '.NS'
                sector = row['Industry']
                
                try:
                    # Fetch data
                    stock_prices, market_prices = fetch_data(ticker, market_ticker, start_date.strftime('%Y-%m-%d'), 'Daily')
                    
                    if stock_prices is not None and len(stock_prices) > 30:
                        # Calculate returns
                        stock_returns = calculate_returns(stock_prices)
                        market_returns = calculate_returns(market_prices)
                        
                        # Calculate beta
                        result = calculate_capm_beta(stock_returns, market_returns, with_alpha=False)
                        
                        if sector not in sector_betas:
                            sector_betas[sector] = []
                        sector_betas[sector].append({
                            'company': row['Company Name'],
                            'symbol': ticker,
                            'beta': result['beta']
                        })
                except:
                    continue
            
            progress_bar.empty()
            
            # Calculate sector statistics
            sector_stats = []
            for sector, stocks in sector_betas.items():
                if len(stocks) >= min_stocks:
                    betas = [s['beta'] for s in stocks]
                    sector_stats.append({
                        'Sector': sector,
                        'Avg Beta': np.mean(betas),
                        'Median Beta': np.median(betas),
                        'Std Dev': np.std(betas),
                        'Min Beta': np.min(betas),
                        'Max Beta': np.max(betas),
                        'Stock Count': len(stocks)
                    })
            
            sector_df = pd.DataFrame(sector_stats).sort_values('Avg Beta', ascending=False)
            
            if len(sector_df) == 0:
                st.warning("No sector data available. Try reducing minimum stocks per sector.")
                return
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Sector Beta Heatmap")
                
                # Create heatmap
                fig = go.Figure(data=go.Bar(
                    x=sector_df['Sector'],
                    y=sector_df['Avg Beta'],
                    marker=dict(
                        color=sector_df['Avg Beta'],
                        colorscale='RdYlGn_r',  # Red (high) to Green (low)
                        showscale=True,
                        colorbar=dict(title="Beta")
                    ),
                    text=sector_df['Avg Beta'].round(3),
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Avg Beta: %{y:.3f}<br>Stocks: %{customdata}<extra></extra>',
                    customdata=sector_df['Stock Count']
                ))
                
                fig.update_layout(
                    title="Average Beta by Sector",
                    xaxis_title="Sector",
                    yaxis_title="Average Beta",
                    xaxis_tickangle=-45,
                    height=500,
                    template="plotly_dark",
                    plot_bgcolor='#1e1e1e',
                    paper_bgcolor='#1e1e1e',
                    font=dict(color='#d4d4d4')
                )
                
                # Add horizontal line at beta = 1
                fig.add_hline(y=1.0, line_dash="dash", line_color="#c792ea", 
                             annotation_text="Market Beta = 1.0", annotation_position="right")
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🏆 Top Risk Sectors")
                top_risk = sector_df.nlargest(5, 'Avg Beta')[['Sector', 'Avg Beta', 'Stock Count']]
                st.dataframe(top_risk, hide_index=True, use_container_width=True)
                
                st.subheader("🛡️ Low Risk Sectors")
                low_risk = sector_df.nsmallest(5, 'Avg Beta')[['Sector', 'Avg Beta', 'Stock Count']]
                st.dataframe(low_risk, hide_index=True, use_container_width=True)
            
            # Full sector statistics table
            st.subheader("📋 Detailed Sector Statistics")
            st.dataframe(
                sector_df.style.background_gradient(subset=['Avg Beta'], cmap='RdYlGn_r'),
                use_container_width=True,
                hide_index=True
            )
            
            # Sector distribution chart
            st.subheader("📈 Sector Beta Distribution")
            
            fig2 = go.Figure()
            
            for sector in sector_df['Sector'].head(10):  # Top 10 sectors
                sector_stocks = sector_betas[sector]
                betas = [s['beta'] for s in sector_stocks]
                
                fig2.add_trace(go.Box(
                    y=betas,
                    name=sector[:20],  # Truncate long names
                    boxmean='sd'
                ))
            
            fig2.update_layout(
                title="Beta Distribution Across Top 10 Sectors",
                yaxis_title="Beta",
                xaxis_tickangle=-45,
                height=500,
                template="plotly_dark",
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#1e1e1e',
                font=dict(color='#d4d4d4'),
                showlegend=False
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Download button
            csv = sector_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sector Analysis",
                data=csv,
                file_name=f"sector_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def main():
    st.title("Beta Analysis Dashboard")
    st.caption("Calculate and visualize stock beta using CAPM or Rolling Beta methods")
    
    # Create tabs for different features
    tab1, tab2 = st.tabs(["📈 Beta Analysis", "🏭 Sector Analysis"])
    
    with tab1:
        run_beta_analysis()
    
    with tab2:
        run_sector_analysis()


def run_beta_analysis():
    """Main beta analysis feature"""
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
    
    # Initialize session state for additional indices
    if 'additional_indices' not in st.session_state:
        st.session_state.additional_indices = []
    
    # Display additional index dropdowns with remove buttons
    for idx, index_name in enumerate(st.session_state.additional_indices):
        all_selected = [market_display] + [st.session_state.additional_indices[i] for i in range(len(st.session_state.additional_indices)) if i != idx]
        available_for_this = [index_name] + [ind for ind in market_indices.keys() if ind not in all_selected]
        
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            new_selection = st.selectbox(
                f"Additional Index {idx + 1}",
                available_for_this,
                index=0,
                key=f"additional_idx_{idx}"
            )
            if new_selection != index_name:
                st.session_state.additional_indices[idx] = new_selection
                st.rerun()
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Align with selectbox
            if st.button("✖", key=f"remove_idx_{idx}", use_container_width=True):
                st.session_state.additional_indices.pop(idx)
                st.rerun()
    
    # Add new index button
    if len(st.session_state.additional_indices) < 2:  # Limit to 2 additional (3 total)
        all_selected = [market_display] + st.session_state.additional_indices
        available_indices = [idx for idx in market_indices.keys() if idx not in all_selected]
        
        if available_indices:
            if st.sidebar.button("➕ Add Index", use_container_width=True, key="add_index_btn"):
                st.session_state.additional_indices.append(available_indices[0])
                st.rerun()
    
    # Company selection based on first index
    if "NSE" in market_display or "BSE" in market_display:
        stock_list = nse_stocks
    else:
        stock_list = us_stocks
    
    # Ticker selection mode
    ticker_mode = st.sidebar.radio("Ticker Selection", ["📋 Select from List", "✏️ Enter Manually"], horizontal=True)
    
    if ticker_mode == "📋 Select from List":
        company_display = st.sidebar.selectbox("Company", list(stock_list.keys()), key="company_dropdown")
        company_ticker = stock_list[company_display]
    else:
        company_ticker = st.sidebar.text_input(
            "Enter Ticker Symbol",
            value="RELIANCE.NS",
            help="Enter any valid ticker (e.g., RELIANCE.NS for NSE, AAPL for US)",
            key="manual_ticker"
        ).upper().strip()
    
    # Date range with preset buttons
    st.sidebar.markdown("**Date Range**")
    
    # Initialize date session state if not exists
    today = datetime.now().date()
    if 'date_start' not in st.session_state:
        st.session_state['date_start'] = today - timedelta(days=365*2)
    if 'date_end' not in st.session_state:
        st.session_state['date_end'] = today
    
    # Preset buttons - stacked in 2 rows for better fit
    preset_row1_col1, preset_row1_col2, preset_row1_col3 = st.sidebar.columns(3)
    preset_row2_col1, preset_row2_col2 = st.sidebar.columns(2)
    
    with preset_row1_col1:
        if st.button("1Y", use_container_width=True, key="preset_1y"):
            st.session_state['date_start'] = today - timedelta(days=365)
            st.session_state['date_end'] = today
            st.rerun()
    with preset_row1_col2:
        if st.button("3Y", use_container_width=True, key="preset_3y"):
            st.session_state['date_start'] = today - timedelta(days=365*3)
            st.session_state['date_end'] = today
            st.rerun()
    with preset_row1_col3:
        if st.button("5Y", use_container_width=True, key="preset_5y"):
            st.session_state['date_start'] = today - timedelta(days=365*5)
            st.session_state['date_end'] = today
            st.rerun()
    with preset_row2_col1:
        if st.button("YTD", use_container_width=True, key="preset_ytd"):
            st.session_state['date_start'] = datetime(today.year, 1, 1).date()
            st.session_state['date_end'] = today
            st.rerun()
    with preset_row2_col2:
        if st.button("Max", use_container_width=True, key="preset_max"):
            st.session_state['date_start'] = datetime(2000, 1, 1).date()
            st.session_state['date_end'] = today
            st.rerun()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state['date_start'],
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime.now().date()
        )
        if start_date != st.session_state['date_start']:
            st.session_state['date_start'] = start_date
    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state['date_end'],
            min_value=datetime(2000, 1, 1).date(),
            max_value=datetime.now().date()
        )
        if end_date != st.session_state['date_end']:
            st.session_state['date_end'] = end_date
    
    # Store in session state
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    
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
            # Fetch stock data once
            stock_prices, _, _ = fetch_data(company_ticker, market_ticker, start_date, frequency)
            
            if stock_prices is None or len(stock_prices) == 0:
                st.error(f"Could not fetch data for {company_ticker}")
                st.info(f"""
                **Troubleshooting:**
                - Ticker: {company_ticker}
                - Date range: {start_date} to {end_date}
                - Try a different date range or stock
                """)
                st.stop()
            
            # Fetch data for primary and additional indices
            all_selected_indices = [market_display] + st.session_state.get('additional_indices', [])
            all_index_data = {}
            listing_date = None
            for index_name in all_selected_indices:
                index_ticker = market_indices[index_name]
                fetched_stock, market_prices, stock_listing = fetch_data(company_ticker, index_ticker, start_date, frequency)
                if market_prices is not None and len(market_prices) > 0:
                    if listing_date is None:  # Store listing date from first fetch
                        stock_prices = fetched_stock
                        listing_date = stock_listing
                    all_index_data[index_name] = {
                        'ticker': index_ticker,
                        'prices': market_prices
                    }
            
            if not all_index_data:
                st.error("Could not fetch data for any selected index")
                st.stop()
            
            # Calculate returns
            stock_returns = calculate_returns(stock_prices)
            
            # Adjust for risk-free rate
            periods_per_year = {"Daily": 252, "Weekly": 52, "Monthly": 12}
            rf_period = risk_free_rate / periods_per_year[frequency]
            
            stock_excess = stock_returns - rf_period
            
            # Store in session state
            st.session_state['data_calculated'] = True
            st.session_state['stock_prices'] = stock_prices
            st.session_state['stock_returns'] = stock_returns
            st.session_state['stock_excess'] = stock_excess
            st.session_state['company_ticker'] = company_ticker
            st.session_state['beta_method'] = beta_method
            st.session_state['rolling_period'] = rolling_period
            st.session_state['chart_color'] = chart_color
            st.session_state['all_index_data'] = all_index_data
            st.session_state['listing_date'] = listing_date
    
    # Display results if data has been calculated
    if st.session_state.get('data_calculated', False):
        # Retrieve from session state
        stock_prices = st.session_state['stock_prices']
        stock_returns = st.session_state['stock_returns']
        stock_excess = st.session_state['stock_excess']
        company_ticker = st.session_state['company_ticker']
        beta_method = st.session_state['beta_method']
        rolling_period = st.session_state['rolling_period']
        chart_color = st.session_state['chart_color']
        all_index_data = st.session_state['all_index_data']
        listing_date = st.session_state.get('listing_date', None)
        
        # If multiple indices, show tabs
        if len(all_index_data) > 1:
            tabs = st.tabs([f"📊 {idx_name}" for idx_name in all_index_data.keys()])
            
            for tab, (index_name, index_data) in zip(tabs, all_index_data.items()):
                with tab:
                    display_beta_results(
                        stock_prices, stock_returns, stock_excess,
                        index_data['prices'], index_data['ticker'], index_name,
                        company_ticker, beta_method, rolling_period, chart_color,
                        risk_free_rate, frequency, start_date, end_date, listing_date
                    )
        else:
            # Single index - display directly
            index_name = list(all_index_data.keys())[0]
            index_data = all_index_data[index_name]
            display_beta_results(
                stock_prices, stock_returns, stock_excess,
                index_data['prices'], index_data['ticker'], index_name,
                company_ticker, beta_method, rolling_period, chart_color,
                risk_free_rate, frequency, start_date, end_date, listing_date
            )


def display_beta_results(stock_prices, stock_returns, stock_excess, market_prices, market_ticker, market_name,
                        company_ticker, beta_method, rolling_period, chart_color, risk_free_rate, frequency, start_date, end_date, listing_date=None):
    """
    Display beta analysis results for a given market index
    """
    # Calculate market returns and excess
    market_returns = calculate_returns(market_prices)
    periods_per_year_dict = {"Daily": 252, "Weekly": 52, "Monthly": 12}
    periods_per_year = periods_per_year_dict[frequency]
    rf_period = risk_free_rate / periods_per_year
    market_excess = market_returns - rf_period
    
    # Calculate Sharpe ratios
    stock_sharpe = calculate_sharpe_ratio(stock_returns, risk_free_rate, periods_per_year)
    market_sharpe = calculate_sharpe_ratio(market_returns, risk_free_rate, periods_per_year)
    
    # Display info including Sharpe ratios and listing date
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Stock", company_ticker)
    with col2:
        st.metric("Market Index", market_name)
    with col3:
        st.metric("Listing Date", listing_date.strftime('%Y-%m-%d') if listing_date else "N/A")
    with col4:
        st.metric("Data Points", len(stock_returns))
    with col5:
        st.metric("Stock Sharpe", f"{stock_sharpe:.3f}", help="Annualized Sharpe Ratio")
    with col6:
        st.metric("Market Sharpe", f"{market_sharpe:.3f}", help="Annualized Sharpe Ratio")
    
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
            
            # Ensure both series have the same index
            common_index = stock_norm.index.intersection(market_norm.index)
            stock_norm = stock_norm.loc[common_index]
            market_norm = market_norm.loc[common_index]
            
            norm_data = pd.DataFrame({
                'Date': common_index,
                f'{company_ticker}_Normalized': stock_norm.values,
                f'{market_ticker}_Normalized': market_norm.values
            })
            norm_csv = create_csv_download(norm_data, f"{company_ticker}_normalized_prices.csv")
            st.download_button("📥 Normalized Prices (CSV)", norm_csv, f"{company_ticker}_normalized_prices.csv", "text/csv", use_container_width=True, key=f"dl_capm_norm_{market_ticker}")
            
        with col_dl2:
            # With Alpha downloads
            st.markdown("**With Alpha**")
            fig1_mpl = plot_capm(result_with, chart_color, with_alpha=True)
            buf1 = save_figure(fig1_mpl)
            st.download_button("📥 Chart (PNG)", buf1, f"{company_ticker}_capm_with_alpha.png", "image/png", use_container_width=True, key=f"dl_capm_with_png_{market_ticker}")
            plt.close(fig1_mpl)
            
            reg_data_with = result_with['data'].copy()
            # For statsmodels with alpha, need to add constant
            X_pred_with = sm.add_constant(reg_data_with[['market']])
            reg_data_with['fitted'] = result_with['model'].predict(X_pred_with)
            reg_csv_with = create_csv_download(reg_data_with, f"{company_ticker}_regression_with_alpha.csv")
            st.download_button("📥 Data (CSV)", reg_csv_with, f"{company_ticker}_regression_with_alpha.csv", "text/csv", use_container_width=True, key=f"dl_capm_with_csv_{market_ticker}")
            
        with col_dl3:
            # Without Alpha downloads
            st.markdown("**Without Alpha**")
            fig2_mpl = plot_capm(result_without, chart_color, with_alpha=False)
            buf2 = save_figure(fig2_mpl)
            st.download_button("📥 Chart (PNG)", buf2, f"{company_ticker}_capm_without_alpha.png", "image/png", use_container_width=True, key=f"dl_capm_without_png_{market_ticker}")
            plt.close(fig2_mpl)
            
            reg_data_without = result_without['data'].copy()
            # For statsmodels without alpha, predict directly
            reg_data_without['fitted'] = result_without['model'].predict(reg_data_without[['market']])
            reg_csv_without = create_csv_download(reg_data_without, f"{company_ticker}_regression_without_alpha.csv")
            st.download_button("📥 Data (CSV)", reg_csv_without, f"{company_ticker}_regression_without_alpha.csv", "text/csv", use_container_width=True, key=f"dl_capm_without_csv_{market_ticker}")
        
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
            # Normalized prices CSV - ensure alignment
            stock_norm = (stock_prices / stock_prices.iloc[0]) * 100
            market_norm = (market_prices / market_prices.iloc[0]) * 100
            if isinstance(stock_norm, pd.DataFrame):
                stock_norm = stock_norm.squeeze()
            if isinstance(market_norm, pd.DataFrame):
                market_norm = market_norm.squeeze()
            
            # Align series to ensure same length
            stock_norm, market_norm = stock_norm.align(market_norm, join='inner')
            
            # Create DataFrame with aligned data
            norm_data = pd.DataFrame({
                'Date': stock_norm.index,
                f'{company_ticker}_Normalized': stock_norm.values,
                f'{market_ticker}_Normalized': market_norm.values
            })
            
            norm_csv = create_csv_download(norm_data, f"{company_ticker}_normalized_prices.csv")
            st.download_button("📥 Normalized Prices (CSV)", norm_csv, f"{company_ticker}_normalized_prices.csv", "text/csv", use_container_width=True, key=f"dl_rolling_norm_{market_ticker}")
            
        with col_dl2:
            # PNG download (matplotlib)
            fig_mpl = plot_rolling(rolling_betas, chart_color, rolling_period)
            buf = save_figure(fig_mpl)
            st.download_button("📥 Rolling Beta Chart (PNG)", buf, f"{company_ticker}_rolling_beta.png", "image/png", use_container_width=True, key=f"dl_rolling_png_{market_ticker}")
            plt.close(fig_mpl)
            
        with col_dl3:
            # CSV data download
            rolling_data = pd.DataFrame({
                'Date': rolling_betas.index,
                'Beta': rolling_betas.values
            })
            rolling_csv = create_csv_download(rolling_data, f"{company_ticker}_rolling_beta.csv")
            st.download_button("📥 Rolling Beta Data (CSV)", rolling_csv, f"{company_ticker}_rolling_beta.csv", "text/csv", use_container_width=True, key=f"dl_rolling_csv_{market_ticker}")
        
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
        
        # COE Backtesting Section
        st.markdown("---")
        st.markdown("#### 💰 Cost of Equity (COE) Backtesting")
        st.caption("Validate COE predictions against actual forward returns using historical Risk-Free rates")
        
        if st.button("🔬 Run COE Backtest", key="coe_backtest"):
            with st.spinner("Loading historical risk-free rates and calculating COE..."):
                    try:
                        # COE backtesting is always done annually for stability
                        # Load annual bond yield data (use monthly as closest to annual)
                        sheet_name = 'India 10-Y Bond Yield Monthly'
                        
                        # Load historical India 10Y G-Sec yields from Excel file
                        excel_path = os.path.join(os.path.dirname(__file__), 'India 10-Year Bond Yield Historical Data.xlsx')
                        bond_data = pd.read_excel(excel_path, sheet_name=sheet_name)
                        
                        # Convert Date column to datetime and set as index
                        bond_data['Date'] = pd.to_datetime(bond_data['Date'])
                        bond_data.set_index('Date', inplace=True)
                        bond_data.sort_index(inplace=True)
                        
                        # Use Price column (yield) and convert to decimal
                        bond_yields = bond_data['Price'] / 100
                        
                        # Get raw returns from session state and resample to annual
                        stock_returns_raw = st.session_state['stock_returns']
                        market_returns_raw = st.session_state['market_returns']
                        
                        # Resample to annual frequency (end of year)
                        stock_prices_raw = st.session_state['stock_prices']
                        market_prices_raw = st.session_state['market_prices']
                        
                        # Ensure prices are Series (not DataFrame)
                        if isinstance(stock_prices_raw, pd.DataFrame):
                            stock_prices_raw = stock_prices_raw.squeeze()
                        if isinstance(market_prices_raw, pd.DataFrame):
                            market_prices_raw = market_prices_raw.squeeze()
                        
                        # Calculate annual returns from prices
                        stock_annual = stock_prices_raw.resample('YE').last()
                        market_annual = market_prices_raw.resample('YE').last()
                        
                        stock_returns_annual = stock_annual.pct_change().dropna()
                        market_returns_annual = market_annual.pct_change().dropna()
                        
                        # Resample rolling beta to annual (take last value of each year)
                        rolling_betas_annual = rolling_betas.resample('YE').last().dropna()
                        
                        # Calculate annual ERP using market returns and annual historical Rf
                        market_excess_annual = []
                        
                        for date in market_returns_annual.index:
                            # Get historical annual Rf for this date
                            if date in bond_yields.index:
                                annual_rf = bond_yields.loc[date]
                            else:
                                available_dates = bond_yields.index[bond_yields.index <= date]
                                if len(available_dates) > 0:
                                    annual_rf = bond_yields.loc[available_dates[-1]]
                                else:
                                    annual_rf = 0.045  # Fallback to 4.5%
                            
                            # Calculate annual excess return (no conversion needed - both are annual)
                            market_excess_annual.append(market_returns_annual.loc[date] - annual_rf)
                        
                        # ERP = average of annual market excess returns
                        erp_annual = np.mean(market_excess_annual)
                        
                        # ERP = average of annual market excess returns
                        erp_annual = np.mean(market_excess_annual)
                        
                        # Calculate annual COE for each year using annual rolling beta and historical annual Rf
                        coe_values = []
                        actual_returns = []
                        dates = []
                        rf_used = []
                        
                        for i in range(len(rolling_betas_annual) - 1):  # -1 because we need next year's return
                            date = rolling_betas_annual.index[i]
                            next_date = rolling_betas_annual.index[i + 1]
                            
                            # Get closest historical annual Rf for this date
                            if date in bond_yields.index:
                                annual_rf = bond_yields.loc[date]
                            else:
                                # Forward fill to nearest available date
                                available_dates = bond_yields.index[bond_yields.index <= date]
                                if len(available_dates) > 0:
                                    annual_rf = bond_yields.loc[available_dates[-1]]
                                else:
                                    continue  # Skip if no historical data available
                            
                            # Calculate annual COE = annual Rf + (Beta × annual ERP)
                            coe = annual_rf + (rolling_betas_annual.iloc[i] * erp_annual)
                            
                            # Get actual annual return for next year using date-based lookup
                            if next_date in stock_returns_annual.index:
                                actual_return = stock_returns_annual.loc[next_date]
                            else:
                                continue  # Skip if next return not available
                            
                            coe_values.append(coe)
                            actual_returns.append(actual_return)
                            dates.append(date)
                            rf_used.append(annual_rf)
                        
                        # Convert to arrays for calculations
                        coe_array = np.array(coe_values)
                        actual_array = np.array(actual_returns)
                        
                        # Calculate error metrics
                        mae = np.mean(np.abs(coe_array - actual_array))
                        rmse = np.sqrt(np.mean((coe_array - actual_array) ** 2))
                        
                        # MAPE (Mean Absolute Percentage Error) - avoid division by zero
                        mape = np.mean(np.abs((actual_array - coe_array) / np.where(actual_array != 0, actual_array, 1))) * 100
                        
                        # SMAPE (Symmetric Mean Absolute Percentage Error)
                        smape = np.mean(2 * np.abs(coe_array - actual_array) / (np.abs(coe_array) + np.abs(actual_array) + 1e-10)) * 100
                        
                        # Hit rate (% correct direction)
                        correct_direction = np.sum((coe_array > 0) == (actual_array > 0))
                        hit_rate = (correct_direction / len(coe_array)) * 100
                        
                        # Display metrics in one row
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("MAE", f"{mae:.6f}", help="Mean Absolute Error")
                        with col2:
                            st.metric("RMSE", f"{rmse:.6f}", help="Root Mean Squared Error")
                        with col3:
                            st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
                        with col4:
                            st.metric("SMAPE", f"{smape:.2f}%", help="Symmetric Mean Absolute Percentage Error")
                        with col5:
                            st.metric("Hit Rate", f"{hit_rate:.1f}%", help="% correct direction predictions")
                        
                        # Create comparison chart with dual y-axes for better visualization
                        from plotly.subplots import make_subplots
                        
                        fig_coe = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add COE predictions on secondary y-axis (left)
                        fig_coe.add_trace(
                            go.Scatter(
                                x=dates,
                                y=coe_array * 100,
                                name='Predicted COE',
                                line=dict(color='#c792ea', width=2.5),
                                mode='lines'
                            ),
                            secondary_y=False
                        )
                        
                        # Add actual returns on primary y-axis (right)
                        fig_coe.add_trace(
                            go.Scatter(
                                x=dates,
                                y=actual_array * 100,
                                name='Actual Forward Returns',
                                line=dict(color='#89ddff', width=2),
                                mode='lines',
                                opacity=0.7
                            ),
                            secondary_y=True
                        )
                        
                        # Add zero lines
                        fig_coe.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, secondary_y=False)
                        fig_coe.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, secondary_y=True)
                        
                        # Update layout
                        fig_coe.update_xaxes(title_text="Date")
                        fig_coe.update_yaxes(title_text="Predicted Annual COE (%)", secondary_y=False, title_font=dict(color='#c792ea'))
                        fig_coe.update_yaxes(title_text="Actual Annual Returns (%)", secondary_y=True, title_font=dict(color='#89ddff'))
                        
                        fig_coe.update_layout(
                            title="Annual Cost of Equity vs Actual Annual Returns (Dual Axis)",
                            hovermode='x unified',
                            template="plotly_dark",
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            font=dict(color='#d4d4d4'),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        
                        st.plotly_chart(fig_coe, use_container_width=True)
                        
                        # Error distribution
                        st.markdown("**Error Distribution**")
                        errors = coe_array - actual_array
                        
                        fig_error = go.Figure()
                        fig_error.add_trace(go.Histogram(
                            x=errors * 100,
                            nbinsx=50,
                            marker_color='#c792ea',
                            name='Prediction Errors'
                        ))
                        
                        fig_error.update_layout(
                            title="Distribution of Prediction Errors",
                            xaxis_title="Error (%)",
                            yaxis_title="Frequency",
                            template="plotly_dark",
                            plot_bgcolor='#1e1e1e',
                            paper_bgcolor='#1e1e1e',
                            font=dict(color='#d4d4d4'),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_error, use_container_width=True)
                        
                        # Download COE data
                        coe_df = pd.DataFrame({
                            'Date': dates,
                            'Rolling_Beta': rolling_betas.iloc[:len(dates)].values,
                            'Historical_Rf_Annual': rf_used,
                            'Predicted_COE': coe_array,
                            'Actual_Forward_Return': actual_array,
                            'Error': errors,
                            'Abs_Error': np.abs(errors)
                        })
                        
                        coe_csv = coe_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download COE Backtest Data (CSV)",
                            data=coe_csv,
                            file_name=f"{company_ticker}_coe_backtest.csv",
                            mime="text/csv",
                            key=f"dl_coe_backtest_{market_ticker}"
                        )
                        
                    except Exception as e:
                        st.error(f"Error loading bond yield data: {str(e)}")
                        st.info("💡 Ensure 'India 10-Year Bond Yield Historical Data.xlsx' is in the same folder as the dashboard.")
    
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
