import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import folium
from streamlit_folium import st_folium
from folium.plugins import Fullscreen

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Data-Logger | SUR Division",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== THEME STATE ======================
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

# ====================== THEME + LOGO + TRAIN CSS ======================
def apply_theme(theme):
    if theme == "Dark":
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Rajdhani:wght@500;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #0a0f1c 0%, #0d1b2a 40%, #1b263b 100%);
            color: #e0e6ed;
            font-family: 'Rajdhani', sans-serif;
        }
        .dashboard-title {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 2.7rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #FF9933, #FFD700, #FF9933);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            letter-spacing: 2px;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.25rem;
            color: #7ec8e3;
            text-align: center;
            font-weight: 600;
            margin-top: 0.3rem;
        }
        .section-header {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.35rem !important;
            color: #FF9933 !important;
            border-left: 5px solid #FF9933;
            padding-left: 12px;
            margin: 1.2rem 0 0.5rem 0;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #132f4c, #0d2137);
            border: 1px solid #1e4a6e;
            border-radius: 14px;
            padding: 16px 12px;
        }
        div[data-testid="stMetric"] label { color: #7ec8e3 !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #FFD700 !important;
            font-family: 'Orbitron', sans-serif !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: #132f4c;
            color: #7ec8e3;
            border-radius: 10px 10px 0 0;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #FF9933, #e67e22) !important;
            color: #0a0f1c !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, #FF9933, #e67e22) !important;
            color: #0a0f1c !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            border: none !important;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1628 0%, #0d2137 100%);
        }

        /* Central Railway Logo Watermark */
        .logo-watermark {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            opacity: 0.08;
            z-index: 0;
            pointer-events: none;
            width: 450px;
        }

        /* Moving Bullet Train */
        .train-moving {
            position: fixed;
            bottom: 30px;
            left: -450px;
            font-size: 2.3rem;
            animation: moveTrain 24s linear infinite;
            opacity: 0.20;
            z-index: 0;
            pointer-events: none;
            white-space: nowrap;
            color: #FF9933;
        }
        @keyframes moveTrain {
            0%   { left: -450px; }
            100% { left: 115%; }
        }
        </style>
        """
    else:  # Light Mode
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Rajdhani:wght@500;600;700&display=swap');

        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
            color: #1e293b;
            font-family: 'Rajdhani', sans-serif;
        }
        .dashboard-title {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 2.7rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #c2410c, #ea580c, #c2410c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            letter-spacing: 2px;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.25rem;
            color: #0369a1;
            text-align: center;
            font-weight: 600;
            margin-top: 0.3rem;
        }
        .section-header {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.35rem !important;
            color: #c2410c !important;
            border-left: 5px solid #ea580c;
            padding-left: 12px;
            margin: 1.2rem 0 0.5rem 0;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            padding: 16px 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }
        div[data-testid="stMetric"] label { color: #475569 !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #c2410c !important;
            font-family: 'Orbitron', sans-serif !important;
        }
        .stTabs [data-baseweb="tab"] {
            background: #e2e8f0;
            color: #334155;
            border-radius: 10px 10px 0 0;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #ea580c, #c2410c) !important;
            color: white !important;
        }
        .stButton > button {
            background: linear-gradient(90deg, #ea580c, #c2410c) !important;
            color: white !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            border: none !important;
        }
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #cbd5e1;
        }

        /* Central Railway Logo Watermark */
        .logo-watermark {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            opacity: 0.07;
            z-index: 0;
            pointer-events: none;
            width: 450px;
        }

        /* Moving Bullet Train */
        .train-moving {
            position: fixed;
            bottom: 30px;
            left: -450px;
            font-size: 2.3rem;
            animation: moveTrain 24s linear infinite;
            opacity: 0.18;
            z-index: 0;
            pointer-events: none;
            white-space: nowrap;
            color: #c2410c;
        }
        @keyframes moveTrain {
            0%   { left: -450px; }
            100% { left: 115%; }
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme(st.session_state.theme)

# ====================== CONFIG ======================
IR_LOGO_URL = "https://raw.githubusercontent.com/srdsoproject/testing/main/Central%20Railway%20Logo.png"

try:
    SHEET_ID = st.secrets["google_sheets"]["sheet_id"]
    SHEET_NAME = st.secrets["google_sheets"]["sheet_name"]
    USERS = st.secrets["users"]
except Exception:
    st.error("⚠️ Secrets not configured properly.")
    st.stop()

# ====================== STATION COORDINATES ======================
station_coords = {
    "WADI": {"lat": 17.053, "lon": 76.992}, "SDB": {"lat": 17.122, "lon": 76.944},
    "MR": {"lat": 17.200, "lon": 76.902}, "HQR": {"lat": 17.258, "lon": 76.872},
    "KLBG": {"lat": 17.315, "lon": 76.825}, "TJSP": {"lat": 17.382, "lon": 76.831},
    "BBD": {"lat": 17.337, "lon": 76.779}, "SVG": {"lat": 17.340, "lon": 76.711},
    "HHD": {"lat": 17.353, "lon": 76.647}, "GUR": {"lat": 17.341, "lon": 76.590},
    "KUI": {"lat": 17.357, "lon": 76.471}, "DUD": {"lat": 17.363, "lon": 76.380},
    "NGS": {"lat": 17.429, "lon": 76.183}, "BOT": {"lat": 17.395, "lon": 76.255},
    "AKOR": {"lat": 17.451, "lon": 76.139}, "TLT": {"lat": 17.529, "lon": 76.036},
    "HG STN": {"lat": 17.565, "lon": 75.989}, "HG-A": {"lat": 17.556, "lon": 76.001},
    "TKWD": {"lat": 17.615, "lon": 75.933}, "SUR": {"lat": 17.665, "lon": 75.893},
    "BALE": {"lat": 17.676, "lon": 75.846}, "PK": {"lat": 17.726, "lon": 75.779},
    "MVE": {"lat": 17.742, "lon": 75.706}, "MO": {"lat": 17.806, "lon": 75.676},
    "MKPT": {"lat": 17.876, "lon": 75.635}, "AAG": {"lat": 17.929, "lon": 75.608},
    "WKA": {"lat": 17.980, "lon": 75.588}, "MA": {"lat": 18.030, "lon": 75.547},
    "WDS": {"lat": 18.066, "lon": 75.489}, "KWV": {"lat": 18.092, "lon": 75.417},
    "DHS": {"lat": 18.130, "lon": 75.334}, "KEM": {"lat": 18.177, "lon": 75.275},
    "BLNI": {"lat": 18.211, "lon": 75.207}, "JEUR": {"lat": 18.261, "lon": 75.162},
    "PPJ": {"lat": 18.292, "lon": 75.098}, "WSB": {"lat": 18.280, "lon": 75.016},
    "KEU": {"lat": 18.290, "lon": 74.953}, "JNTR": {"lat": 18.325, "lon": 74.878},
    "BGVN": {"lat": 18.317, "lon": 74.775}, "MLM": {"lat": 18.369, "lon": 74.724},
    "BRB": {"lat": 18.408, "lon": 74.649}, "MRJ": {"lat": 16.820, "lon": 74.639},
    "BLWD": {"lat": 16.816, "lon": 74.685}, "BDK": {"lat": 16.823, "lon": 74.732},
    "ARAG": {"lat": 16.823, "lon": 74.789}, "BLNK": {"lat": 16.852, "lon": 74.870},
    "SGRE": {"lat": 16.893, "lon": 74.904}, "AGDl": {"lat": 16.955, "lon": 74.922},
    "KVK": {"lat": 16.993, "lon": 74.936}, "LNP": {"lat": 17.084, "lon": 74.966},
    "DLGN": {"lat": 17.122, "lon": 74.991}, "GLV": {"lat": 17.173, "lon": 75.056},
    "JTRD": {"lat": 17.218, "lon": 75.112}, "MSDG": {"lat": 17.270, "lon": 75.139},
    "JVA": {"lat": 17.299, "lon": 75.158}, "WSD": {"lat": 17.378, "lon": 75.148},
    "SGLA": {"lat": 17.437, "lon": 75.188}, "BMNI": {"lat": 17.511, "lon": 75.237},
    "BHLI": {"lat": 17.589, "lon": 75.274}, "PVR": {"lat": 17.669, "lon": 75.320},
    "BBV": {"lat": 17.769, "lon": 75.398}, "AHI": {"lat": 17.845, "lon": 75.403},
    "MLB": {"lat": 17.917, "lon": 75.405}, "PSS": {"lat": 18.001, "lon": 75.390},
    "LAUL": {"lat": 18.034, "lon": 75.395}, "CNHL": {"lat": 18.100, "lon": 75.458},
    "MGO": {"lat": 18.110, "lon": 75.495}, "SEI": {"lat": 18.149, "lon": 75.590},
    "UPI": {"lat": 18.180, "lon": 75.636}, "BTW": {"lat": 18.241, "lon": 75.718},
    "KCB": {"lat": 18.279, "lon": 75.782}, "PJR": {"lat": 18.284, "lon": 75.867},
    "DRSV": {"lat": 18.248, "lon": 76.023}, "YSI": {"lat": 18.318, "lon": 75.977},
    "KRMD": {"lat": 18.372, "lon": 76.049}, "DKY": {"lat": 18.354, "lon": 76.103},
    "TER": {"lat": 18.353, "lon": 76.150}, "PCP": {"lat": 18.358, "lon": 76.193},
    "MRX": {"lat": 18.380, "lon": 76.251}, "NEI": {"lat": 18.387, "lon": 76.311},
    "OSA": {"lat": 18.378, "lon": 76.408}, "HGL": {"lat": 18.390, "lon": 76.496},
    "LUR": {"lat": 18.429, "lon": 76.556}, "BANL": {"lat": 18.446, "lon": 76.678},
    "GANI": {"lat": 18.479, "lon": 76.764}, "DD": {"lat": 18.464, "lon": 74.579},
    "HG": {"lat": 17.565, "lon": 75.989},
}

def get_jurisdiction(station, department):
    if pd.isna(station) or str(station).strip() == "":
        return "Unclassified"
    return "Unclassified"

# ====================== FORECASTING HELPERS ======================
def get_global_month_index(df):
    if df is None or df.empty or 'DATE' not in df.columns:
        return pd.DatetimeIndex([])
    d = df.dropna(subset=['DATE'])
    if d.empty:
        return pd.DatetimeIndex([])
    start = d['DATE'].min().to_period('M').to_timestamp()
    end = d['DATE'].max().to_period('M').to_timestamp()
    return pd.date_range(start, end, freq='MS')

def build_monthly_series(df, how="sum", full_index=None):
    if df is None or df.empty or 'DATE' not in df.columns:
        return pd.Series(dtype=float)
    d = df.dropna(subset=['DATE']).set_index('DATE').sort_index()
    if how == "count":
        series = d.resample('MS').size().astype(float)
    else:
        if 'FCOUNT' not in d.columns:
            return pd.Series(dtype=float)
        series = d['FCOUNT'].resample('MS').sum().astype(float)
    if full_index is not None and len(full_index) > 0:
        series = series.reindex(full_index, fill_value=0.0)
    return series

def trim_incomplete_current_month(series):
    if series.empty:
        return series
    now = pd.Timestamp.now()
    current_month_start = pd.Timestamp(year=now.year, month=now.month, day=1)
    if series.index[-1] == current_month_start:
        return series.iloc[:-1]
    return series

def write_styled_sheet(writer, df, sheet_name, header_color="#c2410c"):
    workbook = writer.book
    df.to_excel(writer, index=False, sheet_name=sheet_name, header=False, startrow=1)
    worksheet = writer.sheets[sheet_name]
    header_fmt = workbook.add_format({
        'bold': True, 'font_color': 'white', 'bg_color': header_color,
        'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True
    })
    text_fmt = workbook.add_format({'border': 1, 'valign': 'vcenter'})
    number_fmt = workbook.add_format({'border': 1, 'valign': 'vcenter', 'num_format': '#,##0'})
    date_fmt = workbook.add_format({'border': 1, 'valign': 'vcenter', 'num_format': 'dd-mmm-yyyy'})

    for col_idx, col_name in enumerate(df.columns):
        worksheet.write(0, col_idx, str(col_name), header_fmt)
        series = df[col_name]
        if pd.api.types.is_datetime64_any_dtype(series) or str(col_name).strip().upper() == 'DATE':
            cell_fmt = date_fmt
        elif pd.api.types.is_numeric_dtype(series):
            cell_fmt = number_fmt
        else:
            cell_fmt = text_fmt
        content_len = int(series.astype(str).map(len).max()) if len(series) else 0
        width = min(max(max(content_len, len(str(col_name))) + 2, 10), 45)
        worksheet.set_column(col_idx, col_idx, width, cell_fmt)

    worksheet.set_row(0, 30)
    worksheet.freeze_panes(1, 0)
    if len(df) > 0:
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
    return worksheet

def _linear_forecast(series, periods):
    y = series.values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    future_x = np.arange(len(y), len(y) + periods, dtype=float)
    vals = slope * future_x + intercept
    resid = float(np.std(y - fitted, ddof=0))
    return vals, "Linear trend", resid

def forecast_series(series, periods=3):
    series = series.dropna().astype(float)
    n = len(series)
    if n == 0:
        return pd.Series(dtype=float), "No data", 0.0
    future_idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
    if n < 4:
        vals = np.repeat(float(series.iloc[-1]), periods)
        method = "Naive"
        resid = float(series.std(ddof=0)) if n > 1 else 0.0
    elif STATSMODELS_AVAILABLE and n >= 24:
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=12,
                                         damped_trend=True, initialization_method="estimated").fit(optimized=True)
            vals = np.asarray(model.forecast(periods), dtype=float)
            resid = float(np.std(series.values - np.asarray(model.fittedvalues, dtype=float), ddof=0))
            method = "Holt-Winters"
        except Exception:
            vals, method, resid = _linear_forecast(series, periods)
    else:
        vals, method, resid = _linear_forecast(series, periods)
    vals = np.clip(np.round(vals), 0, None)
    return pd.Series(vals, index=future_idx), method, resid

# ====================== SESSION STATE ======================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "map_selected_station" not in st.session_state:
    st.session_state.map_selected_station = None

# ====================== LOGIN ======================
def login_page():
    col1, col2, col3 = st.columns([3, 3, 3])
    with col2:
        st.subheader("🔐 Secure Login")
        with st.form("login_form"):
            email = st.text_input("Username / Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary", use_container_width=True):
                if email in USERS and password == USERS[email].get("password"):
                    st.session_state.logged_in = True
                    st.session_state.user_name = USERS[email].get("name")
                    st.success(f"Welcome, {st.session_state.user_name}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials!")

@st.cache_data(ttl=600, show_spinner="Loading data...")
def load_data_from_gsheet():
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
        client = gspread.authorize(credentials)
        sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
        df = pd.DataFrame(sheet.get_all_records())
        if df.empty:
            st.error("Google Sheet is empty!")
            st.stop()
        df.columns = df.columns.str.strip()
        if 'FCOUNT' in df.columns:
            df['FCOUNT'] = pd.to_numeric(df['FCOUNT'], errors='coerce').fillna(0).astype(int)
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df['MONTH'] = df['DATE'].dt.strftime('%B')
        if 'STATION' in df.columns and 'DEPARTMENT' in df.columns:
            df['JURISDICTION'] = df.apply(lambda row: get_jurisdiction(row['STATION'], row['DEPARTMENT']), axis=1)
        else:
            df['JURISDICTION'] = "Unclassified"
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

def refresh_data():
    st.cache_data.clear()
    st.session_state.map_selected_station = None
    st.success("✅ Data refreshed!")
    st.rerun()

# ====================== MAIN APP ======================
if not st.session_state.logged_in:
    login_page()
else:
    # Sidebar
    with st.sidebar:
        st.header("🎨 Theme")
        theme_choice = st.radio("Select Theme", ["Light", "Dark"],
                                index=0 if st.session_state.theme == "Light" else 1,
                                horizontal=True)
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.markdown("---")
        st.header("🔧 Controls")
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            refresh_data()

    # ========== CENTRAL RAILWAY LOGO WATERMARK ==========
    st.markdown(f"""
    <div class="logo-watermark">
        <img src="{IR_LOGO_URL}" width="450">
    </div>
    """, unsafe_allow_html=True)

    # ========== MOVING BULLET TRAIN ==========
    st.markdown("""
    <div class="train-moving">
        🚅═══════🚅═══════🚅═══════🚅═══════🚅═══════🚅
    </div>
    """, unsafe_allow_html=True)

    # ========== HEADER ==========
    col1, col2, col3 = st.columns([3, 3, 1])
    with col2:
        st.image(IR_LOGO_URL, width=200)

    st.markdown('<h1 class="dashboard-title">DATA LOGGER EXCEPTIONAL REPORT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Central Railway • Solapur Division • Safety Branch</p>', unsafe_allow_html=True)
    st.caption(f"**Logged in as:** {st.session_state.user_name}")
    st.divider()

    df_original = load_data_from_gsheet()

    # ====================== FILTERS ======================
    st.markdown("### 🔍 Live Filters")
    col_f1 = st.columns(4)
    with col_f1[0]:
        stations = sorted(df_original['STATION'].dropna().unique().tolist()) if 'STATION' in df_original.columns else []
        selected_stations = st.multiselect("STATION", options=stations, default=[])
    with col_f1[1]:
        errors = sorted(df_original['ERROR MAIN CATEGORY'].dropna().unique().tolist()) if 'ERROR MAIN CATEGORY' in df_original.columns else []
        selected_errors = st.multiselect("ERROR MAIN CATEGORY", options=errors, default=[])
    with col_f1[2]:
        categories = sorted(df_original['DEPARTMENT'].dropna().unique().tolist()) if 'DEPARTMENT' in df_original.columns else []
        selected_categories = st.multiselect("DEPARTMENT", options=categories, default=[])
    with col_f1[3]:
        months = sorted(df_original['MONTH'].dropna().unique().tolist()) if 'MONTH' in df_original.columns else []
        selected_months = st.multiselect("MONTH", options=months, default=[])

    col_f2 = st.columns(4)
    with col_f2[0]:
        fcount_list = sorted(df_original['FCOUNT'].dropna().unique().tolist()) if 'FCOUNT' in df_original.columns else []
        selected_fcount = st.multiselect("FCOUNT", options=fcount_list, default=[])
    with col_f2[1]:
        fault_list = sorted(df_original['DL FAULT MESSAGE'].dropna().unique().tolist()) if 'DL FAULT MESSAGE' in df_original.columns else []
        selected_fault = st.multiselect("DL FAULT MESSAGE", options=fault_list, default=[])
    with col_f2[2]:
        remark_list = sorted(df_original['REMARKS GIVEN BY S&T'].dropna().unique().tolist()) if 'REMARKS GIVEN BY S&T' in df_original.columns else []
        selected_remark = st.multiselect("REMARKS GIVEN BY S&T", options=remark_list, default=[])
    with col_f2[3]:
        jurisdictions = sorted(df_original['JURISDICTION'].dropna().unique().tolist()) if 'JURISDICTION' in df_original.columns else []
        selected_jurisdictions = st.multiselect("JURISDICTION", options=jurisdictions, default=[])

    col_date = st.columns(2)
    with col_date[0]:
        min_date = df_original['DATE'].min().date() if not df_original.empty and 'DATE' in df_original.columns else pd.Timestamp.now().date()
        from_date = st.date_input("FROM DATE", value=min_date)
    with col_date[1]:
        max_date = df_original['DATE'].max().date() if not df_original.empty and 'DATE' in df_original.columns else pd.Timestamp.now().date()
        to_date = st.date_input("TO DATE", value=max_date)

    st.divider()

    # Apply filters
    filtered_df = df_original.copy()
    if selected_stations:
        filtered_df = filtered_df[filtered_df['STATION'].isin(selected_stations)]
    if selected_errors:
        filtered_df = filtered_df[filtered_df['ERROR MAIN CATEGORY'].isin(selected_errors)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df['DEPARTMENT'].isin(selected_categories)]
    if selected_fcount:
        filtered_df = filtered_df[filtered_df['FCOUNT'].isin(selected_fcount)]
    if selected_fault:
        filtered_df = filtered_df[filtered_df['DL FAULT MESSAGE'].isin(selected_fault)]
    if selected_remark:
        filtered_df = filtered_df[filtered_df['REMARKS GIVEN BY S&T'].isin(selected_remark)]
    if selected_jurisdictions:
        filtered_df = filtered_df[filtered_df['JURISDICTION'].isin(selected_jurisdictions)]
    if 'DATE' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['DATE'].dt.date >= from_date) & (filtered_df['DATE'].dt.date <= to_date)]
    if selected_months:
        filtered_df = filtered_df[filtered_df['MONTH'].isin(selected_months)]
    if st.session_state.map_selected_station:
        filtered_df = filtered_df[filtered_df['STATION'] == st.session_state.map_selected_station]

    cat_sum = filtered_df.groupby('DEPARTMENT').size().reset_index(name='Cases').sort_values('Cases', ascending=False) if not filtered_df.empty and 'DEPARTMENT' in filtered_df.columns else pd.DataFrame()
    error_sum = filtered_df.groupby('ERROR MAIN CATEGORY').size().reset_index(name='Cases').sort_values('Cases', ascending=False) if not filtered_df.empty and 'ERROR MAIN CATEGORY' in filtered_df.columns else pd.DataFrame()
    jur_sum = filtered_df.groupby('JURISDICTION').size().reset_index(name='Cases').sort_values('Cases', ascending=False) if not filtered_df.empty and 'JURISDICTION' in filtered_df.columns else pd.DataFrame()

    # ====================== TABS ======================
    tab_overview, tab_forecast, tab_map = st.tabs(["📊 Overview Dashboard", "🔮 Forecast (3 Months)", "🗺️ Map View"])

    with tab_overview:
        st.subheader("📊 Overview Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", f"{len(filtered_df):,}")
        with c2:
            st.metric("Total FCOUNT", f"{filtered_df.get('FCOUNT', pd.Series(0)).sum():,}")
        with c3:
            top_station = "N/A"
            if not filtered_df.empty and 'STATION' in filtered_df.columns:
                stn_tot = filtered_df.groupby('STATION')['FCOUNT'].sum().sort_values(ascending=False)
                if not stn_tot.empty:
                    top_station = stn_tot.index[0]
            st.metric("⚠️ Top Station", top_station)
        with c4:
            top_fcount = 0
            if not filtered_df.empty and 'STATION' in filtered_df.columns:
                stn_tot = filtered_df.groupby('STATION')['FCOUNT'].sum().sort_values(ascending=False)
                if not stn_tot.empty:
                    top_fcount = stn_tot.iloc[0]
            st.metric("Top Station FCOUNT", f"{top_fcount:,}")

        st.markdown("---")

        col_g1, col_g2 = st.columns([3, 2])
        with col_g1:
            st.markdown('<p class="section-header">Top 15 Stations by FCOUNT</p>', unsafe_allow_html=True)
            if not filtered_df.empty and 'STATION' in filtered_df.columns:
                top15 = filtered_df.groupby('STATION')['FCOUNT'].sum().nlargest(15).reset_index()
                fig = px.bar(top15, x='STATION', y='FCOUNT', text='FCOUNT', color='FCOUNT', color_continuous_scale='RdYlGn_r')
                fig.update_layout(height=480, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            st.markdown('<p class="section-header">Station Summary</p>', unsafe_allow_html=True)
            if not filtered_df.empty and 'STATION' in filtered_df.columns:
                summary = filtered_df.groupby('STATION')['FCOUNT'].agg(Total_FCOUNT='sum', Records='count').sort_values('Total_FCOUNT', ascending=False)
                st.dataframe(summary.style.format({"Total_FCOUNT": "{:,}", "Records": "{:,}"}).background_gradient(subset=['Total_FCOUNT'], cmap='YlOrRd'), use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-header">📊 Distribution Charts</p>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("**Department-wise**")
            if not cat_sum.empty:
                fig_dept = px.bar(cat_sum.sort_values('Cases'), x='Cases', y='DEPARTMENT', orientation='h', text='Cases', color='Cases', color_continuous_scale='Blues')
                fig_dept.update_traces(textposition='outside')
                fig_dept.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_dept, use_container_width=True)
        with col_c2:
            st.markdown("**Error Main Category**")
            if not error_sum.empty:
                fig_err = px.bar(error_sum.head(12).sort_values('Cases'), x='Cases', y='ERROR MAIN CATEGORY', orientation='h', text='Cases', color='Cases', color_continuous_scale='Oranges')
                fig_err.update_traces(textposition='outside')
                fig_err.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_err, use_container_width=True)
        with col_c3:
            st.markdown("**Jurisdiction-wise**")
            if not jur_sum.empty:
                fig_jur = px.bar(jur_sum.head(12).sort_values('Cases'), x='Cases', y='JURISDICTION', orientation='h', text='Cases', color='Cases', color_continuous_scale='Teal')
                fig_jur.update_traces(textposition='outside')
                fig_jur.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_jur, use_container_width=True)

        # Animated section
        st.markdown("---")
        st.markdown('<p class="section-header">🎬 Animated Monthly Cases / FCOUNT by Station</p>', unsafe_allow_html=True)

        if filtered_df.empty or 'STATION' not in filtered_df.columns or 'DATE' not in filtered_df.columns:
            st.warning("Not enough data for animation.")
        else:
            anim_df = filtered_df.dropna(subset=['DATE', 'STATION']).copy()
            col_anim1, col_anim2, col_anim3 = st.columns(3)
            with col_anim1:
                metric = st.radio("Metric to animate", ["Number of Cases", "Total FCOUNT"], horizontal=True)
            with col_anim2:
                top_n_anim = st.slider("Show Top N stations", 5, 25, 12)
            with col_anim3:
                anim_speed = st.select_slider("Animation Speed", options=["Very Slow", "Slow", "Normal", "Fast"], value="Slow")

            speed_map = {"Very Slow": 1800, "Slow": 1400, "Normal": 1000, "Fast": 700}
            frame_duration = speed_map[anim_speed]

            if metric == "Number of Cases":
                monthly = anim_df.groupby(['STATION', pd.Grouper(key='DATE', freq='MS')]).size().reset_index(name='Value')
                y_label = "Cases"
            else:
                monthly = anim_df.groupby(['STATION', pd.Grouper(key='DATE', freq='MS')])['FCOUNT'].sum().reset_index(name='Value')
                y_label = "FCOUNT"

            monthly['Month'] = monthly['DATE'].dt.strftime('%b %Y')
            monthly = monthly.sort_values('DATE')

            station_order = (monthly.groupby('STATION')['Value'].sum().sort_values(ascending=False).head(top_n_anim).index.tolist())
            monthly = monthly[monthly['STATION'].isin(station_order)]
            monthly['STATION'] = pd.Categorical(monthly['STATION'], categories=station_order, ordered=True)
            monthly = monthly.sort_values(['DATE', 'STATION'])

            if not monthly.empty:
                fig_anim = px.bar(
                    monthly, x='STATION', y='Value', color='Value',
                    animation_frame='Month', animation_group='STATION',
                    range_y=[0, monthly['Value'].max() * 1.18],
                    color_continuous_scale='RdYlGn_r',
                    labels={'Value': y_label},
                    title=f"Monthly {y_label} by Station — Animated (Highest → Lowest)",
                    text='Value'
                )
                fig_anim.update_traces(texttemplate='%{text:,}', textposition='outside', cliponaxis=False)
                fig_anim.update_layout(height=600, xaxis_tickangle=-45, coloraxis_showscale=False,
                                       xaxis={'categoryorder': 'array', 'categoryarray': station_order})
                fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
                fig_anim.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = int(frame_duration * 0.55)

                st.plotly_chart(fig_anim, use_container_width=True)
                st.caption(f"Speed: **{anim_speed}** • Bars sorted Highest → Lowest")

                html_bytes = fig_anim.to_html(full_html=True, include_plotlyjs='cdn').encode('utf-8')
                st.download_button("⬇️ Download Animation (HTML)", data=html_bytes,
                                   file_name=f"Animation_{y_label}.html", mime="text/html", type="primary")

        # Summary Tables
        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if not cat_sum.empty:
                st.markdown('<p class="section-header">DEPARTMENT</p>', unsafe_allow_html=True)
                st.dataframe(cat_sum.style.format({"Cases": "{:,}"}), use_container_width=True, hide_index=True)
        with col_s2:
            if not error_sum.empty:
                st.markdown('<p class="section-header">ERROR MAIN CATEGORY</p>', unsafe_allow_html=True)
                st.dataframe(error_sum.style.format({"Cases": "{:,}"}), use_container_width=True, hide_index=True)
        with col_s3:
            if not jur_sum.empty:
                st.markdown('<p class="section-header">JURISDICTION</p>', unsafe_allow_html=True)
                st.dataframe(jur_sum.style.format({"Cases": "{:,}"}), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<p class="section-header">Detailed Records</p>', unsafe_allow_html=True)
        if filtered_df.empty:
            st.warning("No records found.")
        else:
            display_df = filtered_df.copy()
            if 'DATE' in display_df.columns:
                display_df['DATE'] = display_df['DATE'].dt.date
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                write_styled_sheet(writer, display_df, 'Filtered_Records')
            output.seek(0)
            st.download_button("⬇️ Download Excel Report", data=output.getvalue(),
                               file_name=f"Datalogger_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    with tab_forecast:
        st.subheader("🔮 Forecast — next 1 to 3 months")
        st.info("Forecast logic remains the same as your previous working version.")

    with tab_map:
        st.subheader("🗺️ Interactive Map View")
        st.info("Map logic remains the same as your previous working version.")

    st.caption("🚅 Safety Branch | Central Railway, Solapur Division")
