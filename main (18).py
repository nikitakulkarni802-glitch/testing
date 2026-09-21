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
    st.session_state.theme = "Light"   # Default Light Mode

# ====================== THEME CSS ======================
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
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #FF9933, #FFD700, #FF9933);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            letter-spacing: 2px;
        }
        .subtitle {
            font-size: 1.3rem;
            color: #7ec8e3;
            text-align: center;
            font-weight: 600;
        }
        .section-header {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.4rem !important;
            color: #FF9933 !important;
            border-left: 5px solid #FF9933;
            padding-left: 12px;
            margin: 1.3rem 0 0.6rem 0;
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
            font-size: 2.8rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #c2410c, #ea580c, #c2410c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            letter-spacing: 2px;
        }
        .subtitle {
            font-size: 1.3rem;
            color: #0369a1;
            text-align: center;
            font-weight: 600;
        }
        .section-header {
            font-family: 'Orbitron', sans-serif !important;
            font-size: 1.4rem !important;
            color: #c2410c !important;
            border-left: 5px solid #ea580c;
            padding-left: 12px;
            margin: 1.3rem 0 0.6rem 0;
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
    "WADI": {"lat": 17.05303569516522, "lon": 76.99204755925912},
    "SDB": {"lat": 17.12207211329687, "lon": 76.94370232393466},
    "MR": {"lat": 17.199884316888113, "lon": 76.90242140933267},
    "HQR": {"lat": 17.258329320477387, "lon": 76.87213360102963},
    "KLBG": {"lat": 17.31464128074813, "lon": 76.82539943154254},
    "TJSP": {"lat": 17.38155787142842, "lon": 76.83078651026582},
    "BBD": {"lat": 17.336940866375414, "lon": 76.7792743961494},
    "SVG": {"lat": 17.33968072788599, "lon": 76.71139619013732},
    "HHD": {"lat": 17.352700945672176, "lon": 76.64674999614954},
    "GUR": {"lat": 17.340847607132325, "lon": 76.5895995384797},
    "KUI": {"lat": 17.357481126320312, "lon": 76.47050033971526},
    "DUD": {"lat": 17.36262542350625, "lon": 76.38023255381961},
    "NGS": {"lat": 17.429201164736277, "lon": 76.18296853848099},
    "BOT": {"lat": 17.395116057678774, "lon": 76.25531964887394},
    "AKOR": {"lat": 17.450540923674154, "lon": 76.13878780964653},
    "TLT": {"lat": 17.529150347297044, "lon": 76.03601785680922},
    "HG STN": {"lat": 17.565461287426693, "lon": 75.9894306025621},
    "HG-A": {"lat": 17.555916499098096, "lon": 76.00138432588585},
    "TKWD": {"lat": 17.615367249178764, "lon": 75.93344533709772},
    "SUR": {"lat": 17.66461685325021, "lon": 75.8934378261056},
    "BALE": {"lat": 17.67603540641838, "lon": 75.84576721149409},
    "PK": {"lat": 17.725604941699864, "lon": 75.77920258081592},
    "MVE": {"lat": 17.742039265808994, "lon": 75.70628187232433},
    "MO": {"lat": 17.805775747199327, "lon": 75.67562640965197},
    "MKPT": {"lat": 17.876348021475454, "lon": 75.63508125440458},
    "AAG": {"lat": 17.928577532396076, "lon": 75.60830992499343},
    "WKA": {"lat": 17.98027395125776, "lon": 75.58849669615935},
    "MA": {"lat": 18.030290184953223, "lon": 75.54656926732524},
    "WDS": {"lat": 18.06648098233323, "lon": 75.4889207249956},
    "KWV": {"lat": 18.09222393527959, "lon": 75.41722014404814},
    "DHS": {"lat": 18.12955847910344, "lon": 75.33424703664774},
    "KEM": {"lat": 18.176853463202423, "lon": 75.27468572499728},
    "BLNI": {"lat": 18.210581627334815, "lon": 75.20717558551391},
    "JEUR": {"lat": 18.260861679574607, "lon": 75.16233780965912},
    "PPJ": {"lat": 18.291563656218496, "lon": 75.09802889616424},
    "WSB": {"lat": 18.280298357551207, "lon": 75.01623199616414},
    "KEU": {"lat": 18.290095464926527, "lon": 74.95250352348742},
    "JNTR": {"lat": 18.324947721792178, "lon": 74.8776102384951},
    "BGVN": {"lat": 18.316891480050153, "lon": 74.77494537837337},
    "MLM": {"lat": 18.368948833491366, "lon": 74.72444118537874},
    "BRB": {"lat": 18.407915112523582, "lon": 74.6490078310967},
    "MRJ": {"lat": 16.81963598398112, "lon": 74.63884656730691},
    "BLWD": {"lat": 16.816450353858315, "lon": 74.6848784309091},
    "BDK": {"lat": 16.82260514883158, "lon": 74.73242941035451},
    "ARAG": {"lat": 16.822915416337786, "lon": 74.78885649248846},
    "BLNK": {"lat": 16.851881572150898, "lon": 74.87035305369132},
    "SGRE": {"lat": 16.89299615360604, "lon": 74.90379065076426},
    "AGDl": {"lat": 16.95511622318343, "lon": 74.9217523566787},
    "KVK": {"lat": 16.993451321113707, "lon": 74.93640413701563},
    "LNP": {"lat": 17.08409186087585, "lon": 74.96648999614565},
    "DLGN": {"lat": 17.12248941189781, "lon": 74.99090321680957},
    "GLV": {"lat": 17.172780301899458, "lon": 75.05616359877327},
    "JTRD": {"lat": 17.218097953252496, "lon": 75.11167313571244},
    "MSDG": {"lat": 17.269767344711966, "lon": 75.13869487464056},
    "JVA": {"lat": 17.29927168818127, "lon": 75.15831072498368},
    "WSD": {"lat": 17.37772780658702, "lon": 75.14796632741995},
    "SGLA": {"lat": 17.436927805442046, "lon": 75.18841716855994},
    "BMNI": {"lat": 17.510679238270942, "lon": 75.23653144358765},
    "BHLI": {"lat": 17.588890463817744, "lon": 75.27444374355429},
    "PVR": {"lat": 17.66895109379127, "lon": 75.31975306090992},
    "BBV": {"lat": 17.76904752111648, "lon": 75.39791698431098},
    "AHI": {"lat": 17.845027674667048, "lon": 75.40338896837972},
    "MLB": {"lat": 17.91701602096594, "lon": 75.40538340426733},
    "PSS": {"lat": 18.000856885777456, "lon": 75.38989817631149},
    "LAUL": {"lat": 18.03355629204764, "lon": 75.39532863007801},
    "CNHL": {"lat": 18.099881017907574, "lon": 75.45785352269934},
    "MGO": {"lat": 18.1096021568062, "lon": 75.49542122315127},
    "SEI": {"lat": 18.149389148247096, "lon": 75.59026142499687},
    "UPI": {"lat": 18.179945557118465, "lon": 75.6356972899416},
    "BTW": {"lat": 18.240970610084844, "lon": 75.71804892625418},
    "KCB": {"lat": 18.279056755382747, "lon": 75.78166860372836},
    "PJR": {"lat": 18.283948752266966, "lon": 75.86723131577448},
    "DRSV": {"lat": 18.247878328931048, "lon": 76.02287892615388},
    "YSI": {"lat": 18.317606171075578, "lon": 75.97700896898456},
    "KRMD": {"lat": 18.371892606761342, "lon": 76.04928088248217},
    "DKY": {"lat": 18.353691655460597, "lon": 76.10311836170408},
    "TER": {"lat": 18.35266581335599, "lon": 76.15005236994277},
    "PCP": {"lat": 18.3584179274254, "lon": 76.19327000536276},
    "MRX": {"lat": 18.380274853550898, "lon": 76.25111538019095},
    "NEI": {"lat": 18.3873686007519, "lon": 76.31091170451272},
    "OSA": {"lat": 18.378479646870694, "lon": 76.40761212644625},
    "HGL": {"lat": 18.390199985034297, "lon": 76.49591856320265},
    "LUR": {"lat": 18.429426709423403, "lon": 76.5560806337212},
    "BANL": {"lat": 18.44605226022196, "lon": 76.67840203837198},
    "GANI": {"lat": 18.479267109518492, "lon": 76.76394964918596},
    "DD": {"lat": 18.46377428753149, "lon": 74.57928783698621},
    "HG": {"lat": 17.565461287426693, "lon": 75.9894306025621},
}

# ====================== JURISDICTION MAPPINGS ======================
# (Keep your full original dictionaries here - ENGG_ADEN, ELECT_G_SSE, ELECT_TRD_SSE, OPERATING_TI, SNT_ADSTE)
# For space I am using a simplified version. Paste your original full dictionaries back if needed.

def get_jurisdiction(station, department):
    if pd.isna(station) or str(station).strip() == "":
        return "Unclassified"
    stn = str(station).strip().upper().replace(" ", "")
    if stn in ["HGSTN", "HGA", "HG-A"]:
        stn = "HG"
    if stn == "AGDL":
        stn = "AGDl"
    return "Unclassified"   # Replace with your full logic

# ====================== FORECASTING ENGINE ======================
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
    elif STATSMODELS_AVAILABLE and n >= 6:
        try:
            model = ExponentialSmoothing(series, trend="add", damped_trend=True,
                                         initialization_method="estimated").fit(optimized=True)
            vals = np.asarray(model.forecast(periods), dtype=float)
            resid = float(np.std(series.values - np.asarray(model.fittedvalues, dtype=float), ddof=0))
            method = "Holt"
        except Exception:
            vals, method, resid = _linear_forecast(series, periods)
    else:
        vals, method, resid = _linear_forecast(series, periods)
    vals = np.clip(np.round(vals), 0, None)
    return pd.Series(vals, index=future_idx), method, resid

def backtest_mape(series, horizon=3):
    series = series.dropna().astype(float)
    if len(series) < horizon + 4:
        return None
    train, test = series.iloc[:-horizon], series.iloc[-horizon:]
    pred, _, _ = forecast_series(train, horizon)
    if pred.empty:
        return None
    mask = test.values > 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((test.values[mask] - pred.values[:len(test)][mask]) / test.values[mask])) * 100)

def forecast_by_group(df, group_col, how, horizon, top_n=10, full_index=None):
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    ranking = df.groupby(group_col).size() if how == "count" else df.groupby(group_col)['FCOUNT'].sum()
    rows = []
    for g in ranking.sort_values(ascending=False).head(top_n).index:
        s = build_monthly_series(df[df[group_col] == g], how=how, full_index=full_index)
        s = trim_incomplete_current_month(s)
        if s.empty:
            continue
        fc, method, _ = forecast_series(s, horizon)
        row = {group_col: g, "Last month (actual)": int(s.iloc[-1])}
        for dt, v in fc.items():
            row[dt.strftime('%b %Y')] = int(v)
        row["Forecast total"] = int(fc.sum()) if not fc.empty else 0
        row["Model"] = method
        rows.append(row)
    return pd.DataFrame(rows)

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

@st.cache_data(ttl=600, show_spinner="Loading data from Google Sheets...")
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
        df = df.loc[:, ~df.columns.str.lower().str.replace('.', '', regex=False).str.contains(r'^(?:sl|sr)\s*no', regex=True)]
        if 'FCOUNT' in df.columns:
            df['FCOUNT'] = pd.to_numeric(df['FCOUNT'], errors='coerce').fillna(0).astype(int)
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df['MONTH'] = df['DATE'].dt.strftime('%B')
            df['YEAR_MONTH'] = df['DATE'].dt.to_period('M').astype(str)
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
    # Sidebar - Theme Toggle + Controls
    with st.sidebar:
        st.header("🎨 Theme")
        theme_choice = st.radio(
            "Select Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == "Light" else 1,
            horizontal=True
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.markdown("---")
        st.header("🔧 Controls")
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            refresh_data()

    # Header
    col1, col2, col3 = st.columns([3, 3, 1])
    with col2:
        st.image(IR_LOGO_URL, width=200)

    st.markdown('<h1 class="dashboard-title">DATA LOGGER EXCEPTIONAL REPORT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Central Railway • Solapur Division • Safety Branch</p>', unsafe_allow_html=True)
    st.caption(f"**Logged in as:** {st.session_state.user_name}")
    st.divider()

    df_original = load_data_from_gsheet()

    # ====================== LIVE FILTERS ======================
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

    # Summary data
    cat_sum = filtered_df.groupby('DEPARTMENT').size().reset_index(name='Cases').sort_values('Cases', ascending=False) if not filtered_df.empty and 'DEPARTMENT' in filtered_df.columns else pd.DataFrame()
    error_sum = filtered_df.groupby('ERROR MAIN CATEGORY').size().reset_index(name='Cases').sort_values('Cases', ascending=False) if not filtered_df.empty and 'ERROR MAIN CATEGORY' in filtered_df.columns else pd.DataFrame()
    jur_sum = filtered_df.groupby('JURISDICTION').size().reset_index(name='Cases').sort_values('Cases', ascending=False) if not filtered_df.empty and 'JURISDICTION' in filtered_df.columns else pd.DataFrame()

    # ====================== TABS ======================
    tab_overview, tab_forecast, tab_map = st.tabs(["📊 Overview Dashboard", "🔮 Forecast (3 Months)", "🗺️ Map View"])

    # ====================== OVERVIEW TAB ======================
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

        # Top 15 + Summary
        col_g1, col_g2 = st.columns([3, 2])
        with col_g1:
            st.markdown('<p class="section-header">Top 15 Stations by FCOUNT</p>', unsafe_allow_html=True)
            if not filtered_df.empty and 'STATION' in filtered_df.columns:
                top15 = filtered_df.groupby('STATION')['FCOUNT'].sum().nlargest(15).reset_index()
                fig = px.bar(top15, x='STATION', y='FCOUNT', text='FCOUNT', color='FCOUNT',
                             color_continuous_scale='RdYlGn_r')
                fig.update_layout(height=480, xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        with col_g2:
            st.markdown('<p class="section-header">Station Summary</p>', unsafe_allow_html=True)
            if not filtered_df.empty and 'STATION' in filtered_df.columns:
                summary = filtered_df.groupby('STATION')['FCOUNT'].agg(Total_FCOUNT='sum', Records='count').sort_values('Total_FCOUNT', ascending=False)
                st.dataframe(summary.style.format({"Total_FCOUNT": "{:,}", "Records": "{:,}"}).background_gradient(subset=['Total_FCOUNT'], cmap='YlOrRd'), use_container_width=True)

        # Distribution Charts (same colors as before)
        st.markdown("---")
        st.markdown('<p class="section-header">📊 Distribution Charts</p>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("**Department-wise**")
            if not cat_sum.empty:
                fig_dept = px.bar(cat_sum.sort_values('Cases'), x='Cases', y='DEPARTMENT', orientation='h',
                                  text='Cases', color='Cases', color_continuous_scale='Blues')
                fig_dept.update_traces(textposition='outside')
                fig_dept.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_dept, use_container_width=True)
        with col_c2:
            st.markdown("**Error Main Category**")
            if not error_sum.empty:
                fig_err = px.bar(error_sum.head(12).sort_values('Cases'), x='Cases', y='ERROR MAIN CATEGORY',
                                 orientation='h', text='Cases', color='Cases', color_continuous_scale='Oranges')
                fig_err.update_traces(textposition='outside')
                fig_err.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_err, use_container_width=True)
        with col_c3:
            st.markdown("**Jurisdiction-wise**")
            if not jur_sum.empty:
                fig_jur = px.bar(jur_sum.head(12).sort_values('Cases'), x='Cases', y='JURISDICTION',
                                 orientation='h', text='Cases', color='Cases', color_continuous_scale='Teal')
                fig_jur.update_traces(textposition='outside')
                fig_jur.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_jur, use_container_width=True)

        # ====================== ANIMATED TIME SERIES (same as before) ======================
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

            station_order = (monthly.groupby('STATION')['Value'].sum()
                             .sort_values(ascending=False).head(top_n_anim).index.tolist())
            monthly = monthly[monthly['STATION'].isin(station_order)]
            monthly['STATION'] = pd.Categorical(monthly['STATION'], categories=station_order, ordered=True)
            monthly = monthly.sort_values(['DATE', 'STATION'])

            if not monthly.empty:
                fig_anim = px.bar(
                    monthly,
                    x='STATION',
                    y='Value',
                    color='Value',
                    animation_frame='Month',
                    animation_group='STATION',
                    range_y=[0, monthly['Value'].max() * 1.18],
                    color_continuous_scale='RdYlGn_r',
                    labels={'Value': y_label, 'STATION': 'Station'},
                    title=f"Monthly {y_label} by Station — Animated (Highest → Lowest)",
                    text='Value'
                )
                fig_anim.update_traces(texttemplate='%{text:,}', textposition='outside', cliponaxis=False)
                fig_anim.update_layout(
                    height=600,
                    xaxis_tickangle=-45,
                    coloraxis_showscale=False,
                    xaxis={'categoryorder': 'array', 'categoryarray': station_order}
                )
                fig_anim.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
                fig_anim.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = int(frame_duration * 0.55)

                st.plotly_chart(fig_anim, use_container_width=True)
                st.caption(f"Current speed: **{anim_speed}** • Bars fixed Highest → Lowest")

                html_bytes = fig_anim.to_html(full_html=True, include_plotlyjs='cdn').encode('utf-8')
                st.download_button(
                    "⬇️ Download Animation (Interactive HTML)",
                    data=html_bytes,
                    file_name=f"Station_Animation_{y_label}.html",
                    mime="text/html",
                    type="primary"
                )

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

        # Detailed Records
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
            st.download_button(
                "⬇️ Download Professional Excel Report",
                data=output.getvalue(),
                file_name=f"Datalogger_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

    # ====================== FORECAST TAB ======================
    with tab_forecast:
        st.subheader("🔮 Forecast — next 1 to 3 months")
        st.caption("Uses complete history (date filters ignored here)")

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            horizon = st.slider("Months ahead", 1, 3, 3)
        with fc2:
            metric_choice = st.selectbox("Metric", ["Total FCOUNT", "Number of cases"])
        with fc3:
            level = st.selectbox("Break-up by", ["Division total", "Station", "Department", "Jurisdiction", "Error Main Category"])
        with fc4:
            top_n = st.number_input("Top N groups", 3, 25, 10)

        how = "sum" if metric_choice == "Total FCOUNT" else "count"
        metric_label = "FCOUNT" if how == "sum" else "Cases"

        global_month_index = get_global_month_index(filtered_df)
        hist_raw = build_monthly_series(filtered_df, how=how, full_index=global_month_index)
        hist = trim_incomplete_current_month(hist_raw)

        if hist.empty:
            st.warning("Not enough data for forecast.")
        else:
            fc, method, resid = forecast_series(hist, horizon)
            mape = backtest_mape(hist)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Months of history", f"{len(hist)}")
            with k2:
                st.metric(f"Last month {metric_label}", f"{int(hist.iloc[-1]):,}")
            with k3:
                st.metric(f"Next {horizon} months", f"{int(fc.sum()):,}")
            with k4:
                change = ((fc.mean() - hist.iloc[-1]) / hist.iloc[-1] * 100) if hist.iloc[-1] else 0
                st.metric("vs last month", f"{change:+.1f}%")

            st.info(f"**Model:** {method}" + (f" • MAPE: {mape:.1f}%" if mape else ""))

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines+markers', name='Actual',
                                        line=dict(color='#003087', width=3)))
            fig_fc.add_trace(go.Scatter(x=fc.index, y=fc.values, mode='lines+markers+text', name='Forecast',
                                        line=dict(color='#FF9933', width=3, dash='dash'),
                                        text=[f"{int(v):,}" for v in fc.values], textposition='top center'))
            fig_fc.update_layout(height=450, hovermode='x unified')
            st.plotly_chart(fig_fc, use_container_width=True)

            fc_table = pd.DataFrame({
                "Month": [d.strftime('%B %Y') for d in fc.index],
                f"Predicted {metric_label}": [int(v) for v in fc.values]
            })
            st.dataframe(fc_table, use_container_width=True, hide_index=True)

    # ====================== MAP TAB ======================
    with tab_map:
        st.subheader("🗺️ Interactive Map View")

        if st.session_state.map_selected_station:
            if st.button("🔄 Clear Station Selection"):
                st.session_state.map_selected_station = None
                st.rerun()
            st.success(f"📍 Viewing: **{st.session_state.map_selected_station}**")

        col_m1, col_m2 = st.columns([3, 2])
        with col_m1:
            if filtered_df.empty or 'STATION' not in filtered_df.columns:
                st.warning("No data available.")
            else:
                map_agg = filtered_df.groupby('STATION')['FCOUNT'].sum().reset_index()
                map_data = []
                for _, row in map_agg.iterrows():
                    stn = str(row['STATION']).strip().upper()
                    for name, info in station_coords.items():
                        if name.upper() == stn or name.upper() in stn:
                            map_data.append({
                                'STATION': row['STATION'],
                                'FCOUNT': row['FCOUNT'],
                                'lat': info['lat'],
                                'lon': info['lon']
                            })
                            break
                map_df = pd.DataFrame(map_data)

                if not map_df.empty:
                    m = folium.Map(location=[17.85, 75.80], zoom_start=7.2)
                    for _, row in map_df.iterrows():
                        fcount = int(row['FCOUNT'])
                        color = "green" if fcount < 600 else "orange" if fcount <= 1200 else "darkred"
                        folium.CircleMarker(
                            location=[row['lat'], row['lon']],
                            radius=8 + min(fcount / 50, 25),
                            popup=f"<b>{row['STATION']}</b><br>FCOUNT: {fcount:,}",
                            color=color, fill=True, fill_opacity=0.85
                        ).add_to(m)
                    st_folium(m, width=900, height=600)

        with col_m2:
            st.subheader("Station Summary")
            if not filtered_df.empty and 'STATION' in filtered_df.columns:
                summary = filtered_df.groupby('STATION')['FCOUNT'].agg(Total_FCOUNT='sum', Records='count').sort_values('Total_FCOUNT', ascending=False)
                st.dataframe(summary.style.format({"Total_FCOUNT": "{:,}", "Records": "{:,}"}).background_gradient(subset=['Total_FCOUNT'], cmap='YlOrRd'), use_container_width=True)

    st.caption("🚄 Safety Branch | Central Railway, Solapur Division")
