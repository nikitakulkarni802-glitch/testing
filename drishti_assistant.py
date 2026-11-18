"""
drishti_assistant.py
====================
Free, unlimited, rule-based chat assistant for the DRISHTI dashboard.

* No external AI service, no API key, no usage limit - runs inside the app.
* Data never leaves the server.
* Understands plain-English questions about Station / Department / Error /
  Jurisdiction / Date, does small talk, draws charts, shows tables, and
  produces Excel / PDF / PPTX / PNG / Word / CSV downloads.
* "Trainable": a `teach` command adds synonyms, and every question the bot
  could not answer (or got a thumbs-down) is logged so you can add new rules.

Integration (3 small edits in the main app) is described at the bottom of
this file and in the chat message.
"""
from __future__ import annotations

import difflib
import json
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    _IST = timezone(timedelta(hours=5, minutes=30))

ALIAS_FILE = "assistant_aliases.json"
LOG_FILE = "assistant_log.csv"

# --------------------------------------------------------------------------
# Static knowledge
# --------------------------------------------------------------------------
DIM_COL = {
    "station": "STATION",
    "error": "ERROR MAIN CATEGORY",
    "dept": "DEPARTMENT",
    "jur": "JURISDICTION",
    "fault": "DL FAULT MESSAGE",
    "month": None,  # computed from DATE
}
DIM_LABEL = {"station": "Station", "error": "Error category", "dept": "Department",
             "jur": "Jurisdiction", "fault": "Fault message", "month": "Month"}
DIM_PLURAL = {"station": "stations", "error": "error categories", "dept": "departments",
              "jur": "jurisdictions", "fault": "fault messages", "month": "months"}

NOUN_PATS = [
    ("station", r"stations?|stns?|locations?"),
    ("error", r"errors?|categor(?:y|ies)|fault types?|failures?"),
    ("dept", r"departments?|depts?"),
    ("jur", r"jurisdictions?|sections?|adens?|sses?|adstes?"),
    ("month", r"months?"),
    ("fault", r"messages?"),
]
_NOUN_RX = re.compile(r"\b(?:" + "|".join(f"(?P<{k}>{p})" for k, p in NOUN_PATS) + r")\b")

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
_MONTH_RX = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
             r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20}

DEPT_ALIAS = {"s&t": "S&T", "snt": "S&T", "optg": "OPTG", "operating": "OPTG", "operations": "OPTG",
              "engg": "ENGG", "engineering": "ENGG", "trd": "TRD", "ohe": "TRD", "traction": "TRD",
              "elect": "ELECT", "electrical": "ELECT"}
DEPT_KEYS = {"S&T": ["S&T", "SNT", "SIGNAL"], "OPTG": ["OPTG", "OPERATING"],
             "ENGG": ["ENGG", "ENGINEERING"], "TRD": ["TRD", "TRACTION", "OHE"], "ELECT": ["ELECT"]}

# Words people use for places -> station code (only used when the code exists in data)
DEFAULT_ALIASES = {
    "solapur": "SUR", "sholapur": "SUR", "kalaburagi": "KLBG", "gulbarga": "KLBG",
    "kurduwadi": "KWV", "latur road": "LUR", "pandharpur": "PVR", "miraj": "MRJ",
    "hotgi": "HG", "daund": "DD", "barshi road": "BRB", "akkalkot road": "AKOR",
    "akalkot road": "AKOR", "dudhani": "DUD", "shahabad": "SDB",
}
AMBIGUOUS_CODES = {"BOT"}  # codes that are also English words -> need uppercase / "at BOT"

# Used only for the optional severity weight in the risk score.
SEVERITY = {"SIGNAL FAILURE": 4, "POINT FAILURE": 4, "TRACK CIRCUIT FAILURE": 3,
            "AXLE COUNTER FAILURE": 3, "LC GATE FAILURE": 4, "POWER SUPPLY FAILURE": 3,
            "COMMUNICATION FAILURE": 2, "OTHER": 1}

STOPWORDS = set("""a an the of for in on at to from by with and or is are was were be been it this that
these those me my we you your our show give tell please can could would will what which who how many much
all any some do does did there here about as into than then them they their cases case records record
data list find get display view""".split())


# --------------------------------------------------------------------------
# Small data classes / helpers
# --------------------------------------------------------------------------
@dataclass
class Period:
    start: pd.Timestamp
    end: pd.Timestamp
    label: str


@dataclass
class Reply:
    text: str
    table: pd.DataFrame | None = None
    fig: object | None = None
    file: dict | None = None
    chips: list = field(default_factory=list)
    intent: str = "unknown"
    understood: bool = True


def _fmt(n) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return str(n)


def _pct_change(new, old):
    if old in (0, None) or pd.isna(old):
        return None
    return (new - old) / old * 100.0


def _arrow(p):
    if p is None:
        return ""
    return "▲" if p > 0 else ("▼" if p < 0 else "●")


def _num(tok: str):
    tok = str(tok).strip().lower()
    if tok.isdigit():
        return int(tok)
    return NUM_WORDS.get(tok)


def _loose_pat(value):
    """Regex that matches a value ignoring punctuation / spacing differences."""
    toks = re.findall(r"[a-z0-9&]+", str(value).lower())
    if not toks:
        return None
    body = r"[^A-Za-z0-9&]+".join(re.escape(t) for t in toks)
    return re.compile(r"(?<![A-Za-z0-9&])" + body + r"(?![A-Za-z0-9&])", re.I)


def _month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts + pd.offsets.MonthEnd(0)).replace(hour=23, minute=59, second=59)


def _style(fig, h=420, title=None):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#0d1b2a", margin=dict(t=55 if title else 20, b=30, l=10, r=20),
                      title=title, title_x=0.0)
    return fig


def _hbar(tbl, label_col, val_col, title, scale="Blues", h=None):
    d = tbl.sort_values(val_col, ascending=True)
    fig = px.bar(d, x=val_col, y=label_col, orientation="h", text=val_col,
                 color=val_col, color_continuous_scale=scale)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(type="category", title="")
    fig.update_xaxes(title="Cases")
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, h or max(300, 32 * len(d) + 110), title)


# --------------------------------------------------------------------------
# Time parsing
# --------------------------------------------------------------------------
def parse_periods(ql: str, anchor: pd.Timestamp, as_range: bool = True) -> list[Period]:
    a = anchor.normalize()
    out: list[Period] = []

    def month_period(y, m):
        s = pd.Timestamp(year=y, month=m, day=1)
        return Period(s, _month_end(s), s.strftime("%b %Y"))

    m = re.search(r"\b(?:last|past|previous)\s+(\d+|[a-z]+)\s+(day|week|month|year)s?\b", ql)
    if m and _num(m.group(1)):
        n, unit = _num(m.group(1)), m.group(2)
        end = a.replace(hour=23, minute=59, second=59)
        if unit == "day":
            start = a - pd.Timedelta(days=n - 1)
        elif unit == "week":
            start = a - pd.Timedelta(days=7 * n - 1)
        elif unit == "month":
            start = a.replace(day=1) - pd.DateOffset(months=n - 1)
        else:
            start = a.replace(month=1, day=1) - pd.DateOffset(years=n - 1)
        return [Period(start, end, f"last {n} {unit}{'s' if n > 1 else ''}")]

    if re.search(r"\b(?:last|previous|prior)\s+month\b", ql):
        s = a.replace(day=1) - pd.DateOffset(months=1)
        return [Period(s, _month_end(s), s.strftime("%b %Y"))]
    if re.search(r"\bthis\s+month\b|\bcurrent\s+month\b", ql):
        s = a.replace(day=1)
        return [Period(s, a.replace(hour=23, minute=59, second=59), s.strftime("%b %Y") + " (so far)")]
    if re.search(r"\blast\s+year\b|\bprevious\s+year\b", ql):
        y = a.year - 1
        return [Period(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31, 23, 59, 59), str(y))]
    if re.search(r"\bthis\s+year\b|\bcurrent\s+year\b", ql):
        return [Period(pd.Timestamp(a.year, 1, 1), a.replace(hour=23, minute=59, second=59), f"{a.year} (so far)")]
    if re.search(r"\blast\s+week\b|\bthis\s+week\b", ql):
        n = 7
        return [Period(a - pd.Timedelta(days=n - 1), a.replace(hour=23, minute=59, second=59), "last 7 days")]
    if re.search(r"\byesterday\b", ql):
        d = a - pd.Timedelta(days=1)
        return [Period(d, d.replace(hour=23, minute=59, second=59), d.strftime("%d %b %Y"))]
    if re.search(r"\btoday\b", ql):
        return [Period(a, a.replace(hour=23, minute=59, second=59), a.strftime("%d %b %Y"))]

    found = []
    for mm in re.finditer(r"\b(" + _MONTH_RX + r")\b(?:[\s,\-']*(20\d{2}))?", ql):
        tok, yr = mm.group(1), mm.group(2)
        if tok == "may" and not yr:
            before = ql[:mm.start()].split()[-1:] or [""]
            if before[0] not in {"in", "for", "of", "during", "since", "from", "to", "and", "vs", "versus", "between"}:
                continue
        mnum = [i for i, name in enumerate(MONTHS, 1) if name.startswith(tok[:3])][0]
        if yr:
            y = int(yr)
        else:
            y = a.year if mnum <= a.month else a.year - 1
        found.append((y, mnum))
    if found:
        if as_range and len(found) >= 2:
            p1, p2 = month_period(*found[0]), month_period(*found[-1])
            if p2.start >= p1.start:
                return [Period(p1.start, p2.end, f"{p1.label} – {p2.label}")]
        seen, res = set(), []
        for y, mnum in found:
            if (y, mnum) not in seen:
                seen.add((y, mnum))
                res.append(month_period(y, mnum))
        return res

    ym = re.search(r"\b(20\d{2})\b", ql)
    if ym:
        y = int(ym.group(1))
        return [Period(pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31, 23, 59, 59), str(y))]
    return out


# --------------------------------------------------------------------------
# Intent detection
# --------------------------------------------------------------------------
FORMAT_PATS = [
    ("pptx", r"\b(ppt|pptx|powerpoint|presentation|slides?|slide deck)\b"),
    ("docx", r"\b(word|docx|word doc(?:ument)?|ms word)\b"),
    ("excel", r"\b(excel|xlsx|xls|spreadsheet)\b"),
    ("pdf", r"\bpdf\b"),
    ("png", r"\b(png|snapshot|image|picture|infographic)\b"),
    ("csv", r"\bcsv\b"),
]


def detect_format(core: str):
    for fmt, pat in FORMAT_PATS:
        if re.search(pat, core):
            return fmt
    return None


def detect_intent(core: str) -> str:
    fmt = detect_format(core)
    verb = re.search(r"\b(download|export|generate|create|make|give|send|get|prepare|need|want|save|build)\b", core)
    if fmt and (verb or len(core.split()) <= 5
                or re.search(r"\b(report|file|sheet|document|summary|snapshot|deck|presentation|slides?|infographic)\b", core)):
        return "download"
    if re.search(r"\b(download|export)\b", core):
        return "report_menu"
    if re.search(r"\bwhat.?if\b|\bscenario\b|\bif (?:we|the|top)\b.*\b(?:reduce|cut|lower|improve)\b", core):
        return "whatif"
    if re.search(r"forecast|predict|projection|\bnext\s+(?:\d+\s+|[a-z]+\s+)?months?\b|upcoming|expected", core):
        return "forecast"
    if re.search(r"\brisk\b|critical stations?|high.?risk|priorit|inspection (?:list|plan)", core):
        return "risk"
    if re.search(r"anomal|spike|unusual|sudden|surge|abnormal|outlier|\bjump", core):
        return "anomaly"
    if re.search(r"chronic|repeat|recurr|persistent|every month|again and again|habitual|frequent(?:ly)?", core):
        return "chronic"
    if re.search(r"\bpending\b|no remarks?|without remarks?|remarks? (?:missing|blank|not given)|not (?:replied|attended)", core):
        return "pending"
    if re.search(r"\bcompare\b|comparison|\bvs\.?\b|versus|difference between", core):
        return "compare"
    if re.search(r"trend|over time|month.?wise|monthly|month by month|timeline|progress|growth|increasing|decreasing|improving|worsening", core):
        return "trend"
    if re.search(r"data (?:up ?to|until|available|range|coverage)|latest (?:record|data)|last updated|how (?:much|old) (?:data|is the data)|date range", core):
        return "datainfo"
    if re.search(r"insight|summari[sz]e|summary|overview|brief(?:ing)?|highlights?|\bstatus\b|key points|executive|tell me about|analy[sz]", core):
        return "insights"
    if re.search(r"\breport\b", core):
        return "report_menu"
    if re.search(r"\b(?:top|bottom|highest|most|worst|maximum|max|leading|biggest|ranking|rank|lowest|least|minimum|fewest|best)\b", core):
        return "top"
    if re.search(r"how many|\bcount\b|\btotal\b|number of|how much|\bsum\b", core):
        return "count"
    if re.search(r"\bwise\b|breakdown|break.?up|distribution|split|\bby\s+(?:station|dept|department|error|jurisdiction|month)|\bper\s+(?:station|dept|department|error|jurisdiction|month)|\bshare\b|percentage", core):
        return "breakdown"
    if re.search(r"\blist\b|show|display|records?|details?|rows|\bfind\b|\bview\b|\bopen\b", core):
        return "list"
    if re.search(r"chart|graph|plot|visual", core):
        return "breakdown"
    return "unknown"


def _find_dim(core: str, default: str = "station") -> str:
    m = re.search(r"\b(?P<d>station|stn|error|department|dept|jurisdiction|month)s?[\s\-]?wise\b", core)
    if m:
        return {"stn": "station", "dept": "dept", "department": "dept", "jurisdiction": "jur",
                "error": "error", "month": "month", "station": "station"}[m.group("d")]
    hits = [(m.start(), m.lastgroup) for m in _NOUN_RX.finditer(core)]
    if not hits:
        return default
    by = re.search(r"\b(?:by|per)\b", core)
    if by:
        after = [h for h in hits if h[0] > by.start()]
        if after:
            return after[0][1]
    trig = re.search(r"\b(?:top|bottom|which|highest|most|lowest|least|worst|best|how many)\b", core)
    if trig:
        after = [h for h in hits if h[0] > trig.start()]
        if after:
            return after[0][1]
    return hits[0][1]


def _parse_n(core: str, default: int = 5) -> int:
    m = re.search(r"\b(?:top|bottom|first|worst|best)\s+(\d+|[a-z]+)\b", core)
    if m and _num(m.group(1)):
        return max(1, min(_num(m.group(1)), 50))
    m = re.search(r"\b(\d+)\s+(?:stations?|errors?|departments?|jurisdictions?|categories|months?)\b", core)
    if m:
        return max(1, min(int(m.group(1)), 50))
    return default


# --------------------------------------------------------------------------
# Small talk
# --------------------------------------------------------------------------
_DATA_SIGNAL = re.compile(
    r"\b(top|bottom|how many|count|total|show|list|compare|trend|forecast|risk|download|export|report|chart|graph|"
    r"station|error|department|jurisdiction|cases?|records?|summary|insights?|month|anomal|pending|teach|reset|"
    r"highest|lowest|most|least|excel|pdf|word|ppt)\b")

SMALLTALK = [
    ("howareyou", r"how (?:are|r) (?:you|u)|how'?s it going|how do you do|kaise ho|kaisa hai|kasa aahes|kya haal|what'?s up|wassup|\bsup\b", [
        "I'm running smoothly, thanks for asking! 😊 How about you, {name}? Ready to dig into some DRISHTI data?",
        "All systems normal on my side! 🚄 What would you like to look at today?",
        "Doing great — no faults on my line today! 😄 What shall we check, {name}?"]),
    ("whoareyou", r"who (?:are|r) (?:you|u)|your name|introduce yourself|about yourself|what are you", [
        "I'm the **DRISHTI Assistant** — a built-in helper for the Safety Branch, Solapur Division. I read the data-logger records "
        "and can answer questions, draw charts, build tables, forecast, and prepare Excel / PDF / Word / PowerPoint reports. 🚄",
        "I'm DRISHTI's chat assistant. Think of me as a colleague who has memorised every row in the data logger sheet, "
        "never takes a tea break, and loves charts. ☕➡️📊"]),
    ("capabilities", r"what can you do|\bhelp\b|capabilit|features|what do you do|how (?:can|do) i use|guide me|commands|\boptions\b|\bmenu\b", None),
    ("thanks", r"\bthank|\bthanks\b|\bthx\b|shukriya|dhanyavad|aabhar|great job|good job|well done|awesome|nice work|excellent|superb|perfect|brilliant", [
        "You're welcome, {name}! 😊 Anything else you'd like to check?",
        "Happy to help! Ask me anything else about the data any time. 👍",
        "Glad that helped! Want me to turn it into a report?"]),
    ("bye", r"\b(?:bye|goodbye|good night|see you|cya|take care|tata)\b", [
        "Goodbye, {name}! Drive safe, keep the trains running. 🚄",
        "See you later! I'll be right here when you need the next report. 👋"]),
    ("joke", r"\bjokes?\b|make me laugh|something funny|\bfunny\b", [
        "Why did the signal go to therapy? It kept flying back to danger. 🚦😄",
        "Why was the track circuit so calm? It never lost its connection. 🛤️",
        "What did the data logger say to the station? \"I've got my eye on you.\" 👀"]),
    ("fact", r"fun fact|did you know|railway fact|something new|surprise me|trivia|something interesting", None),
    ("date", r"what(?:'s| is) (?:the )?(?:date|time|day)|today'?s date|current time|what day", None),
    ("ack", r"^(?:ok|okay|k|fine|cool|hmm+|yes|yeah|yep|sure|alright|got it|understood|nice|good|hm+|haan|ha)\W*$", [
        "👍 What would you like to look at next?",
        "Great! Just tell me what to pull up — a station, a month, an error type…",
        "Sure thing. I'm ready when you are."]),
    ("sorry", r"^(?:sorry|my bad|oops|pardon)", [
        "No problem at all, {name}! 😊",
        "No worries — ask away."]),
    ("tired", r"\bi(?:'m| am) (?:tired|stressed|bored|busy|exhausted)", [
        "Long day? Let me do the heavy lifting — tell me what you need and I'll pull the numbers together. 💪",
        "Take a breath, {name}. Give me a question and I'll have the answer ready in a second. ☕"]),
    ("insult", r"stupid|useless|dumb|idiot|bad bot|you suck|worst bot", [
        "Sorry I let you down, {name}. Please press 👎 on the answer so the team can improve me — and try rephrasing; "
        "type **help** to see what I can answer."]),
]


def _tod_greeting() -> str:
    h = datetime.now(_IST).hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 17 else "Good evening")


# --------------------------------------------------------------------------
# Excel / Word builders (self-contained)
# --------------------------------------------------------------------------
def _build_excel(d: pd.DataFrame) -> bytes:
    out = BytesIO()
    d2 = d.copy()
    if "DATE" in d2.columns:
        d2["DATE"] = d2["DATE"].dt.date
    sheets = {"Records": d2}
    for col, name in (("STATION", "By_Station"), ("DEPARTMENT", "By_Department"),
                      ("ERROR MAIN CATEGORY", "By_Error"), ("JURISDICTION", "By_Jurisdiction")):
        if col in d.columns and not d.empty:
            sheets[name] = d.groupby(col).size().reset_index(name="Cases").sort_values("Cases", ascending=False)
    if "DATE" in d.columns and not d.dropna(subset=["DATE"]).empty:
        m = d.dropna(subset=["DATE"]).set_index("DATE").resample("MS").size().reset_index(name="Cases")
        m["DATE"] = m["DATE"].dt.strftime("%b %Y")
        sheets["Monthly_Trend"] = m.rename(columns={"DATE": "Month"})
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        hf = w.book.add_format({"bold": True, "font_color": "white", "bg_color": "#0277bd",
                                "border": 1, "align": "center", "valign": "vcenter"})
        for name, df in sheets.items():
            nm = name[:31]
            df.to_excel(w, index=False, sheet_name=nm, header=False, startrow=1)
            ws = w.sheets[nm]
            for i, c in enumerate(df.columns):
                ws.write(0, i, str(c), hf)
                width = len(str(c))
                if len(df):
                    width = max(width, int(df[c].astype(str).str.len().max()))
                ws.set_column(i, i, min(width + 2, 45))
            ws.freeze_panes(1, 0)
            if len(df):
                ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    return out.getvalue()


def _build_docx(d: pd.DataFrame, scope_label: str, user_name: str, insight_lines: list[str]) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(0.9)
    navy = RGBColor(0x0D, 0x21, 0x37)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("DRISHTI")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(28), navy
    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run("Data Logger Report of Identified Significant Technical Happenings & Insights")
    r.font.size = Pt(10)
    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run("Safety Branch  •  Central Railway, Solapur Division")
    r.bold, r.font.size = True, Pt(11)

    p = doc.add_paragraph()
    p.add_run(f"Scope: {scope_label}\n").bold = True
    p.add_run(f"Generated: {datetime.now(_IST).strftime('%d %b %Y, %H:%M')} IST   |   By: {user_name or 'DRISHTI Assistant'}")

    def shade(cell, hex_fill):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), hex_fill)
        tcPr.append(shd)

    def add_table(df, headers):
        tb = doc.add_table(rows=1, cols=len(headers))
        tb.style = "Table Grid"
        for i, h in enumerate(headers):
            c = tb.rows[0].cells[i]
            c.text = ""
            run = c.paragraphs[0].add_run(h)
            run.bold, run.font.color.rgb, run.font.size = True, RGBColor(255, 255, 255), Pt(9)
            shade(c, "0D2137")
        for _, row in df.iterrows():
            cells = tb.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(9)
        doc.add_paragraph()

    doc.add_heading("Key insights", level=1)
    for line in insight_lines or ["No data for the selected scope."]:
        doc.add_paragraph(re.sub(r"\*\*", "", line), style="List Bullet")

    def block(col, title, n=10):
        if col in d.columns and not d.empty:
            vc = d[col].dropna().value_counts().head(n)
            tot = len(d)
            tb = pd.DataFrame({title: vc.index.astype(str), "Cases": [f"{v:,}" for v in vc.values],
                               "% of total": [f"{v / tot * 100:.1f}%" for v in vc.values]})
            doc.add_heading(f"Top {title.lower()}", level=1)
            add_table(tb, list(tb.columns))
            return vc
        return None

    vc = block("STATION", "Stations")
    if vc is not None and len(vc):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        v = vc.head(10)[::-1]
        ax.barh(v.index.astype(str), v.values, color="#0277bd")
        ax.set_xlabel("Cases")
        ax.set_title("Top stations by cases", loc="left", fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=160)
        plt.close(fig)
        buf.seek(0)
        doc.add_picture(buf, width=Inches(6.0))
    block("ERROR MAIN CATEGORY", "Error categories")
    block("DEPARTMENT", "Departments")
    block("JURISDICTION", "Jurisdictions", 12)

    f = doc.add_paragraph()
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = f.add_run("CONFIDENTIAL – For Official Use Only  |  Safety Branch, Solapur Division")
    r.italic, r.font.size = True, Pt(8)
    out = BytesIO()
    doc.save(out)
    return out.getvalue()


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class AssistantEngine:
    ENT_KEYS = (("stations", "STATION"), ("errors", "ERROR MAIN CATEGORY"),
                ("depts", "DEPARTMENT"), ("jurs", "JURISDICTION"))

    def __init__(self, df: pd.DataFrame, user_name: str = "", tools: dict | None = None):
        d = df if df is not None else pd.DataFrame()
        if "DATE" in d.columns and not pd.api.types.is_datetime64_any_dtype(d["DATE"]):
            d = d.copy()
            d["DATE"] = pd.to_datetime(d["DATE"], errors="coerce")
        self.df = d
        self.user = (user_name or "").strip()
        self.first = self.user.split()[0] if self.user else "there"
        self.tools = tools or {}

        has_date = "DATE" in d.columns and d["DATE"].notna().any()
        self.anchor = d["DATE"].max() if has_date else pd.Timestamp.now()
        self.first_date = d["DATE"].min() if has_date else self.anchor
        self.partial_last = self.anchor.day < self.anchor.days_in_month

        def uniq(col):
            if col not in d.columns:
                return []
            s = d[col].dropna()
            s = s[s.astype(str).str.strip() != ""]
            return sorted(set(map(str, s.unique().tolist())))

        self.stations = uniq("STATION")
        self.errors = uniq("ERROR MAIN CATEGORY")
        self.depts = uniq("DEPARTMENT")
        self.jurs = uniq("JURISDICTION")
        self.kind_of = {}
        for kind, vals in (("stations", self.stations), ("errors", self.errors),
                           ("depts", self.depts), ("jurs", self.jurs)):
            for v in vals:
                self.kind_of.setdefault(v.upper(), (kind, v))
        self.aliases = self._load_aliases()

    # ------------------------------------------------------------ aliases
    def _load_aliases(self):
        al = {}
        for k, v in DEFAULT_ALIASES.items():
            if v.upper() in self.kind_of:
                al[k] = self.kind_of[v.upper()][1]
        try:
            if os.path.exists(ALIAS_FILE):
                with open(ALIAS_FILE, "r", encoding="utf-8") as f:
                    for k, v in json.load(f).items():
                        if str(v).upper() in self.kind_of:
                            al[str(k).lower()] = self.kind_of[str(v).upper()][1]
        except Exception:
            pass
        return al

    def _teach(self, phrase: str, target: str) -> Reply:
        phrase = phrase.strip().strip("'\"").lower()
        target = target.strip().strip("'\"")
        key = target.upper()
        if key not in self.kind_of:
            pool = list(self.kind_of.keys())
            close = difflib.get_close_matches(key, pool, n=3, cutoff=0.6)
            msg = f"I couldn't find **{target}** in the data."
            if close:
                msg += " Did you mean: " + ", ".join(f"`{self.kind_of[c][1]}`" for c in close) + "?"
            return Reply(msg, intent="teach", understood=False)
        kind, canon = self.kind_of[key]
        self.aliases[phrase] = canon
        try:
            saved = {}
            if os.path.exists(ALIAS_FILE):
                with open(ALIAS_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
            saved[phrase] = canon
            with open(ALIAS_FILE, "w", encoding="utf-8") as f:
                json.dump(saved, f, indent=2, ensure_ascii=False)
            note = ""
        except Exception:
            note = " (I'll remember it for this session only.)"
        label = {"stations": "station", "errors": "error category", "depts": "department", "jurs": "jurisdiction"}[kind]
        return Reply(f"Learned! From now on **\"{phrase}\"** means the {label} **{canon}**.{note}",
                     chips=[f"Top 5 error categories at {canon}" if kind == "stations" else f"Top 5 stations for {canon}"],
                     intent="teach")

    # ------------------------------------------------------------ entities
    def _extract(self, q: str):
        qw = q
        ents = {"stations": [], "errors": [], "depts": [], "jurs": []}

        def add(key, val):
            if val not in ents[key]:
                ents[key].append(val)

        for phrase, target in sorted(self.aliases.items(), key=lambda kv: -len(kv[0])):
            pat = _loose_pat(phrase)
            if pat and pat.search(qw):
                kind, canon = self.kind_of[target.upper()]
                add(kind, canon)
                qw = pat.sub(" ", qw)

        for j in sorted(self.jurs, key=len, reverse=True):
            variants = [j]
            if "(" in j:
                variants.append(j.split("(")[0])
            for v in variants:
                pat = _loose_pat(v)
                if pat and pat.search(qw):
                    add("jurs", j)
                    qw = pat.sub(" ", qw)
                    break

        for e in sorted(self.errors, key=len, reverse=True):
            pat = _loose_pat(e)
            if pat and pat.search(qw):
                add("errors", e)
                qw = pat.sub(" ", qw)
        if not ents["errors"]:
            for e in self._fuzzy_errors(qw):
                add("errors", e)

        for alias, target in DEPT_ALIAS.items():
            pat = re.compile(r"(?<![A-Za-z0-9&/])" + re.escape(alias) + r"(?![A-Za-z0-9&/])", re.I)
            if pat.search(qw):
                for v in self.depts:
                    if any(k in v.upper() for k in DEPT_KEYS[target]):
                        add("depts", v)
                qw = pat.sub(" ", qw)
        for dv in self.depts:
            pat = _loose_pat(dv)
            if pat and pat.search(qw):
                add("depts", dv)
                qw = pat.sub(" ", qw)

        for s in sorted(self.stations, key=len, reverse=True):
            su = s.upper().strip()
            body = re.escape(su)
            guard_l, guard_r = r"(?<![A-Za-z0-9/\-])", r"(?![A-Za-z0-9/\-])"
            if len(su) <= 2 or su in AMBIGUOUS_CODES:
                cs = re.compile(guard_l + body + guard_r)
                prep = re.compile(r"\b(?:at|for|in|station|stn)\s+" + re.escape(s) + r"\b", re.I)
                m = cs.search(qw) or prep.search(qw)
                if m:
                    add("stations", s)
                    qw = (cs.sub(" ", qw) if cs.search(qw) else prep.sub(" ", qw))
            else:
                pat = re.compile(guard_l + body + guard_r, re.I)
                if pat.search(qw):
                    add("stations", s)
                    qw = pat.sub(" ", qw)
        return ents, qw

    def _fuzzy_errors(self, text: str):
        words = re.findall(r"[a-z0-9&]+", text.lower())
        if not words:
            return []
        hits = []
        for e in self.errors:
            vw = [w for w in re.findall(r"[a-z0-9&]+", e.lower()) if len(w) > 2]
            if not vw:
                continue
            matched = sum(1 for w in vw if w in words or difflib.get_close_matches(w, words, n=1, cutoff=0.84))
            frac = matched / len(vw)
            if (len(vw) >= 2 and matched >= 2 and frac >= 0.75) or (len(vw) == 1 and matched == 1):
                hits.append((frac, e))
        hits.sort(reverse=True)
        return [e for _, e in hits[:2]]

    # ------------------------------------------------------------ scope
    def _scope(self, ents, period=None, df=None):
        d = self.df if df is None else df
        for key, col in self.ENT_KEYS:
            if ents.get(key) and col in d.columns:
                d = d[d[col].isin(ents[key])]
        if period is not None and "DATE" in d.columns:
            d = d[(d["DATE"] >= period.start) & (d["DATE"] <= period.end)]
        return d

    def _all_range(self):
        return f"{self.first_date.strftime('%d %b %Y')} – {self.anchor.strftime('%d %b %Y')}"

    def _desc(self, ents, period):
        parts = []
        for key, _ in self.ENT_KEYS:
            if ents.get(key):
                parts.append(", ".join(ents[key]))
        if period:
            parts.append(period.label)
        if not parts:
            return f"across the whole division ({self._all_range()})"
        return "for " + " · ".join(parts)

    def _tail(self, ents, period):
        vals = [v for key, _ in self.ENT_KEYS for v in ents.get(key, [])]
        t = (" for " + ", ".join(vals)) if vals else ""
        if period:
            t += " in " + period.label.replace(" (so far)", "")
        return t

    @staticmethod
    def _prior_period(period: Period) -> Period:
        """Previous period of the same length; calendar-month aligned when the period is whole months."""
        s, e = period.start, period.end
        if s.day == 1 and e.normalize() == _month_end(e).normalize():
            n = (e.year - s.year) * 12 + (e.month - s.month) + 1
            ps = s - pd.DateOffset(months=n)
            return Period(ps, s - pd.Timedelta(seconds=1), "previous period")
        span = e - s + pd.Timedelta(seconds=1)
        return Period(s - span, s - pd.Timedelta(seconds=1), "previous period")

    def _group_counts(self, d, dim):
        if dim == "month":
            s = d.dropna(subset=["DATE"])["DATE"].dt.to_period("M").astype(str)
            return s.value_counts().sort_index()
        col = DIM_COL[dim]
        if col not in d.columns:
            return pd.Series(dtype=int)
        s = d[col]
        s = s[s.notna() & (s.astype(str).str.strip() != "")]
        return s.value_counts()

    def _empty(self, ents, period):
        return Reply(f"I couldn't find any records {self._desc(ents, period)}. 🔍\n\n"
                     f"The data runs from **{self._all_range()}**. Try a different month, or drop one of the filters.",
                     chips=["Top 5 stations", "Summary of last month"], intent="empty", understood=True)

    # ------------------------------------------------------------ monthly helpers
    def _monthly(self, d, start=None, end=None):
        dd = d.dropna(subset=["DATE"])
        start = (start or self.first_date).to_period("M").to_timestamp()
        end = (end or self.anchor).to_period("M").to_timestamp()
        idx = pd.date_range(start, end, freq="MS")
        if dd.empty:
            return pd.Series(0.0, index=idx)
        s = dd.set_index("DATE").resample("MS").size().astype(float)
        return s.reindex(idx, fill_value=0.0)

    def _trim_partial(self, s):
        if len(s) and self.partial_last and s.index[-1] == self.anchor.to_period("M").to_timestamp():
            return s.iloc[:-1]
        return s

    def _forecast(self, s, periods):
        fx = self.tools.get("forecast_series")
        if fx:
            try:
                fc, method, resid = fx(s, periods)
                if len(fc):
                    return fc, method, resid
            except Exception:
                pass
        s = s.dropna().astype(float)
        n = len(s)
        idx = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")
        if n < 4:
            vals, method, resid = np.repeat(float(s.iloc[-1]), periods), "Naive (last month)", float(s.std(ddof=0)) if n > 1 else 0.0
        else:
            tail = s.iloc[-12:]
            x = np.arange(len(tail), dtype=float)
            slope, icpt = np.polyfit(x, tail.values, 1)
            fut = np.arange(len(tail), len(tail) + periods, dtype=float)
            vals = slope * fut + icpt
            resid = float(np.std(tail.values - (slope * x + icpt)))
            method = "Linear trend on last 12 months"
        return pd.Series(np.clip(np.round(vals), 0, None), index=idx), method, resid

    # ------------------------------------------------------------ insights
    def _insight_lines(self, d, ents, period):
        lines = []
        total = len(d)
        if total == 0:
            return lines
        lines.append(f"**{_fmt(total)} cases** {self._desc(ents, period)}.")
        # change vs previous equal period / previous month
        try:
            if period:
                prev = self._prior_period(period)
                prev_n = len(self._scope(ents, prev))
                ch = _pct_change(total, prev_n)
                if ch is not None:
                    lines.append(f"{_arrow(ch)} **{ch:+.1f}%** versus the previous period of the same length ({_fmt(prev_n)} cases).")
            else:
                s = self._trim_partial(self._monthly(d))
                if len(s) >= 2:
                    ch = _pct_change(s.iloc[-1], s.iloc[-2])
                    if ch is not None:
                        lines.append(f"{_arrow(ch)} Last full month ({s.index[-1].strftime('%b %Y')}) had **{_fmt(s.iloc[-1])}** cases, "
                                     f"**{ch:+.1f}%** versus {s.index[-2].strftime('%b %Y')} ({_fmt(s.iloc[-2])}).")
        except Exception:
            pass
        for key, dim, word in (("stations", "station", "station"), ("errors", "error", "error category"),
                               ("depts", "dept", "department"), ("jurs", "jur", "jurisdiction")):
            if ents.get(key):
                continue
            vc = self._group_counts(d, dim)
            if len(vc) == 0:
                continue
            lines.append(f"Top {word}: **{vc.index[0]}** with {_fmt(vc.iloc[0])} cases ({vc.iloc[0] / total * 100:.1f}%).")
            if dim == "station" and len(vc) >= 3:
                t3 = vc.head(3).sum() / total * 100
                lines.append(("⚠️ Concentration alert: " if t3 >= 40 else "Spread: ") +
                             f"top 3 stations account for **{t3:.1f}%** of cases (across {len(vc)} stations).")
        dd = d.dropna(subset=["DATE"])
        if len(dd) >= 7:
            wd = dd["DATE"].dt.day_name().value_counts()
            lines.append(f"Busiest weekday: **{wd.index[0]}** ({wd.iloc[0] / len(dd) * 100:.0f}% of cases).")
            bd = dd["DATE"].dt.date.value_counts()
            lines.append(f"Busiest single day: **{pd.Timestamp(bd.index[0]).strftime('%d %b %Y')}** with {_fmt(bd.iloc[0])} cases.")
        rem = self._remark_col(d)
        if rem:
            blank = self._blank_mask(d[rem]).sum()
            if blank:
                lines.append(f"📝 **{_fmt(blank)}** cases ({blank / total * 100:.1f}%) have no S&T remark yet.")
        return lines

    def _remark_col(self, d):
        for c in d.columns:
            if c.strip().upper().startswith("REMARKS"):
                return c
        return None

    @staticmethod
    def _blank_mask(s):
        t = s.astype(str).str.strip().str.lower()
        return s.isna() | t.isin(["", "nan", "none", "na", "n/a", "-", "nil", "null"])

    # ------------------------------------------------------------ main entry
    def welcome(self) -> Reply:
        return Reply(
            f"{_tod_greeting()}, {self.first}! 👋 I'm the **DRISHTI Assistant**.\n\n"
            "Ask me about stations, departments, error categories, jurisdictions or months — in plain English. "
            "I can show **tables and charts**, spot **spikes and repeat offenders**, **forecast** the next months, "
            "and prepare **Excel / PDF / Word / PowerPoint** reports.\n\nTry one of these, or just type below:",
            chips=["Top 5 stations", "Summary of last month", "Which stations are at risk?",
                   "Forecast next 3 months", "What can you do?"], intent="welcome")

    def respond(self, query: str, ctx: dict | None = None) -> Reply:
        ctx = ctx if ctx is not None else {}
        try:
            return self._respond(query, ctx)
        except Exception as e:  # never crash the dashboard
            return Reply("Oops — something went wrong while working that out. 😅 "
                         "Please try rephrasing, or press 👎 so the team can fix it.\n\n"
                         f"`{type(e).__name__}: {e}`", intent="error", understood=False)

    def _respond(self, query: str, ctx: dict) -> Reply:
        q = (query or "").strip()
        if not q:
            return self.welcome()
        ql = q.lower()

        if re.match(r"^\s*(reset|clear|start over|new chat|forget (this|that))\W*$", ql):
            ctx.clear()
            return Reply("Done — I've cleared the conversation context. What would you like to look at?",
                         chips=["Top 5 stations", "Summary of last month"], intent="reset")

        m = re.match(r"^\s*(?:teach|learn|remember)\s*:?\s*(.+?)\s*(?:=|->|=>|means|is)\s*(.+?)\s*$", q, re.I)
        if m:
            return self._teach(m.group(1), m.group(2))

        # greeting prefix
        greet_rx = r"^\s*(?:hi+|hello+|hey+|hii+|namaste|namaskar|नमस्ते|नमस्कार|good\s+(?:morning|afternoon|evening)|gm)\b[\s,!.]*"
        gm = re.match(greet_rx, ql)
        rest = ql[gm.end():] if gm else ql
        if gm:
            filler = re.sub(r"\b(there|bot|assistant|drishti|ai|sir|madam|friend|buddy|everyone|team|claude)\b|[^\w]", "", rest)
            if not filler or (self.first.lower() != "there" and rest.strip(" ,!.") == self.first.lower()):
                hello = ("नमस्ते" if re.match(r"^\s*(namaste|namaskar|नमस्ते|नमस्कार)", ql) else _tod_greeting())
                return Reply(random.choice([
                    f"{hello}, {self.first}! 😊 How can I help you with DRISHTI today?",
                    f"{hello} {self.first}! 👋 Ready when you are — what shall we look at?",
                    f"Hello {self.first}! 🚄 Ask me about any station, error or month.",
                ]), chips=["Top 5 stations", "Summary of last month", "What can you do?"], intent="greet")
            q, ql = q[gm.end():].strip(), rest

        # other small talk (only if no data words)
        if not _DATA_SIGNAL.search(ql) and len(ql.split()) <= 9:
            for name, pat, replies in SMALLTALK:
                if re.search(pat, ql):
                    return self._smalltalk(name, replies)

        ents, qw = self._extract(q)
        core = qw.lower()
        intent = detect_intent(core)
        periods = parse_periods(ql, self.anchor, as_range=(intent != "compare"))
        period = periods[0] if periods else None
        want_chart = bool(re.search(r"chart|graph|plot|visual|diagram", core))
        has_entities = any(ents.values())

        # follow-ups: only clearly elliptical phrasing ("and for October?", "what about TRD?", "show that as a chart")
        ctx_dim = None
        followup_rx = re.compile(r"^\W*(?:and|also|now|then|same|only|but|what about|how about|what of|for|in|at|during)\b")
        pure_chart = bool(re.match(
            r"^\W*(?:(?:show|make|give|draw|plot|display)\s+)?(?:me\s+)?(?:it|that|this|the same)?\s*"
            r"(?:as|in|with)?\s*(?:a\s+)?(?:chart|graph|plot|visual|diagram)s?\W*$", core.strip()))
        if ctx.get("intent") and pure_chart:
            intent, ents, period = ctx["intent"], ctx.get("ents", ents), ctx.get("period")
            ctx_dim, want_chart = ctx.get("dim"), True
        elif ctx.get("intent") and intent == "unknown" and (has_entities or periods) and followup_rx.match(core):
            intent = ctx["intent"]
            ents = {k: (ents[k] or ctx.get("ents", {}).get(k, [])) for k in ents}
            if not periods and ctx.get("period"):
                period = ctx["period"]
            ctx_dim = ctx.get("dim")

        default_dim = "error" if ents["stations"] else "station"
        dim = _find_dim(core, ctx_dim or default_dim)
        if dim == "station" and ents["stations"] and intent in ("top", "breakdown"):
            dim = "error"
        n = _parse_n(core, ctx.get("n", 5) if ctx_dim else 5)

        if intent == "capabilities" or re.search(r"^\W*(help|menu)\W*$", core):
            return self._smalltalk("capabilities", None)

        handler = {
            "download": lambda: self._h_download(ents, period, detect_format(core)),
            "report_menu": lambda: self._h_report_menu(ents, period),
            "whatif": lambda: self._h_whatif(core, ents, period),
            "forecast": lambda: self._h_forecast(core, ents, period),
            "risk": lambda: self._h_risk(ents, period),
            "anomaly": lambda: self._h_anomaly(ents, dim if dim != "month" else "station"),
            "chronic": lambda: self._h_chronic(ents, period, dim if dim in ("station", "error", "jur", "dept") else "station"),
            "pending": lambda: self._h_pending(ents, period, dim if re.search(r"\bby\b|wise", core) else "jur"),
            "compare": lambda: self._h_compare(ents, periods, dim),
            "trend": lambda: self._h_trend(ents, period),
            "datainfo": lambda: self._h_datainfo(),
            "insights": lambda: self._h_insights(ents, period),
            "top": lambda: self._h_top(ents, period, dim if dim != "month" else "station", n,
                                       bool(re.search(r"\b(bottom|lowest|least|fewest|minimum)\b", core))),
            "breakdown": lambda: self._h_top(ents, period, dim, 25, False, breakdown=True),
            "count": lambda: self._h_count(ents, period, want_chart),
            "list": lambda: self._h_list(ents, period, want_chart),
        }.get(intent)

        if handler is None:
            if has_entities or periods:
                handler = lambda: self._h_list(ents, period, want_chart)
                intent = "list"
            else:
                return self._h_search(core, q)
        reply = handler()
        reply.intent = reply.intent if reply.intent != "unknown" else intent

        if reply.understood and intent not in ("empty", "report_menu", "download", "datainfo", "capabilities", "search"):
            ctx.update({"intent": intent, "ents": ents, "period": period, "dim": dim, "n": n})
        return reply

    # ------------------------------------------------------------ small talk
    def _smalltalk(self, name, replies) -> Reply:
        chips = ["Top 5 stations", "Summary of last month", "What can you do?"]
        if name == "capabilities":
            txt = (
                "Here's what I can do — just type it in your own words:\n\n"
                "**📊 Numbers & rankings**\n"
                "- *Top 10 stations in October* · *Bottom 5 stations* · *How many cases at SUR?*\n"
                "- *Which department has the most Track Circuit Failure?*\n\n"
                "**📈 Trends & predictions**\n"
                "- *Monthly trend for KLBG* · *Forecast next 3 months* · *Compare September vs October*\n"
                "- *What if top 3 stations reduce cases by 30%?*\n\n"
                "**🎯 Safety insights**\n"
                "- *Which stations are at risk?* · *Any unusual spikes?* · *Repeat offenders last 6 months*\n"
                "- *Pending remarks by jurisdiction*\n\n"
                "**📑 Reports & files**\n"
                "- *Excel report for ENGG last 3 months* · *PDF health report* · *Word report* · *PowerPoint summary* · *PNG snapshot*\n\n"
                "**🧠 Teach me**\n"
                "- *teach: sholapur = SUR* — I'll remember your words for places and categories.\n\n"
                "I understand English best (Hindi/Marathi only for greetings and small talk). "
                "Say *reset* to start a fresh conversation."
            )
            return Reply(txt, chips=["Top 5 stations", "Which stations are at risk?", "Any unusual spikes?",
                                     "Forecast next 3 months"], intent="capabilities")
        if name == "fact":
            facts = [
                "The first passenger train in India ran between Bori Bunder (Bombay) and Thane on 16 April 1853 — "
                "and Central Railway grew out of that very line. 🚂",
                "Central Railway's headquarters is the historic Chhatrapati Shivaji Maharaj Terminus in Mumbai. 🏛️",
            ]
            df_fact = self._data_fact()
            if df_fact:
                facts.append(df_fact)
            return Reply(random.choice(facts), chips=chips, intent="fact")
        if name == "date":
            now = datetime.now(_IST)
            return Reply(f"It's **{now.strftime('%A, %d %B %Y')}**, {now.strftime('%I:%M %p')} IST. 🕒 "
                         f"The latest record in the data is dated **{self.anchor.strftime('%d %b %Y')}**.",
                         chips=chips, intent="date")
        txt = random.choice(replies).format(name=self.first)
        return Reply(txt, chips=chips, intent=name)

    def _data_fact(self):
        dd = self.df.dropna(subset=["DATE"]) if "DATE" in self.df.columns else pd.DataFrame()
        if len(dd) < 20:
            return None
        wd = dd["DATE"].dt.day_name().value_counts()
        mo = dd["DATE"].dt.strftime("%B").value_counts()
        return (f"📊 Data fact: **{wd.index[0]}** is the busiest weekday in the data logger records "
                f"({wd.iloc[0] / len(dd) * 100:.0f}% of cases), and **{mo.index[0]}** is the busiest calendar month.")

    # ------------------------------------------------------------ handlers
    def _h_datainfo(self):
        dd = self.df
        txt = (f"The data has **{_fmt(len(dd))} records**, from **{self.first_date.strftime('%d %b %Y')}** to "
               f"**{self.anchor.strftime('%d %b %Y')}**, covering **{len(self.stations)} stations**, "
               f"{len(self.depts)} departments and {len(self.errors)} error categories.")
        if self.partial_last:
            txt += f"\n\nThe latest month ({self.anchor.strftime('%B %Y')}) is still in progress."
        return Reply(txt, chips=["Top 5 stations", "Summary of last month"], intent="datainfo")

    def _h_count(self, ents, period, want_chart):
        d = self._scope(ents, period)
        if d.empty:
            return self._empty(ents, period)
        n = len(d)
        total = len(self.df)
        txt = f"**{_fmt(n)} cases** {self._desc(ents, period)} — {n / total * 100:.1f}% of all {_fmt(total)} records."
        if period:
            prev = self._prior_period(period)
            pn = len(self._scope(ents, prev))
            ch = _pct_change(n, pn)
            if ch is not None:
                txt += f"\n\n{_arrow(ch)} **{ch:+.1f}%** versus the previous period of the same length ({_fmt(pn)} cases)."
        fig = tbl = None
        if want_chart:
            dim = "station" if not ents["stations"] else "error"
            vc = self._group_counts(d, dim).head(10)
            tbl = pd.DataFrame({DIM_LABEL[dim]: vc.index, "Cases": vc.values.astype(int)})
            fig = _hbar(tbl, DIM_LABEL[dim], "Cases", f"Top {DIM_PLURAL[dim]} {self._desc(ents, period)}")
        t = self._tail(ents, period)
        return Reply(txt, tbl, fig, chips=[f"Top 5 stations{t}", f"Monthly trend{t}", f"Excel report{t}"], intent="count")

    def _h_top(self, ents, period, dim, n, ascending, breakdown=False):
        d = self._scope(ents, period)
        if d.empty:
            return self._empty(ents, period)
        vc = self._group_counts(d, dim)
        if dim == "month":
            vc = vc.sort_index()
        elif ascending:
            vc = vc.sort_values(ascending=True)
        if len(vc) == 0:
            return Reply(f"There's no {DIM_LABEL[dim].lower()} information in these records.", intent="top", understood=False)
        total = len(d)
        vc = vc.head(n)
        tbl = pd.DataFrame({DIM_LABEL[dim]: vc.index.astype(str), "Cases": vc.values.astype(int),
                            "Share %": (vc.values / total * 100).round(1)})
        tbl.insert(0, "Rank", range(1, len(tbl) + 1))
        word = "Bottom" if ascending else "Top"
        first = tbl.iloc[0]
        if breakdown:
            head = f"**Cases by {DIM_LABEL[dim].lower()} {self._desc(ents, period)}** ({_fmt(total)} cases in total)."
        else:
            head = f"**{word} {len(tbl)} {DIM_PLURAL[dim]} {self._desc(ents, period)}** ({_fmt(total)} cases in total)."
        body = (f"\n\n**{first[DIM_LABEL[dim]]}** {'has the fewest' if ascending else 'leads'} with "
                f"**{_fmt(first['Cases'])} cases** ({first['Share %']}%).")
        if not ascending and len(tbl) >= 3 and dim != "month":
            body += f" The top 3 together account for **{tbl['Share %'].head(3).sum():.1f}%**."
        scale = {"station": "Blues", "error": "Oranges", "dept": "GnBu", "jur": "Purples", "month": "Blues", "fault": "Reds"}[dim]
        if dim == "month":
            fig = px.bar(tbl, x=DIM_LABEL[dim], y="Cases", text="Cases", color="Cases", color_continuous_scale="Blues")
            fig.update_layout(coloraxis_showscale=False)
            fig = _style(fig, 400, f"Cases by month {self._desc(ents, period)}")
        else:
            fig = _hbar(tbl, DIM_LABEL[dim], "Cases", f"{word} {DIM_PLURAL[dim]} {self._desc(ents, period)}", scale)
        name = str(first[DIM_LABEL[dim]])
        t = self._tail(ents, period)
        if dim == "station":
            chips = [f"Monthly trend for {name}", f"Top 5 error categories for {name}", f"Forecast for {name}",
                     f"Excel report{t}"]
        elif dim == "error":
            chips = [f"Top 5 stations for {name}", f"Monthly trend for {name}", f"Excel report{t}"]
        else:
            chips = [f"Top 5 stations for {name}", f"Top 5 error categories for {name}", f"Excel report{t}"]
        return Reply(head + body, tbl, fig, chips=chips, intent="top")

    def _h_list(self, ents, period, want_chart):
        d = self._scope(ents, period)
        if d.empty:
            return self._empty(ents, period)
        d = d.sort_values("DATE", ascending=False) if "DATE" in d.columns else d
        show = d.head(200).copy()
        if "DATE" in show.columns:
            show["DATE"] = show["DATE"].dt.date
        pref = ["DATE", "STATION", "DEPARTMENT", "JURISDICTION", "ERROR MAIN CATEGORY", "DL FAULT MESSAGE"]
        rem = self._remark_col(show)
        if rem:
            pref.append(rem)
        cols = [c for c in pref if c in show.columns] + [c for c in show.columns if c not in pref and c not in ("MONTH", "YEAR_MONTH")]
        txt = f"Found **{_fmt(len(d))} records** {self._desc(ents, period)}."
        if len(d) > 200:
            txt += " Showing the 200 most recent here — use the Excel report for the full list."
        fig = None
        if want_chart:
            dim = "station" if not ents["stations"] else "error"
            vc = self._group_counts(d, dim).head(10)
            fig = _hbar(pd.DataFrame({DIM_LABEL[dim]: vc.index, "Cases": vc.values.astype(int)}),
                        DIM_LABEL[dim], "Cases", f"Top {DIM_PLURAL[dim]} {self._desc(ents, period)}")
        t = self._tail(ents, period)
        return Reply(txt, show[cols], fig, chips=[f"Top 5 stations{t}", f"Monthly trend{t}", f"Excel report{t}"], intent="list")

    def _h_trend(self, ents, period):
        d = self._scope(ents, period)
        if d.empty:
            return self._empty(ents, period)
        s = self._monthly(d, period.start if period else None, period.end if period else None)
        if len(s) < 2:
            return Reply(f"There's only one month of data {self._desc(ents, period)} ({_fmt(s.sum())} cases), "
                         "so there's no trend to draw yet. Try a longer period.", intent="trend")
        peak = s.idxmax()
        txt = (f"**Monthly trend {self._desc(ents, period)}** — {_fmt(s.sum())} cases over {len(s)} months "
               f"(average **{s.mean():.1f}** per month).\n\nPeak: **{peak.strftime('%b %Y')}** with {_fmt(s.max())} cases.")
        full = self._trim_partial(s)
        if len(full) >= 2:
            ch = _pct_change(full.iloc[-1], full.iloc[-2])
            if ch is not None:
                txt += (f" Last full month ({full.index[-1].strftime('%b %Y')}): **{_fmt(full.iloc[-1])}**, "
                        f"{_arrow(ch)} {ch:+.1f}% vs the month before.")
        if len(full) >= 4:
            tail = full.iloc[-6:]
            slope = np.polyfit(np.arange(len(tail)), tail.values, 1)[0]
            rel = slope / tail.mean() if tail.mean() else 0
            word = "rising 📈" if rel > 0.05 else ("falling 📉" if rel < -0.05 else "fairly stable ➖")
            txt += f"\n\nOver the last {len(tail)} full months the trend is **{word}**."
        if self.partial_last and s.index[-1] == self.anchor.to_period("M").to_timestamp():
            txt += f"\n\n_({s.index[-1].strftime('%b %Y')} is still in progress — data up to {self.anchor.strftime('%d %b')}.)_"
        tbl = pd.DataFrame({"Month": s.index.strftime("%b %Y"), "Cases": s.values.astype(int)})
        fig = px.line(x=s.index, y=s.values, markers=True, labels={"x": "Month", "y": "Cases"})
        fig.update_traces(line=dict(color="#0277bd", width=3), marker=dict(size=8))
        fig = _style(fig, 400, f"Monthly cases {self._desc(ents, period)}")
        t = self._tail(ents, None)
        return Reply(txt, tbl, fig, chips=[f"Forecast next 3 months{t}", f"Top 5 stations{self._tail(ents, period)}", f"Any unusual spikes{t}"],
                     intent="trend")

    def _h_compare(self, ents, periods, dim):
        groups, kind = [], None
        if len(periods) >= 2:
            groups = [(p.label, self._scope(ents, p)) for p in periods[:2]]
            kind = "period"
        else:
            for key, _ in self.ENT_KEYS:
                if len(ents.get(key, [])) >= 2:
                    for v in ents[key][:3]:
                        e2 = {k: list(vv) for k, vv in ents.items()}
                        e2[key] = [v]
                        groups.append((v, self._scope(e2, periods[0] if periods else None)))
                    kind = "entity"
                    break
        if len(groups) < 2:
            return Reply("What should I compare? Give me **two months** (e.g. *compare September vs October*) "
                         "or **two stations / departments / errors** (e.g. *compare SUR vs KLBG*).",
                         chips=["Compare last month vs this month", "Compare SUR vs KLBG"], intent="compare", understood=False)
        if all(g[1].empty for g in groups):
            return self._empty(ents, periods[0] if periods else None)
        totals = [len(g[1]) for g in groups]
        lines = " vs ".join(f"**{g[0]}: {_fmt(len(g[1]))}**" for g in groups)
        txt = f"**Comparison** — {lines} cases."
        ch = _pct_change(totals[1], totals[0])
        if ch is not None:
            txt += f"\n\n{groups[1][0]} is {_arrow(ch)} **{abs(ch):.1f}% {'higher' if ch > 0 else 'lower'}** than {groups[0][0]}."
        if kind == "period":
            cdim = dim if dim not in ("month",) else "station"
            if cdim == "station" and ents["stations"]:
                cdim = "error"
            a = self._group_counts(groups[0][1], cdim)
            b = self._group_counts(groups[1][1], cdim)
            tbl = pd.concat([a.rename(groups[0][0]), b.rename(groups[1][0])], axis=1).fillna(0).astype(int)
            tbl["Change"] = tbl.iloc[:, 1] - tbl.iloc[:, 0]
            tbl = tbl.sort_values(tbl.columns[0], ascending=False).head(12)
            tbl.index.name = DIM_LABEL[cdim]
            tbl = tbl.reset_index()
            if len(tbl):
                up = tbl.sort_values("Change", ascending=False).iloc[0]
                dn = tbl.sort_values("Change").iloc[0]
                if up["Change"] > 0:
                    txt += f"\n\nBiggest increase: **{up[DIM_LABEL[cdim]]}** (+{int(up['Change'])})."
                if dn["Change"] < 0:
                    txt += f" Biggest drop: **{dn[DIM_LABEL[cdim]]}** ({int(dn['Change'])})."
            melt = tbl.melt(id_vars=DIM_LABEL[cdim], value_vars=[groups[0][0], groups[1][0]], var_name="Period", value_name="Cases")
            fig = px.bar(melt, x=DIM_LABEL[cdim], y="Cases", color="Period", barmode="group",
                         color_discrete_sequence=["#90caf9", "#0277bd"])
            fig = _style(fig, 430, f"{groups[0][0]} vs {groups[1][0]} by {DIM_LABEL[cdim].lower()}")
        else:
            fig = go.Figure()
            palette = ["#0277bd", "#ef6c00", "#2e7d32"]
            rows = []
            for i, (name, gd) in enumerate(groups):
                s = self._monthly(gd)
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=name,
                                         line=dict(color=palette[i % 3], width=3)))
                top_e = self._group_counts(gd, "error")
                rows.append({"Item": name, "Total cases": len(gd),
                             "Avg / month": round(s.mean(), 1) if len(s) else 0,
                             "Top error": top_e.index[0] if len(top_e) else "-"})
            fig = _style(fig, 420, "Monthly cases — comparison")
            tbl = pd.DataFrame(rows)
        return Reply(txt, tbl, fig, chips=["Forecast next 3 months", "Top 5 stations"], intent="compare")

    def _h_insights(self, ents, period):
        d = self._scope(ents, period)
        if d.empty:
            return self._empty(ents, period)
        lines = self._insight_lines(d, ents, period)
        dim = "error" if ents["stations"] else "station"
        vc = self._group_counts(d, dim).head(8)
        tbl = pd.DataFrame({DIM_LABEL[dim]: vc.index, "Cases": vc.values.astype(int),
                            "Share %": (vc.values / len(d) * 100).round(1)})
        fig = _hbar(tbl, DIM_LABEL[dim], "Cases", f"Top {DIM_PLURAL[dim]} {self._desc(ents, period)}")
        t = self._tail(ents, period)
        return Reply("**📋 Briefing**\n\n" + "\n".join(f"- {x}" for x in lines), tbl, fig,
                     chips=[f"Word report{t}", f"PDF health report{t}", f"Any unusual spikes", "Which stations are at risk?"],
                     intent="insights")

    def _h_forecast(self, core, ents, period):
        d = self._scope(ents, None)
        if d.empty:
            return self._empty(ents, None)
        m = re.search(r"\bnext\s+(\d+|[a-z]+)\s+months?\b", core) or re.search(r"\b(\d+)\s+months?\b", core)
        h = max(1, min(_num(m.group(1)) or 3, 6)) if m else 3
        hist = self._trim_partial(self._monthly(d))
        if len(hist) < 3:
            return Reply("I need at least **3 full months** of history to forecast. "
                         f"There are only {len(hist)} {self._desc(ents, None)}.", intent="forecast", understood=False)
        fc, method, resid = self._forecast(hist, h)
        lo = [max(0, v - 1.96 * resid * np.sqrt(i + 1)) for i, v in enumerate(fc.values)]
        hi = [v + 1.96 * resid * np.sqrt(i + 1) for i, v in enumerate(fc.values)]
        tbl = pd.DataFrame({"Month": fc.index.strftime("%B %Y"), "Predicted cases": fc.values.astype(int),
                            "Lower": np.round(lo).astype(int), "Upper": np.round(hi).astype(int)})
        last = hist.iloc[-1]
        ch = _pct_change(fc.mean(), last)
        txt = (f"**Forecast {self._desc(ents, None)}** — about **{_fmt(fc.sum())} cases** over the next {h} month{'s' if h > 1 else ''} "
               f"(≈ {fc.mean():.0f} per month")
        txt += f", {_arrow(ch)} {ch:+.1f}% vs last full month's {_fmt(last)})." if ch is not None else ")."
        txt += f"\n\nModel: _{method}_, based on {len(hist)} months of history."
        if len(hist) < 12:
            txt += " ⚠️ That's a short history, so treat this as indicative only."
        fig = go.Figure()
        ax = [hist.index[-1]] + list(fc.index)
        ay = [float(hist.iloc[-1])] + [float(v) for v in fc.values]
        up = [float(hist.iloc[-1])] + hi
        dn = [float(hist.iloc[-1])] + lo
        fig.add_trace(go.Scatter(x=ax + ax[::-1], y=up + dn[::-1], fill="toself", fillcolor="rgba(2,136,209,0.18)",
                                 line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="95% range"))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Actual",
                                 line=dict(color="#0277bd", width=3)))
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines+markers+text", name="Forecast",
                                 line=dict(color="#0288d1", width=3, dash="dash"),
                                 text=[""] + [f"{int(v):,}" for v in fc.values], textposition="top center"))
        fig = _style(fig, 430, f"Forecast {self._desc(ents, None)}")
        t = self._tail(ents, None)
        return Reply(txt, tbl, fig, chips=[f"Monthly trend{t}", "What if top 3 stations reduce cases by 30%?", f"Excel report{t}"],
                     intent="forecast")

    def _h_whatif(self, core, ents, period):
        d = self._scope(ents, None)
        hist = self._trim_partial(self._monthly(d)) if not d.empty else pd.Series(dtype=float)
        if len(hist) < 3:
            return Reply("I need at least 3 full months of history for a what-if scenario.", intent="whatif", understood=False)
        pm = re.search(r"(\d{1,2})\s*%", core)
        red = int(pm.group(1)) if pm else 30
        red = max(5, min(red, 90))
        n = _parse_n(core, 3)
        last_start = hist.index[-1]
        last = d[(d["DATE"] >= last_start) & (d["DATE"] < last_start + pd.DateOffset(months=1))]
        if last.empty or "STATION" not in last.columns:
            return Reply("There are no records in the last full month to base the scenario on.", intent="whatif", understood=False)
        share = (last.groupby("STATION").size() / len(last)).sort_values(ascending=False).head(n)
        fc, method, _ = self._forecast(hist, 3)
        factor = 1.0 - share.sum() * red / 100.0
        scen = (fc * factor).clip(lower=0).round()
        saved = int(fc.sum() - scen.sum())
        txt = (f"**What-if:** if the top **{n}** stations ({', '.join(share.index)}) cut their cases by **{red}%**, "
               f"the next 3 months fall from **{_fmt(fc.sum())}** to **{_fmt(scen.sum())}** — "
               f"about **{_fmt(saved)} fewer cases** ({saved / fc.sum() * 100:.1f}% overall).\n\n"
               f"These stations produced {share.sum() * 100:.1f}% of last full month's cases.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Actual", line=dict(color="#0277bd", width=3)))
        fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Original forecast", line=dict(color="#0288d1", dash="dash", width=2)))
        fig.add_trace(go.Scatter(x=scen.index, y=scen.values, name="Scenario", line=dict(color="#1e8449", width=3)))
        fig = _style(fig, 420, f"If top {n} stations reduce cases by {red}%")
        tbl = pd.DataFrame({"Month": fc.index.strftime("%B %Y"), "Original": fc.values.astype(int), "Scenario": scen.values.astype(int)})
        return Reply(txt, tbl, fig, chips=["Which stations are at risk?", "Forecast next 3 months"], intent="whatif")

    def risk_table(self, d, lookback=90):
        if d.empty or "STATION" not in d.columns:
            return None
        cutoff = self.anchor - pd.Timedelta(days=lookback)
        r = d[d["DATE"] >= cutoff]
        if r.empty:
            return None
        months = pd.period_range(cutoff, self.anchor, freq="M")
        piv = (r.assign(ym=r["DATE"].dt.to_period("M")).groupby(["STATION", "ym"]).size()
               .unstack(fill_value=0).reindex(columns=months, fill_value=0))
        x = np.arange(len(months), dtype=float)
        xm = x - x.mean()
        slope = (piv.values * xm).sum(axis=1) / (xm ** 2).sum() if (xm ** 2).sum() else np.zeros(len(piv))
        out = pd.DataFrame({"STATION": piv.index, "Cases (recent)": piv.sum(axis=1).values, "Trend slope": slope})
        sev_map = self.tools.get("severity_map") or SEVERITY
        if "ERROR MAIN CATEGORY" in r.columns:
            sv = r["ERROR MAIN CATEGORY"].map(lambda v: sev_map.get(str(v).strip().upper(), 1))
            out["Avg severity"] = out["STATION"].map(sv.groupby(r["STATION"]).mean()).fillna(1.0)
        else:
            out["Avg severity"] = 1.0

        def norm(s):
            return pd.Series(50.0, index=s.index) if s.max() == s.min() else 100 * (s - s.min()) / (s.max() - s.min())

        vol, trend, sevs = norm(out["Cases (recent)"]), norm(out["Trend slope"].clip(lower=0)), norm(out["Avg severity"])
        sev_used = out["Avg severity"].nunique() > 1
        w = (0.40, 0.30, 0.30) if sev_used else (0.55, 0.45, 0.0)
        out["Risk score"] = (w[0] * vol + w[1] * trend + w[2] * sevs).round(1)
        out["Band"] = pd.cut(out["Risk score"], bins=[-1, 40, 65, 85, 101], labels=["Low", "Medium", "High", "Critical"]).astype(str)
        out = out.sort_values("Risk score", ascending=False).reset_index(drop=True)
        out.attrs["sev_used"] = sev_used
        out.attrs["lookback"] = lookback
        return out

    def _h_risk(self, ents, period):
        base = self._scope({k: ([] if k == "stations" else v) for k, v in ents.items()}, None)
        rt = self.risk_table(base)
        if rt is None or rt.empty:
            return Reply("There isn't enough recent data (last 90 days) to calculate risk scores.", intent="risk", understood=False)
        wanted = ents.get("stations", [])
        if wanted:
            sel = rt[rt["STATION"].isin(wanted)].copy()
            if sel.empty:
                return Reply(f"No recent cases for {', '.join(wanted)} in the last {rt.attrs['lookback']} days — risk is effectively **Low**.",
                             intent="risk")
            sel.insert(0, "Rank", sel.index + 1)
            r0 = sel.iloc[0]
            txt = (f"**{r0['STATION']}** has a risk score of **{r0['Risk score']}** ({r0['Band']}), "
                   f"ranked **#{int(r0['Rank'])} of {len(rt)}** stations over the last {rt.attrs['lookback']} days "
                   f"({_fmt(r0['Cases (recent)'])} cases, trend {r0['Trend slope']:+.2f}/month).")
            tbl = sel
            fig = None
        else:
            top = rt.head(10).copy()
            top.insert(0, "Rank", range(1, len(top) + 1))
            names = ", ".join(f"**{row['STATION']}** ({row['Risk score']} · {row['Band']})"
                              for _, row in top.head(3).iterrows())
            txt = (f"**Stations needing attention** (last {rt.attrs['lookback']} days, score 0–100): {names}.")
            rising = top[top["Trend slope"] > 0].head(3)["STATION"].tolist()
            if rising:
                txt += f"\n\n📈 Rising fastest: {', '.join(rising)}."
            tbl = top
            fig = px.bar(top.sort_values("Risk score"), x="Risk score", y="STATION", orientation="h", text="Risk score",
                         color="Risk score", color_continuous_scale="RdYlGn_r", range_color=[0, 100])
            fig.update_yaxes(type="category", title="")
            fig.update_layout(coloraxis_showscale=False)
            fig = _style(fig, max(320, 36 * len(top) + 110), "Top stations by risk score")
        txt += ("\n\n_Score = volume + rising trend" + (" + error severity._" if rt.attrs["sev_used"] else
                " (error-severity weighting is off because the severity list doesn't match your error categories)._"))
        first = tbl["STATION"].iloc[0]
        return Reply(txt, tbl, fig, chips=[f"Monthly trend for {first}", f"Top 5 error categories for {first}", "Any unusual spikes"],
                     intent="risk")

    def _h_anomaly(self, ents, dim):
        d = self._scope(ents, None)
        if d.empty:
            return self._empty(ents, None)
        hist_s = self._trim_partial(self._monthly(d))
        if len(hist_s) < 4:
            return Reply("I need at least 4 full months of history to detect spikes.", intent="anomaly", understood=False)
        months = list(hist_s.index)
        last_m, base_m = months[-1], months[-7:-1]
        col = DIM_COL[dim]
        dd = d.dropna(subset=["DATE"]).assign(_m=lambda x: x["DATE"].dt.to_period("M").dt.to_timestamp())
        piv = dd.groupby([col, "_m"]).size().unstack(fill_value=0).reindex(columns=months[-7:], fill_value=0)
        rows = []
        for name, r in piv.iterrows():
            latest = r[last_m]
            base = r[base_m]
            mu, sd = base.mean(), base.std(ddof=0)
            thr = mu + 2 * sd if sd > 0 else max(mu * 2, mu + 3)
            if latest >= 3 and latest > thr:
                rows.append({DIM_LABEL[dim]: name, f"{last_m.strftime('%b %Y')}": int(latest),
                             "Usual / month": round(mu, 1), "Extra cases": round(latest - mu, 1),
                             "Signal": "🆕 New hotspot" if mu == 0 else "🔺 Spike"})
        title = f"in {last_m.strftime('%b %Y')} versus the previous {len(base_m)} months"
        if not rows:
            return Reply(f"✅ **No unusual spikes** {title} {self._desc(ents, None)}. Every {DIM_LABEL[dim].lower()} is within its normal range.",
                         chips=["Which stations are at risk?", "Forecast next 3 months"], intent="anomaly")
        tbl = pd.DataFrame(rows).sort_values("Extra cases", ascending=False).head(15)
        names = ", ".join(f"**{x}**" for x in tbl[DIM_LABEL[dim]].head(4))
        txt = f"⚠️ **{len(tbl)} unusual spike{'s' if len(tbl) > 1 else ''}** {title}: {names}."
        melt = tbl.head(8).melt(id_vars=DIM_LABEL[dim], value_vars=[last_m.strftime("%b %Y"), "Usual / month"],
                                var_name="Measure", value_name="Cases")
        fig = px.bar(melt, x=DIM_LABEL[dim], y="Cases", color="Measure", barmode="group",
                     color_discrete_sequence=["#c62828", "#90a4ae"])
        fig = _style(fig, 400, "Spikes vs usual level")
        first = tbl.iloc[0][DIM_LABEL[dim]]
        chips = [f"Monthly trend for {first}", f"Top 5 error categories for {first}"] if dim == "station" else [f"Top 5 stations for {first}"]
        return Reply(txt, tbl, fig, chips=chips, intent="anomaly")

    def _h_chronic(self, ents, period, dim):
        d = self._scope(ents, None)
        if d.empty:
            return self._empty(ents, None)
        if period is None:
            end = self.anchor.to_period("M").to_timestamp()
            start = end - pd.DateOffset(months=5)
            win_label = f"the last 6 months ({start.strftime('%b %Y')} – {end.strftime('%b %Y')})"
        else:
            start, end = period.start.to_period("M").to_timestamp(), period.end.to_period("M").to_timestamp()
            win_label = period.label
        w = d[(d["DATE"] >= start) & (d["DATE"] <= _month_end(end))]
        n_months = len(pd.date_range(start, end, freq="MS"))
        if w.empty or n_months < 2:
            return self._empty(ents, period)
        col = DIM_COL[dim]
        g = w.dropna(subset=["DATE"]).assign(_m=lambda x: x["DATE"].dt.to_period("M")).groupby(col)
        tbl = pd.DataFrame({"Months active": g["_m"].nunique(), "Total cases": g.size()})
        tbl["Avg / active month"] = (tbl["Total cases"] / tbl["Months active"]).round(1)
        tbl = tbl.sort_values(["Months active", "Total cases"], ascending=False).head(15)
        tbl.index.name = DIM_LABEL[dim]
        tbl = tbl.reset_index()
        every = tbl[tbl["Months active"] == n_months][DIM_LABEL[dim]].tolist()
        txt = f"**Repeat offenders over {win_label}** ({n_months} months)."
        if every:
            txt += (f"\n\n🔁 **{len(every)} {DIM_PLURAL[dim]}** had cases in **every** month: "
                    f"{', '.join(every[:8])}{'…' if len(every) > 8 else ''}.")
        else:
            txt += "\n\nNo single one appeared in every month; the most persistent are listed below."
        fig = px.bar(tbl.sort_values("Months active"), x="Total cases", y=DIM_LABEL[dim], orientation="h",
                     color="Months active", color_continuous_scale="Reds", text="Total cases")
        fig.update_yaxes(type="category", title="")
        fig = _style(fig, max(320, 32 * len(tbl) + 110), f"Persistence over {n_months} months")
        first = str(tbl.iloc[0][DIM_LABEL[dim]])
        return Reply(txt, tbl, fig, chips=[f"Monthly trend for {first}", "Which stations are at risk?"], intent="chronic")

    def _h_pending(self, ents, period, dim):
        d = self._scope(ents, period)
        rem = self._remark_col(d)
        if not rem:
            return Reply("I couldn't find a remarks column in the data, so I can't check pending remarks.", intent="pending", understood=False)
        p = d[self._blank_mask(d[rem])]
        if p.empty:
            return Reply(f"🎉 Every case {self._desc(ents, period)} already has a remark.", intent="pending")
        vc = self._group_counts(p, dim).head(12)
        tbl = pd.DataFrame({DIM_LABEL[dim]: vc.index, "Pending": vc.values.astype(int)})
        txt = (f"**{_fmt(len(p))} cases** {self._desc(ents, period)} have **no remark** yet "
               f"({len(p) / len(d) * 100:.1f}% of {_fmt(len(d))}).\n\n"
               f"_(I treat a blank remarks cell as pending.)_\n\nMost pending: **{tbl.iloc[0][DIM_LABEL[dim]]}** ({_fmt(tbl.iloc[0]['Pending'])}).")
        fig = _hbar(tbl, DIM_LABEL[dim], "Pending", f"Pending remarks by {DIM_LABEL[dim].lower()}", "Reds")
        out = p.sort_values("DATE", ascending=False).head(200).copy()
        out["DATE"] = out["DATE"].dt.date
        pref = [c for c in ["DATE", "STATION", "DEPARTMENT", "JURISDICTION", "ERROR MAIN CATEGORY", "DL FAULT MESSAGE"] if c in out.columns]
        file = {"name": f"DRISHTI_Pending_Remarks_{datetime.now(_IST).strftime('%d%b%Y_%H%M')}.csv",
                "data": p.assign(DATE=p["DATE"].dt.date).to_csv(index=False).encode("utf-8"), "mime": "text/csv"}
        return Reply(txt, tbl, fig, file=file, chips=["Pending remarks by department", "Pending remarks by station"], intent="pending")

    # ------------------------------------------------------------ downloads
    def _h_report_menu(self, ents, period):
        t = self._tail(ents, period)
        return Reply("Sure! Which format would you like? 👇", chips=[f"Excel report{t}", f"PDF health report{t}",
                     f"Word report{t}", f"PowerPoint summary{t}", f"PNG snapshot{t}"], intent="report_menu")

    def _h_download(self, ents, period, fmt):
        d = self._scope(ents, period)
        if d.empty:
            return self._empty(ents, period)
        has = d["DATE"].notna().any()
        f = (period.start if period else (d["DATE"].min() if has else self.first_date)).date()
        t = (min(period.end, self.anchor) if period else (d["DATE"].max() if has else self.anchor)).date()
        stamp = datetime.now(_IST).strftime("%d%b%Y_%H%M")
        tag = (ents["stations"] + ents["depts"] + ents["errors"] + ents["jurs"] or ["ALL"])[0]
        tag = re.sub(r"[^A-Za-z0-9]+", "-", str(tag))[:20]
        scope = self._desc(ents, period)
        mimes = {"excel": ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                 "pdf": ("pdf", "application/pdf"),
                 "pptx": ("pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
                 "png": ("png", "image/png"),
                 "docx": ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                 "csv": ("csv", "text/csv")}
        ext, mime = mimes[fmt]
        try:
            if fmt == "csv":
                dd = d.copy()
                dd["DATE"] = dd["DATE"].dt.date
                data = dd.to_csv(index=False).encode("utf-8")
            elif fmt == "excel":
                tool = self.tools.get("excel")
                data = tool(d) if tool else _build_excel(d)
            elif fmt == "docx":
                data = _build_docx(d, f"{scope} · {f.strftime('%d %b %Y')} – {t.strftime('%d %b %Y')}", self.user,
                                   self._insight_lines(d, ents, period))
            else:
                tool = self.tools.get(fmt)
                if not tool:
                    return Reply(f"The **{fmt.upper()}** export isn't connected to the assistant yet (see the integration step). "
                                 "I can give you Excel, Word or CSV right now.",
                                 chips=[f"Excel report{self._tail(ents, period)}", f"Word report{self._tail(ents, period)}"],
                                 intent="download", understood=False)
                data = tool(d, pd.Timestamp(f), pd.Timestamp(t))
            data = data.getvalue() if hasattr(data, "getvalue") else data
        except ImportError as e:
            if "docx" in str(e).lower():
                return Reply("Word export needs the **python-docx** package. Add `python-docx` to your `requirements.txt`, "
                             "redeploy, and ask again. Meanwhile I can give you Excel or PDF.",
                             chips=[f"Excel report{self._tail(ents, period)}"], intent="download", understood=False)
            raise
        name = f"DRISHTI_{ {'excel': 'Report', 'docx': 'Report', 'pdf': 'HealthReport', 'pptx': 'Executive_Summary', 'png': 'Snapshot', 'csv': 'Data'}[fmt] }_{tag}_{stamp}.{ext}"
        label = {"excel": "Excel", "docx": "Word", "pdf": "PDF", "pptx": "PowerPoint", "png": "PNG", "csv": "CSV"}[fmt]
        return Reply(f"✅ Your **{label}** file is ready — {_fmt(len(d))} cases {scope}. "
                     "Use the download button below.",
                     file={"name": name, "data": data, "mime": mime},
                     chips=[f"Summary{self._tail(ents, period)}", "Top 5 stations"], intent="download")

    # ------------------------------------------------------------ free-text fallback
    def _has_text_tokens(self, core):
        toks = [w for w in re.findall(r"[a-z0-9]{3,}", core) if w not in STOPWORDS]
        return len(toks) >= 2 and not re.search(r"\b(and|for|in|at|what about|how about)\b", core)

    def _h_search(self, core, q):
        toks = [w for w in re.findall(r"[a-z0-9&]{3,}", core) if w not in STOPWORDS]
        text_cols = [c for c in ("STATION", "DL FAULT MESSAGE", "ERROR MAIN CATEGORY", "JURISDICTION", "DEPARTMENT")
                     if c in self.df.columns]
        rem = self._remark_col(self.df)
        if rem:
            text_cols.append(rem)
        if toks and text_cols:
            blob = self.df[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
            mask = pd.Series(True, index=self.df.index)
            for tk in toks:
                mask &= blob.str.contains(re.escape(tk), na=False)
            res = self.df[mask]
            if res.empty and len(toks) > 1:
                mask = pd.Series(False, index=self.df.index)
                for tk in toks:
                    mask |= blob.str.contains(re.escape(tk), na=False)
                res = self.df[mask]
                note = " (matching **any** of your words)"
            else:
                note = ""
            if not res.empty and len(res) < len(self.df):
                ents = {k: [] for k, _ in self.ENT_KEYS}
                vc = self._group_counts(res, "station").head(8)
                tbl = pd.DataFrame({"Station": vc.index, "Cases": vc.values.astype(int)})
                show = res.sort_values("DATE", ascending=False).head(100).copy()
                show["DATE"] = show["DATE"].dt.date
                txt = (f"I didn't recognise a specific command, so I searched the records for **{' '.join(toks)}**{note}: "
                       f"found **{_fmt(len(res))} matching records**.")
                fig = _hbar(tbl, "Station", "Cases", "Matches by station") if len(tbl) else None
                return Reply(txt, show, fig, chips=["Top 5 stations", "What can you do?"], intent="search")
        pool = self.stations + self.errors + self.depts
        words = re.findall(r"[a-zA-Z&]{3,}", q)
        guess = []
        for w in words:
            guess += difflib.get_close_matches(w.upper(), [p.upper() for p in pool], n=1, cutoff=0.75)
        txt = ("Hmm, I didn't quite get that. 🤔 I'm best with questions about stations, departments, error categories, "
               "jurisdictions and months.")
        if guess:
            txt += f"\n\nDid you mean: {', '.join(f'`{g}`' for g in dict.fromkeys(guess))}?"
        txt += "\n\nTry something like:\n- *Top 5 stations in October*\n- *Monthly trend for KLBG*\n- *Which stations are at risk?*\n\nType **help** for the full list."
        return Reply(txt, chips=["Top 5 stations", "Summary of last month", "What can you do?"], intent="unknown", understood=False)


# --------------------------------------------------------------------------
# Training log
# --------------------------------------------------------------------------
def log_interaction(user, query, reply: Reply, fb=None):
    row = {"ts": datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S"), "user": user or "", "query": query,
           "intent": reply.intent, "understood": reply.understood, "feedback": fb or ""}
    try:
        new = not os.path.exists(LOG_FILE)
        pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=new, index=False)
    except Exception:
        pass
    return row


# --------------------------------------------------------------------------
# Streamlit tab
# --------------------------------------------------------------------------
def render_assistant_tab(df_all, df_filtered, user_name, tools=None, show_training_log=True):
    """Call inside `with tab_assistant:` in the main app."""
    import streamlit as st

    ss = st.session_state
    ss.setdefault("ast_msgs", [])
    ss.setdefault("ast_ctx", {})
    ss.setdefault("ast_pending", None)
    ss.setdefault("ast_log", [])

    st.subheader("🤖 DRISHTI Assistant")
    st.caption("Free • unlimited • runs inside this app, so your data never leaves it. "
               "Ask in plain English — e.g. *top 5 stations in October*, *forecast next 3 months*, *word report for ENGG*.")

    c1, c2 = st.columns([4, 1])
    with c1:
        use_f = st.checkbox("Answer using my current dashboard filters", value=False, key="ast_use_filters",
                            help="Off = the assistant looks at ALL records, ignoring the filters above.")
    with c2:
        if st.button("🗑️ Clear chat", use_container_width=True, key="ast_clear"):
            ss.ast_msgs, ss.ast_ctx, ss.ast_pending = [], {}, None
            st.rerun()

    df = df_filtered if use_f else df_all
    if df is None or df.empty:
        st.warning("No data available for the assistant.")
        return

    sig = (len(df), str(df["DATE"].min()) if "DATE" in df.columns else "", str(df["DATE"].max()) if "DATE" in df.columns else "",
           int(pd.util.hash_pandas_object(df.index).sum()), use_f, user_name)
    if ss.get("ast_engine_sig") != sig:
        ss.ast_engine = AssistantEngine(df, user_name, tools)
        ss.ast_engine_sig = sig
    engine: AssistantEngine = ss.ast_engine

    if not ss.ast_msgs:
        w = engine.welcome()
        ss.ast_msgs.append({"role": "assistant", "q": "", **w.__dict__})

    last_idx = len(ss.ast_msgs) - 1
    for i, m in enumerate(ss.ast_msgs):
        with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🚄"):
            if m["role"] == "user":
                st.markdown(m["text"])
                continue
            st.markdown(m["text"])
            if m.get("fig") is not None:
                st.plotly_chart(m["fig"], use_container_width=True, key=f"ast_fig_{i}")
            if m.get("table") is not None and len(m["table"]):
                st.dataframe(m["table"], use_container_width=True, hide_index=True)
            if m.get("file"):
                f = m["file"]
                st.download_button(f"⬇️ Download {f['name']}", data=f["data"], file_name=f["name"], mime=f["mime"],
                                   key=f"ast_dl_{i}", type="primary")
            if i > 0:
                fb = m.get("fb")
                if fb:
                    st.caption("Thanks for the feedback 🙏")
                else:
                    b1, b2, _ = st.columns([1, 1, 12])
                    if b1.button("👍", key=f"ast_up_{i}"):
                        m["fb"] = "up"
                        log_interaction(user_name, m.get("q", ""), Reply(m["text"], intent=m.get("intent", ""),
                                        understood=m.get("understood", True)), "up")
                        st.rerun()
                    if b2.button("👎", key=f"ast_dn_{i}"):
                        m["fb"] = "down"
                        log_interaction(user_name, m.get("q", ""), Reply(m["text"], intent=m.get("intent", ""),
                                        understood=m.get("understood", True)), "down")
                        st.rerun()
            if i == last_idx and m.get("chips"):
                cols = st.columns(min(len(m["chips"]), 4))
                for j, chip in enumerate(m["chips"][:8]):
                    if cols[j % len(cols)].button(chip, key=f"ast_chip_{i}_{j}"):
                        ss.ast_pending = chip
                        st.rerun()

    q = st.chat_input("Ask DRISHTI anything… e.g. “top 5 stations last month”")
    if ss.ast_pending:
        q, ss.ast_pending = ss.ast_pending, None
    if q:
        ss.ast_msgs.append({"role": "user", "text": q})
        reply = engine.respond(q, ss.ast_ctx)
        ss.ast_msgs.append({"role": "assistant", "q": q, **reply.__dict__})
        ss.ast_msgs = ss.ast_msgs[-40:]
        log_interaction(user_name, q, reply)
        st.rerun()

    if show_training_log:
        with st.expander("🛠️ Training log — questions the assistant couldn't answer or got 👎"):
            try:
                log = pd.read_csv(LOG_FILE)
                bad = log[(log["understood"] == False) | (log["feedback"] == "down")]  # noqa: E712
                st.caption("Use these to add new rules, or teach synonyms with **teach: phrase = VALUE**.")
                st.dataframe(bad.tail(100), use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download full chat log (CSV)", data=log.to_csv(index=False).encode("utf-8"),
                                   file_name="drishti_assistant_log.csv", mime="text/csv", key="ast_log_dl")
            except Exception:
                st.caption("Nothing logged yet.")


# --------------------------------------------------------------------------
# INTEGRATION (3 edits in the main app)
# --------------------------------------------------------------------------
# 1) Near the other imports:
#       from drishti_assistant import render_assistant_tab
#
# 2) Replace the st.tabs line with:
#       tab_overview, tab_forecast, tab_map, tab_assistant = st.tabs(
#           ["📊 Overview Dashboard", "🔮 Forecast (3 Months)", "🗺️ Map View", "🤖 Assistant"])
#
# 3) Just before the final st.caption("🚄 DRISHTI | Safety Branch ..."):
#       with tab_assistant:
#           render_assistant_tab(
#               df_original, filtered_df, st.session_state.user_name,
#               tools={
#                   "pdf":  lambda d, f, t: generate_one_page_report(d, f, t, st.session_state.user_name),
#                   "pptx": lambda d, f, t: generate_executive_summary_pptx(d, f, t, st.session_state.user_name),
#                   "png":  lambda d, f, t: generate_summary_image(d, f, t, st.session_state.user_name),
#                   "forecast_series": forecast_series,
#               },
#           )
# And add `python-docx` to requirements.txt (only needed for Word reports).
