"""
drishti_assistant_ui.py
Compact chat UI for the DRISHTI Assistant.
Keeps drishti_assistant.py (engine) untouched and only replaces the tab layout.
Main app:  from drishti_assistant_ui import render_assistant_tab
"""
import pandas as pd

from drishti_assistant import (
    AssistantEngine, Reply, _AST_CSS, _get_llm, _tod_greeting, log_interaction,
)

_UI_CSS = """
<style>
.ast-title{margin:0 !important;font-size:1.25rem !important;}
.ast-status{padding:4px 10px;font-size:.9rem;margin:0;}
div[data-testid="stChatMessage"]{padding:10px 14px !important;margin-bottom:8px !important;}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *{font-size:.98rem !important;line-height:1.5 !important;}
</style>
"""


def render_assistant_tab(df_all, df_filtered, user_name, tools=None, show_training_log=True):
    import streamlit as st

    ss = st.session_state
    ss.setdefault("ast_msgs", [])
    ss.setdefault("ast_ctx", {})
    ss.setdefault("ast_pending", None)
    st.markdown(_AST_CSS, unsafe_allow_html=True)
    st.markdown(_UI_CSS, unsafe_allow_html=True)

    use_f = bool(ss.get("ast_use_filters", False))
    df = df_filtered if use_f else df_all
    llm = _get_llm(st)
    if df is None or df.empty:
        st.warning("No data available for the assistant.")
        return

    sig = (len(df), str(df["DATE"].min()) if "DATE" in df.columns else "",
           str(df["DATE"].max()) if "DATE" in df.columns else "",
           int(pd.util.hash_pandas_object(df.index).sum()), use_f, user_name, llm.ready())
    if ss.get("ast_engine_sig") != sig:
        ss.ast_engine = AssistantEngine(df, user_name, tools, llm=llm)
        ss.ast_engine_sig = sig
        ss.pop("ast_questions", None)
    engine = ss.ast_engine
    if "ast_questions" not in ss:
        ss.ast_questions = engine.sample_questions()

    # ---------------- compact top bar ----------------
    kind, html = llm.status()
    badge = {"ok": "🟢 AI ready", "off": "⚪ Rules only", "warn": "🟠 AI limited"}.get(kind, "")
    t1, t2, t3, t4 = st.columns([4, 1.7, 1.7, 1.2])
    with t1:
        st.markdown(f'<div class="ast-title">🤖 DRISHTI Assistant '
                    f'<span class="ast-ver">{badge}</span></div>', unsafe_allow_html=True)
    with t2:
        with st.popover("💡 Questions", use_container_width=True):
            cats = list(ss.ast_questions.keys())
            cat = st.selectbox("Category", cats, key="ast_cat", label_visibility="collapsed")
            for qi, qq in enumerate(ss.ast_questions[cat]):
                if st.button(qq, key=f"astq_{qi}", use_container_width=True):
                    ss.ast_pending = qq
                    st.rerun()
    with t3:
        with st.popover("⚙️ Settings", use_container_width=True):
            st.checkbox("Use my dashboard filters", value=False, key="ast_use_filters",
                        help="Off = all records, ignoring the filters above.")
            st.checkbox("🤖 Use AI for open questions", value=True, key="ast_ai_on",
                        disabled=not llm.configured())
            st.checkbox("Gemini as backup", value=True, key="ast_gemini_on", disabled=not llm.gemini_key)
            st.markdown(f'<div class="ast-status {kind}">{html}</div>', unsafe_allow_html=True)
            st.caption(f"AI answers this session: {llm.session_calls} / {llm.max_calls}. "
                       "Only summary numbers are sent, never raw records.")
            st.caption("Tip: `teach: sholapur = SUR` teaches a word; `reset` starts over.")
    with t4:
        if st.button("🗑️ Clear", use_container_width=True, key="ast_clear"):
            ss.ast_msgs, ss.ast_ctx, ss.ast_pending = [], {}, None
            st.rerun()

    if not ss.ast_msgs:
        w = engine.welcome()
        w.text = (f"{_tod_greeting()}, {engine.first}! 👋 Ask me about stations, errors, departments "
                  "or months. Tap **💡 Questions** for ideas.")
        ss.ast_msgs.append({"role": "assistant", "q": "", **w.__dict__})

    # ---------------- scrollable chat box; input sits right below it ----------------
    box = st.container(height=520, border=True)
    last_idx = len(ss.ast_msgs) - 1
    with box:
        for i, m in enumerate(ss.ast_msgs):
            with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🚄"):
                if m["role"] == "user":
                    st.markdown(m["text"])
                    continue
                st.markdown(m["text"])
                if m.get("fig") is not None:
                    st.plotly_chart(m["fig"], use_container_width=True, key=f"ast_fig_{i}")
                if m.get("table") is not None and len(m["table"]):
                    with st.expander(f"📋 Table ({len(m['table'])} rows)", expanded=(i == last_idx)):
                        st.dataframe(m["table"], use_container_width=True, hide_index=True, height=260)
                if m.get("file"):
                    f = m["file"]
                    st.download_button(f"⬇️ Download {f['name']}", data=f["data"], file_name=f["name"],
                                       mime=f["mime"], key=f"ast_dl_{i}", type="primary")
                if i == last_idx and i > 0:
                    cols = st.columns([1, 1, 8])
                    if m.get("fb"):
                        cols[2].caption("Thanks for the feedback 🙏")
                    else:
                        for c, key, val in ((cols[0], f"ast_up_{i}", "up"), (cols[1], f"ast_dn_{i}", "down")):
                            if c.button("👍" if val == "up" else "👎", key=key):
                                m["fb"] = val
                                log_interaction(user_name, m.get("q", ""),
                                                Reply(m["text"], intent=m.get("intent", ""),
                                                      understood=m.get("understood", True)), val)
                                st.rerun()
                if i == last_idx and m.get("chips"):
                    chips = m["chips"][:3]
                    cols = st.columns(len(chips))
                    for j, chip in enumerate(chips):
                        if cols[j].button(chip, key=f"ast_chip_{i}_{j}", use_container_width=True):
                            ss.ast_pending = chip
                            st.rerun()

    q = st.chat_input("Type your question… e.g. top 5 stations last month")
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
        with st.expander("🛠️ Training log"):
            try:
                from drishti_assistant import LOG_FILE
                log = pd.read_csv(LOG_FILE)
                bad = log[(log["understood"] == False) | (log["feedback"] == "down")]  # noqa: E712
                st.dataframe(bad.tail(100), use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download chat log (CSV)", data=log.to_csv(index=False).encode("utf-8"),
                                   file_name="drishti_assistant_log.csv", mime="text/csv", key="ast_log_dl")
            except Exception:
                st.markdown("Nothing logged yet.")
