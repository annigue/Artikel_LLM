#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import datetime as dt

import streamlit as st
import streamlit.components.v1 as components

# Optional: .env laden, falls vorhanden
try:
    from dotenv import load_dotenv  # pip install python-dotenv (optional)
    load_dotenv()
except Exception:
    pass

# Unsere Pipeline importieren
try:
    from humanizer_claude import generate_article
except ImportError as e:
    st.error("Konnte `humanizer_claude` nicht importieren. "
             "Lege `app.py` in den gleichen Ordner wie `humanizer_claude.py`.")
    st.stop()

# ---------- Utils ----------

def extract_slug(md: str) -> str:
    m = re.search(r'^---[\s\S]*?\bslug:\s*"?(?P<slug>[^"\n]+)"?[\s\S]*?---', md, re.MULTILINE)
    if m and m.group("slug").strip():
        return m.group("slug").strip()
    # Fallback: H1 -> slug
    m2 = re.search(r'^#\s+(.+)$', md, re.MULTILINE)
    title = (m2.group(1) if m2 else "artikel").lower()
    slug = re.sub(r'[^a-z0-9\-]+', '-', re.sub(r'\s+', '-', title))
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug or "artikel"

def copy_to_clipboard_button(text: str, label: str = "In Zwischenablage kopieren"):
    # Kleines JS-Snippet, um Text in die Zwischenablage zu legen
    clicked = st.button(label, type="secondary")
    if clicked:
        components.html(
            f"""
            <script>
                const txt = {json.dumps(text)};
                navigator.clipboard.writeText(txt);
            </script>
            """,
            height=0,
        )
        st.toast("Kopiert.", icon="✅")

# ---------- App-Layout ----------

st.set_page_config(
    page_title="Camping-Rezept-Generator",
    page_icon="🍳",
    layout="wide",
)

st.title("🍳 Camping-Rezept-Generator (Claude)")

with st.sidebar:
    st.subheader("🔑 API & Modell")
    api_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    st.caption("ANTHROPIC_API_KEY in der Umgebung wird verwendet.")
    if not api_ok:
        st.error("ANTHROPIC_API_KEY nicht gefunden. Bitte per Umgebungsvariable setzen.")
    model = st.selectbox(
        "Modell",
        [
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
        ],
        index=0,
    )
    st.divider()
    st.subheader("⚙️ Länge & Optionen")
    col_len1, col_len2 = st.columns(2)
    with col_len1:
        min_words = st.number_input("Min. Wörter", min_value=400, max_value=3000, value=700, step=50)
    with col_len2:
        max_words = st.number_input("Max. Wörter", min_value=min_words, max_value=6000, value=1000, step=50)

    show_draft = st.toggle("Draft zusätzlich anzeigen", value=False)
    st.caption("Der Draft ist die unveredelte Rohfassung vor Clean/Style-Pass.")

st.markdown("#### ✍️ Parameter")

with st.form("article_form"):
    topic = st.text_input("Thema / Titel-Idee*", placeholder="z. B. Shakshuka unterwegs im Camper")
    details = st.text_input(
        "Pflichtdetails* (kommasepariert)",
        placeholder="z. B. Campingkocher, reife Tomaten, 12–14 Min, wenig Abwasch",
    )
    col_seo1, col_seo2 = st.columns(2)
    with col_seo1:
        primary_kw = st.text_input("Primär-Keyword (optional, Standard = Thema)", value="")
    with col_seo2:
        secondary_kws = st.text_input("Sekundär-Keywords (optional, kommagetrennt)", value="")

    col_trip1, col_trip2 = st.columns(2)
    with col_trip1:
        destination = st.text_input("Reiseziel (optional)", placeholder="z. B. Israel")
    with col_trip2:
        travel_angle = st.selectbox(
            "Reiseperspektive",
            ["Vanlife/Rundreise", "Roadtrip", "Städtetrip", "Campen", "Trekking"],
            index=0,
        )

    st.caption("Felder mit * sind Pflicht.")
    submitted = st.form_submit_button("🧪 Artikel generieren", type="primary")

# ---------- Aktion ----------

if submitted:
    if not topic or not details:
        st.error("Bitte **Thema** und **Pflichtdetails** ausfüllen.")
        st.stop()

    with st.spinner("Generiere Artikel …"):
        try:
            result = generate_article(
                topic=topic,
                details=details,
                primary_kw=primary_kw,
                secondary_kws=secondary_kws,
                model=model,
                min_words=int(min_words),
                max_words=int(max_words),
                destination=destination,
                travel_angle=travel_angle,
            )
        except Exception as e:
            st.error(f"Fehler beim Generieren: {e}")
            st.stop()

    final_md = result.get("final", "").strip()
    draft_md = result.get("draft", "").strip()
    metrics = result.get("metrics", {})
    ok = result.get("passed_heuristics", False)

    if not final_md:
        st.error("Kein Text erhalten. Bitte Eingaben prüfen und erneut versuchen.")
        st.stop()

    # Anzeige
    slug = extract_slug(final_md)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"{slug or 'artikel'}-{ts}.md"

    st.success(f"Fertig. Heuristiken/Checks ok: **{ok}**")

    tabs = st.tabs(["📰 Gerenderte Ansicht", "🧾 Roh-Markdown", "📊 Metriken", "🧪 Draft" if show_draft else ""])
    # Entferne leere Tabs (Streamlit benötigt fixe Länge)
    tabs = [t for t in tabs if t]

    with tabs[0]:
        st.markdown(final_md)

        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
        with dl_col1:
            st.download_button("⬇️ Markdown herunterladen", data=final_md.encode("utf-8"),
                               file_name=file_name, mime="text/markdown")
        with dl_col2:
            copy_to_clipboard_button(final_md, "📋 Kopieren")

    with tabs[1]:
        st.text_area("Markdown", value=final_md, height=500)

    if len(tabs) >= 3:
        with tabs[2]:
            st.json(metrics, expanded=False)

    if show_draft and len(tabs) >= 4:
        with tabs[3]:
            st.text_area("Draft (Rohfassung)", value=draft_md, height=400)

    st.info("Tipp: Nutze die Download-Schaltfläche für die saubere Weitergabe oder den Kopieren-Button für die Zwischenablage.")
else:
    st.caption("Hinweis: Zeitangaben in **Pflichtdetails** (z. B. „12–14 Min“) werden in der Plausibilitäts-Prüfung berücksichtigt.")
