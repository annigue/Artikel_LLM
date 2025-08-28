#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit-App: Camping-Rezept-Generator (Claude)
- Erfordert: streamlit, python-dotenv (optional), markdown, pyyaml
  pip install streamlit python-dotenv markdown pyyaml
"""

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
    from humanizer_claude import generate_article  # muss bg_mode unterstützen
except ImportError as e:
    st.error("Konnte `humanizer_claude` nicht importieren. "
             "Lege `app.py` in den gleichen Ordner wie `humanizer_claude.py`.")
    st.stop()

# HTML/Frontmatter-Helfer
try:
    import markdown as _md
except Exception:
    _md = None

try:
    import yaml as _yaml
except Exception:
    _yaml = None


# ---------- Utils ----------

FRONTMATTER_RE = re.compile(r"^---\s*\n([\s\S]*?)\n---\s*", re.MULTILINE)

def split_frontmatter(md_text: str):
    """Gibt (meta_dict, body_md) zurück."""
    m = FRONTMATTER_RE.match(md_text.strip())
    if not m:
        return {}, md_text
    raw_yaml = m.group(1)
    body_md = md_text[m.end():]
    meta = {}
    if _yaml:
        try:
            meta = _yaml.safe_load(raw_yaml) or {}
        except Exception:
            meta = {}
    return meta, body_md

def remove_leading_h1(md_body: str) -> str:
    """Erste H1 (# ...) entfernen (WP setzt Titel separat)."""
    lines = md_body.lstrip().splitlines()
    if lines and re.match(r"^\s*#\s+.+", lines[0]):
        return "\n".join(lines[1:]).lstrip()
    return md_body

def markdown_to_wp_html(md_body: str) -> str:
    """Markdown → HTML (für Gutenberg/Klassik-Editor geeignet)."""
    if not _md:
        raise RuntimeError("Paket 'markdown' fehlt. Installiere: pip install markdown")
    return _md.markdown(md_body, extensions=["extra", "sane_lists"])

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

    st.subheader("🧩 WordPress-Export")
    wp_keep_h1 = st.toggle("H1 im HTML belassen (sonst entfernt)", value=False)
    emit_meta_json = st.toggle("meta.json erzeugen", value=True)

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

    # Steuerung für „Hintergrund & Tipps“
    bg_mode_label = st.selectbox(
        "Start von „Hintergrund & Tipps“",
        ["Automatisch", "Tipps zuerst (keine Anekdote)", "Story/Brücke"],
        index=0,
        help="„Tipps zuerst“ vermeidet die immer gleiche Anekdoten-Öffnung."
    )
    bg_mode_map = {
        "Automatisch": "auto",
        "Tipps zuerst (keine Anekdote)": "tips",
        "Story/Brücke": "story"
    }
    bg_mode = bg_mode_map[bg_mode_label]

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
                bg_mode=bg_mode,  # Steuerung für Hintergrundmodus
            )
        except Exception as e:
            st.error(f"Fehler beim Generieren: {e}")
            st.stop()

    final_md = result.get("final", "").strip()
    if not final_md:
        st.error("Kein Text erhalten. Bitte Eingaben prüfen und erneut versuchen.")
        st.stop()

    # WP-HTML & Meta erzeugen
    try:
        meta, body_md = split_frontmatter(final_md)
        body_md_for_wp = remove_leading_h1(body_md) if not wp_keep_h1 else body_md
        wp_html = markdown_to_wp_html(body_md_for_wp)

        # Fallback-Titel aus H1, falls im YAML nichts steht
        m_h1 = re.search(r'^#\s+(.+)$', body_md, re.MULTILINE)
        h1_title = (m_h1.group(1).strip() if m_h1 else "")

        meta_out = {
            "seo_title": meta.get("seo_title", h1_title),
            "meta_description": meta.get("meta_description", ""),
            "slug": meta.get("slug", extract_slug(final_md)),
            "primary_keyword": meta.get("primary_keyword", ""),
            "secondary_keywords": meta.get("secondary_keywords", []),
            "post_title": h1_title or meta.get("seo_title", ""),
        }
        meta_json_str = json.dumps(meta_out, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Fehler beim WP-Export: {e}")
        wp_html, meta_json_str = "", ""

    draft_md = result.get("draft", "").strip()
    metrics = result.get("metrics", {})
    ok = result.get("passed_heuristics", False)

    # Anzeige
    slug = extract_slug(final_md)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    st.success(f"Fertig. Heuristiken/Checks ok: **{ok}**")

    # Tabs
    tab_labels = ["📰 Gerenderte Ansicht", "🧾 Roh-Markdown", "🧩 WordPress-HTML", "📊 Metriken"]
    if show_draft:
        tab_labels.append("🧪 Draft")
    tabs = st.tabs(tab_labels)

    # Tab 0: gerenderter Markdown (YAML nicht anzeigen)
    with tabs[0]:
        try:
            st.markdown(body_md)  # nur Body, ohne Frontmatter
        except Exception:
            st.markdown(final_md)  # Fallback
        dl_col1, dl_col2, _ = st.columns([1, 1, 2])
        with dl_col1:
            st.download_button(
                "⬇️ Markdown herunterladen",
                data=final_md.encode("utf-8"),
                file_name=f"{slug}-{ts}.md",
                mime="text/markdown"
            )
        with dl_col2:
            copy_to_clipboard_button(final_md, "📋 Markdown kopieren")

    # Tab 1: Roh-Markdown
    with tabs[1]:
        st.text_area("Markdown", value=final_md, height=500)

    # Tab 2: WordPress-HTML
    with tabs[2]:
        if wp_html:
            st.caption("Vorschau (HTML gerendert):")
            components.html(wp_html, height=600, scrolling=True)
            st.divider()
            st.caption("Roh-HTML (zum Kopieren/Einfügen in Gutenberg „HTML“-Block oder Code-Editor):")
            st.text_area("HTML", value=wp_html, height=300)
            col_html1, col_html2, col_html3 = st.columns([1, 1, 2])
            with col_html1:
                st.download_button(
                    "⬇️ WordPress-HTML",
                    data=wp_html.encode("utf-8"),
                    file_name=f"{slug}-{ts}.html",
                    mime="text/html"
                )
            with col_html2:
                copy_to_clipboard_button(wp_html, "📋 HTML kopieren")
            if emit_meta_json and meta_json_str:
                st.download_button(
                    "⬇️ meta.json",
                    data=meta_json_str.encode("utf-8"),
                    file_name=f"{slug}-{ts}.meta.json",
                    mime="application/json"
                )
        else:
            st.warning("Kein HTML generiert.")

    # Tab 3: Metriken
    with tabs[3]:
        st.json(metrics, expanded=False)

    # Tab 4: Draft (optional)
    if show_draft:
        with tabs[4]:
            st.text_area("Draft (Rohfassung)", value=draft_md, height=400)

else:
    st.caption("Hinweis: Zeitangaben in **Pflichtdetails** (z. B. „12–14 Min“) werden in der Plausibilitäts-Prüfung berücksichtigt.")
