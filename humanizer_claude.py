#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
humanizer_pipeline.py (Claude-Version) — strukturstrenger & doppelfrei
----------------------------------------------------------------------
- Feste Artikelstruktur + SEO-Block (YAML Frontmatter)
- Ich-Erzählstimme + Du-Hinweise
- Grammatik/Style-Clean-Pass
- Wortlängenrange (min/max) + Heuristiken
- Auto-Repair (bis 6 Versuche) inkl. Struktur-Verschlankung
- Reiseziel-Mini-Story am H2-Start „Hintergrund & Tipps“
- Nach JEDEM Modellschritt Normalisierung (Überschriften/Listen/Dedupe)

Erfordert:
  pip install anthropic
  ANTHROPIC_API_KEY in der Umgebung

Beispiel:
python humanizer_pipeline.py \
  --topic "Protein-Porridge mit Erdnuss-Banane" \
  --details "Haferflocken, Erdnussbutter, Banane, Campingkocher, 10–12 Min köcheln" \
  --primary_kw "Protein-Porridge" \
  --secondary_kws "Erdnussbutter, Banane, Campingfrühstück" \
  --destination "Norwegen" \
  --travel_angle "Vanlife" \
  --out porridge.md --min-words 700 --max-words 1000
"""

import argparse
import json
import os
import re
import statistics
import sys
from typing import List, Dict, Any, Tuple

# ---------------- Anthropic (Claude) ----------------

try:
    import anthropic
except ImportError:
    print("Das 'anthropic'-Paket fehlt. Bitte installieren: pip install anthropic")
    sys.exit(1)

# Optionaler Key-Check (wenn du willst aktivieren):
# if not os.getenv("ANTHROPIC_API_KEY"):
#     print("Fehlt: ANTHROPIC_API_KEY")
#     sys.exit(1)

# ---------------- Schalter & Limits ----------------

MAX_TIPS = 4
MAX_VARIANTS = 4
MIN_STEPS = 6
MAX_STEPS = 8
MAX_BULLETS_PER_LIST = 4
ALLOWED_H2 = {"Einleitung", "Hintergrund & Tipps"}
ALLOWED_H3_IN_REZEPT = {"Zutaten", "Schritt für Schritt", "Zeiten & Portionen"}

FORBIDDEN_H2 = {
    "Packliste", "Timing", "Timing & Planung", "Troubleshooting",
    "Schnelle Abwandlungen", "Planung", "Packliste für die Zubereitung unterwegs"
}

# ---------------- Stil & Guides ----------------

STYLEGUIDE_DE = """
Schreibe wie „camp-kochen.de“: pragmatisch, persönlich, ohne Füllfloskeln.
- **Ich-Perspektive** (ich/mir/mich/mein) in allen Abschnitten.
- Leseransprache „du“ ist erlaubt für Tipps/Handgriffe; Erzähler bleibt „ich“.
- Variiere Satzlängen. Kurze Sätze sind okay.
- Konkrete Details (Mengen, Zeiten, Geräusche/Textur), keine Leerphrasen.
- Vermeide: „In diesem Artikel“, „abschließend“, „insgesamt“, „innovativ“, „köstlich“,
  „einfach zuzubereiten“, „im Folgenden“, „es ist wichtig zu beachten“, „nachstehend“.
- Aktive Verben. Keine sterile Aufzählung.
- Länge: zwischen {min_words} und {max_words} Wörtern. Keine Wortzahl nennen.
"""

STYLE_EXAMPLES_DE = """
Beispiel (Ton + Bild):
Wie seine kleinen Cousinen, die Windbeutel, besteht dieser Kuchen aus Brandteig und einer Vanillecreme. Erst kocht man einen Pudding, dann mischt man weiche Butter darunter. Zack, fühlt man sich wie im Hochgebirge. Puderzucker wie Schneesturm.
"""

NEGATIVE_LIST_DE = [
    "In diesem Artikel", "abschließend", "insgesamt", "innovativ", "köstlich",
    "einfach zuzubereiten", "im Folgenden", "es ist wichtig zu beachten",
    "nachstehend", "zusammenfassend", "Fazit"
]

SYSTEM_PROMPT_DE = """Du bist Redakteur:in für camp-kochen.de.
Schreibe persönlich, konkret, in Ich-Perspektive mit Du-Hinweisen, ohne Floskeln.
Halte den Struktur-Guide strikt ein.
"""

KONSISTENZ_GUIDE_DE = """
Kohärenz & Story-Führung:
- „Einleitung“ und „Hintergrund & Tipps“ bilden eine Erzählung; im Hintergrund zuerst an die Einleitung andocken.
- Reiseziel (falls vorhanden) in Einleitung **und** im ersten Satz des Hintergrunds nennen.
- Keine konkurrierenden Start-Szenen; wenn Wechsel: sauber als Rückblende markieren („Meine erste Begegnung…“).
- Konsistentes Vokabular: Camper, Kocher, Pfanne, Vanlife/Rundreise.
"""

STRUCTURE_GUIDE = """
Erzeuge einen Artikel in **Markdown** mit **genau** dieser Struktur:

- YAML-Frontmatter (oberhalb des Inhalts):
  ---
  seo_title: "<max 60 Zeichen, enthält Primär-Keyword>"
  meta_description: "<max 155 Zeichen, Nutzen/USP, enthält Primär-Keyword>"
  slug: "<kebab-case-ohne-Sonderzeichen>"
  primary_keyword: "<Primär-Keyword>"
  secondary_keywords: ["<Sek1>", "<Sek2>", "<Sek3>"]
  ---
- # H1: Rezeptname (Primär-Keyword enthalten)
- ## Einleitung
  - 1–2 Absätze, Primär-Keyword in den ersten 100 Wörtern.
- ## Hintergrund & Tipps
  - Anschluss an die Einleitung (oder klare Rückblende).
  - 2–4 **Bullet-Tipps** (keine Doppelungen), kompakt formuliert.
- ## Rezept: <Kurzbezeichnung>
  - ### Zutaten (Liste mit Mengenangaben)
  - ### Schritt für Schritt (nummerierte Liste, 6–8 Schritte)
  - ### Zeiten & Portionen (Zubereitung, Gesamtzeit, Portionen)

Optional:
- **Genau ein** H2 „Varianten“ (max. 4 kompakte Bullets).

**Keine weiteren H2/H3** außer den oben genannten.
**Keine Packliste, Timing, Troubleshooting oder ähnliche Zusatzabschnitte.**
"""

# ---------------- Prompts ----------------

DRAFT_TEMPLATE_DE = """Erstelle einen Rohentwurf gemäß Stil-, Struktur- und Konsistenz-Guide.
Thema: {topic}
Pflichtdetails: {details}
Primär-Keyword: {primary_kw}
Sekundär-Keywords (optional): {secondary_kws}

Stilguide:
{styleguide}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Beispiele (Tonfall):
{examples}

Gib **nur** den finalen Markdown-Artikel in der geforderten Struktur zurück (inkl. YAML-SEO-Block).
"""

EDIT_TEMPLATE_DE = """Überarbeite den Text gemäß den Guides.
- **Ich-Perspektive** durchgängig (Einleitung & Hintergrund starten in Ich-Form).
- Leser als „du“ ansprechen (für Tipps), ohne Erzählerrolle zu ändern.
- Entferne Floskeln: {negative}
- Bewahre die **exakte Struktur** (YAML-SEO-Block, H1; H2/H3 nur wie erlaubt).
- **Keine zusätzlichen H2/H3** außer: Einleitung, Hintergrund & Tipps, Rezept: … (+ Variants H2 optional) und die drei H3 im Rezept.
- Hintergrund: **2–4 Bullet-Tipps**, kompakt, keine Doppelungen.
- Schritt für Schritt: **{min_steps}–{max_steps} Schritte**.
- Zutaten mit Mengenangaben.
- Länge **{min_words}–{max_words}** Wörter.

Stilguide:
{styleguide}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Beispiele:
{examples}

Text:
{draft}
"""

CLEAN_TEMPLATE_DE = """Korrigiere Deutsch (Rechtschreibung, Grammatik, Zeichensetzung).
- Erzählerstimme: **Ich**. „Du“-Ansprache für Hinweise okay.
- Keine Strukturänderung, keine Floskeln.
Gib NUR den bereinigten Text zurück.

Text:
{raw}
"""

COHERENCE_LINE = "\n\nWICHTIG: Einleitung und „Hintergrund & Tipps“ bilden **eine** Erzählung; im Hintergrund zuerst **an die Einleitung andocken**, nicht neu anfangen."

# ---------------- Heuristiken ----------------

BANNED_PATTERNS = [re.compile(re.escape(p), flags=re.IGNORECASE) for p in NEGATIVE_LIST_DE]
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-']+", flags=re.UNICODE)
SECOND_PERSON_WORDS = ["du", "dich", "dir", "dein", "deine", "deinen", "deinem", "deiner", "deines"]
FORMAL_ADDRESS_WORDS = [r"\bSie\b", r"\bIhnen\b", r"\bIhr\b", r"\bIhre\b", r"\bIhrem\b", r"\bIhren\b", r"\bIhrer\b", r"\bIhres\b"]

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)

def word_count(text: str) -> int:
    return len(tokenize(text))

def type_token_ratio(text: str) -> float:
    tokens = [t.lower() for t in tokenize(text)]
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def sentence_lengths(text: str) -> List[int]:
    sentences = re.split(r"[\.!\?\n]+", text)
    return [len(tokenize(s)) for s in sentences if tokenize(s)] or [0]

def variance_sentence_length(text: str) -> float:
    lens = sentence_lengths(text)
    return statistics.pstdev(lens) if len(lens) > 1 else 0.0

def count_first_person(text: str) -> int:
    return len(re.findall(r"\b(ich|wir|mich|uns|mein|unser)\b", text.lower()))

def count_numbers(text: str) -> int:
    return len(re.findall(r"\b\d+[.,]?\d*\b", text))

def has_banned_phrases(text: str) -> bool:
    return any(p.search(text) for p in BANNED_PATTERNS)

def count_second_person(text: str) -> int:
    t = text.lower()
    return sum(len(re.findall(fr"\b{w}\b", t)) for w in SECOND_PERSON_WORDS)

def count_formal_address(text: str) -> int:
    return sum(len(re.findall(w, text)) for w in FORMAL_ADDRESS_WORDS)

# ---------------- Struktur-Prüfungen ----------------

def extract_section(md: str, header: str) -> str:
    pat = rf"^##\s+{re.escape(header)}\s*\n([\s\S]*?)(?=\n##\s+|$)"
    m = re.search(pat, md, re.MULTILINE)
    return m.group(1).strip() if m else ""

def yaml_present(md: str) -> bool:
    return bool(re.search(r"^---\s*[\s\S]*?---\s*", md.strip()))

def seo_lengths(md: str) -> Dict[str, Any]:
    title = ""
    desc = ""
    m = re.search(r'seo_title:\s*"([^"]+)"', md)
    if m: title = m.group(1)
    m = re.search(r'meta_description:\s*"([^"]+)"', md)
    if m: desc = m.group(1)
    return {
        "title_len": len(title),
        "title_ok": 10 <= len(title) <= 60,
        "meta_len": len(desc),
        "meta_ok": 50 <= len(desc) <= 155
    }

def check_structure(md: str, primary_kw: str) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["yaml_frontmatter"] = yaml_present(md)

    m_h1 = re.search(r"^#\s+(.+)", md, re.MULTILINE)
    h1 = m_h1.group(1).strip() if m_h1 else ""
    checks["h1_present"] = bool(h1)
    checks["h1_contains_primary_kw"] = (primary_kw.lower() in h1.lower()) if primary_kw else True

    # H2 erlaubt
    h2_all = re.findall(r"^##\s+(.+)$", md, re.MULTILINE)
    checks["h2_all"] = h2_all
    # Rezept-H3
    rezept_block = re.search(r"^##\s+Rezept:.*?(?=\n##\s+|$)", md, re.MULTILINE | re.DOTALL)
    h3_in_rezept = re.findall(r"^###\s+(.+)$", rezept_block.group(0), re.MULTILINE) if rezept_block else []
    checks["h3_in_rezept"] = h3_in_rezept

    # Pflicht-H2 und H3 vorhanden?
    checks["h2_einleitung"] = "Einleitung" in h2_all
    checks["h2_bg_tipps"] = "Hintergrund & Tipps" in h2_all
    checks["h2_rezept"] = any(h.startswith("Rezept:") for h in h2_all)
    checks["h3_zutaten"] = "Zutaten" in h3_in_rezept
    checks["h3_schritte"] = "Schritt für Schritt" in h3_in_rezept
    checks["h3_zeiten"] = "Zeiten & Portionen" in h3_in_rezept

    # Steps zählen
    steps = re.findall(r"^\d+\.\s", md, re.MULTILINE)
    checks["steps_count"] = len(steps)
    checks["steps_ok"] = MIN_STEPS <= len(steps) <= MAX_STEPS

    # Keyword in ersten 100 Worten
    body_without_yaml = re.sub(r"^---[\s\S]*?---", "", md).strip()
    first100 = " ".join(tokenize(body_without_yaml)[:100]).lower()
    checks["kw_in_first100"] = (primary_kw.lower() in first100) if primary_kw else True

    # Verbotene/zusätzliche H2/H3
    forbidden_h2_present = any(h in FORBIDDEN_H2 for h in h2_all)
    extra_h2 = [h for h in h2_all if (h not in ALLOWED_H2 and not h.startswith("Rezept:") and h != "Varianten")]
    extra_h3 = [h for h in h3_in_rezept if h not in ALLOWED_H3_IN_REZEPT]
    checks["forbidden_h2_present"] = forbidden_h2_present
    checks["extra_h2"] = extra_h2
    checks["extra_h3"] = extra_h3

    return checks

FIRST_PERSON_SET = {"ich","mir","mich","mein","meine","meinem","meinen","meiner","meines"}

def ich_in_first100(md: str) -> bool:
    body_without_yaml = re.sub(r"^---[\s\S]*?---", "", md).strip()
    tokens = [t.lower() for t in tokenize(body_without_yaml)]
    window = tokens[:100]
    return any(tok in FIRST_PERSON_SET for tok in window)

def contains_destination(text: str, dest: str) -> bool:
    if not dest:
        return True
    return dest.lower() in text.lower()

def coherence_checks(md: str, destination: str) -> Dict[str, Any]:
    intro = extract_section(md, "Einleitung")
    bg = extract_section(md, "Hintergrund & Tipps")

    intro_has_dest = contains_destination(intro, destination)
    bg_has_dest = contains_destination(bg[:200], destination) if bg else True

    first_bg_sentence = ""
    if bg:
        first_bg_sentence = re.split(r"[.!?]\s", bg.strip(), maxsplit=1)[0].lower()
    bridge_markers = ["meine erste begegnung", "ein paar tage zuvor", "später", "damals", "hier", "dort"]
    bridge_ok = any(k in first_bg_sentence for k in bridge_markers) or (
        bool(destination) and destination.lower() in first_bg_sentence
    )

    ok = intro_has_dest and bg_has_dest and bridge_ok
    return {"intro_has_dest": intro_has_dest, "bg_has_dest": bg_has_dest, "bridge_ok": bridge_ok, "ok": ok}

# ---------------- Normalisierung ----------------

def dedupe_lines_keep_order(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for ln in lines:
        key = re.sub(r"\s+", " ", ln.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(ln.strip())
    return out

def cap_bullets(block: str, cap: int) -> str:
    bullets = re.findall(r"^\s*[-*]\s+.+$", block, re.MULTILINE)
    if len(bullets) <= cap:
        return block
    # Keep first `cap` bullets
    kept = 0
    lines = block.splitlines()
    out = []
    for ln in lines:
        if re.match(r"^\s*[-*]\s+.+$", ln) and kept < cap:
            out.append(ln)
            kept += 1
        elif re.match(r"^\s*[-*]\s+.+$", ln) and kept >= cap:
            continue
        else:
            out.append(ln)
    return "\n".join(out)

def renumber_steps(block: str) -> str:
    steps = re.findall(r"^\s*(\d+)\.\s+.+$", block, re.MULTILINE)
    if not steps:
        return block
    lines = block.splitlines()
    new_lines = []
    idx = 1
    for ln in lines:
        if re.match(r"^\s*\d+\.\s+.+$", ln):
            if idx > MAX_STEPS:
                # drop extra steps
                continue
            ln = re.sub(r"^\s*\d+\.\s+", f"{idx}. ", ln)
            idx += 1
        new_lines.append(ln)
    return "\n".join(new_lines)

def unify_tips_in_background(md: str) -> str:
    bg = extract_section(md, "Hintergrund & Tipps")
    if not bg:
        return md
    # Sammle Tipp-Zeilen („Tipp …:“) + Bullets
    tip_lines = re.findall(r"(?im)^tipp\s*\d*\s*:\s*(.+)$", bg)
    bullet_lines = re.findall(r"(?m)^\s*[-*]\s+(.+)$", bg)
    tips = tip_lines + bullet_lines
    tips = [re.sub(r"[.!?]\s*$", "", t).strip() for t in tips]
    tips = dedupe_lines_keep_order(tips)[:MAX_TIPS]

    # baue neuen BG-Block: erster Absatz (ohne alte Tipp-Zeilen) + konsolidierte Bulletliste (max MAX_TIPS)
    bg_no_tips = []
    for ln in bg.splitlines():
        if re.match(r"(?i)^\s*tipp\s*\d*\s*:", ln) or re.match(r"^\s*[-*]\s+.+$", ln):
            continue
        bg_no_tips.append(ln)
    bg_no_tips_txt = "\n".join(bg_no_tips).strip()
    if tips:
        bullets = "\n".join(f"- {t}" for t in tips)
        bg_clean = bg_no_tips_txt + ("\n\n" if bg_no_tips_txt else "") + bullets
    else:
        bg_clean = bg_no_tips_txt

    # cap bullets
    bg_clean = cap_bullets(bg_clean, MAX_TIPS)

    # ersetze im md
    return re.sub(
        rf"(^##\s+Hintergrund\s*&\s*Tipps\s*\n)[\s\S]*?(?=\n##\s+|$)",
        r"\1" + bg_clean + "\n",
        md, flags=re.MULTILINE
    )

def drop_forbidden_sections(md: str) -> str:
    # Entferne nicht erlaubte H2 vollständig
    for h in list(FORBIDDEN_H2):
        md = re.sub(rf"^##\s+{re.escape(h)}\s*\n[\s\S]*?(?=\n##\s+|$)", "", md, flags=re.MULTILINE)
    # Entferne alle extra H2 außer erlaubte und „Rezept:“ / „Varianten“
    h2_all = re.findall(r"^##\s+(.+)$", md, re.MULTILINE)
    for h in h2_all:
        if h in ALLOWED_H2 or h == "Varianten" or h.startswith("Rezept:"):
            continue
        md = re.sub(rf"^##\s+{re.escape(h)}\s*\n[\s\S]*?(?=\n##\s+|$)", "", md, flags=re.MULTILINE)
    return md

def keep_only_variants_once(md: str) -> str:
    # Nur eine H2 „Varianten“ zulassen und maximal MAX_VARIANTS Bullets behalten
    parts = re.split(r"(^##\s+Varianten\s*$)", md, flags=re.MULTILINE)
    if len(parts) <= 1:
        return md
    # Teile wie [..., "## Varianten", content, ..., "## Varianten", content, ...]
    first_idx = None
    content_after = ""
    i = 0
    while i < len(parts):
        if parts[i].strip().startswith("## Varianten"):
            if first_idx is None:
                first_idx = i
                # nimm den ersten Content-Block
                if i + 1 < len(parts):
                    content_after = parts[i+1]
            else:
                # drop spätere Varianten-Blocks
                if i + 1 < len(parts):
                    parts[i] = ""
                    parts[i+1] = ""
        i += 1
    # cap bullets im ersten
    if first_idx is not None:
        content_after = cap_bullets(content_after, MAX_VARIANTS)
        # wieder einsetzen
        parts[first_idx+1] = content_after
    return "".join(parts)

def clamp_list_lengths(md: str) -> str:
    # Begrenzt beliebige Bulletlisten auf MAX_BULLETS_PER_LIST
    def _cap_block(m):
        return cap_bullets(m.group(0), MAX_BULLETS_PER_LIST)
    return re.sub(r"(^(\s*[-*]\s+.+\n)+)", _cap_block, md, flags=re.MULTILINE)

def normalize_blanklines(md: str) -> str:
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = md.strip() + "\n"
    return md

def renumber_and_limit_steps(md: str) -> str:
    # Finde Rezept-Block und renummeriere dort die Schritte
    rezept = re.search(r"(^##\s+Rezept:[\s\S]*?$)", md, flags=re.MULTILINE)
    if not rezept:
        return md
    block = rezept.group(1)
    block = re.sub(
        r"(^###\s+Schritt\s+für\s+Schritt\s*\n)([\s\S]*?)(?=\n###\s+|\Z)",
        lambda m: m.group(1) + renumber_steps(m.group(2)),
        block, flags=re.MULTILINE
    )
    # setze Block zurück
    md = re.sub(r"(^##\s+Rezept:[\s\S]*?$)", block, md, flags=re.MULTILINE)
    return md

def dedupe_whole_doc_lines(md: str) -> str:
    lines = md.splitlines()
    # Nicht YAML block dedupen
    if md.strip().startswith("---"):
        parts = re.split(r"(^---[\s\S]*?---\s*)", md, maxsplit=1)
        yaml = parts[1]
        rest = parts[2] if len(parts) > 2 else ""
        rest_lines = dedupe_lines_keep_order(rest.splitlines())
        return yaml + "\n" + "\n".join(rest_lines) + "\n"
    else:
        return "\n".join(dedupe_lines_keep_order(lines)) + "\n"

def normalize_markdown(md: str) -> str:
    md = drop_forbidden_sections(md)
    md = unify_tips_in_background(md)
    md = keep_only_variants_once(md)
    md = clamp_list_lengths(md)
    md = renumber_and_limit_steps(md)
    md = dedupe_whole_doc_lines(md)
    md = normalize_blanklines(md)
    return md

# ---------------- Quality-Eval ----------------

def evaluate_quality(text: str, targets: Dict[str, Any], primary_kw: str, destination: str = "") -> Tuple[bool, Dict[str, Any]]:
    style_metrics = {
        "ttr": type_token_ratio(text),
        "var_sentence_len": variance_sentence_length(text),
        "first_person": count_first_person(text),
        "numbers": count_numbers(text),
        "words": word_count(text),
        "second_person": count_second_person(text),
        "formal_address": count_formal_address(text),
        "has_banned": has_banned_phrases(text),
    }
    structure = check_structure(text, primary_kw)
    seo_meta = seo_lengths(text)
    coherence = coherence_checks(text, destination)
    ich_intro_ok = ich_in_first100(text)

    # Überschriften/Liste-Disziplin
    headings_ok = (
        not structure["forbidden_h2_present"] and
        len(structure["extra_h2"]) == 0 and
        len(structure["extra_h3"]) == 0
    )

    ok = (
        not style_metrics["has_banned"] and
        style_metrics["ttr"] >= targets.get("min_ttr", 0.45) and
        style_metrics["var_sentence_len"] >= targets.get("min_var_sentence_len", 7.0) and
        style_metrics["first_person"] >= targets.get("min_first_person", 6) and
        style_metrics["numbers"] >= targets.get("min_numbers", 3) and
        targets["min_words"] <= style_metrics["words"] <= targets["max_words"] and
        style_metrics["second_person"] >= targets.get("min_second_person", 2) and
        style_metrics["formal_address"] <= targets.get("max_formal_address", 0) and
        ich_intro_ok and
        all([
            structure["yaml_frontmatter"], structure["h1_present"],
            structure["h2_einleitung"], structure["h2_bg_tipps"], structure["h2_rezept"],
            structure["h3_zutaten"], structure["h3_schritte"], structure["h3_zeiten"],
            structure["steps_ok"], structure["kw_in_first100"],
            seo_meta["title_ok"], seo_meta["meta_ok"]
        ]) and
        coherence["ok"] and
        headings_ok
    )
    return ok, {
        "style_metrics": style_metrics,
        "structure": structure,
        "seo_meta": seo_meta,
        "coherence": coherence,
        "ich_intro_ok": ich_intro_ok,
        "headings_ok": headings_ok
    }

# ---------------- Anthropic Call ----------------

def _anthropic_text_from_content(resp: "anthropic.types.Message") -> str:
    parts: List[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()

def call_claude(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.8,
    top_p: float = 0.9,
    num_predict: int = 6000,
) -> str:
    client = anthropic.Anthropic()
    system_txt = ""
    msg_list: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_txt = f"{system_txt}\n\n{content}".strip() if system_txt else content
        else:
            msg_list.append({"role": role, "content": content})
    if not any(m["role"] == "user" for m in msg_list):
        msg_list.append({"role": "user", "content": ""})
    resp = client.messages.create(
        model=model, max_tokens=num_predict, temperature=temperature, top_p=top_p,
        system=system_txt or None, messages=msg_list,
    )
    return _anthropic_text_from_content(resp)

# ---------------- Repair-Prompts ----------------

def build_expand_prompt(base_text: str, min_words: int, max_words: int, negative: str, style: str, structure: str, examples: str, consistency: str) -> str:
    return f"""Erweitere substanziell auf {min_words}–{max_words} Wörter.
- **Keine neuen H2/H3**. Erlaubt sind nur die im Struktur-Guide.
- Hintergrund mit **max. {MAX_TIPS} kompakten Bullets**, keine Doppelungen.
- Schritt für Schritt auf **{MIN_STEPS}–{MAX_STEPS}** begrenzen.
- Vermeide Floskeln: {negative}

Stilguide:
{style}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Beispiele:
{examples}

Text:
{base_text}
"""

def build_condense_prompt(base_text: str, min_words: int, max_words: int, negative: str, style: str, structure: str, examples: str, consistency: str) -> str:
    return f"""Kürze präzise auf {min_words}–{max_words} Wörter.
- Keine Dopplungen, keine Zusatz-H2.
- Hintergrund auf **max. {MAX_TIPS} Bullets** konsolidieren.
- Schritte auf **{MIN_STEPS}–{MAX_STEPS}** begrenzen.

Stilguide:
{style}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Beispiele:
{examples}

Text:
{base_text}
"""

def build_du_rewrite_prompt(base_text: str, negative: str, style: str) -> str:
    return f"""Passe die Leseransprache auf **du** an (du/dich/dir/dein …).
- Erzähler bleibt **ich** (Ich-Perspektive unverändert).
- Entferne formelle Anreden. Vermeide Floskeln: {negative}
Gib nur den Text zurück.

Stil:
{style}

Text:
{base_text}
"""

def build_ich_rewrite_prompt(base_text: str, negative: str, style: str) -> str:
    return f"""Schreibe konsequent in **Ich-Perspektive** (ich/mir/mich/mein …).
- Einleitung und Hintergrund starten in Ich-Form; „du“ nur als Adressat für Tipps.
- Ohne Strukturänderung; Floskeln vermeiden: {negative}
Gib nur den Text zurück.

Stil:
{style}

Text:
{base_text}
"""

def build_structure_fix_prompt(base_text: str, primary_kw: str, negative: str, style: str, structure: str, examples: str, consistency: str) -> str:
    return f"""Bringe den Artikel exakt in die geforderte Struktur:
- YAML-SEO vollständig & valide (seo_title ≤60, meta 50–155, slug kebab-case).
- H1 enthält Primär-Keyword: "{primary_kw}".
- Erlaubte H2/H3 **ausschließlich** laut Struktur-Guide.
- Hintergrund: **2–4 Bullets**, keine Doppelungen.
- Schritte: **{MIN_STEPS}–{MAX_STEPS}**.
- Floskeln vermeiden: {negative}

Stil:
{style}

Struktur:
{structure}

Konsistenz:
{consistency}

Beispiele:
{examples}

Text:
{base_text}
"""

def build_coherence_fix_prompt(base_text: str, destination: str, style: str, structure: str, consistency: str, negative: str) -> str:
    dest_line = f"- Nenne **{destination}** in Einleitung und im ersten Satz des Hintergrunds.\n" if destination else ""
    return f"""Füge Einleitung & Hintergrund zu einer einheitlichen Erzählung:
- Erste Sätze im Hintergrund docken klar an die Einleitung an (oder saubere Rückblende).
{dest_line}- Keine neue Startszene. Keine Extra-H2/H3. Floskeln vermeiden: {negative}

Stil:
{style}

Struktur:
{structure}

Konsistenz:
{consistency}

Text:
{base_text}
"""

def build_simplify_structure_prompt(base_text: str, style: str, structure: str, consistency: str, negative: str) -> str:
    return f"""Verschlanke die Struktur strikt:
- **Entferne** alle H2 außer: Einleitung, Hintergrund & Tipps, Rezept: …, (optional) Varianten.
- Im Rezept nur H3: Zutaten, Schritt für Schritt, Zeiten & Portionen.
- Hintergrund: bündele verstreute Tipps zu **max. {MAX_TIPS} kompakten Bullets**, ohne Doppelungen.
- Schritte auf **{MIN_STEPS}–{MAX_STEPS}** begrenzen.
- Floskeln vermeiden: {negative}. Ich-Perspektive beibehalten.
Gib nur den bereinigten Artikel zurück.

Stil:
{style}

Struktur:
{structure}

Konsistenz:
{consistency}

Text:
{base_text}
"""

# ---------------- Ziel-Mapping (Auto-Guess) ----------------

def guess_destination(text: str) -> str:
    s = text.lower()
    mapping = {
        "shakshuka": "Israel", "pad thai": "Thailand", "carbonara": "Italien",
        "khachapuri": "Georgien", "arepas": "Kolumbien", "laksa": "Malaysia",
        "ratatouille": "Frankreich", "paella": "Spanien", "chili": "USA", "bibimbap": "Südkorea",
    }
    for dish, dest in mapping.items():
        if dish in s:
            return dest
    return ""

# ---------------- Pipeline ----------------

def generate_article(
    topic: str,
    details: str,
    primary_kw: str = "",
    secondary_kws: str = "",
    model: str = "claude-3-haiku-20240307",
    min_words: int = 700,
    max_words: int = 1000,
    destination: str = "",
    travel_angle: str = "Vanlife/Rundreise",
) -> Dict[str, Any]:

    system = SYSTEM_PROMPT_DE
    style = STYLEGUIDE_DE.format(min_words=min_words, max_words=max_words).strip()
    structure = STRUCTURE_GUIDE.strip()
    examples = STYLE_EXAMPLES_DE.strip()
    negative = ", ".join(NEGATIVE_LIST_DE)
    consistency = KONSISTENZ_GUIDE_DE.strip()

    pk = primary_kw.strip() if primary_kw.strip() else topic
    sk = secondary_kws.strip() if secondary_kws.strip() else "Rezept, Outdoor, Kochen, Backen"

    # Reiseziel-Story Zusatz
    dest = (destination or guess_destination(f"{topic} {pk}")).strip()
    angle = (travel_angle or "Vanlife/Rundreise").strip()
    travel_hook = ""
    if dest:
        travel_hook = f"""
Zusatzanforderung (Reiseziel-Story):
- Beginne **H2 „Hintergrund & Tipps“** mit Anschluss an die Einleitung (kein neuer Start).
- Nenne **{dest}** im ersten Satz; Bezug auf **{angle}** (z. B. Campen/Roadtrip) und 1–2 konkrete Details (Geräusch, Geruch, Licht).
- Erkläre kurz, **warum das Gericht dort passt**. Danach die Bullet-Tipps.
"""

    # Pass 1: Draft
    draft = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": DRAFT_TEMPLATE_DE.format(
                topic=topic, details=details, primary_kw=pk, secondary_kws=sk,
                styleguide=style, structure=structure, examples=examples, consistency=consistency
            ) + (travel_hook or "") + COHERENCE_LINE},
        ],
        num_predict=4096, temperature=0.8, top_p=0.9
    )
    draft = normalize_markdown(draft)

    # Pass 1.5: Clean
    cleaned = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": CLEAN_TEMPLATE_DE.format(raw=draft)},
        ],
        temperature=0.4, top_p=0.9, num_predict=4096
    )
    cleaned = normalize_markdown(cleaned)

    # Pass 2: Edit
    edited = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": EDIT_TEMPLATE_DE.format(
                styleguide=style, negative=negative, draft=cleaned,
                min_words=min_words, max_words=max_words,
                min_steps=MIN_STEPS, max_steps=MAX_STEPS,
                structure=structure, examples=examples, consistency=consistency
            ) + (travel_hook or "") + COHERENCE_LINE},
        ],
        temperature=0.6, top_p=0.9, num_predict=4096
    )
    edited = normalize_markdown(edited)

    # Qualität prüfen
    targets = dict(
        min_ttr=0.45,
        min_var_sentence_len=7.0,
        min_first_person=6,
        min_numbers=3,
        min_words=min_words,
        max_words=max_words,
        min_second_person=2,
        max_formal_address=0,
    )
    ok, metrics = evaluate_quality(edited, targets, pk, destination=dest)

    # Auto-Repair
    attempts = 0
    while not ok and attempts < 6:
        attempts += 1
        sm = metrics["style_metrics"]
        stc = metrics["structure"]
        seo = metrics["seo_meta"]
        ich_intro_ok = metrics.get("ich_intro_ok", False)
        coh = metrics.get("coherence", {"ok": True})
        headings_ok = metrics.get("headings_ok", True)

        if sm["first_person"] < targets["min_first_person"] or not ich_intro_ok:
            repair_prompt = build_ich_rewrite_prompt(edited, negative, style)
        elif not coh.get("ok", True):
            repair_prompt = build_coherence_fix_prompt(edited, dest, style, structure, consistency, negative)
        elif not headings_ok:
            repair_prompt = build_simplify_structure_prompt(edited, style, structure, consistency, negative)
        elif sm["formal_address"] > 0 or sm["second_person"] < targets["min_second_person"]:
            repair_prompt = build_du_rewrite_prompt(edited, negative, style)
        elif not all([
            stc["yaml_frontmatter"], stc["h1_present"],
            stc["h2_einleitung"], stc["h2_bg_tipps"], stc["h2_rezept"],
            stc["h3_zutaten"], stc["h3_schritte"], stc["h3_zeiten"],
            stc["steps_ok"], stc["kw_in_first100"], seo["title_ok"], seo["meta_ok"]
        ]):
            repair_prompt = build_structure_fix_prompt(edited, pk, negative, style, structure, examples, consistency)
        elif sm["words"] < min_words:
            repair_prompt = build_expand_prompt(edited, min_words, max_words, negative, style, structure, examples, consistency)
        elif sm["words"] > max_words:
            repair_prompt = build_condense_prompt(edited, min_words, max_words, negative, style, structure, examples, consistency)
        else:
            repair_prompt = build_simplify_structure_prompt(edited, style, structure, consistency, negative)

        edited = call_claude(
            model,
            [{"role": "system", "content": system},
             {"role": "user", "content": repair_prompt}],
            temperature=0.5, top_p=0.9, num_predict=4096
        )
        edited = normalize_markdown(edited)
        ok, metrics = evaluate_quality(edited, targets, pk, destination=dest)

    # Harte Absicherung bei zu kurz
    if not ok and metrics["style_metrics"]["words"] < min_words:
        for _ in range(2):
            edited = call_claude(
                model,
                [{"role": "system", "content": system},
                 {"role": "user", "content": build_expand_prompt(
                     edited, min_words, max_words, negative, style, structure, examples, consistency
                 )}],
                temperature=0.5, top_p=0.9, num_predict=4096
            )
            edited = normalize_markdown(edited)
            ok, metrics = evaluate_quality(edited, targets, pk, destination=dest)
            if ok:
                break

    return {"draft": draft, "final": edited, "metrics": metrics, "passed_heuristics": ok}

# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(description="Artikel-Generator (Claude) — strikte Struktur & Dedupe.")
    parser.add_argument("--topic", required=True, help="Thema / Titel-Idee")
    parser.add_argument("--details", required=True, help="Pflichtdetails (kommasepariert)")
    parser.add_argument("--primary_kw", default="", help="Primär-Keyword (SEO). Standard: topic")
    parser.add_argument("--secondary_kws", default="", help="Sekundär-Keywords (kommagetrennt)")
    parser.add_argument("--model", default="claude-3-haiku-20240307", help="Anthropic-Modellname")
    parser.add_argument("--min-words", type=int, default=700, help="Minimale Wortzahl")
    parser.add_argument("--max-words", type=int, default=1000, help="Maximale Wortzahl")
    parser.add_argument("--destination", default="", help="Reiseziel für die Hintergrund-Story (z. B. Israel)")
    parser.add_argument("--travel_angle", default="Vanlife/Rundreise", help="Reiseperspektive (z. B. Campen, Roadtrip)")
    parser.add_argument("--out", default="out.md", help="Zieldatei (Markdown)")
    parser.add_argument("--show-draft", action="store_true", help="Draft zusätzlich speichern")
    args = parser.parse_args()

    try:
        result = generate_article(
            args.topic, args.details,
            primary_kw=args.primary_kw, secondary_kws=args.secondary_kws,
            model=args.model, min_words=args.min_words, max_words=args.max_words,
            destination=args.destination, travel_angle=args.travel_angle
        )
    except anthropic.APIStatusError as e:
        print("Anthropic API-Fehler:", e)
        sys.exit(2)
    except Exception as e:
        print("Fehler:", e)
        sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(result["final"].strip() + "\n")

    if args.show_draft:
        draft_path = re.sub(r"\.md$", "_draft.md", args.out)
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(result["draft"].strip() + "\n")

    print("# --- Humanizer Pipeline (Struktur+SEO / Claude) ---")
    print("Heuristiken/Struktur/SEO OK:", result["passed_heuristics"])
    print("Metriken:", json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Finaler Text gespeichert in: {args.out}")
    print("Kohärenz:", json.dumps(result["metrics"].get("coherence", {}), ensure_ascii=False))

if __name__ == "__main__":
    main()
