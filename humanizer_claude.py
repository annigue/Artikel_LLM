#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
humanizer_claude.py (Claude-Version)
--------------------------------------
Zwei-Pass-Schreibpipeline mit:
- fester Artikelstruktur + SEO-Block (YAML Frontmatter)
- Du-Ansprache & Sprach-/Grammatik-Clean-Pass
- Wortlängen-Range (min/max) + Heuristiken
- Auto-Repair (Ich/DU/Kohärenz/Plausibilität/Struktur/SEO/Länge), bis zu 6 Versuche
- Reiseziel-Mini-Story mit sauberem Anschluss an die Einleitung

Erfordert:
  pip install anthropic
  ANTHROPIC_API_KEY als Umgebungsvariable

Beispiel:
python humanizer_claude.py \
  --topic "Shakshuka" \
  --details "Eier, Tomate, Levante Küche, wenig Abwasch" \
  --primary_kw "Schakshuka Rezept" \
  --secondary_kws "Eier, Tomate, Levante Küche" \
  --destination "Polen" \
  --travel_angle "Rundreise im Camper" \
  --out shakshuka.md --min-words 700 --max-words 1000
"""

import argparse
import json
import os
import re
import statistics
import sys
from typing import List, Dict, Any, Tuple, Optional

# --- Anthropic (Claude) ---
try:
    import anthropic
except ImportError:
    print("Das 'anthropic'-Paket fehlt. Bitte installieren: pip install anthropic")
    sys.exit(1)

# -------------------- Stil + Beispiele --------------------

STYLEGUIDE_DE = """
Schreibe wie „camp-kochen.de“: pragmatisch, persönlich, ohne Füllfloskeln.
- Erzählerstimme: **Ich-Perspektive** (ich/mir/mich/mein) in allen Abschnitten.
- Leseransprache „du“ ist erlaubt für Tipps/Handgriffe, aber der Erzähler bleibt „ich“.
- Variiere Satzlängen. Kurze Sätze sind erlaubt.
- Konkrete Details (Mengen, Zeiten, Geräusche/Textur), keine Leerphrasen.
- Vermeide: „In diesem Artikel“, „abschließend“, „insgesamt“, „innovativ“, „köstlich“,
  „einfach zuzubereiten“, „im Folgenden“, „es ist wichtig zu beachten“, „nachstehend“.
- Aktive Verben. Keine sterile Aufzählung.
- Länge: zwischen {min_words} und {max_words} Wörtern. Gib keine Wortzahl aus.
"""

STYLE_EXAMPLES_DE = """
Beispiel 1:
Versprochen, so ein essbares Microadventure sorgt nochmal für eine Extraportion Umami.

Beispiel 2 (Erklärung + Bildsprache):
Unsere cremige Süßkartoffelsuppe bringt extra viel Farbe ins trübe Herbstwetter und zusammen mit den schnellen, rauchigen und knusprigen Rosenkohl-Chips, die einen tollen Kontrast zur süßen Gemüsebasis bilden, wirds richtig aufregend in der Schüssel. 

Beispiel 3 (Hürde entkrampfen, Humor):
Ich rede jetzt aber nicht davon, bei den nächsten drölf Schneeflocken sofort die 
ganze Family in den Minivan zu packen und zusammen mit 827 anderen, denen die Decke auf den Kopf fällt, den nächstgelegenen und eigentlich viel zu popelig-kurzen Schlittenhügel mit einer fünfsekündigen Abfahrt und matschigem Auslauf anzusteuern. 

Beispiel 4 (fachlich + bildlich):
Dafür schneiden wir den Strunk großzügig ab und kerben den Stil rundherum leicht schräg mit einem kleinen Küchenmesser ein. Am einfachsten gehts mit einem kurzen, scharfen Schälmesser. So lassen sich die äußeren Blätter leicht ablösen."""

NEGATIVE_LIST_DE = [
    "In diesem Artikel", "abschließend", "insgesamt", "innovativ", "köstlich",
    "einfach zuzubereiten", "im Folgenden", "es ist wichtig zu beachten",
    "nachstehend", "zusammenfassend", "Fazit", "Revolutioniere", "Tauche ein", "Erfahre mehr über", 
    "Auf eine Reise gehen durch", "spannende Einblicke", "Die Macht von", "Entfessele die Kraft",
    "Meine erste Begegnung", "Nicht nur", "Hier sind einige", "Im Laufe der Jahre", "Ich habe festgelstellt"
]

SYSTEM_PROMPT_DE = """Du bist Redakteur:in für camp-kochen.de.
Deine Aufgabe: hilfreiche, konkrete, persönliche Texte mit natürlichem Rhythmus verfassen.
Erzählerstimme ist **Ich-Perspektive**; Leseransprache „du“ nur für Hinweise/Tipps.
Halte den Stilguide strikt ein.
"""

KONSISTENZ_GUIDE_DE = """
Kohärenz & Story-Führung:
- Halte Einleitung und „Hintergrund & Tipps“ in derselben Szene/Erzählzeit, ODER markiere Wechsel sauber als Rückblende (z. B. „Meine erste Begegnung …“, „Ein paar Tage zuvor …“).
- Wenn ein Reiseziel angegeben ist, nenne es in der Einleitung **und** zu Beginn von „Hintergrund & Tipps“.
- Vermeide konkurrierende Start-Szenen; nutze im Hintergrund einen Übergangssatz, der klar an die Einleitung anknüpft.
- Nutze ein konsistentes Vokabular: „Camper“, „Kocher“, „Pfanne“, „Vanlife“, "Rundreise"
"""

PLAUSIBILITY_GUIDE_DE = """
Plausibilität & Camping-Kontext:
- Nutze typische Outdoor-Ausrüstung (Pfanne, Campingkocher, Deckel). Kein „Backofen/Ofen“ ohne explizite Anforderung.
- „Wenig Abwasch“: vermeide unnötige zusätzliche Schüsseln/Schalen. Wenn möglich direkt in der Pfanne arbeiten.
- Bleib bei hausüblichen Mengen und Zeiten. Wenn im Input ein Zeitfenster genannt ist (z. B. 12–14 Min), nutze es.
- Keine erfundenen Markennamen oder exakten Adressen. Nenne nur allgemein bekannte Orte/Landschaften (z. B. Tel Aviv, Negev, Totes Meer).
- sei präzise bei der Beschreibung von Details, bildlich beschreiben, vermeide Übertreibungen
"""

# -------------------- Struktur-Guide (verbindlich) --------------------

STRUCTURE_GUIDE = """
Erzeuge einen vollständigen Artikel in **Markdown** mit exakt dieser Struktur:

- **SEO-Block** (oberhalb des Inhalts), im YAML-Frontmatter-Format:
  ---
  seo_title: "<max 60 Zeichen, Primär-Keyword enthalten>"
  meta_description: "<max 155 Zeichen, Nutzen/USP, Primär-Keyword enthalten>"
  slug: "<kebab-case-ohne-Sonderzeichen>"
  primary_keyword: "<Primär-Keyword>"
  secondary_keywords: ["<Sek1>", "<Sek2>", "<Sek3>"]
  ---
- **H1**: Rezeptname (Primär-Keyword enthalten)
- **H2 Einleitung**: 1–2 Absätze, natürliche Einbindung des Primär-Keywords in den ersten 100 Wörtern.
- **H2 Hintergrund & Tipps**: Back-Story, typische Fehler entkrampfen, 2–4 konkrete Tipps aus Erfahrung.
- **H2 Rezept: <Kurzbezeichnung>**
  - **H3 Zutaten** (Liste mit Mengenangaben)
  - **H3 Schritt für Schritt** (nummerierte Liste, 6–10 Schritte)
  - **H3 Zeiten & Portionen** (Zubereitung, Gesamtzeit, Ruhe-/Backzeit, Portionen)
Optional am Ende: 1–2 interne Link-Ideen (ohne URL).
"""

# -------------------- Prompt-Templates --------------------

DRAFT_TEMPLATE_DE = """Erstelle einen Rohentwurf gemäß Stilguide, Struktur-Guide und Beispielen.
Thema: {topic}
Pflichtdetails: {details}
Primär-Keyword: {primary_kw}
Sekundär-Keywords (optional): {secondary_kws}
Gib **nur** den finalen Markdown-Artikel in der geforderten Struktur zurück (inkl. YAML-SEO-Block).
Stilguide:
{styleguide}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Plausibilitäts-Guide:
{plausibility}

Beispiele (Tonfallreferenz):
{examples}
"""

EDIT_TEMPLATE_DE = """Überarbeite den Text gemäß Stil- und Struktur-Guide.
- Schreibe durchgängig in **Ich-Perspektive**; beginne **Einleitung** sowie **Hintergrund & Tipps** in der Ich-Form.
- Leser darf als „du“ adressiert werden, aber nicht die Erzählerrolle übernehmen.
- Entferne Floskeln aus der Negativliste: {negative}
- Wahrung der Struktur (YAML-SEO-Block, H1, H2/H3 exakt wie vorgegeben)
- Nutze Primär-Keyword in H1 + Einleitung (erste 100 Wörter) + mindestens 1 H2.
- Variiere Satzlängen; erlaube rhetorische Fragen.
- Füge 1–2 konkrete Beobachtungen/Details ein (Geräusch, Textur, Zahl).
- Liste „Schritt für Schritt“ nummeriert (6–10 Schritte).
- Mengenangaben bei Zutaten prüfen/ergänzen.
- Länge strikt {min_words}–{max_words} Wörter (keine Wortzahl ausgeben).
- Konsequent Du-Ansprache (keine formelle Anrede).
Gib **nur** den finalen Markdown-Artikel zurück.

Stilguide:
{styleguide}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Plausibilitäts-Guide:
{plausibility}

Beispiele (Tonfallreferenz):
{examples}

Text:
{draft}
"""

CLEAN_TEMPLATE_DE = """Korrigiere den folgenden Text in deutscher Sprache:
- Behebe Rechtschreibung, Grammatik, Zeichensetzung und schiefe Formulierungen.
- Schreibe konsequent in der **Ich-Perspektive** (ich/mir/mich/mein). Ersetze unpersönliche „man“-Konstruktionen
  und unpassende „wir“-Formen durch „ich“, außer wenn „wir“ inhaltlich zwingend ist.
- „Du“-Ansprache für Hinweise/Tipps bleibt erlaubt.
- Vermeide formelle Anrede (Sie/Ihnen/Ihr ...).
- Behalte Inhalt und Struktur bei.
Gib NUR den bereinigten Text zurück.

Text:
{raw}
"""

COHERENCE_LINE_STORY = (
    "\n\nWICHTIG: Einleitung und „Hintergrund & Tipps“ bilden eine **einheitliche Szene**; "
    "im Hintergrund zuerst **an die Einleitung andocken** (Übergangssatz), "
    "alternativ sauber markierte Rückblende."
)
COHERENCE_LINE_TIPS = (
    "\n\nWICHTIG: „Hintergrund & Tipps“ **ohne Anekdote eröffnen**. "
    "Starte direkt mit 2–4 **konkreten Technik-/Praxis-Tipps**; "
    "keine Formulierungen wie „Meine erste Begegnung …“."
)

# -------------------- Heuristiken + Struktur/SEO-Checks --------------------

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

# Strengere Ich-Metrik (ohne "wir/unser")
def count_first_person(text: str) -> int:
    return len(re.findall(r"\b(ich|mir|mich|mein|meine|meinem|meinen|meiner|meines)\b", text.lower()))

def count_numbers(text: str) -> int:
    return len(re.findall(r"\b\d+[.,]?\d*\b", text))

def has_banned_phrases(text: str) -> bool:
    return any(p.search(text) for p in BANNED_PATTERNS)

def count_second_person(text: str) -> int:
    t = text.lower()
    return sum(len(re.findall(fr"\b{w}\b", t)) for w in SECOND_PERSON_WORDS)

def count_formal_address(text: str) -> int:
    return sum(len(re.findall(w, text)) for w in FORMAL_ADDRESS_WORDS)

def check_structure(md: str, primary_kw: str) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    # YAML frontmatter
    checks["yaml_frontmatter"] = bool(re.search(r"^---\s*[\s\S]*?---\s*", md.strip()))
    # H1
    m_h1 = re.search(r"^#\s+(.+)", md, re.MULTILINE)
    h1 = m_h1.group(1).strip() if m_h1 else ""
    checks["h1_present"] = bool(h1)
    checks["h1_contains_primary_kw"] = (primary_kw.lower() in h1.lower()) if primary_kw else True
    # Required H2/H3
    checks["h2_einleitung"] = bool(re.search(r"^##\s+Einleitung", md, re.MULTILINE))
    checks["h2_bg_tipps"] = bool(re.search(r"^##\s+Hintergrund\s*&\s*Tipps", md, re.MULTILINE))
    checks["h2_rezept"] = bool(re.search(r"^##\s+Rezept:", md, re.MULTILINE))
    checks["h3_zutaten"] = bool(re.search(r"^###\s+Zutaten", md, re.MULTILINE))
    checks["h3_schritte"] = bool(re.search(r"^###\s+Schritt\s+für\s+Schritt", md, re.MULTILINE))
    checks["h3_zeiten"] = bool(re.search(r"^###\s+Zeiten\s*&\s*Portionen", md, re.MULTILINE))
    # Numbered steps count
    steps = re.findall(r"^\d+\.\s", md, re.MULTILINE)
    checks["steps_count"] = len(steps)
    checks["steps_ok"] = 6 <= len(steps) <= 12
    # Keyword in first 100 words (body without YAML)
    body_without_yaml = re.sub(r"^---[\s\S]*?---", "", md).strip()
    first100 = " ".join(tokenize(body_without_yaml)[:100]).lower()
    checks["kw_in_first100"] = (primary_kw.lower() in first100) if primary_kw else True
    return checks

def extract_section(md: str, header: str) -> str:
    pat = rf"^##\s+{re.escape(header)}\s*\n([\s\S]*?)(?=\n##\s+|$)"
    m = re.search(pat, md, re.MULTILINE)
    return m.group(1).strip() if m else ""

def contains_destination(text: str, dest: str) -> bool:
    if not dest:
        return True
    return dest.lower() in text.lower()

def coherence_checks(md: str, destination: str, bg_mode: str = "auto") -> Dict[str, Any]:
    intro = extract_section(md, "Einleitung")
    bg = extract_section(md, "Hintergrund & Tipps")

    intro_has_dest = contains_destination(intro, destination)
    bg_has_dest = contains_destination(bg[:200], destination) if bg else True

    first_bg_sentence = ""
    if bg:
        first_bg_sentence = re.split(r"[.!?]\s", bg.strip(), maxsplit=1)[0].lower()
    bridge_markers = ["meine erste begegnung","ein paar tage zuvor","später","damals","hier","dort"]
    bridge_ok = any(k in first_bg_sentence for k in bridge_markers) or (
        bool(destination) and destination.lower() in first_bg_sentence
    )

    # Story-Modus verlangt Brücke + Ziel; Tips-Modus nicht.
    if bg_mode == "story":
        ok = intro_has_dest and bg_has_dest and bridge_ok
    else:
        ok = True  # im tips/auto (wenn auto → use_story False) erzwingen wir NICHT die Brücke

    return {
        "intro_has_dest": intro_has_dest,
        "bg_has_dest": bg_has_dest,
        "bridge_ok": (bridge_ok if bg_mode == "story" else True),
        "ok": ok
    }


# -------------------- Plausibilitäts-Checks --------------------

MINUTE_PAT = re.compile(r"(\d+)\s*(?:min|minute|minuten)\b", re.IGNORECASE)
RANGE_PAT  = re.compile(r"(\d+)\s*(?:–|-|bis)\s*(\d+)\s*(?:min|minute|minuten)\b", re.IGNORECASE)

def details_time_target(details: str) -> Optional[Tuple[int, int]]:
    """Suche z. B. '12–14 Min' oder '12-14 Minuten' im details-String."""
    m = RANGE_PAT.search(details)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)
    m2 = MINUTE_PAT.search(details)
    if m2:
        v = int(m2.group(1))
        return (v, v)
    return None

def parse_minutes_in_text(text: str) -> List[int]:
    vals: List[int] = []
    for lo, hi in RANGE_PAT.findall(text):
        vals.extend([int(lo), int(hi)])
    for v in MINUTE_PAT.findall(text):
        try:
            vals.append(int(v))
        except:
            pass
    return vals

DISH_WORDS = ["schüssel", "schale", "becher"]
POT_WORDS  = ["topf", "töpfe", "töpfen"]
OVEN_WORDS = ["backofen", "ofen"]
KOCHER_WORDS = ["kocher", "campingkocher"]
PFANNE_WORDS = ["pfanne", "guss", "gusseisen"]

def count_matches(words: List[str], text: str) -> int:
    t = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(w)}\b", t)) for w in words)

def extract_portions(md: str) -> Optional[int]:
    block = extract_section(md, "Zeiten & Portionen")
    if not block:
        return None
    m = re.search(r"(?:portion|portionen)\s*[:\-]?\s*(\d+)", block, re.IGNORECASE)
    return int(m.group(1)) if m else None

def plausibility_checks(md: str, details: str) -> Dict[str, Any]:
    text_lower = md.lower()

    # 1) Ausrüstung: Kocher/Pfanne vorhanden; Ofen verboten (falls nicht angefragt)
    equipment_ok = (count_matches(KOCHER_WORDS, text_lower) + count_matches(PFANNE_WORDS, text_lower) >= 1)
    oven_ok = True
    if "backofen" not in details.lower() and "ofen" not in details.lower():
        oven_ok = (count_matches(OVEN_WORDS, text_lower) == 0)

    # 2) „Wenig Abwasch“ → begrenze Schüsseln/Schalen
    abwasch_ok = True
    if "wenig abwasch" in details.lower():
        bowls = count_matches(DISH_WORDS, text_lower)
        pots  = count_matches(POT_WORDS, text_lower)
        # Grobe Heuristik: max 1 zusätzliche Schüssel, 1 Topf (Pfanne ist ok)
        abwasch_ok = (bowls <= 1 and pots <= 1)

    # 3) Zeiten (Details vs. Text)
    target = details_time_target(details)
    timing_ok = True
    found_minutes: List[int] = parse_minutes_in_text(md)
    if target:
        lo, hi = target
        # Toleranz ±2 Minuten
        timing_ok = any(lo - 2 <= m <= hi + 2 for m in found_minutes)

    # 4) Portionen pragmatisch (1–8)
    portions = extract_portions(md)
    portions_ok = (portions is None) or (1 <= portions <= 8)

    ok = equipment_ok and oven_ok and abwasch_ok and timing_ok and portions_ok
    return {
        "ok": ok,
        "equipment_ok": equipment_ok,
        "oven_ok": oven_ok,
        "abwasch_ok": abwasch_ok,
        "timing_ok": timing_ok,
        "portions_ok": portions_ok,
        "target_time": target,
        "found_minutes": found_minutes,
        "portions": portions,
    }

# -------------------- LLM Call (Claude) --------------------

def _anthropic_text_from_content(resp: "anthropic.types.Message") -> str:
    """Extrahiere zusammenhängenden Text aus der Anthropic-Antwort."""
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
    """
    Ruft Claude (Anthropic Messages API) auf.
    - messages: Liste mit Rollen 'system', 'user' (optional 'assistant').
    """
    client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY aus der Umgebung

    # Ein 'system' (optional) + abwechselnde user/assistant-Messages
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
        model=model,
        max_tokens=num_predict,
        temperature=temperature,
        top_p=top_p,
        system=system_txt or None,
        messages=msg_list,
    )
    return _anthropic_text_from_content(resp)

FIRST_PERSON_SET = {"ich","mir","mich","mein","meine","meinem","meinen","meiner","meines"}

def ich_in_first100(md: str) -> bool:
    body_without_yaml = re.sub(r"^---[\s\S]*?---", "", md).strip()
    tokens = [t.lower() for t in tokenize(body_without_yaml)]
    window = tokens[:100]
    return any(tok in FIRST_PERSON_SET for tok in window)

# -------------------- Repair Prompts --------------------

def build_expand_prompt(base_text: str, min_words: int, max_words: int, negative: str, style: str, structure: str, examples: str) -> str:
    return f"""Erweitere den Artikel substanziell auf {min_words}–{max_words} Wörter (keine Wortzahl ausgeben).
- Bewahre die **vorgegebene Struktur** (YAML-SEO-Block, H1, H2/H3, nummerierte Schritte).
- Ergänze sinnvolle Abschnitte (Varianten, Packliste, Timing, Troubleshooting, schnelle Abwandlungen).
- Vermeide Wiederholungen und Floskeln (vermeide: {negative}). Aktive Verben, konkrete Zahlen/Zeiten/Mengen.

Stilguide:
{style}

Struktur-Guide:
{structure}

Beispiele (Ton):
{examples}

Text:
{base_text}
"""

def build_condense_prompt(base_text: str, min_words: int, max_words: int, negative: str, style: str, structure: str, examples: str) -> str:
    return f"""Kürze präzise auf {min_words}–{max_words} Wörter (keine Wortzahl ausgeben).
- Erhalte Kerninfos, persönliche Note und konkrete Details.
- Bewahre die **vorgegebene Struktur** (YAML-SEO-Block, H1, H2/H3, nummerierte Schritte).
- Entferne Wiederholungen und Floskeln (vermeide: {negative}). Variiere Satzlängen.

Stilguide:
{style}

Struktur-Guide:
{structure}

Beispiele (Ton):
{examples}

Text:
{base_text}
"""

def build_du_rewrite_prompt(base_text: str, negative: str, style: str) -> str:
    return f"""Passe die Leseransprache auf **du** an (du/dich/dir/dein …) und entferne alle formellen Anreden (Sie/Ihnen/Ihr …).
- **Erhalte die Ich-Perspektive als Erzähler** unverändert.
- Korrigiere Grammatik/Zeichensetzung, keine strukturellen Änderungen.
- Gib NUR den umgeschriebenen Text zurück. Vermeide Floskeln (vermeide: {negative}).

Stilguide:
{style}

Text:
{base_text}
"""

def build_ich_rewrite_prompt(base_text: str, negative: str, style: str) -> str:
    return f"""Schreibe den Text konsequent in der **Ich-Perspektive** um (ich/mir/mich/mein ...).
- Beginne **Einleitung** und **Hintergrund & Tipps** in der Ich-Form.
- „Du“-Ansprache nur für Hinweise/Tipps, aber der Erzähler bleibt „ich“.
- Entferne Floskeln (vermeide: {negative}). Korrigiere Grammatik und Zeichensetzung.
- Behalte Inhalt, Struktur und YAML-SEO-Block bei.
Gib NUR den umgeschriebenen Text zurück.

Stilguide:
{style}

Text:
{base_text}
"""

def build_structure_fix_prompt(base_text: str, primary_kw: str, negative: str, style: str, structure: str, examples: str) -> str:
    return f"""Bringe den Artikel exakt in die geforderte **Struktur** und verbessere SEO:
- YAML-SEO-Block vollständig + valide (seo_title <=60, meta_description 50–155 Zeichen, slug in kebab-case).
- H1 enthält das Primär-Keyword: "{primary_kw}".
- H2 „Einleitung“, H2 „Hintergrund & Tipps“, H2 „Rezept: …“.
- Unter Rezept: H3 „Zutaten“, H3 „Schritt für Schritt“ (6–10 nummerierte Schritte), H3 „Zeiten & Portionen“.
- Primär-Keyword natürlich in den ersten 100 Wörtern.
- Du-Ansprache, keine Floskeln (vermeide: {negative}), aktive Verben.
Gib nur den finalen Markdown-Artikel zurück.

Stilguide:
{style}

Struktur-Guide:
{structure}

Beispiele (Ton):
{examples}

Text:
{base_text}
"""

def build_coherence_fix_prompt(base_text: str, destination: str, style: str, structure: str,
                               consistency: str, negative: str, bg_mode: str = "auto") -> str:
    if bg_mode == "story":
        dest_line = f"- Nenne **{destination}** im ersten Satz von „Hintergrund & Tipps“.\n" if destination else ""
        return f"""Bringe Einleitung und „Hintergrund & Tipps“ in eine **einheitliche Szene**:
- Nutze einen **Übergangssatz** zur Einleitung oder eine klar markierte Rückblende.
{dest_line}- Behalte Struktur & YAML-SEO-Block bei. Entferne Floskeln (vermeide: {negative}). Ich-Perspektive.
- Konsistentes Vokabular: Camper, Kocher, Pfanne.

Stilguide:
{style}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Text:
{base_text}
"""
    # tips-Modus: explizit Anekdoten vermeiden
    return f"""Eröffne „Hintergrund & Tipps“ **ohne Anekdote**:
- Starte direkt mit 2–4 **konkreten Praxis-Tipps** (häufige Fehler, Kniffe, Varianten).
- **Vermeide** Formulierungen wie „Meine erste Begegnung …“, „Damals …“.
- Behalte Struktur & YAML-SEO-Block. Entferne Floskeln (vermeide: {negative}). Ich-Perspektive.

Stilguide:
{style}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Text:
{base_text}
"""

def build_plausibility_fix_prompt(base_text: str, details: str, destination: str, style: str, structure: str, consistency: str, plausibility: str) -> str:
    time_hint = ""
    tgt = details_time_target(details)
    if tgt:
        lo, hi = tgt
        time_hint = f"- Nutze im Rezept das Zeitfenster **{lo}–{hi} Minuten** (±2 Min Toleranz) deutlich in den Schritten oder im Zeitenblock.\n"
    dest_hint = f"- Halte das Reiseziel **{destination}** konsistent in Einleitung und erstem Satz von „Hintergrund & Tipps“.\n" if destination else ""
    return f"""Mache den Text camping-plausibel:
- Verwende **Pfanne** und **(Camping)Kocher**; vermeide **Backofen/Ofen** (nicht angefragt).
- Bei „wenig Abwasch“: maximal 1 zusätzliche Schüssel, wenn nötig.
{time_hint}{dest_hint}- Portionen realistisch (1–8) im Block „Zeiten & Portionen“.
- Behalte **Ich-Perspektive**, Struktur und YAML-SEO-Block.

Stilguide:
{style}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Plausibilitäts-Guide:
{plausibility}

Text:
{base_text}
"""

# -------------------- Pipeline --------------------

def guess_destination(text: str) -> str:
    s = text.lower()
    mapping = {
        "shakshuka": "Israel",
        "pad thai": "Thailand",
        "carbonara": "Italien",
        "khachapuri": "Georgien",
        "arepas": "Kolumbien",
        "laksa": "Malaysia",
        "ratatouille": "Frankreich",
        "paella": "Spanien",
        "chili": "USA",
        "bibimbap": "Südkorea",
    }
    for dish, dest in mapping.items():
        if dish in s:
            return dest
    return ""

def _wants_story(bg_mode: str, destination: str, topic: str, primary_kw: str) -> bool:
    if bg_mode == "story":
        return True
    if bg_mode == "tips":
        return False
    # auto-Heuristik: bei sehr einfachen, generischen Gerichten keine Anekdote
    txt = f"{topic} {primary_kw}".lower()
    simple_markers = [
        "spiegelei","rührei","porridge","haferbrei","nudeln","pasta",
        "salat","sandwich","brot","reis","kartoffeln","omelett","pfannkuchen",
        "tomatensoße","butterbrot"
    ]
    if any(w in txt for w in simple_markers):
        return False
    # ansonsten nur, wenn ein Reiseziel angegeben ist
    return bool(destination.strip())


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
    bg_mode: str = "auto",   # <— NEU
) -> Dict[str, Any]:

    system = SYSTEM_PROMPT_DE
    style = STYLEGUIDE_DE.format(min_words=min_words, max_words=max_words).strip()
    structure = STRUCTURE_GUIDE.strip()
    examples = STYLE_EXAMPLES_DE.strip()
    negative = ", ".join(NEGATIVE_LIST_DE)
    consistency = KONSISTENZ_GUIDE_DE.strip()
    plausibility = PLAUSIBILITY_GUIDE_DE.strip()

    pk = primary_kw.strip() if primary_kw.strip() else topic
    sk = secondary_kws.strip() if secondary_kws.strip() else "Rezept, Outdoor, Kochen, Backen"

    # --- Reiseziel-Story-Hook ---
    dest = (destination or guess_destination(f"{topic} {pk}")).strip()
    angle = (travel_angle or "Vanlife/Rundreise").strip()
    use_story = _wants_story(bg_mode, destination, topic, pk)

    travel_hook = ""
    if use_story:
        travel_hook = f"""
Zusatzanforderung (Reise-Story, nur wenn sinnvoll):
- Beginne H2 „Hintergrund & Tipps“ mit **Übergang** zur Einleitung oder sauber markierter **Rückblende**.
- Nenne **{destination}** im ersten Satz von „Hintergrund & Tipps“ und verknüpfe die Szene mit der Einleitung.
- Beziehe dich auf **{angle}** und nenne 1–2 konkrete Orte/Details. Danach folgen die Tipps.
"""

    coherence_line = COHERENCE_LINE_STORY if use_story else COHERENCE_LINE_TIPS


    # Pass 1: Draft
    draft = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": DRAFT_TEMPLATE_DE.format(
                topic=topic, details=details, primary_kw=pk, secondary_kws=sk,
                styleguide=style, structure=structure, examples=examples,
                consistency=consistency, plausibility=plausibility
            ) + (travel_hook or "") + coherence_line},
        ],
        num_predict=4096, temperature=0.8, top_p=0.9
    )

    # Pass 1.5: Clean
    cleaned = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": CLEAN_TEMPLATE_DE.format(raw=draft)},
        ],
        temperature=0.4, top_p=0.9, num_predict=4096
    )

    # Pass 2: Style-Edit
    edited = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": EDIT_TEMPLATE_DE.format(
                styleguide=style, negative=negative, draft=cleaned,
                min_words=min_words, max_words=max_words,
                structure=structure, examples=examples,
                consistency=consistency, plausibility=plausibility
            ) + (travel_hook or "") + coherence_line},
        ],
        temperature=0.6, top_p=0.9, num_predict=4096
    )

    # Qualität prüfen (Stil + Struktur + SEO + Kohärenz + Plausibilität)
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
    ok, metrics = evaluate_quality(edited, targets, pk, destination=dest, details=details, bg_mode=bg_mode)

    # Auto-Repair (bis zu 6 Versuche)
    attempts = 0
    while not ok and attempts < 6:
        attempts += 1
        sm = metrics["style_metrics"]
        stc = metrics["structure"]
        seo = metrics["seo_meta"]
        ich_intro_ok = metrics.get("ich_intro_ok", False)
        coh = metrics.get("coherence", {"ok": True})
        pl = metrics.get("plausibility", {"ok": True})

        if sm["first_person"] < targets["min_first_person"] or not ich_intro_ok:
            repair_prompt = build_ich_rewrite_prompt(edited, negative, style)
        elif not coh.get("ok", True):
            repair_prompt = build_coherence_fix_prompt(edited, dest, style, structure, consistency, negative, bg_mode=bg_mode)
        elif not pl.get("ok", True):
            repair_prompt = build_plausibility_fix_prompt(edited, details, dest, style, structure, consistency, plausibility)
        elif sm["formal_address"] > 0 or sm["second_person"] < targets["min_second_person"]:
            repair_prompt = build_du_rewrite_prompt(edited, negative, style)
        elif not all([
            stc["yaml_frontmatter"], stc["h1_present"],
            stc["h2_einleitung"], stc["h2_bg_tipps"], stc["h2_rezept"],
            stc["h3_zutaten"], stc["h3_schritte"], stc["h3_zeiten"],
            stc["steps_ok"], stc["kw_in_first100"],
            seo["title_ok"], seo["meta_ok"]
        ]):
            repair_prompt = build_structure_fix_prompt(edited, pk, negative, style, structure, examples)
        elif sm["words"] < min_words:
            repair_prompt = build_expand_prompt(edited, min_words, max_words, negative, style, structure, examples)
        elif sm["words"] > max_words:
            repair_prompt = build_condense_prompt(edited, min_words, max_words, negative, style, structure, examples)
        else:
            repair_prompt = f"""Überarbeite den Artikel gezielt:
- Halte Ich-Perspektive streng durch.
- Entferne restliche Floskeln ({negative}) und erhöhe Satzlängen-Varianz.
- Füge ≥2 konkrete Zahlen/Zeiten/Mengen hinzu.
- Halte Länge {min_words}–{max_words}, bewahre Struktur & YAML-SEO-Block.

Struktur-Guide:
{structure}

Text:
{edited}
"""

        edited = call_claude(
            model,
            [{"role": "system", "content": system},
             {"role": "user", "content": repair_prompt}],
            temperature=0.5, top_p=0.9, num_predict=4096
        )
        ok, metrics = evaluate_quality(edited, targets, pk, destination=dest, details=details, bg_mode=bg_mode)

    # Harte Absicherung: Falls immer noch zu kurz → 2 Force-Expand-Runden
    if not ok and metrics["style_metrics"]["words"] < min_words:
        for _ in range(2):
            edited = call_claude(
                model,
                [{"role": "system", "content": system},
                 {"role": "user", "content": build_expand_prompt(
                     edited, min_words, max_words, negative, style, structure, examples
                 )}],
                temperature=0.5, top_p=0.9, num_predict=4096
            )
            ok, metrics = evaluate_quality(edited, targets, pk, destination=dest, details=details, bg_mode=bg_mode)
            if ok:
                break

    return {
        "draft": draft,
        "final": edited,
        "metrics": metrics,
        "passed_heuristics": ok,
    }

# -------------------- Evaluation Wrapper --------------------

def seo_lengths(md: str) -> Dict[str, Any]:
    title = ""
    desc = ""
    m = re.search(r'seo_title:\s*"([^"]+)"', md)
    if m:
        title = m.group(1)
    m = re.search(r'meta_description:\s*"([^"]+)"', md)
    if m:
        desc = m.group(1)
    return {
        "title_len": len(title),
        "title_ok": 10 <= len(title) <= 60,
        "meta_len": len(desc),
        "meta_ok": 50 <= len(desc) <= 155
    }

def evaluate_quality(
    text: str,
    targets: Dict[str, Any],
    primary_kw: str,
    destination: str = "",
    details: str = "",
    bg_mode: str = "auto"
) -> Tuple[bool, Dict[str, Any]]:
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
    coherence = coherence_checks(text, destination, bg_mode=bg_mode)
    plaus = plausibility_checks(text, details)
    ich_intro_ok = ich_in_first100(text)

    ok = (
        not style_metrics["has_banned"]
        and style_metrics["ttr"] >= targets.get("min_ttr", 0.45)
        and style_metrics["var_sentence_len"] >= targets.get("min_var_sentence_len", 7.0)
        and style_metrics["first_person"] >= targets.get("min_first_person", 6)
        and style_metrics["numbers"] >= targets.get("min_numbers", 3)
        and targets["min_words"] <= style_metrics["words"] <= targets["max_words"]
        and style_metrics["second_person"] >= targets.get("min_second_person", 2)
        and style_metrics["formal_address"] <= targets.get("max_formal_address", 0)
        and ich_intro_ok
        and all(
            [
                structure["yaml_frontmatter"],
                structure["h1_present"],
                structure["h2_einleitung"],
                structure["h2_bg_tipps"],
                structure["h2_rezept"],
                structure["h3_zutaten"],
                structure["h3_schritte"],
                structure["h3_zeiten"],
                structure["steps_ok"],
                structure["kw_in_first100"],
                seo_meta["title_ok"],
                seo_meta["meta_ok"],
            ]
        )
        and coherence["ok"]
        and plaus["ok"]
    )
    return ok, {
        "style_metrics": style_metrics,
        "structure": structure,
        "seo_meta": seo_meta,
        "coherence": coherence,
        "plausibility": plaus,
        "ich_intro_ok": ich_intro_ok,
    }

# -------------------- WP/HTML Export --------------------

try:
    import markdown as _md
except ImportError:
    _md = None

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

FRONTMATTER_RE = re.compile(r"^---\s*\n([\s\S]*?)\n---\s*", re.MULTILINE)

def split_frontmatter(md_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Trennt YAML-Frontmatter vom Markdown-Body.
    Gibt (meta, body_md) zurück. Wenn kein Frontmatter vorhanden: ({}, originaler Text).
    """
    m = FRONTMATTER_RE.match(md_text.strip())
    if not m:
        return {}, md_text
    raw_yaml = m.group(1)
    body_md = md_text[m.end():]
    meta: Dict[str, Any] = {}
    if _yaml:
        try:
            meta = _yaml.safe_load(raw_yaml) or {}
        except Exception:
            meta = {}
    return meta, body_md

def remove_leading_h1(md_body: str) -> str:
    """
    Entfernt die allererste H1-Zeile (# ...) am Dokumentanfang (falls vorhanden).
    """
    lines = md_body.lstrip().splitlines()
    if lines and re.match(r"^\s*#\s+.+", lines[0]):
        return "\n".join(lines[1:]).lstrip()
    return md_body

def markdown_to_wp_html(md_body: str) -> str:
    """
    Wandelt den Markdown-Body (ohne Frontmatter) in HTML um – kompatibel mit WP (Gutenberg & Classic).
    """
    if not _md:
        raise RuntimeError("Das Paket 'markdown' fehlt. Bitte installieren: pip install markdown")
    # 'extra' bündelt u. a. Abkürzungen, Listen, Tabellen; 'sane_lists' verbessert Listenkonvertierung
    return _md.markdown(md_body, extensions=["extra", "sane_lists"])

def write_wp_outputs(
    final_markdown: str,
    out_md_path: str,
    strip_h1: bool = True,
    emit_meta_json: bool = True,
    html_out_path: Optional[str] = None,
    meta_json_path: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Schreibt neben der .md eine .html (WP-Content) und optional meta.json (SEO/Slug).
    - strip_h1=True: entfernt die erste H1 aus dem HTML-Content (Titel kommt in WP ins Titel-Feld).
    """
    # 1) Frontmatter abtrennen
    meta, body_md = split_frontmatter(final_markdown)

    # 2) Optional H1 im Body entfernen (WP setzt Titel separat)
    body_md_for_wp = remove_leading_h1(body_md) if strip_h1 else body_md

    # 3) HTML rendern
    wp_html = markdown_to_wp_html(body_md_for_wp)

    # 4) Pfade festlegen
    base_no_ext = re.sub(r"\.md$", "", out_md_path)
    html_out_path = html_out_path or (base_no_ext + ".html")
    meta_json_path = meta_json_path or (base_no_ext + ".meta.json")

    # 5) HTML schreiben
    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(wp_html.strip() + "\n")

    # 6) Meta schreiben (optional)
    if emit_meta_json:
        # Fallbacks, falls Frontmatter fehlt
        h1 = ""
        m_h1 = re.search(r"^#\s+(.+)", body_md, re.MULTILINE)
        if m_h1:
            h1 = m_h1.group(1).strip()

        meta_out = {
            "seo_title": meta.get("seo_title", h1 or ""),
            "meta_description": meta.get("meta_description", ""),
            "slug": meta.get("slug", ""),
            "primary_keyword": meta.get("primary_keyword", ""),
            "secondary_keywords": meta.get("secondary_keywords", []),
            # Praktisch: Titel fürs WP Titel-Feld
            "post_title": h1 or meta.get("seo_title", ""),
        }
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2)

    return html_out_path, meta_json_path if emit_meta_json else None, wp_html


# -------------------- CLI --------------------

def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Die Umgebungsvariable 'ANTHROPIC_API_KEY' fehlt. Bitte in der Shell oder .env setzen.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Artikel-Generator mit fester Struktur + SEO + Ich/DU (Claude).")
    parser.add_argument("--bg-mode", choices=["auto","tips","story"], default="auto",
    help="Start von 'Hintergrund & Tipps': 'tips' ohne Anekdote, 'story' mit Brücke, 'auto' wählt passend")
    parser.add_argument("--topic", required=True, help="Thema / Titel-Idee")
    parser.add_argument("--details", required=True, help="Pflichtdetails (kommasepariert, z. B. 'Campingkocher, 12–14 Min, wenig Abwasch')")
    parser.add_argument("--primary_kw", default="", help="Primär-Keyword (SEO). Standard: topic")
    parser.add_argument("--secondary_kws", default="", help="Sekundär-Keywords (kommagetrennt)")
    parser.add_argument("--model", default="claude-3-haiku-20240307", help="Anthropic-Modellname")
    parser.add_argument("--min-words", type=int, default=700, help="Minimale Wortzahl")
    parser.add_argument("--max-words", type=int, default=1000, help="Maximale Wortzahl")
    parser.add_argument("--destination", default="", help="Reiseziel für die Hintergrund-Story (z. B. Israel)")
    parser.add_argument("--travel_angle", default="Vanlife/Rundreise", help="Reiseperspektive (z. B. Campen, Roadtrip, Städtetrip)")
    parser.add_argument("--out", default="out.md", help="Zieldatei (Markdown)")
    parser.add_argument("--show-draft", action="store_true", help="Draft zusätzlich speichern")
    parser.add_argument("--html-out", default="", help="Pfad für die generierte WordPress-HTML-Datei (Standard: <out>.html)")
    parser.add_argument("--wp-keep-h1", action="store_true", help="H1 im HTML belassen (Standard: H1 wird für WP entfernt)")
    parser.add_argument("--no-meta-json", action="store_true", help="Kein meta.json schreiben")

    args = parser.parse_args()

    try:
        result = generate_article(
            args.topic, args.details, 
            bg_mode=args.bg_mode,
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

        # HTML/WordPress-Export
    try:
        html_path, meta_path, _html_preview = write_wp_outputs(
            final_markdown=result["final"].strip(),
            out_md_path=args.out,
            strip_h1=(not args.wp_keep_h1),
            emit_meta_json=(not args.no_meta_json),
            html_out_path=(args.html_out if args.html_out else None),
        )
        print(f"WordPress-HTML gespeichert in: {html_path}")
        if meta_path:
            print(f"WP-Metadaten (Frontmatter) gespeichert in: {meta_path}")
    except Exception as e:
        print("Fehler beim HTML/WP-Export:", e)

if __name__ == "__main__":
    main()
