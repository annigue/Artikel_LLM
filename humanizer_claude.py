#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
humanizer_pipeline.py (Claude-Version)
--------------------------------------
Zwei-Pass-Schreibpipeline mit:
- fester Artikelstruktur + SEO-Block (YAML Frontmatter)
- Clean-Pass (Rechtschreibung/Grammatik, Ich-Perspektive, du im Rezeptteil)
- Wortlängen-Range (min/max) + Heuristiken
- Auto-Repair (Ich/du/Struktur/SEO/Story/Expand/Kürzen), bis zu 6 Versuche
- Reise-Mini-Story ODER Herkunftsnote am Anfang von „Hintergrund & Tipps“
- Kohärenz-Check (Einleitung ↔ Hintergrund andocken) + Sensorik-Heuristik

Erfordert:
  pip install anthropic
  ANTHROPIC_API_KEY als Umgebungsvariable

Beispiel:
python humanizer_pipeline.py \
  --topic "Shakshuka" \
  --details "Tomaten, Paprika, Eier, Campingkocher" \
  --primary_kw "Shakshuka Rezept" \
  --secondary_kws "Israel, Levante, Camping" \
  --destination "Israel" \
  --travel_angle "Roadtrip / Vanlife" \
  --story_mode auto --story_len medium --story_places "Carmel-Markt,Jaffa,Route 90" \
  --out shakshuka.md --min-words 700 --max-words 1000
"""

import argparse
import json
import os
import re
import statistics
import sys
from typing import List, Dict, Any, Tuple

# --- Anthropic (Claude) ---
try:
    import anthropic
except ImportError:
    print("Das 'anthropic'-Paket fehlt. Bitte installieren: pip install anthropic")
    sys.exit(1)

# Optionaler ENV-Check (nicht hart abbrechen, nur Hinweis)
if not os.getenv("ANTHROPIC_API_KEY"):
    print("Hinweis: 'ANTHROPIC_API_KEY' ist nicht gesetzt. Setze ihn als Umgebungsvariable, sonst schlägt der API-Call fehl.")

# -------------------- Stil + Beispiele --------------------

STYLEGUIDE_DE = """
Schreibe wie „camp-kochen.de“: pragmatisch, persönlich, ohne Füllfloskeln.
- Erzählerstimme: **Ich-Perspektive** (ich/mir/mich/mein) in allen Abschnitten außer dem Rezeptteil.
- Leseransprache „du“ ist erlaubt für Hinweise/Handgriffe; im Abschnitt **„Schritt für Schritt“: direkte du-/Imperativ-Form**.
- Variiere Satzlängen. Kurze Sätze sind erlaubt.
- Konkrete Details (Mengen, Zeiten, Geräusche/Textur), keine Leerphrasen.
- Vermeide: „In diesem Artikel“, „abschließend“, „insgesamt“, „innovativ“, „köstlich“,
  „einfach zuzubereiten“, „im Folgenden“, „es ist wichtig zu beachten“, „nachstehend“.
- Aktive Verben. Keine sterile Aufzählung.
- Länge: zwischen {min_words} und {max_words} Wörtern. Gib keine Wortzahl aus.
"""

STYLE_EXAMPLES_DE = """
Beispiel 1:
dieses polnische Karpatka Windbeutelkuchen-Rezept mit Puddingcreme-Füllung hat bisher alle meine Besucherherzen verzaubert. Kein Wunder, stellt euch mal vor, einen riesig großen Windbeutel zu vernaschen.

Beispiel 2 (Erklärung + Bildsprache):
Wie seine kleinen Cousinen, die Windbeutel oder Eclairs, besteht dieser Kuchen aus zwei Elementen: Den Böden, nämlich aus Brandteig oder pâte à choux (wie die Franzosen zu sagen pflegen) und einer feinen Vanille-Buttercremefüllung. Für letztere kocht man zunächst einen Pudding und mischt diesen nach dem Abkühlen mit weicher Butter. Dann wird alles zusammengesetzt und zack, fühlt man sich wie in den Bergen. Im schneebedeckten Karpaten Hochgebirge beispielsweise, daher auch der Name. Im englischsprachigen Raum liest man auch häufig von Mountain Cakes in diesem Zusammenhang.

Beispiel 3 (Hürde entkrampfen, Humor):
Ähnlich wie beim Hefeteig lese ich bei Lesern häufig, dass sie sich bis dato nicht recht an Brandteig heran trauen. Ich verstehe das ja. Brand klingt erst mal nach tatütata und die Tatsache, dass der Teig beispielsweise im Topf abgebrannt wird und dort einen weißen Belag hinterlassen soll klingt so gar nicht nach gesundem Backverstand. Kommen die Eier zum Teigkloß, kommen weitere Zweifel, denn zunächst streuben sich beide Parteien partout, zusammenzufinden und man gedenkt kurz, aufzugeben und sich lieber ne flotte Tomatenstulle zu machen. Aber nein, bitte bleibt am Ball, das wird. Wer einmal den Brandteig-Dreh heraus hat, wird dies demnächst im Schlaf backen können.

Beispiel 4 (fachlich + bildlich):
Man macht sich beim Backen mit Brandteig verschiedene physikalische Eigenschaften zu Nutze, um ein luftiges Gebäck zu erhalten, ohne jedoch Backtriebmittel wie Backpulver, Natron oder Hefe verwenden zu müssen. Beim Backen kann dann die gebundene Feuchtigkeit nicht als Wasserdampf durch die Kruste aus verkleisterter Stärke (durch das Abbrennen) entweichen, es entstehen Hohlräume, die das Gebäck schön aufplustern. So kommt der Kuchen auch zu seinem Gebirge, bevor ein Schneesturm aus Puderzucker darüber herfällt.
"""

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
Im Abschnitt **„Schritt für Schritt“** konsequent **du-/Imperativ-Form**.
Halte den Stilguide strikt ein.
"""

KONSISTENZ_GUIDE_DE = """
Kohärenz & Story-Führung:
- Halte Einleitung und „Hintergrund & Tipps“ in derselben Szene/Erzählzeit, ODER markiere Wechsel sauber als Rückblende (z. B. „Meine erste Begegnung …“, „Ein paar Tage zuvor …“).
- Wenn ein Reiseziel angegeben ist, nenne es in der Einleitung **und** zu Beginn von „Hintergrund & Tipps“.
- Vermeide konkurrierende Start-Szenen; nutze im Hintergrund einen Übergangssatz, der klar an die Einleitung anknüpft.
- Nutze ein konsistentes Vokabular: „Camper“, „Kocher“, „Pfanne“, „Vanlife/Rundreise“.
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
- **H2 Hintergrund & Tipps**: (Szene oder Herkunftsnote + 2–4 Sätze praktische Tipps als Fließtext, **keine Liste**).
- **H2 Rezept: <Kurzbezeichnung>**
  - **H3 Zutaten** (Liste mit Mengenangaben)
  - **H3 Schritt für Schritt** (nummerierte Liste, 6–10 Schritte, du-/Imperativ-Form)
  - **H3 Zeiten & Portionen** (Zubereitung, Gesamtzeit, Ruhe-/Backzeit, Portionen)
**Keine zusätzlichen H2/H3 außer den genannten.** Optional am Ende: 1–2 interne Link-Ideen (ohne URL), als Fließtext.
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

Beispiele (Tonfallreferenz):
{examples}
"""

EDIT_TEMPLATE_DE = """Überarbeite den Text gemäß Stil- und Struktur-Guide.
- **Ich-Perspektive** durchgängig, **Einleitung** und **Hintergrund & Tipps** starten in Ich-Form und docken aneinander an.
- Leser darf als „du“ adressiert werden, aber nicht die Erzählerrolle übernehmen.
- **Rezeptteil „Schritt für Schritt“: du-/Imperativ-Form**, nummeriert (6–10 Schritte).
- Tipps in „Hintergrund & Tipps“ als **Fließtext (2–4 Sätze, keine Liste)**.
- Entferne Floskeln aus der Negativliste: {negative}
- Wahrung der Struktur (YAML-SEO-Block, H1, H2/H3 exakt wie vorgegeben), **keine zusätzlichen H2/H3**.
- Nutze Primär-Keyword in H1 + Einleitung (erste 100 Wörter) + mindestens 1 H2.
- Variiere Satzlängen; erlaube rhetorische Fragen.
- Füge 1–2 konkrete Beobachtungen/Details ein (Geräusch, Textur, Zahl).
- Mengenangaben bei Zutaten prüfen/ergänzen.
- Länge strikt {min_words}–{max_words} Wörter (keine Wortzahl ausgeben).
Gib **nur** den finalen Markdown-Artikel zurück.

Stilguide:
{styleguide}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Beispiele (Tonfallreferenz):
{examples}

Text:
{draft}
"""

CLEAN_TEMPLATE_DE = """Korrigiere den folgenden Text in deutscher Sprache:
- Behebe Rechtschreibung, Grammatik, Zeichensetzung und schiefe Formulierungen.
- Schreibe konsequent in der **Ich-Perspektive** (ich/mir/mich/mein). Ersetze unpersönliche „man“-Konstruktionen
  und unpassende „wir“-Formen durch „ich“, außer wenn „wir“ inhaltlich zwingend ist.
- „Du“-Ansprache für Hinweise/Tipps bleibt erlaubt; im Abschnitt **„Schritt für Schritt“** klare **du-/Imperativ-Form**.
- Vermeide formelle Anrede (Sie/Ihnen/Ihr ...).
- Behalte Inhalt und Struktur bei.
Gib NUR den bereinigten Text zurück.

Text:
{raw}
"""

COHERENCE_LINE = "\n\nWICHTIG: Einleitung und „Hintergrund & Tipps“ bilden **eine** Erzählung; im Hintergrund zuerst **an die Einleitung andocken**, nicht neu anfangen."

# -------------------- Heuristiken + Struktur/SEO-Checks --------------------

BANNED_PATTERNS = [re.compile(re.escape(p), flags=re.IGNORECASE) for p in NEGATIVE_LIST_DE]
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-']+", flags=re.UNICODE)
SECOND_PERSON_WORDS = ["du", "dich", "dir", "dein", "deine", "deinen", "deinem", "deiner", "deines"]
FORMAL_ADDRESS_WORDS = [r"\bSie\b", r"\bIhnen\b", r"\bIhr\b", r"\bIhre\b", r"\bIhrem\b", r"\bIhren\b", r"\bIhrer\b", r"\bIhres\b"]

QUICK_DISH_HINTS = {
    "porridge","toast","sandwich","omelett","omelette","rührei","wrap",
    "smoothie","quark","skyr","joghurt","overnight oats","grießbrei"
}

SENSORY_LEX = [
    "duft", "geruch", "aroma", "rauch", "rauschen", "wind", "knistern",
    "brutzeln", "singen", "salzluft", "wärme", "kühle", "licht", "dämmerung",
    "glut", "sand", "staub", "samtig", "knackig", "schmelzend"
]

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

def is_quick_recipe(topic: str, details: str) -> bool:
    s = f"{topic} {details}".lower()
    return any(k in s for k in QUICK_DISH_HINTS)

def first_paragraph(text: str) -> str:
    return text.split("\n\n", 1)[0] if text else ""

def count_sensory_words(text: str) -> int:
    t = text.lower()
    return sum(t.count(w) for w in SENSORY_LEX)

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
    checks["h3_schritte"] = bool(re.search(r"^###\s+Schritt\s+für\s+Schritt", md, reMULTILINE:=re.MULTILINE))
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

def coherence_checks(md: str, destination: str) -> Dict[str, Any]:
    intro = extract_section(md, "Einleitung")
    bg = extract_section(md, "Hintergrund & Tipps")

    # Ziel in Einleitung & am Anfang des Hintergrunds
    intro_has_dest = contains_destination(intro, destination)
    bg_has_dest = contains_destination(bg[:200], destination) if bg else True

    # Hintergrund beginnt mit Brücke / Anschluss an Einleitung
    first_bg_sentence = ""
    if bg:
        first_bg_sentence = re.split(r"[.!?]\s", bg.strip(), maxsplit=1)[0].lower()
    bridge_markers = ["meine erste begegnung", "ein paar tage zuvor", "später", "damals", "hier", "dort"]
    bridge_ok = any(k in first_bg_sentence for k in bridge_markers) or (
        bool(destination) and destination.lower() in first_bg_sentence
    )

    # Kennzahlen erste Hintergrund-Passage
    bg_para = first_paragraph(bg)
    bg_sentences = [s for s in re.split(r"[.!?]+", bg_para) if tokenize(s)]
    bg_sensory = count_sensory_words(bg_para)

    ok = intro_has_dest and bg_has_dest and bridge_ok
    return {
        "intro_has_dest": intro_has_dest,
        "bg_has_dest": bg_has_dest,
        "bridge_ok": bridge_ok,
        "bg_first_para_sentence_count": len(bg_sentences),
        "bg_first_para_sensory": bg_sensory,
        "ok": ok
    }

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

# -------------------- LLM Call (Claude) --------------------

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
    client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY aus der Umgebung
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

# -------------------- Story Hooks --------------------

def sentence_span(story_len: str) -> Tuple[int, int]:
    if story_len == "short": return (3, 4)
    if story_len == "long":  return (8, 10)
    return (5, 7)  # medium

def build_travel_story_hook(dest: str, angle: str, places_csv: str, smin: int, smax: int) -> str:
    place_line = f"- Verwende diese konkreten Bezüge: {places_csv}.\n" if places_csv.strip() else ""
    return f"""
Zusatzanforderung (Reise-Mini-Story):
- Beginne **„Hintergrund & Tipps“** mit **{smin}–{smax} Sätzen** in **Ich-Perspektive** als **konkrete Szene** (Zeitpunkt, Ort, Handlung).
- Verknüpfe **direkt** mit der Einleitung (kein neuer Start) oder markiere sauber als Rückblende.
- Erwähne **{dest}** im ersten Satz und beziehe dich auf **{angle}**.
{place_line}- Nutze mind. **2 Sinnesdetails** (Geruch/Geräusch/Licht/Temperatur) und **1 Logistikdetail** (Straße/Markt/Stellplatz).
- Schliesse mit **warum das Gericht genau dort passt** und leite zur Tipp-Phase über.
"""

def build_history_hook(topic: str, primary_kw: str, smin: int, smax: int) -> str:
    label = primary_kw or topic
    return f"""
Alternative (kulinarische Herkunft):
- Starte **„Hintergrund & Tipps“** mit **{smin}–{smax} Sätzen** zur **Herkunft/Kultur** von „{label}“.
- **Keine exakten Jahreszahlen/umstrittenen Behauptungen**; allgemeiner Kontext (Region, typische Zutaten, Essanlass), wertfrei.
- Danach **direkt** in praktische Tipps übergehen (2–3 Sätze, keine Liste).
"""

def build_story_enrich_prompt(base_text: str, mode: str, dest: str, angle: str,
                              smin: int, smax: int, places: str,
                              style: str, structure: str, consistency: str, negative: str) -> str:
    if mode == "travel":
        p_line = f"- Nutze diese Bezüge: {places}.\n" if places.strip() else ""
        return f"""Erweitere die Reise-Mini-Story am Anfang von „Hintergrund & Tipps“:
- Schreibe **{smin}–{smax} Sätze** in Ich-Perspektive als **konkrete Szene** (Zeitpunkt, Ort, Handlung).
- Erster Satz nennt **{dest}**, Bezug auf **{angle}**. {p_line}- Mindestens **2 Sinnesdetails** und **1 Logistikdetail**.
- Schliesse mit **warum das Gericht dazu passt** und leite in **2–3 Sätze** praktische Tipps über (Fließtext, keine Liste).
- Kein neuer Szenenstart; sauber an die Einleitung andocken.

Stilguide:
{style}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Vermeide Floskeln: {negative}

Text:
{base_text}
"""
    else:  # history
        return f"""Ersetze/ergänze den Auftakt von „Hintergrund & Tipps“ durch eine **{smin}–{smax}-sätzige** Herkunfts-/Kultur-Mini-Note:
- Allgemeiner Kontext (Region, typische Zutaten, Essanlass), **keine exakten Jahreszahlen/umstrittenen Behauptungen**.
- Danach direkt und knapp in praktische Tipps (2–3 Sätze, keine Liste) überleiten.
- Ich-Perspektive in der Überleitung beibehalten.

Stilguide:
{style}

Struktur-Guide:
{structure}

Konsistenz-Guide:
{consistency}

Vermeide Floskeln: {negative}

Text:
{base_text}
"""

# -------------------- Pipeline-Checks --------------------

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

    # Story-Targets
    story_type = targets.get("story_type", "auto")
    min_story_sent = targets.get("story_min_sent", 0)
    min_sensory = targets.get("story_min_sensory", 0)
    story_ok = True
    if story_type in ("travel", "history"):
        story_ok = (
            coherence["bg_first_para_sentence_count"] >= min_story_sent and
            (coherence["bg_first_para_sensory"] >= min_sensory if story_type == "travel" else True)
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
        story_ok
    )
    return ok, {
        "style_metrics": style_metrics,
        "structure": structure,
        "seo_meta": seo_meta,
        "coherence": coherence,
        "story_requirements": {
            "type": story_type,
            "min_sentences": min_story_sent,
            "min_sensory": (min_sensory if story_type == "travel" else 0),
            "met": story_ok
        },
        "ich_intro_ok": ich_intro_ok
    }

# -------------------- Ziel-Ort Heuristik --------------------

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

# -------------------- Haupt-Pipeline --------------------

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
    story_mode: str = "auto",
    story_len: str = "medium",
    story_places: str = "",
) -> Dict[str, Any]:

    system = SYSTEM_PROMPT_DE
    style = STYLEGUIDE_DE.format(min_words=min_words, max_words=max_words).strip()
    structure = STRUCTURE_GUIDE.strip()
    examples = STYLE_EXAMPLES_DE.strip()
    negative = ", ".join(NEGATIVE_LIST_DE)
    consistency = KONSISTENZ_GUIDE_DE.strip()

    pk = primary_kw.strip() if primary_kw.strip() else topic
    sk = secondary_kws.strip() if secondary_kws.strip() else "Rezept, Outdoor, Kochen, Backen"

    # --- Story-Modus bestimmen ---
    auto_quick = is_quick_recipe(topic, details)
    if story_mode == "auto":
        eff_story = "history" if auto_quick else "travel"
    else:
        eff_story = story_mode

    smin, smax = sentence_span(story_len)

    dest = (destination or guess_destination(f"{topic} {pk}")).strip()
    angle = (travel_angle or "Vanlife/Rundreise").strip()

    hook_text = ""
    if eff_story == "travel" and dest:
        hook_text = build_travel_story_hook(dest, angle, story_places, smin, smax)
    elif eff_story == "history":
        hook_text = build_history_hook(topic, pk, max(3, smin-1), max(5, smax-2))

    # Pass 1: Draft
    draft = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": DRAFT_TEMPLATE_DE.format(
                topic=topic, details=details, primary_kw=pk, secondary_kws=sk,
                styleguide=style, structure=structure, examples=examples, consistency=consistency
            ) + (hook_text or "") + COHERENCE_LINE},
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
                structure=structure, examples=examples, consistency=consistency
            ) + (hook_text or "") + COHERENCE_LINE},
        ],
        temperature=0.6, top_p=0.9, num_predict=4096
    )

    # Qualität prüfen (Stil + Struktur + SEO + Story)
    targets = dict(
        min_ttr=0.45,
        min_var_sentence_len=7.0,
        min_first_person=6,
        min_numbers=3,
        min_words=min_words,
        max_words=max_words,
        min_second_person=2,
        max_formal_address=0,
        story_type=eff_story,
        story_min_sent=(smin if eff_story in ("travel","history") else 0),
        story_min_sensory=(1 if eff_story == "travel" else 0),
    )
    ok, metrics = evaluate_quality(edited, targets, pk, destination=dest)

    # Auto-Repair (bis zu 6 Versuche)
    attempts = 0
    while not ok and attempts < 6:
        attempts += 1
        sm = metrics["style_metrics"]
        stc = metrics["structure"]
        seo = metrics["seo_meta"]
        ich_intro_ok = metrics.get("ich_intro_ok", False)
        coh = metrics.get("coherence", {"ok": True})
        story_req = metrics.get("story_requirements", {"type": "auto", "met": True})

        if sm["first_person"] < targets["min_first_person"] or not ich_intro_ok:
            repair_prompt = build_ich_rewrite_prompt(edited, negative, style)
        elif story_req.get("type") in ("travel","history") and not story_req.get("met"):
            repair_prompt = build_story_enrich_prompt(
                edited, story_req["type"], dest, angle, smin, smax, story_places,
                style, structure, consistency, negative
            )
        elif not coh.get("ok", True):
            repair_prompt = build_story_enrich_prompt(
                edited, (eff_story if eff_story in ("travel","history") else "history"),
                dest, angle, smin, smax, story_places, style, structure, consistency, negative
            )
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
- Halte Ich-Perspektive streng durch; im Rezeptteil **du-/Imperativ-Form**.
- Entferne restliche Floskeln ({negative}) und erhöhe Satzlängen-Varianz.
- Füge ≥2 konkrete Zahlen/Zeiten/Mengen hinzu.
- Halte Länge {min_words}–{max_words}, bewahre Struktur & YAML-SEO-Block (keine zusätzlichen H2/H3).

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
        ok, metrics = evaluate_quality(edited, targets, pk, destination=dest)

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
            ok, metrics = evaluate_quality(edited, targets, pk, destination=dest)
            if ok:
                break

    return {
        "draft": draft,
        "final": edited,
        "metrics": metrics,
        "passed_heuristics": ok,
    }

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Artikel-Generator mit fester Struktur + SEO + Ich/du (Claude).")
    parser.add_argument("--topic", required=True, help="Thema / Titel-Idee")
    parser.add_argument("--details", required=True, help="Pflichtdetails (kommasepariert)")
    parser.add_argument("--primary_kw", default="", help="Primär-Keyword (SEO). Standard: topic")
    parser.add_argument("--secondary_kws", default="", help="Sekundär-Keywords (kommagetrennt)")
    parser.add_argument("--model", default="claude-3-haiku-20240307", help="Anthropic-Modellname")
    parser.add_argument("--min-words", type=int, default=700, help="Minimale Wortzahl")
    parser.add_argument("--max-words", type=int, default=1000, help="Maximale Wortzahl")
    parser.add_argument("--destination", default="", help="Reiseziel für die Hintergrund-Story (z. B. Israel)")
    parser.add_argument("--travel_angle", default="Vanlife/Rundreise", help="Reiseperspektive (z. B. Campen, Roadtrip, Städtetrip)")
    parser.add_argument("--story_mode", choices=["auto","travel","history","none"], default="auto", help="Erzähllogik")
    parser.add_argument("--story_len", choices=["short","medium","long"], default="medium", help="Länge der Mini-Story")
    parser.add_argument("--story_places", default="", help="Konkrete Orte/Routen/Details, kommasepariert")
    parser.add_argument("--out", default="out.md", help="Zieldatei (Markdown)")
    parser.add_argument("--show-draft", action="store_true", help="Draft zusätzlich speichern")
    args = parser.parse_args()

    try:
        result = generate_article(
            args.topic, args.details,
            primary_kw=args.primary_kw, secondary_kws=args.secondary_kws,
            model=args.model, min_words=args.min_words, max_words=args.max_words,
            destination=args.destination, travel_angle=args.travel_angle,
            story_mode=args.story_mode, story_len=args.story_len, story_places=args.story_places
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

    print("# --- Humanizer Pipeline (Struktur + SEO / Claude) ---")
    print("Heuristiken/Struktur/Story OK:", result["passed_heuristics"])
    print("Metriken:", json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Finaler Text gespeichert in: {args.out}")
    print("Kohärenz:", json.dumps(result["metrics"].get("coherence", {}), ensure_ascii=False))
    print("Story-Reqs:", json.dumps(result["metrics"].get("story_requirements", {}), ensure_ascii=False))

if __name__ == "__main__":
    main()
