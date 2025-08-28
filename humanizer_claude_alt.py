#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
humanizer_pipeline.py (Claude-Version)
--------------------------------------
Zwei-Pass-Schreibpipeline mit:
- fester Artikelstruktur + SEO-Block (YAML Frontmatter)
- Du-Ansprache & Sprach-/Grammatik-Clean-Pass
- Wortlängen-Range (min/max) + Heuristiken
- Auto-Repair (Expand/Kürzen/Du/Struktur/SEO), max. 3 Versuche

Erfordert:
  pip install anthropic
  ANTHROPIC_API_KEY als Umgebungsvariable

Beispiel:
python humanizer_pipeline.py \
  --topic "Karpatka unterwegs im Van" \
  --details "Brandteig, Puddingcreme, Campingkocher, 12–14 Min kochen, wenig Abwasch" \
  --primary_kw "Karpatka Rezept" \
  --secondary_kws "Brandteig, Puddingcreme, Windbeutelkuchen" \
  --out karpatka.md --min-words 700 --max-words 1000
"""

import argparse
import json
import re
import statistics
import sys
from typing import List, Dict, Any, Tuple
# Removed unused import

# --- Anthropic (Claude) ---
try:
    import anthropic
except ImportError as e:
    raise SystemExit(
        "Das 'anthropic'-Paket fehlt. Bitte installieren: pip install anthropic"
    )

# -------------------- Stil + Beispiele --------------------

STYLEGUIDE_DE = """
Schreibe wie „camp-kochen.de“: pragmatisch, persönlich, ohne Füllfloskeln.
- Perspektive: Ich/Wir draußen beim Kochen; direkte Du-Ansprache.
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
    "nachstehend", "zusammenfassend", "Fazit"
]

SYSTEM_PROMPT_DE = """Du bist Redakteur:in für camp-kochen.de.
Deine Aufgabe: hilfreiche, konkrete, persönliche Texte mit natürlichem Rhythmus verfassen.
Halte den Stilguide strikt ein und schreibe konsequent in der Du-Ansprache.
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

Beispiele (Tonfallreferenz):
{examples}
"""

EDIT_TEMPLATE_DE = """Überarbeite den Text gemäß Stil- und Struktur-Guide.
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

Beispiele (Tonfallreferenz):
{examples}

Text:
{draft}
"""

CLEAN_TEMPLATE_DE = """Korrigiere den folgenden Text in deutscher Sprache:
- Behebe Rechtschreibung, Grammatik, Zeichensetzung und schiefe Formulierungen.
- Ersetze formelle Anrede (Sie/Ihnen/Ihr ...) konsequent durch die Du-Form (du/dich/dir/dein ...).
- Vereinheitliche Ton: direkt, klar, modern, ohne Füllfloskeln.
- Behalte Inhalt und Struktur bei.
Gib NUR den bereinigten Text zurück.

Text:
{raw}
"""

# -------------------- Heuristiken + Struktur/SEO-Checks --------------------

BANNED_PATTERNS = [re.compile(re.escape(p), flags=re.IGNORECASE) for p in NEGATIVE_LIST_DE]
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-']+", flags=re.UNICODE)
SECOND_PERSON_WORDS = ["du","dich","dir","dein","deine","deinen","deinem","deiner","deines"]
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

def check_structure(md: str, primary_kw: str) -> Dict[str, Any]:
    checks = {}
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

def evaluate_quality(text: str, targets: Dict[str, Any], primary_kw: str) -> Tuple[bool, Dict[str, Any]]:
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

    ok = (
        not style_metrics["has_banned"] and
        style_metrics["ttr"] >= targets.get("min_ttr", 0.45) and
        style_metrics["var_sentence_len"] >= targets.get("min_var_sentence_len", 7.0) and
        style_metrics["first_person"] >= targets.get("min_first_person", 3) and
        style_metrics["numbers"] >= targets.get("min_numbers", 3) and
        targets["min_words"] <= style_metrics["words"] <= targets["max_words"] and
        style_metrics["second_person"] >= targets.get("min_second_person", 4) and
        style_metrics["formal_address"] <= targets.get("max_formal_address", 0) and
        all([
            structure["yaml_frontmatter"], structure["h1_present"],
            structure["h2_einleitung"], structure["h2_bg_tipps"], structure["h2_rezept"],
            structure["h3_zutaten"], structure["h3_schritte"], structure["h3_zeiten"],
            structure["steps_ok"], structure["kw_in_first100"],
            seo_meta["title_ok"], seo_meta["meta_ok"]
        ])
    )
    return ok, {"style_metrics": style_metrics, "structure": structure, "seo_meta": seo_meta}

# -------------------- LLM Call (Claude) --------------------

def _anthropic_text_from_content(resp: anthropic.types.Message) -> str:
    """Extrahiere zusammenhängenden Text aus der Anthropic-Antwort."""
    parts = []
    for block in resp.content:
        if block.type == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()

def call_claude(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.8,
    top_p: float = 0.9,
    # Removed unused parameter
    num_predict: int = 6000
) -> str:
    """
    Ruft Claude (Anthropic Messages API) auf.
    - 'messages' enthält Rollen 'system' und 'user' (optional auch 'assistant' für Verlaufsfortsetzung).
    - 'system' wird an 'system' übergeben; die restlichen werden als Messages geschickt.
    """
    client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY aus der Umgebung

    # Anthropic erwartet genau ein 'system' (optional) + messages mit user/assistant-Turns.
    system_txt = ""
    msg_list = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            # falls mehrere system-Msgs: zusammenführen
            if system_txt:
                system_txt += "\n\n" + m.get("content", "")
            else:
                system_txt = m.get("content", "")
        else:
            msg_list.append({"role": role, "content": m.get("content", "")})

    # Fallback: mindestens eine user-Message muss vorhanden sein
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
    return f"""Schreibe den Text konsequent in der Du-Ansprache um (du/dich/dir/dein ...), entferne alle formellen Anreden (Sie/Ihnen/Ihr ...).
Korrigiere Grammatik, Rechtschreibung und Zeichensetzung.
Keine Erklärungen, nur den umgeschriebenen Text.

Tabuwörter/Floskeln: {negative}

Stilguide:
{style}

Text:
{base_text}
"""

def build_structure_fix_prompt(base_text: str, primary_kw: str, negative: str, style: str, structure: str, examples: str) -> str:
    return f"""Bringe den Artikel exakt in die geforderte **Struktur** und verbessere SEO:
- YAML-SEO-Block vollständig + valide (seo_title ≤60, meta_description 50–155 Zeichen, slug in kebab-case).
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
        "bibimbap": "Südkorea"
    }
    for dish, dest in mapping.items():
        if dish in s:
            return dest
    return ""

def generate_article(topic: str, details: str, primary_kw: str = "", secondary_kws: str = "",
                     model: str = "claude-3-5-sonnet-20240620",
                     min_words: int = 700, max_words: int = 1000,
                     destination: str = "", travel_angle: str = "Vanlife/Rundreise") -> Dict[str, Any]:

    system = SYSTEM_PROMPT_DE
    style = STYLEGUIDE_DE
    structure = STRUCTURE_GUIDE
    examples = STYLE_EXAMPLES_DE
    negative = ", ".join(NEGATIVE_LIST_DE)
    pk = primary_kw.strip() if primary_kw.strip() else topic
    sk = secondary_kws.strip() if secondary_kws.strip() else "Rezept, Outdoor, Kochen, Backen"

    # --- Reiseziel-Story-Hook ---
    dest = (destination or guess_destination(f"{topic} {pk}")).strip()
    angle = (travel_angle or "Vanlife/Rundreise").strip()
    travel_hook = ""
    if dest:
        travel_hook = f"""
Zusatzanforderung (Reiseziel-Story):
- Beginne **H2 „Hintergrund & Tipps“** mit einer **4–6 Sätze** langen, persönlichen Mini-Story über **{dest}**.
- Beziehe dich auf **{angle}** (z. B. Campen/Roadtrip), nenne **1–2 konkrete Orte/Routen/Details** (Geräusch, Geruch, Licht).
- Erkläre in 1–2 Sätzen, **warum das Gericht dazu passt**. Danach folgen die eigentlichen Tipps.
"""

            {"role": "system", "content": SYSTEM_PROMPT_DE},
    draft = call_claude(
        model,
        [
                    styleguide=STYLEGUIDE_DE, structure=STRUCTURE_GUIDE, examples=STYLE_EXAMPLES_DE
            {"role": "user", "content":
                DRAFT_TEMPLATE_DE.format(
                    topic=topic, details=details, primary_kw=pk, secondary_kws=sk,
                    styleguide=style, structure=structure, examples=examples
                ) + (travel_hook or "")
            },
        ],
        num_predict=8000, temperature=0.8, top_p=0.9
    )

    # Pass 1.5
    cleaned = call_claude(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": CLEAN_TEMPLATE_DE.format(raw=draft) + (travel_hook or "")},
        ],
        temperature=0.4, top_p=0.9, num_predict=8000
    )

    # Pass 2
    edited = call_claude(
        model,
                    styleguide=STYLEGUIDE_DE, negative=", ".join(NEGATIVE_LIST_DE), draft=cleaned,
            {"role": "system", "content": system},
            {"role": "user", "content":
                EDIT_TEMPLATE_DE.format(
                    styleguide=style, negative=negative, draft=cleaned,
                    min_words=min_words, max_words=max_words,
                    structure=structure, examples=examples
                ) + (travel_hook or "")
            },
        ],
        temperature=0.6, top_p=0.9, num_predict=8000
    )
    # Qualität prüfen (Stil + Struktur + SEO)
    targets = dict(
        min_ttr=0.45,
        min_var_sentence_len=7.0,
        min_first_person=3,
        min_numbers=3,
        min_words=min_words,
        max_words=max_words,
        min_second_person=4,
        max_formal_address=0,
    )
    ok, metrics = evaluate_quality(edited, targets, pk)

    # Auto-Repair (max 3 Versuche)
    attempts = 0
    while not ok and attempts < 6:
        attempts += 1
        sm = metrics["style_metrics"]
        repair_prompt = build_du_rewrite_prompt(edited, ", ".join(NEGATIVE_LIST_DE), STYLEGUIDE_DE)
        seo = metrics["seo_meta"]

        if sm["formal_address"] > 0 or sm["second_person"] < targets["min_second_person"]:
            repair_prompt = build_du_rewrite_prompt(edited, negative, style)
        elif not all([
            stc["yaml_frontmatter"], stc["h1_present"], stc["h2_einleitung"],
            repair_prompt = build_structure_fix_prompt(edited, pk, ", ".join(NEGATIVE_LIST_DE), STYLEGUIDE_DE, STRUCTURE_GUIDE, STYLE_EXAMPLES_DE)
            stc["h3_schritte"], stc["h3_zeiten"], stc["steps_ok"], stc["kw_in_first100"],
            repair_prompt = build_expand_prompt(edited, min_words, max_words, ", ".join(NEGATIVE_LIST_DE), STYLEGUIDE_DE, STRUCTURE_GUIDE, STYLE_EXAMPLES_DE)
        ]):
            repair_prompt = build_condense_prompt(edited, min_words, max_words, ", ".join(NEGATIVE_LIST_DE), STYLEGUIDE_DE, STRUCTURE_GUIDE, STYLE_EXAMPLES_DE)
        elif sm["words"] < min_words:
            repair_prompt = build_expand_prompt(edited, min_words, max_words, negative, style, structure, examples)
        elif sm["words"] > max_words:
            repair_prompt = build_condense_prompt(edited, min_words, max_words, negative, style, structure, examples)
        else:
            repair_prompt = f"""Überarbeite den Artikel gezielt:
- Entferne restliche Floskeln ({negative}) und erhöhe Satzlängen-Varianz.
- Füge ≥2 konkrete Zahlen/Zeiten/Mengen hinzu.
- Halte Länge {min_words}–{max_words}, bewahre Struktur & YAML-SEO-Block.
- Du-Ansprache, aktive Verben.

Struktur-Guide:
{structure}

Text:
            [{"role": "system", "content": SYSTEM_PROMPT_DE},

        edited = call_claude(
            model,
            [{"role": "system", "content": system},
             {"role": "user", "content": repair_prompt}],
            temperature=0.5,
            top_p=0.9,
            num_predict=4096
        )
        ok, metrics = evaluate_quality(edited, targets, pk)

    if not ok:
        # falls noch zu kurz → 2 Force-Expand-Runden versuchen
        if metrics["style_metrics"]["words"] < min_words:
            for _ in range(2):
                edited = call_claude(
                    model,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": build_expand_prompt(
                            edited, min_words, max_words, negative, style, structure, examples
                        )}
                    ],
                    temperature=0.5, top_p=0.9, num_predict=8000
                )
                ok, metrics = evaluate_quality(edited, targets, pk)
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
    parser = argparse.ArgumentParser(description="Artikel-Generator mit fester Struktur + SEO + Du-Ansprache (Claude).")
    parser.add_argument("--topic", required=True, help="Thema / Titel-Idee")
    parser.add_argument("--details", required=True, help="Pflichtdetails (kommasepariert)")
    parser.add_argument("--primary_kw", default="", help="Primär-Keyword (SEO). Standard: topic")
    parser.add_argument("--secondary_kws", default="", help="Sekundär-Keywords (kommagetrennt)")
    parser.add_argument("--model", default="claude-3-haiku-20240307", help="Anthropic-Modellname")
    parser.add_argument("--lang", default="de", help="Sprache (nur 'de')")
    parser.add_argument("--min-words", type=int, default=700, help="Minimale Wortzahl")
    parser.add_argument("--max-words", type=int, default=1000, help="Maximale Wortzahl")
    parser.add_argument("--out", default="out.md", help="Zieldatei (Markdown)")
    parser.add_argument("--show-draft", action="store_true", help="Draft zusätzlich speichern")
    args = parser.parse_args()

    try:
        result = generate_article(
            args.topic, args.details,
            primary_kw=args.primary_kw, secondary_kws=args.secondary_kws,
            model=args.model, lang=args.lang,
            min_words=args.min_words, max_words=args.max_words
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
    print("Heuristiken/Struktur/SEO OK:", result["passed_heuristics"])
    print("Metriken:", json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Finaler Text gespeichert in: {args.out}")

if __name__ == "__main__":
    main()
