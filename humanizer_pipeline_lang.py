#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
humanizer_pipeline.py (length-fixed)
------------------------------------
Zwei-Pass-Schreibpipeline (Draft -> Stil-Edit -> Heuristik-Check + Auto-Repair) für natürlich klingende Artikel.
Backend: Ollama REST API (lokal), z. B. mit `llama3.1:8b-instruct` oder `mistral:latest`.

Änderungen ggü. deiner Version:
- Harte Wortlängen-Ziele (min/max) als Heuristik + Auto-Repair-Loop.
- `num_predict` standardmäßig hoch gesetzt, damit das Modell genug Token generiert.
- Zusätzliche Repair-Prompts (Expand/Condense), bis Zielbereich erreicht ist.
- CLI-Flags `--min-words` und `--max-words` zum Feintuning.

Install & Start:
    1) https://ollama.com/download
    2) Modell ziehen, z. B.:  ollama pull llama3.1:8b-instruct
    3) Server läuft automatisch lokal auf http://localhost:11434
Run:
    python humanizer_pipeline.py --topic "One-Pot-Pasta im Van" --details "Kochzeit 12-14 Min, Kichererbsen, Tomate, wenig Abwasch, Kocher mit kleinem Topf" --out out.md

Optional:
    --model "llama3.1:latest"
    --lang de  (oder en -> derzeit nur de aktiv)
    --min-words 700 --max-words 1000
"""

import argparse
import json
import math
import random
import re
import statistics
import sys
from typing import List, Dict, Any, Tuple, Optional
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

STYLEGUIDE_DE = """
Schreibe wie „camp-kochen.de“: pragmatisch, persönlich, ohne Füllfloskeln, 700–1000 Wörter.
- Perspektive: Ich/Wir beim Draußen-Kochen, kleine Beobachtungen. Gegenüber duzen.
- Variiere Satzlängen. Kurze Sätze sind erlaubt.
- Konkrete Details (Mengen, Zeiten, Geräusche/Anfühlen), keine Leerphrasen.
- Vermeide: „In diesem Artikel“, „abschließend“, „insgesamt“, „innovativ“, „köstlich“, „einfach zuzubereiten“, „im Folgenden“, „es ist wichtig zu beachten“, „nachstehend“.
- Nutze aktive Verben. Keine Aufzählungshölle.
- SEO: 1 prägnante H2, 2–3 H3; Keywords natürlich einbinden, kein Keyword-Stuffing.
- Ton: unaufgeregt, modern, minimalistisch.
- Länge: schreibe zwischen {min_words} und {max_words} Wörtern. Gib keine Wortzahl aus.
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
    "einfach zuzubereiten", "im Folgenden", "es ist wichtig zu beachten", "nachstehend",
    "zusammenfassend", "Fazit"
]

SYSTEM_PROMPT_DE = """Du bist Redakteurin für camp-kochen.de.
Deine Aufgabe: hilfreiche, konkrete, persönliche Texte mit natürlichem Rhythmus verfassen.
Halte den folgenden Stilguide strikt ein.
"""

DRAFT_TEMPLATE_DE = """Erstelle einen Rohentwurf für einen Artikel gemäß Stilguide.
Thema: {topic}
Pflichtdetails: {details}
Zielgruppe: Menschen auf Reisen, die draußen im Van/Zelt/Camper kochen.
Länge: zwischen {min_words} und {max_words} Wörtern. Gib NUR den Artikeltext zurück, ohne Vorbemerkungen oder Wortzahlen.
Stilguide:
{styleguide}
"""

EDIT_TEMPLATE_DE = """Überarbeite den folgenden Text gemäß Stilguide:
- Entferne Floskeln aus der Negativliste: {negative}
- Variiere Satzlängen (kurz + lang).
- Füge 1–2 konkrete Beobachtungen aus der Draußen-Situation ein (Geräusche, Geruch, Textur).
- Lasse 1 persönliche Meinung stehen.
- Verwende aktive Verben.
- Halte die Länge zwischen {min_words} und {max_words} Wörtern (keine Wortzahl ausgeben).
- Gib NUR den finalen Text zurück.
Stilguide:
{styleguide}

Beispiele (Tonfallreferenz):
{examples}

Text:
{draft}
"""

# -------------------- Heuristics --------------------

# Zusätzliche Heuristik-Ziele
SECOND_PERSON_WORDS = [
    "du", "dich", "dir", "dein", "deine", "deinen", "deinem", "deiner", "deines"
]
# Vorsicht: "Ihr/ihr" kann auch Possessiv/Plural sein; wir zielen hier v. a. auf formelle Anrede mit großem S.
FORMAL_ADDRESS_WORDS = [
    r"Sie", r"Ihnen", r"Ihr", r"Ihre", r"Ihrem", r"Ihren", r"Ihrer", r"Ihres"
]

BANNED_PATTERNS = [re.compile(re.escape(p), flags=re.IGNORECASE) for p in NEGATIVE_LIST_DE]

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-']+", flags=re.UNICODE)

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)

def word_count(text: str) -> int:
    return len(tokenize(text))

def type_token_ratio(text: str) -> float:
    tokens = [t.lower() for t in tokenize(text)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def sentence_lengths(text: str) -> List[int]:
    # naive split: ., !, ? and newlines.
    sentences = re.split(r"[\.!\?\n]+", text)
    lens = [len(tokenize(s)) for s in sentences if tokenize(s)]
    return lens or [0]

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
    # Zähle explizit groß geschriebene formelle Anredewörter
    return sum(len(re.findall(w, text)) for w in FORMAL_ADDRESS_WORDS)

def fails_heuristics(text: str, targets: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    ttr = type_token_ratio(text)
    var = variance_sentence_length(text)
    fp = count_first_person(text)
    nums = count_numbers(text)
    wc = word_count(text)
    du_ct = count_second_person(text)
    formal_ct = count_formal_address(text)

    results = {
        "ttr": ttr,
        "var_sentence_len": var,
        "first_person": fp,
        "numbers": nums,
        "words": wc,
        "second_person": du_ct,
        "formal_address": formal_ct,
        "has_banned": has_banned_phrases(text),
    }
    fail = (
        results["has_banned"] or
        results["ttr"] < targets.get("min_ttr", 0.45) or
        results["var_sentence_len"] < targets.get("min_var_sentence_len", 7.0) or
        results["first_person"] < targets.get("min_first_person", 3) or
        results["numbers"] < targets.get("min_numbers", 3) or
        results["words"] < targets.get("min_words", 700) or
        results["words"] > targets.get("max_words", 1000) or
        results["second_person"] < targets.get("min_second_person", 4) or
        results["formal_address"] > targets.get("max_formal_address", 0)
    )
    return fail, results

# -------------------- LLM Call --------------------

CLEAN_TEMPLATE_DE = """Korrigiere den folgenden Text in deutscher Sprache:
- Behebe Rechtschreibung, Grammatik, Zeichensetzung und schiefe Formulierungen.
- Ersetze förmliche Anrede (Sie/Ihnen/Ihr ...) **konsequent** durch die lockere Du-Form (du/dich/dir/dein ...).
- Vereinheitliche Ton: direkt, klar, modern, ohne Füllfloskeln.
- Behalte Inhalt, Reihenfolge und Fakten bei.
- Gib NUR den bereinigten Text zurück, ohne Erklärungen.

Text:
{raw}
"""

# -------------------- LLM Call --------------------

def call_ollama(model: str, messages: List[Dict[str, str]], temperature: float = 0.8, top_p: float = 0.9, repeat_penalty: float = 1.05, num_predict: int = 2048) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            # Wichtig: hoch genug, damit 700–1000 Wörter möglich sind
            "num_predict": num_predict,
        },
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns { "message": { "content": "..." }, ... }
    return data.get("message", {}).get("content", "").strip()

# -------------------- Repair helpers --------------------

def build_expand_prompt(base_text: str, min_words: int, max_words: int, negative: str, style: str) -> str:
    return f"""Erweitere den Text substanziell, bis die Länge zwischen {min_words} und {max_words} Wörtern liegt (keine Wortzahl ausgeben).
- Ergänze sinnvolle Abschnitte zu: Zutatenersatz/Varianten, Packliste für Kocher & Topf, Timing/Feuerkontrolle, Troubleshooting, schnelle Abwandlungen.
- Bewahre Ton und Struktur, keine Wiederholungen, keine Füllfloskeln (vermeide: {negative}).
- Nutze aktive Verben und streue konkrete Zahlen/Zeiten/Mengen ein.

Stilguide:
{style}

Text:
{base_text}
"""

def build_condense_prompt(base_text: str, min_words: int, max_words: int, negative: str, style: str) -> str:
    return f"""Kürze den Text präzise auf {min_words}–{max_words} Wörter (keine Wortzahl ausgeben).
- Erhalte Kerninfos, persönliche Note und konkrete Details.
- Entferne Wiederholungen und Füllfloskeln (vermeide: {negative}).
- Variiere Satzlängen.

Stilguide:
{style}

Text:
{base_text}
"""

# -------------------- Pipeline --------------------

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

def generate_article(topic: str, details: str, model: str = "llama3.1:latest", lang: str = "de", min_words: int = 700, max_words: int = 1000) -> Dict[str, Any]:
    if lang.lower().startswith("de"):
        system = SYSTEM_PROMPT_DE
        style = STYLEGUIDE_DE.format(min_words=min_words, max_words=max_words).strip()
        draft_prompt = DRAFT_TEMPLATE_DE.format(topic=topic, details=details, styleguide=style, min_words=min_words, max_words=max_words)
        negative = ", ".join(NEGATIVE_LIST_DE)
        edit_template = EDIT_TEMPLATE_DE
    else:
        raise ValueError("Only 'de' supported in this template.")

    # Pass 1: Draft
    draft = call_ollama(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": draft_prompt},
        ],
    )

    # Pass 1.5: Cleaning (Korrekturen + Duzen-Erzwingung)
    cleaned = call_ollama(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": CLEAN_TEMPLATE_DE.format(raw=draft)},
        ],
        temperature=0.4,
        num_predict=2048,
    )

    # Pass 2: Style Edit
    edited = call_ollama(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": edit_template.format(styleguide=style, negative=negative, draft=cleaned, min_words=min_words, max_words=max_words, examples=STYLE_EXAMPLES_DE)},
        ],
        temperature=0.6,
        num_predict=2048,
    )

    # Heuristics
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
    fail, metrics = fails_heuristics(edited, targets)

    # Auto-Repair loop (max 3 Versuche)
    attempts = 0
    while fail and attempts < 3:
        attempts += 1
        wc = metrics.get("words", 0)
        if metrics.get("formal_address", 0) > 0 or metrics.get("second_person", 0) < 4:
            repair_prompt = build_du_rewrite_prompt(edited, negative, style)
        elif wc < min_words:
            repair_prompt = build_expand_prompt(edited, min_words, max_words, negative, style)
        elif wc > max_words:
            repair_prompt = build_condense_prompt(edited, min_words, max_words, negative, style)
        else:
            # sonst allgemeine Reparatur
            repair_prompt = f"""Überarbeite den Text erneut. Ziele:
- Entferne restliche Floskeln aus: {negative}
- Erhöhe Satzlängen-Varianz (mix sehr kurz + deutlich länger).
- Füge mind. 2 konkrete Zahlen/Zeiten/Mengen hinzu.
- Stelle konsequent die Du-Ansprache sicher und vermeide formelle Anrede vollständig.
- Halte die Länge strikt zwischen {min_words} und {max_words} Wörtern (keine Wortzahl ausgeben).

Text:
{edited}"""
        edited = call_ollama(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0.5,
            num_predict=2048,
        )
        fail, metrics = fails_heuristics(edited, targets)

def main():
    parser = argparse.ArgumentParser(description="Humanizer Pipeline")
    parser.add_argument("--topic", required=True, help="The topic of the article")
    parser.add_argument("--details", required=True, help="Details for the article")
    parser.add_argument("--model", default="llama3.1:latest", help="Model to use")
    parser.add_argument("--lang", default="de", help="Language of the article")
    parser.add_argument("--min-words", type=int, default=700, help="Minimum word count")
    parser.add_argument("--max-words", type=int, default=1000, help="Maximum word count")
    parser.add_argument("--out", required=True, help="Output file path")
    parser.add_argument("--show-draft", action="store_true", help="Save draft version")
    args = parser.parse_args()

    try:
        result = generate_article(args.topic, args.details, model=args.model, lang=args.lang, min_words=args.min_words, max_words=args.max_words)
    except requests.exceptions.ConnectionError:
        print("Fehler: Konnte nicht mit Ollama verbinden. Läuft der Server auf http://localhost:11434 und ist das Modell installiert?")
        sys.exit(2)
    except Exception as e:
        print("Fehler:", e)
        sys.exit(1)

    # Save
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(result["final"].strip() + "\n")

    if args.show_draft:
        draft_path = re.sub(r"\.md$", "_draft.md", args.out)
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(result["draft"].strip() + "\n")

    # Console summary
    print("# --- Humanizer Pipeline ---")
    print("Heuristiken bestanden:", result["passed_heuristics"])
    print("Metriken:", json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Finaler Text gespeichert in: {args.out}")

if __name__ == "__main__":
    main()
