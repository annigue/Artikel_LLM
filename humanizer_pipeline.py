#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
humanizer_pipeline.py
---------------------
Zwei-Pass-Schreibpipeline (Draft -> Stil-Edit -> Heuristik-Check) für natürlich klingende Artikel.
Backend: Ollama REST API (lokal), z. B. mit `llama3.1:8b-instruct` oder `mistral:latest`.
Install & Start:
    1) https://ollama.com/download
    2) Modell ziehen, z. B.:  ollama pull llama3.1:8b-instruct
    3) Server läuft automatisch lokal auf http://localhost:11434
Run:
    python humanizer_pipeline.py --topic "One-Pot-Pasta im Van" --details "Kochzeit 12-14 Min, Kichererbsen, Tomate, wenig Abwasch, Kocher mit kleinem Topf" --out out.md

Optional:
    --model "llama3.1:latest"
    --lang de  (oder en)
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

SYSTEM_PROMPT_DE = """Du bist Redakteur:in für camp-kochen.de.
Deine Aufgabe: hilfreiche, konkrete, persönliche Texte mit natürlichem Rhythmus verfassen.
Halte den folgenden Stilguide strikt ein.
"""

DRAFT_TEMPLATE_DE = """Erstelle einen Rohentwurf für einen Artikel (700–1000 Wörter) gemäß Stilguide.
Thema: {topic}
Pflichtdetails: {details}
Zielgruppe: Menschen auf Reisen, die draußen im Van/Zelt/Camper kochen.
Gib NUR den Artikeltext zurück, ohne Vorbemerkungen.
Stilguide:
{styleguide}
"""

EDIT_TEMPLATE_DE = """Überarbeite den folgenden Text gemäß Stilguide:
- Entferne Floskeln aus der Negativliste: {negative}
- Variiere Satzlängen (kurz + lang).
- Füge 1–2 konkrete Beobachtungen aus der Draußen-Situation ein (Geräusche, Geruch, Textur).
- Lasse 1 persönliche Meinung stehen.
- Verwende aktive Verben.
- Gib NUR den finalen Text zurück.
Stilguide:
{styleguide}

Beispiele (Tonfallreferenz):
{examples}

Text:
{draft}
"""

# -------------------- Heuristics --------------------

BANNED_PATTERNS = [re.compile(re.escape(p), flags=re.IGNORECASE) for p in NEGATIVE_LIST_DE]

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-']+", flags=re.UNICODE)

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)

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

def fails_heuristics(text: str, targets: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    ttr = type_token_ratio(text)
    var = variance_sentence_length(text)
    fp = count_first_person(text)
    nums = count_numbers(text)

    results = {
        "ttr": ttr,
        "var_sentence_len": var,
        "first_person": fp,
        "numbers": nums,
        "has_banned": has_banned_phrases(text),
    }
    fail = (
        results["has_banned"] or
        results["ttr"] < targets.get("min_ttr", 0.45) or
        results["var_sentence_len"] < targets.get("min_var_sentence_len", 7.0) or
        results["first_person"] < targets.get("min_first_person", 3) or
        results["numbers"] < targets.get("min_numbers", 3)
    )
    return fail, results

# -------------------- LLM Call --------------------

def call_ollama(model: str, messages: List[Dict[str, str]], temperature: float = 0.8, top_p: float = 0.9, repeat_penalty: float = 1.05) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        },
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns { "message": { "content": "..." }, ... }
    return data.get("message", {}).get("content", "").strip()

# -------------------- Pipeline --------------------

def generate_article(topic: str, details: str, model: str = "llama3.1:latest", lang: str = "de") -> Dict[str, Any]:
    if lang.lower().startswith("de"):
        system = SYSTEM_PROMPT_DE
        style = STYLEGUIDE_DE.strip()
        draft_prompt = DRAFT_TEMPLATE_DE.format(topic=topic, details=details, styleguide=style)
        edit_prompt_tmpl = EDIT_TEMPLATE_DE
        negative = ", ".join(NEGATIVE_LIST_DE)
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

    # Pass 2: Style Edit
    edited = call_ollama(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": edit_prompt_tmpl.format(styleguide=style, negative=negative, draft=draft)},
        ],
    )

    # Heuristics
    targets = dict(min_ttr=0.45, min_var_sentence_len=7.0, min_first_person=3, min_numbers=3)
    fail, metrics = fails_heuristics(edited, targets)

    # Optional Auto-Repair if failed
    if fail:
        repair_prompt = f"""Überarbeite den Text erneut. Ziele:
- Entferne restliche Floskeln aus: {negative}
- Erhöhe Satzlängen-Varianz (mix sehr kurz + deutlich länger).
- Füge mind. 2 konkrete Zahlen/Zeiten/Mengen hinzu.
- Lass die Ich/Wir-Perspektive sichtbar (mind. 3 Vorkommen).
- Gib NUR den finalen Text zurück.

Text:
{edited}"""
        edited = call_ollama(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": repair_prompt},
            ],
        )
        # Re-check
        fail, metrics = fails_heuristics(edited, targets)

    return {
        "draft": draft,
        "final": edited,
        "metrics": metrics,
        "passed_heuristics": not fail,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=True, help="Kurzes Thema/Arbeitstitel")
    parser.add_argument("--details", type=str, required=True, help="Pflichtdetails (kommasepariert)")
    parser.add_argument("--model", type=str, default="llama3.1:8b-instruct", help="Ollama Modellname")
    parser.add_argument("--lang", type=str, default="de", help="Sprachcode, derzeit nur 'de'")
    parser.add_argument("--out", type=str, default="out.md", help="Datei für finalen Text")
    parser.add_argument("--show-draft", action="store_true", help="Draft zusätzlich abspeichern")
    args = parser.parse_args()

    try:
        result = generate_article(args.topic, args.details, model=args.model, lang=args.lang)
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
