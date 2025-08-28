#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
humanizer_pipeline_with_examples.py
-----------------------------------
Zwei-Pass-Schreibpipeline (Draft -> Stil-Edit -> Heuristik-Check) mit
eingebetteten **Few-Shot-Beispielen** aus deinem Stil (Tonfall, Rhythmus, Humor).
Backend: Ollama REST API (lokal).
"""

import argparse
import json
import re
import statistics
from typing import List, Dict, Any, Tuple
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

# --- Dein Stilguide (ergänzt um Ton & Rhythmus aus den Beispielen) ---
STYLEGUIDE_DE = """
Schreibe persönlich, anschaulich und leicht verspielt – ohne Floskeln.
- Ton: warm, neugierig, gelegentlich humorvoll (dezente Wortspiele, Metaphern).
- Perspektive: Ich/Wir; direkte Ansprache ist okay (du/ihr).
- Rhythmus: gemischte Satzlängen; kurze Sätze dürfen pointieren. Rhetorische Fragen willkommen.
- Zeige konkrete Details (Mengen, Zeiten, Texturen, Geräusche). Sensorik vor Floskeln.
- Erkläre Handgriffe verständlich und entkrampfe vermeintlich „schwierige“ Schritte.
- Vermeide: „In diesem Artikel“, „abschließend“, „insgesamt“, „innovativ“, „einfach zuzubereiten“, „Fazit“.
- SEO: 1 H2 + 2–3 H3 mit natürlichen Zwischenüberschriften. Kein Keyword-Stuffing.
- Länge: 700–1000 Wörter.
"""

NEGATIVE_LIST_DE = [
    "In diesem Artikel", "abschließend", "insgesamt", "innovativ",
    "einfach zuzubereiten", "im Folgenden", "es ist wichtig zu beachten",
    "nachstehend", "zusammenfassend", "Fazit", "abschließend lässt sich sagen"
]

# --- Deine Beispielabsätze als Few-Shot-Referenz für Tonfall ---
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

SYSTEM_PROMPT_DE = """Du bist Redakteur:in mit Sinn für Wärme, klare Erklärungen und leichten Humor.
Halte den Stilguide ein und orientiere dich am Ton der Beispiele (Rhythmus, Bildsprache, Ermutigung).
"""

DRAFT_TEMPLATE_DE = """Erstelle einen Rohentwurf (700–1000 Wörter) gemäß Stilguide.
Nutze den Tonfall und Rhythmus der Beispiele.
Thema: {topic}
Pflichtdetails: {details}
Zielgruppe: Leute, die Reisen und draußen beim campen im Zelt/Van/Camper kochen wollen.
Gib NUR den Artikeltext zurück.
Stilguide:
{styleguide}

Beispiele (Tonfallreferenz):
{examples}
"""

EDIT_TEMPLATE_DE = """Überarbeite den Text gemäß Stilguide und Beispielen.
- Entferne Floskeln aus der Negativliste: {negative}
- Variiere Satzlängen (kurz + lang), erlaube rhetorische Fragen.
- Füge 1–2 konkrete Beobachtungen/Details ein (Geräusch, Textur, Zahl).
- Lasse 1 persönliche Meinung/Ermutigung stehen.
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

def tokenize(text: str):
    return WORD_RE.findall(text)

def type_token_ratio(text: str) -> float:
    tokens = [t.lower() for t in tokenize(text)]
    return (len(set(tokens)) / len(tokens)) if tokens else 0.0

def sentence_lengths(text: str):
    sentences = re.split(r"[\.!\?\n]+", text)
    lens = [len(tokenize(s)) for s in sentences if tokenize(s)]
    return lens or [0]

def variance_sentence_length(text: str) -> float:
    lens = sentence_lengths(text)
    return (statistics.pstdev(lens) if len(lens) > 1 else 0.0)

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
        results["numbers"] < targets.get("min_numbers", 2)
    )
    return fail, results

# -------------------- LLM Call --------------------

def call_ollama(model: str, messages: List[Dict[str, str]], temperature: float = 0.8, top_p: float = 0.9, repeat_penalty: float = 1.05) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "options": {"temperature": temperature, "top_p": top_p, "repeat_penalty": repeat_penalty},
        "stream": False
    }
    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()

# -------------------- Pipeline --------------------

def generate_article(topic: str, details: str, model: str = "llama3.1:8b-instruct", lang: str = "de") -> Dict[str, Any]:
    if not lang.lower().startswith("de"):
        raise ValueError("Dieses Template unterstützt aktuell nur Deutsch.")
    system = SYSTEM_PROMPT_DE
    style = STYLEGUIDE_DE.strip()
    examples = STYLE_EXAMPLES_DE.strip()
    negative = ", ".join(NEGATIVE_LIST_DE)

    draft = call_ollama(model, [
        {"role": "system", "content": system},
        {"role": "user", "content": DRAFT_TEMPLATE_DE.format(styleguide=style, examples=examples, topic=topic, details=details)},
    ])

    edited = call_ollama(model, [
        {"role": "system", "content": system},
        {"role": "user", "content": EDIT_TEMPLATE_DE.format(styleguide=style, examples=examples, negative=negative, draft=draft)},
    ])

    targets = dict(min_ttr=0.45, min_var_sentence_len=7.0, min_first_person=3, min_numbers=2)
    fail, metrics = fails_heuristics(edited, targets)

    if fail:
        repair_prompt = f"""Überarbeite den Text erneut. Ziele:
- Entferne restliche Floskeln ({negative}).
- Erhöhe Satzlängen-Varianz.
- Füge mind. 1 zusätzliche Zahl/Zeit/Menge ein.
- Ich/Wir-Perspektive sichtbar halten (≥ 3 Vorkommen).
- Gib NUR den finalen Text zurück.
Beispiele (Ton):
{examples}

Text:
{edited}"""
        edited = call_ollama(model, [
            {"role": "system", "content": system},
            {"role": "user", "content": repair_prompt},
        ])
        fail, metrics = fails_heuristics(edited, targets)

    return {"draft": draft, "final": edited, "metrics": metrics, "passed_heuristics": not fail}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--model", default="llama3.1:8b-instruct")
    parser.add_argument("--lang", default="de")
    parser.add_argument("--out", default="out.md")
    parser.add_argument("--show-draft", action="store_true")
    args = parser.parse_args()

    try:
        result = generate_article(args.topic, args.details, model=args.model, lang=args.lang)
    except requests.exceptions.ConnectionError:
        print("Fehler: Konnte nicht mit Ollama verbinden. Läuft http://localhost:11434 und ist das Modell installiert?")
        exit(2)
    except Exception as e:
        print("Fehler:", e)
        exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(result["final"].strip() + "\n")

    if args.show_draft:
        draft_path = re.sub(r"\.md$", "_draft.md", args.out)
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(result["draft"].strip() + "\n")

    print("# --- Humanizer Pipeline (mit Beispielen) ---")
    print("Heuristiken bestanden:", result["passed_heuristics"])
    print("Metriken:", json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Finaler Text gespeichert in: {args.out}")

if __name__ == "__main__":
    main()
