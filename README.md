# Humanizer Pipeline (Deutsch, Ollama)

Diese kleine Pipeline erzeugt Artikel, die **weniger „KI-haft“** klingen, indem sie:
1. einen **Rohentwurf** schreibt,
2. einen **Stil-Editor-Pass** mit klaren Regeln anwendet,
3. **Heuristiken** prüft (Blockliste, Satzlängen-Varianz, TTR, Ich/Wir, Zahlen) und bei Bedarf **Auto-Rewrite** auslöst.

## Voraussetzungen
- **Ollama** lokal: https://ollama.com/download
- Ein Instruct-Modell, z. B.:
  ```bash
  ollama pull llama3.1:8b-instruct
  # oder
  ollama pull mistral:latest
  ```
- Python 3.9+ und `requests`:
  ```bash
  pip install requests
  ```

## Start
```bash
python humanizer_pipeline.py \
  --topic "One-Pot-Pasta im Van" \
  --details "Kochzeit 12-14 Min, Kichererbsen, Tomate, wenig Abwasch, Kocher mit kleinem Topf" \
  --model "llama3.1:8b-instruct" \
  --out out.md --show-draft
```

- Der finale Text landet in `out.md` (Draft optional in `out_draft.md`).
- Die Heuristiken und Metriken werden in der Konsole angezeigt.

## Inhalte anpassen
- **STYLEGUIDE_DE**: Passe den Stil an deine Marke an (camp-kochen.de ist als Default gesetzt).
- **NEGATIVE_LIST_DE**: Ergänze/ändere verbotene Floskeln.
- **Ziele** der Heuristiken (`min_ttr`, `min_var_sentence_len`, `min_first_person`, `min_numbers`) kannst du feintunen.

## Warum Ollama?
- Läuft **lokal**, keine API-Kosten.
- Modelle: Llama, Mistral, Qwen etc.
- Wenn du lieber HF/Transformers nutzen willst, kannst du die `call_ollama()`-Funktion gegen einen OpenAI-kompatiblen Client tauschen.

## Nächste Ausbaustufe (optional)
- **LoRA-Fine-Tuning** mit deinen Texten, um Stimme/Wortwahl noch genauer zu treffen.
- **Evaluator** als kleiner Klassifikator (LogReg oder Mini-Transformer), um „KI-haft“ vs. „menschlich“ auf deinem Domain-Datensatz zu unterscheiden.

Viel Spaß – und sag Bescheid, wenn ich dir eine Variante für **Transformers + PEFT (LoRA)** bauen soll oder den Stilguide mit echten Beispielabsätzen von dir füttern darf.
