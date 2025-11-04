## What Happens in the Prompt Stays in the Prompt: Quantifying Persona Consistency in LLM-Generated Persuasive Text

### Purpose
This project measures how a large language model (LLM) adapts persuasive messaging when it is told who the recipient is. The user always asks for help convincing a friend to join a girls’ trip to Las Vegas; only the persona description of that friend changes. By holding the core request constant and varying the persona context, we can quantify how tailoring impacts similarity across responses.

### Repository Layout
- `data/model_inputs/personas.jsonl` - canonical persona descriptions (key + neutral sentence).
- `data/bronze/` - raw model outputs (`responses.jsonl`).
- `data/silver/` - cleaned tabular data (`responses.csv`).
- `data/gold/` - analysis outputs (per-message deltas, persona summaries, centroid similarities).
- `figures/` - diagnostic plots generated during cosine-similarity analysis.
- `power_analysis.py` - effect-size driven sample estimator.
- `generate_responses.py` - orchestrates persona prompts and records Ollama responses.
- `clean_responses.py` - extracts the text inside parentheses and computes word counts.
- `analyze_cosine_similarity.py` - embeds responses, compares personas, and saves statistics/plots.

### Personas (default descriptions)
- **planner** - budget-conscious organizer who wants clear logistics and cost transparency.
- **party_captain** - high-energy friend who hates missing out on memorable group experiences.
- **homebody** - prefers relaxed, tasteful plans and low-key environments.
- **wellness_weekender** - balances light nightlife with restorative activities.
- **culture_seeker** - values substance, enrichment, and memorable experiences over pure spectacle.
- **control** - no persona cue (blank description).

Edit `data/model_inputs/personas.jsonl` if you need to adjust or extend personas; each line must remain a JSON object with `persona_key` and `persona_description` fields.

### Experiment Workflow
1. **Size the sample** - Run `python power_analysis.py` to estimate the total and per-persona responses required. The script assumes a one-way ANOVA with Cohen’s *f* = 0.25, α = 0.05, power = 0.80, and six groups (five personas + control). Adjust the constants if you expect a different effect size or design.
2. **Collect responses** - Pull or install your Ollama model (default `llama3`), then execute `python generate_responses.py --repeat N`, where `N` is the number of runs you want per persona. The script reads personas from `data/model_inputs/personas.jsonl`, builds the standard prompt (mentioning the persona sentence and requesting a verbatim reply inside parentheses), and writes every response to `data/bronze/responses.jsonl`, overwriting any previous file.
3. **Clean and summarize** - Run `python clean_responses.py`. It parses `data/bronze/responses.jsonl`, keeps the substring after the first `(` (trimming trailing `)`/quotes), counts words, and saves `data/silver/responses.csv` with columns `persona_key`, `iteration`, `response`, and `word_count`.
4. **Analyze cosine similarity** - Run `python analyze_cosine_similarity.py`. The script loads `data/silver/responses.csv`, generates sentence embeddings with `all-MiniLM-L6-v2`, and evaluates how each message aligns with (a) its own persona’s centroid versus (b) all other personas combined. Outputs include:
   - `data/gold/delta_similarity_per_message.csv` - within/between cosine scores and Δ per entry.
   - `data/gold/delta_summary_by_persona.csv` - per-persona means, standard deviations, counts, and 95% confidence intervals.
   - `data/gold/persona_centroid_similarity_matrix.csv` - cosine similarities between persona centroids.
   - `figures/delta_histogram.png`, `figures/delta_qqplot.png`, `figures/persona_centroid_similarity_heatmap.png` - distribution checks and centroid heatmap.
5. **Interpret findings** - Review the gold-layer CSVs and figures to see whether persona cues reliably increase within-persona similarity (Δ > 0), whether specific personas diverge, and which personas cluster together.

### Dependencies
- Python 3.9+
- Stats & data: `statsmodels`, `numpy`, `pandas`, `scipy`, `seaborn`, `matplotlib`
- Embeddings: `sentence-transformers`, `torch`
- Ollama runtime with the target local model (`ollama run llama3 ...`).

Install Python packages via `pip install statsmodels numpy pandas scipy seaborn matplotlib sentence-transformers torch` (or your preferred environment manager). Ensure the `figures/` directory exists before running the analysis if you want plots saved.

### Notes & Tips
- The request prompt is intentionally deterministic aside from the persona description; rerun `generate_responses.py` with the same `--repeat` to regenerate a fresh dataset.
- If the LLM omits parentheses, `clean_responses.py` records a blank `response` and zero `word_count`, making it easy to filter incomplete outputs.
- You can swap `--model` on `generate_responses.py` to compare different Ollama models or rewrite the template if you experiment with alternative phrasing - just keep the parenthetical instruction if you rely on the cleaner as-is.
- `analyze_cosine_similarity.py` downloads the embedding model on first use; allow extra time for that initial run.

This pipeline keeps raw, cleaned, and analyzed data separate (bronze → silver → gold) so you can reproduce each stage, re-run with new personas, or plug additional metrics into the analysis step without contaminating earlier results.

### AI Use Disclosure
The structure and formatting of this README were generated with AI assistance and then reviewed for accuracy on 11/4/2025.