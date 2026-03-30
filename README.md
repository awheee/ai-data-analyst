# AI Data Analyst (Streamlit)

Upload a dataset (`.csv` / `.xlsx`, plus best-effort `.pdf` / `.docx`) and ask questions in plain English. The app returns a computed table and (when possible) a chart.

LLM calls use **[Groq](https://console.groq.com)** (OpenAI-compatible API).

## Supported uploads
- `CSV` (`.csv`)
- `Excel` (`.xlsx`, best-effort via pandas)
- `PDF` (`.pdf`): best-effort table extraction; if no tables are found, it will show a friendly message (no charts)
- `DOCX` (`.docx`): best-effort table extraction; if no tables are found, it will show a friendly message (no charts)

## How it stays “safe”
- The LLM returns a JSON template spec (no arbitrary code execution).
- The backend runs only a small allow-list of deterministic pandas operations.
- **Dataset summary** (when you ask in chat): accurate column profile + optional sample rows are computed in Python (not guessed by the model).
- Out-of-scope questions get an explicit **“I can’t answer this…”** style response listing what *is* supported.
- File size is capped at **100MB**; very large tables are sampled for responsiveness.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Configure Groq API key

### Option A: `.streamlit/secrets.toml` (local Streamlit)
Create or edit `ai-data-analyst/.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Do **not** commit real keys to Git.

### Option B: environment variable

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

### Option C: Streamlit Cloud
In your app’s **Secrets**, add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### Optional: model override
Default model is `llama-3.1-8b-instant`. To use another Groq model:

```bash
export GROQ_MODEL="llama-3.3-70b-versatile"
```

## Deploy to Streamlit Cloud
1. Push this folder to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io) and connect your GitHub repo.
3. Select `app.py` as the entrypoint.
4. Add `GROQ_API_KEY` in secrets (see above).
5. Deploy.

---------------
