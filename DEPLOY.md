# Deploy to GitHub + Streamlit Cloud

## Already done locally

- `.gitignore` excludes `.streamlit/secrets.toml` and other junk.
- Initial commit is on branch `main`.

## Push to GitHub (you run these once)

1. Create an empty repository on GitHub (no README) named e.g. `ai-data-analyst`.
2. In this folder:

```bash
cd /Users/acosmovi/zuckerberg/ai-data-analyst
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git push -u origin main
```

Use SSH if you prefer: `git@github.com:YOUR_USER/YOUR_REPO.git`

## Streamlit Community Cloud

1. Open https://share.streamlit.io and sign in with GitHub.
2. **New app** → select this repository and branch **main**.
3. **Main file path:** `app.py`
4. **App settings → Secrets** — paste:

```toml
GROQ_API_KEY = "gsk_...."
```

5. Save / deploy. Your app URL will look like `https://YOUR_APP.streamlit.app`.

See also [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example).
