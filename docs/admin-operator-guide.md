# FirstLook Scan — Admin & Operator Guide

This guide is for operators hosting or maintaining the FirstLook Scan demo.

---

## Deployment Model

The demo is hosted on **Streamlit Community Cloud** with:
- GitHub-based deployment
- Secrets managed in Streamlit settings
- No local secrets committed

---

## Required Secrets

Configured in Streamlit Cloud → App → Settings → Secrets:

```toml
OPENAI_API_KEY = "sk-..."
DEMO_ACCESS_CODE = "demo-code"
