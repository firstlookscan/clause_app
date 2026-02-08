# FirstLook Scan

**FirstLook Scan** is a deal triage tool that scans contracts and surfaces
deal-relevant clauses that routinely drive diligence friction and post-close risk.

It is designed to answer one core question quickly:

> **Which contracts should we look at first?**

---

## What FirstLook Scan Does

- Ingests contracts (`.pdf`, `.docx`, `.txt`)
- Extracts *verbatim* excerpts for key clause categories
- Assigns triage-level risk (High / Medium / Low)
- Organizes findings issue-first (not document-first)
- Produces a contract risk heatmap and deal-level summary

**Important:**  
FirstLook Scan is a **triage tool**, not legal advice and not a replacement for legal review.

---

## Clause Categories (v1)

FirstLook Scan currently looks for five clause types commonly escalated in early diligence:

- **Termination**
- **Change of Control**
- **MFN (Most Favored Nation)**
- **Revenue / Spend Commitments**
- **Exclusivity**

Each finding includes:
- `excerpt` (verbatim contract language)
- `risk_level` (High / Medium / Low)
- `confidence` (0.0–1.0)
- `why_this_matters` (plain-English rationale)

---

## How Risk Is Interpreted

Risk levels reflect **deal triage urgency**, not legal enforceability.

- **High** → commonly escalated, potential deal friction
- **Medium** → notable, context-dependent
- **Low** → present but typically limited

Confidence is used for filtering noise, not as a guarantee of correctness.

---

## Dashboard Views

### Contract Risk Heatmap
- Rows = contracts
- Columns = clause types
- Cells = *worst risk found* for that clause type in that contract
- Contracts sorted by total risk score

### Top Risks
- Sorted by risk level, then confidence
- Helps teams identify what to review first

### Issue-First Drilldown
- Findings grouped by clause type
- Displays verbatim excerpts and rationale

---

## Demo Mode

The hosted demo runs in **guard-railed demo mode**:
- Access-code protected
- File size and count limits
- Scan limits per session
- Confidence floor applied
- Includes a curated “Sample Deal” dataset

**Do not upload confidential documents into the demo environment.**

---

## Local Development

### Install
```bash
pip install -r requirements.txt

