# IncomeLens-AI — Pre-Migration Audit

**Date:** October 26, 2026  
**Auditor:** Senior Software Architect & Lead Release Engineer  
**Scope:** Integration of `D:\income-lens-ai-frontend-development` into `D:\incomelens-ai`

---

## 1. Environment & Path Inventory

| Component | Path / Location | Details |
| :--- | :--- | :--- |
| **Current Workspace** | `D:\incomelens-ai` | Primary ML & analytics repository |
| **Git Status** | Branch `master`, commit `19698cd` | Working tree contains uncommitted files; clean commit baseline |
| **Python Virtual Env** | `D:\incomelens-ai\.venv` | Python 3.14.7, FastAPI 0.159.1, Uvicorn 0.46.0, scikit-learn 1.9.0, SHAP 0.52.0 |
| **Legacy Frontend** | `D:\incomelens-ai\app\` | Streamlit multi-page app (`Home.py` + 4 pages in `app/pages/`) |
| **New Frontend Source**| `D:\income-lens-ai-frontend-development` | Next.js 16.4.3, React 19, Tailwind CSS v4, TypeScript 5.7.3, shadcn/ui |
| **Target Frontend Dest**| `D:\incomelens-ai\frontend\` | Official integrated frontend directory inside repository |
| **Backend Service** | `src/incomelens/` | Core ML library (`inference.py`, `data.py`, `evaluation.py`, `explainability.py`, `fairness.py`, `provenance.py`) |
| **HTTP API Layer** | `src/incomelens/api.py` *(To be established)* | FastAPI application bridging frontend with model and reporting artifacts |

---

## 2. Model & Data Artifact Verification

| Asset | Path | Size | Status |
| :--- | :--- | :---: | :---: |
| Production Model | `models/random_forest_final.joblib` | 50,084,507 bytes | Verified (SHA-256 `f164a611...`) |
| Grouped Random Forest | `models/random_forest_grouped.joblib` | 39,018,267 bytes | Verified |
| Grouped Logistic Reg. | `models/logistic_regression_grouped.joblib` | 8,106 bytes | Verified |
| Training Split | `data/processed/train.csv` | 3,190,538 bytes | Verified ($N = 29,313$) |
| Validation Split | `data/processed/validation.csv` | 1,064,781 bytes | Verified ($N = 9,758$) |
| Held-Out Test Split | `data/processed/test.csv` | 1,064,895 bytes | Verified ($N = 9,771$) |
| Validation Comparison | `reports/grouped_validation_comparison.csv` | 291 bytes | Verified (LR vs RF) |
| Feature Importance CSV | `reports/feature_importance.csv` | 6,054 bytes | Verified (105 features) |
| Fairness Metrics CSV | `reports/fairness_metrics.csv` | 2,406 bytes | Verified (13 subgroups) |
| Final Test Metrics CSV | `reports/final_test_metrics.csv` | 205 bytes | Verified (0.8683 accuracy) |
| Confusion Matrix CSV | `reports/confusion_matrix.csv` | 78 bytes | Verified (7080, 417, 870, 1404) |
| Feature Importance PNG | `reports/figures/feature_importance.png` | 293,175 bytes | Verified (300 DPI) |
| SHAP Summary PNG | `reports/figures/shap_summary.png` | 406,590 bytes | Verified (300 DPI) |

---

## 3. Pre-Migration Known Issues & Gaps

1. **Disconnected Frontend:** The newly generated frontend in `D:\income-lens-ai-frontend-development` contains mockup state in `workspace-app.tsx` and does not communicate with the Python backend.
2. **Missing HTTP API:** The Python backend currently interacts only through Python API calls and Streamlit UI. A standardized REST API (FastAPI) is needed to serve model predictions, explainability, fairness, and dataset metadata.
3. **Streamlit Redundancy:** The existing Streamlit app in `app/` will be maintained as legacy/backup while the Next.js app in `frontend/` becomes the official frontend.
4. **Form Schema Mismatch:** The prototype predict form in the new frontend included only 6 fields, whereas the production model pipeline strictly expects the 14 raw survey attributes.
5. **Node.js Environment:** Node v24.14.0 and npm 11.9.0 are present; dependencies need to be installed in `frontend/` using `npx pnpm install` or `npm install`.

---

## 4. Migration Strategy & Safety Guardrails

- **Backup:** Create `D:\migration_backups\incomelens_pre_migration_<timestamp>\` capturing the existing frontend configuration and reports before modification.
- **Copy Target:** Safely replicate `D:\income-lens-ai-frontend-development` into `D:\incomelens-ai\frontend\`, excluding cache and build folders.
- **Source Preservation:** The original `D:\income-lens-ai-frontend-development` folder will remain intact until all migration acceptance tests pass.
- **Model Protection:** `models/*.joblib` and `data/processed/*.csv` are treated as immutable read-only assets.
