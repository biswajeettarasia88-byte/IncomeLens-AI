# IncomeLens AI — Python Environment Audit Report

**Audit Date:** October 20, 2026  
**Auditor:** Senior Machine Learning Engineer  
**Project Path:** `D:\incomelens-ai`  
**Report Destination:** `reports/environment_audit.md`

---

## 1. Executive Summary

An audit of the Python runtime and dependency environment for `incomelens-ai` was conducted.

### Key Conclusions:
1. **Healthy Active Runtime:** The project's active virtual environment (`.venv`) runs on **Python 3.14.7** (64-bit Windows AMD64). All core libraries (`pandas 3.0.6`, `numpy 2.5.4`, `scikit-learn 1.9.1`, `shap 0.54.1`, `joblib 1.6.0`, `matplotlib 3.11.2`, `pytest 9.1.1`) are installed and functional.
2. **Pip Integrity Verified:** Running `python -m pip check` reports **"No broken requirements found"** (zero broken or conflicting requirements).
3. **Artifact Compatibility Intact:** The serialized production model (`models/random_forest_final.joblib`) was compiled with the exact scikit-learn (`1.9.1`) and joblib (`1.6.0`) versions in `.venv`. Test inference runs with zero deprecation or version warnings.
4. **No Environment Rebuild Warranted:** There is no evidence of environment corruption. Changing Python versions or wiping `.venv` is unnecessary and carries high risk of invalidating the serialized model.
5. **Gaps Identified:**
   - `ucimlrepo` is a direct import in `scripts/01_audit_data.py` and is installed in `.venv` (version `0.0.7`), but is **omitted from `requirements.txt`**.
   - `scipy` is directly imported in `src/incomelens/explainability.py`, installed as a transitive dependency (`1.18.1`), but not declared in `requirements.txt`.
   - `streamlit` is required by `app/`, but is **not installed in `.venv`** and **omitted from `requirements.txt`**.
   - `seaborn` (`0.13.2`) is declared and installed but **never imported** anywhere in the project.
   - An orphaned dummy package (`package_name 0.1`) and an empty secondary environment (`.venv-1`) exist.

---

## 2. Environment & Interpreter Identification

### 2.1 Python Executable & Runtime
- **Active Project Executable:** `D:\incomelens-ai\.venv\Scripts\python.exe`
- **System Python Executable:** `C:\Users\BISWAJEET\AppData\Local\Programs\Python\Python314\python.exe`
- **Python Version:** `3.14.7` (tags/v3.14.7:823f032, Aug 5 2026, 10:51:32) [MSC v.1944 64 bit (AMD64)]
- **Platform:** Windows AMD64

### 2.2 Virtual Environment Analysis
Two virtual environments are present on disk:

1. **`D:\incomelens-ai\.venv` (Active Project Environment):**
   - Config file: `D:\incomelens-ai\.venv\pyvenv.cfg`
   - Base prefix: `C:\Users\BISWAJEET\AppData\Local\Programs\Python\Python314`
   - Command used to create: `C:\Users\BISWAJEET\AppData\Local\Programs\Python\Python314\python.exe -m venv D:\incomelens-ai\.venv`
   - Contains all project dependencies (32 installed packages).
   - Confirmed as active by bytecode caches (`*.cpython-314.pyc`) and pytest execution.

2. **`D:\incomelens-ai\.venv-1` (Orphaned / Redundant Environment):**
   - Config file: `D:\incomelens-ai\.venv-1\pyvenv.cfg`
   - Created via `uv 0.11.13` (`seed = true`).
   - Contains only bare `pip 26.2.1`. Not utilized by any script or test runner.

---

## 3. Installed Packages vs. Project References

### 3.1 Primary Packages Audit
The table below details all primary packages referenced by the codebase, their installed versions, and their declaration status:

| Package | Installed Version | Declared in `requirements.txt` | Direct Import in Codebase | Usage / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **pandas** | `3.0.6` | Yes (unpinned) | Yes | Used across all scripts, data preparation, preprocessing, and tests. |
| **numpy** | `2.5.4` | Yes (unpinned) | Yes | Used in `explainability.py` (dense matrix conversion, random sampling). |
| **scikit-learn** | `1.9.1` | Yes (unpinned) | Yes (`sklearn`) | Core ML pipeline: preprocessing, imputer, scaler, OHE, RF, metrics. |
| **shap** | `0.54.1` | Yes (unpinned) | Yes | TreeExplainer, summary plot, SHAP matrix generation. |
| **joblib** | `1.6.0` | Yes (unpinned) | Yes | Model persistence and loading across scripts, explainability, tests. |
| **matplotlib** | `3.11.2` | Yes (unpinned) | Yes | Figure generation in `scripts/10` and `explainability.py`. |
| **pytest** | `9.1.1` | Yes (unpinned) | Yes | Test execution and test fixture annotations. |
| **scipy** | `1.18.1` | **No** | Yes | `from scipy import sparse` in `src/incomelens/explainability.py`. |
| **ucimlrepo** | `0.0.7` | **No** | Yes | `from ucimlrepo import fetch_ucirepo` in `scripts/01_audit_data.py`. |
| **seaborn** | `0.13.2` | Yes (unpinned) | **No** | Declared in requirements, but 0 imports found in codebase. |
| **streamlit** | *Not Installed* | **No** | *Planned* | Required by `app/Home.py` and `app/pages/*.py` (all currently 0 bytes). |
| **package_name**| `0.1` | **No** | **No** | Orphaned placeholder package in `site-packages`. |

### 3.2 Complete Distribution Inventory in `.venv` (32 Packages)
Audited via `importlib.metadata`:

1. `certifi == 2026.10.19` (HTTP SSL certificate authority; pulled by `ucimlrepo`)
2. `cloudpickle == 3.1.2` (Object serialization; pulled by `shap`)
3. `colorama == 0.4.6` (Terminal text coloring; pulled by `pytest`)
4. `contourpy == 1.4.0` (Contour calculations; pulled by `matplotlib`)
5. `cycler == 0.12.1` (Color cycler; pulled by `matplotlib`)
6. `fonttools == 4.67.0` (Font manipulation; pulled by `matplotlib`)
7. `iniconfig == 2.3.1` (INI parser; pulled by `pytest`)
8. `joblib == 1.6.0` (Parallel execution & serialization; direct dependency)
9. `kiwisolver == 1.5.1` (Constraint solver; pulled by `matplotlib`)
10. `matplotlib == 3.11.2` (Plotting library; direct dependency)
11. `narwhals == 2.27.1` (Lightweight dataframe compatibility; pulled by `pandas`)
12. `numpy == 2.5.4` (Numerical arrays; direct dependency)
13. `package_name == 0.1` (Orphaned template artifact)
14. `packaging == 26.3` (Version parsing; pulled by `matplotlib`, `pytest`, `shap`)
15. `pandas == 3.0.6` (Dataframes; direct dependency)
16. `pillow == 13.0.0` (Image processing; pulled by `matplotlib`)
17. `pip == 26.2.1` (Package installer)
18. `pluggy == 1.7.0` (Plugin management; pulled by `pytest`)
19. `Pygments == 2.21.0` (Syntax highlighting; pulled by `pytest`)
20. `pyparsing == 3.3.3` (Grammar parser; pulled by `matplotlib`)
21. `pytest == 9.1.1` (Testing framework; direct dependency)
22. `python-dateutil == 2.9.0.post0` (Date utilities; pulled by `pandas`, `matplotlib`)
23. `scikit-learn == 1.9.1` (Machine learning library; direct dependency)
24. `scipy == 1.18.1` (Scientific computing; direct import in `explainability.py`, pulled by `scikit-learn`)
25. `seaborn == 0.13.2` (Statistical plotting; declared in `requirements.txt`, unused in code)
26. `shap == 0.54.1` (Model explainability; direct dependency)
27. `six == 1.17.0` (Python 2/3 compatibility; pulled by `python-dateutil`)
28. `slicer == 0.0.8` (Array slicing; pulled by `shap`)
29. `threadpoolctl == 3.7.0` (Thread-pool controller; pulled by `scikit-learn`)
30. `tqdm == 4.70.1` (Progress bars; pulled by `shap`)
31. `tzdata == 2026.5` (Timezone data; pulled by `pandas`)
32. `ucimlrepo == 0.0.7` (UCI repository fetcher; direct import in `scripts/01_audit_data.py`)

---

## 4. AST Import Analysis vs. `requirements.txt`

An Abstract Syntax Tree (AST) scan of all `.py` files outside virtual environment directories revealed:
```
All top-level imported packages:
['incomelens', 'joblib', 'matplotlib', 'numpy', 'pandas', 'pathlib', 'pytest', 'scipy', 'shap', 'sklearn', 'sys', 'ucimlrepo']
```

### Discrepancy Breakdown:
1. **Declared but Unused:**
   - `seaborn`: Listed in `requirements.txt` and installed, but never imported in any file in `scripts/`, `src/`, or `tests/`.
2. **Used but Undeclared:**
   - `ucimlrepo`: Imported in `scripts/01_audit_data.py`. If a new developer runs `pip install -r requirements.txt`, data downloading will immediately crash with `ModuleNotFoundError`.
   - `scipy`: Directly imported in `src/incomelens/explainability.py` (`from scipy import sparse`).
3. **Application Stack Missing:**
   - `streamlit`: The directory `app/` is structured as a multi-page Streamlit application (`Home.py`, `pages/1_*.py`), but `streamlit` is neither listed in `requirements.txt` nor installed in `.venv`.
4. **Local Package Not Installed in Editable Mode:**
   - `incomelens`: All scripts rely on `sys.path.insert(0, str(ROOT / "src"))` because no `pyproject.toml` or `setup.py` exists to register `incomelens` in `.venv`.

---

## 5. Compatibility, Model Integrity, & Risk Analysis

### 5.1 Model Deserialization & Scikit-Learn Version Pinning
The production model artifact `models/random_forest_final.joblib` (50,084,507 bytes) was serialized with:
- `scikit-learn`: `1.9.1`
- `joblib`: `1.6.0`

Inference test verification:
```python
model = joblib.load('models/random_forest_final.joblib')
predictions = model.predict(test_sample)
```
Output: Runs cleanly with **zero warnings** (`InconsistentVersionWarning` = none).

> **CRITICAL DIRECTIVE:** Do **not** upgrade `scikit-learn` or `joblib` indiscriminately. Serialized scikit-learn models (`Pipeline`, `ColumnTransformer`, `RandomForestClassifier`) are not guaranteed to be backward-compatible across minor/major version bumps. Re-serializing or updating the library without retraining would risk breaking the production model artifact.

### 5.2 Python 3.14 Compatibility
Python `3.14.7` is functioning smoothly with the installed packages:
- Scikit-learn pipelines, imputer, one-hot encoding, and random forest inference operate without deprecation warnings.
- SHAP `TreeExplainer` successfully calculates feature contributions.
- Pytest discovers and runs the 7 existing test cases in 1.83s.
- There is **no justification** for downgrading or replacing Python at this stage.

---

## 6. Smallest Safe Dependency Recommendations

To resolve the identified discrepancies without destabilizing the working environment or altering models/datasets, the following minimal adjustments are recommended:

### Recommendation 1: Update `requirements.txt` (Minimal, Safe Addition)
Update `requirements.txt` to include missing direct dependencies, add minimal pins matching the active environment, and add `streamlit` for the planned UI:

```text
pandas>=3.0.0,<4.0.0
numpy>=2.0.0,<3.0.0
scikit-learn==1.9.1
joblib==1.6.0
matplotlib>=3.10.0,<4.0.0
seaborn>=0.13.0,<1.0.0
pytest>=9.0.0,<10.0.0
shap>=0.54.0,<1.0.0
scipy>=1.18.0,<2.0.0
ucimlrepo>=0.0.7
streamlit>=1.40.0
```

*Rationale:*
- Explicitly pins `scikit-learn==1.9.1` and `joblib==1.6.0` to preserve binary model serialization compatibility.
- Adds `ucimlrepo` so data extraction scripts do not fail in fresh clones.
- Adds `scipy` as an explicit dependency.
- Adds `streamlit` so the multi-page dashboard can be built and launched.

### Recommendation 2: Install `streamlit` in `.venv`
Run:
```bash
.\.venv\Scripts\python.exe -m pip install "streamlit>=1.40.0"
```
*Rationale:* Installs only the single missing package needed for the UI, leaving all other packages strictly untouched.

### Recommendation 3: Add `pyproject.toml` & Editable Install
Add standard packaging metadata:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "incomelens"
version = "0.1.0"
dependencies = [
    "pandas",
    "numpy",
    "scikit-learn==1.9.1",
    "joblib==1.6.0",
    "matplotlib",
    "scipy",
    "shap",
]
```
Then run:
```bash
.\.venv\Scripts\python.exe -m pip install -e .
```
*Rationale:* Eliminates the fragile `sys.path.insert(0, str(ROOT / "src"))` pattern and enables imports like `from incomelens.preprocessing import build_preprocessor` across tests, notebooks, and scripts.

### Recommendation 4: Remove / Ignore Redundant `.venv-1`
Delete or add `.venv-1/` to `.gitignore` to avoid confusion.
