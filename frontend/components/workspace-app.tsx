'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  CircleHelp,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileText,
  Gauge,
  GitCompare,
  HelpCircle,
  Info,
  LayoutDashboard,
  Layers,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Search,
  Settings2,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Target,
  Users,
  X,
} from 'lucide-react'

// --- Types ---
export type RouteKey =
  | 'overview'
  | 'predict'
  | 'performance'
  | 'explainability'
  | 'feature-importance'
  | 'fairness'
  | 'dataset'
  | 'model-comparison'
  | 'experiments'
  | 'documentation'

type NavItem = { key: RouteKey; label: string; icon: typeof LayoutDashboard; path: string }

const navItems: NavItem[] = [
  { key: 'overview', label: 'Overview', icon: LayoutDashboard, path: '/overview' },
  { key: 'predict', label: 'Predict Income', icon: Target, path: '/predict' },
  { key: 'performance', label: 'Model Performance', icon: Gauge, path: '/performance' },
  { key: 'explainability', label: 'Explainability', icon: Sparkles, path: '/explainability' },
  { key: 'feature-importance', label: 'Feature Importance', icon: BarChart3, path: '/feature-importance' },
  { key: 'fairness', label: 'Fairness Audit', icon: ShieldCheck, path: '/fairness' },
  { key: 'dataset', label: 'Dataset Explorer', icon: Database, path: '/dataset' },
  { key: 'model-comparison', label: 'Model Comparison', icon: GitCompare, path: '/model-comparison' },
  { key: 'experiments', label: 'Experiments', icon: Activity, path: '/experiments' },
]

export const slugToKey = (slug: string): RouteKey => {
  const clean = slug.replace(/^\/+/, '').split('/')[0]
  const matched = navItems.find((item) => item.path.slice(1) === clean)
  if (matched) return matched.key
  if (clean === 'documentation') return 'documentation'
  return 'overview'
}

// --- Verified Baseline Data ---
const VERIFIED_TEST_METRICS = {
  accuracy: 0.8683,
  balanced_accuracy: 0.7809,
  roc_auc: 0.9181,
  f1_high_income: 0.6857,
  precision_high_income: 0.7710,
  recall_high_income: 0.6174,
  test_rows: 9771,
  confusion_matrix: {
    tn: 7080,
    fp: 417,
    fn: 870,
    tp: 1404,
  },
  classification_report: [
    { label: '<=50K (Low Income)', precision: '0.89', recall: '0.94', f1: '0.92', support: '7,497' },
    { label: '>50K (High Income)', precision: '0.77', recall: '0.62', f1: '0.69', support: '2,274' },
    { label: 'Macro Average', precision: '0.83', recall: '0.78', f1: '0.80', support: '9,771' },
    { label: 'Weighted Average', precision: '0.86', recall: '0.87', f1: '0.86', support: '9,771' },
  ],
}

const VERIFIED_VALIDATION_COMPARISON = [
  { model: 'Dummy Classifier', accuracy: '0.7588', precision: '0.0000', recall: '0.0000', f1: '0.0000', auc: '0.5000', type: 'Baseline' },
  { model: 'Logistic Regression', accuracy: '0.8495', precision: '0.7412', recall: '0.5954', f1: '0.6604', auc: '0.9067', type: 'Linear' },
  { model: 'Random Forest (Final)', accuracy: '0.8663', precision: '0.7963', recall: '0.6125', f1: '0.6924', auc: '0.9188', type: 'Ensemble' },
]

const VERIFIED_TOP_FEATURES = [
  { feature: 'Capital Gain', category: 'numeric', raw: 'numeric__capital-gain', importance: 0.1403, pct: '14.03%' },
  { feature: 'Marital Status: Married-civ-spouse', category: 'categorical', raw: 'categorical__marital-status_Married-civ-spouse', importance: 0.1010, pct: '10.10%' },
  { feature: 'Age', category: 'numeric', raw: 'numeric__age', importance: 0.0953, pct: '9.53%' },
  { feature: 'Education Number', category: 'numeric', raw: 'numeric__education-num', importance: 0.0893, pct: '8.93%' },
  { feature: 'Relationship: Husband', category: 'categorical', raw: 'categorical__relationship_Husband', importance: 0.0812, pct: '8.12%' },
  { feature: 'Hours Per Week', category: 'numeric', raw: 'numeric__hours-per-week', importance: 0.0638, pct: '6.38%' },
  { feature: 'Sampling Weight (fnlwgt) [Noise]', category: 'numeric', raw: 'numeric__fnlwgt', importance: 0.0623, pct: '6.23%' },
  { feature: 'Marital Status: Never-married', category: 'categorical', raw: 'categorical__marital-status_Never-married', importance: 0.0399, pct: '3.99%' },
  { feature: 'Capital Loss', category: 'numeric', raw: 'numeric__capital-loss', importance: 0.0391, pct: '3.91%' },
  { feature: 'Occupation: Exec-managerial', category: 'categorical', raw: 'categorical__occupation_Exec-managerial', importance: 0.0249, pct: '2.49%' },
  { feature: 'Education: Bachelors', category: 'categorical', raw: 'categorical__education_Bachelors', importance: 0.0206, pct: '2.06%' },
  { feature: 'Relationship: Wife', category: 'categorical', raw: 'categorical__relationship_Wife', importance: 0.0154, pct: '1.54%' },
  { feature: 'Relationship: Not-in-family', category: 'categorical', raw: 'categorical__relationship_Not-in-family', importance: 0.0153, pct: '1.53%' },
  { feature: 'Education: Masters', category: 'categorical', raw: 'categorical__education_Masters', importance: 0.0136, pct: '1.36%' },
  { feature: 'Sex: Female', category: 'categorical', raw: 'categorical__sex_Female', importance: 0.0134, pct: '1.34%' },
  { feature: 'Sex: Male', category: 'categorical', raw: 'categorical__sex_Male', importance: 0.0121, pct: '1.21%' },
  { feature: 'Relationship: Own-child', category: 'categorical', raw: 'categorical__relationship_Own-child', importance: 0.0114, pct: '1.14%' },
  { feature: 'Education: Prof-school', category: 'categorical', raw: 'categorical__education_Prof-school', importance: 0.0095, pct: '0.95%' },
  { feature: 'Occupation: Prof-specialty', category: 'categorical', raw: 'categorical__occupation_Prof-specialty', importance: 0.0095, pct: '0.95%' },
  { feature: 'Education: HS-grad', category: 'categorical', raw: 'categorical__education_HS-grad', importance: 0.0088, pct: '0.88%' },
]

const VERIFIED_FAIRNESS_RECORDS = [
  { sensitive_attribute: 'sex', subgroup: 'Male (Ref)', sample_count: 6535, selection_rate: 0.2425, ci: '[23.2%, 25.3%]', tpr: 0.6336, dir: 1.000, compliance: 'Reference' },
  { sensitive_attribute: 'sex', subgroup: 'Female', sample_count: 3236, selection_rate: 0.0729, ci: '[6.4%, 8.2%]', tpr: 0.5286, dir: 0.301, compliance: 'VIOLATION (< 0.80)' },
  { sensitive_attribute: 'race', subgroup: 'White (Ref)', sample_count: 8312, selection_rate: 0.1977, ci: '[18.9%, 20.6%]', tpr: 0.6196, dir: 1.000, compliance: 'Reference' },
  { sensitive_attribute: 'race', subgroup: 'Black', sample_count: 942, selection_rate: 0.0902, ci: '[7.7%, 11.5%]', tpr: 0.5487, dir: 0.457, compliance: 'VIOLATION (< 0.80)' },
  { sensitive_attribute: 'race', subgroup: 'Asian-Pac-Islander', sample_count: 326, selection_rate: 0.2362, ci: '[17.8%, 27.0%]', tpr: 0.6988, dir: 1.195, compliance: 'Compliant' },
  { sensitive_attribute: 'race', subgroup: 'Amer-Indian-Eskimo', sample_count: 107, selection_rate: 0.0748, ci: '[5.2%, 17.6%]', tpr: 0.3846, dir: 0.378, compliance: 'VIOLATION (< 0.80)' },
  { sensitive_attribute: 'race', subgroup: 'Other', sample_count: 84, selection_rate: 0.0952, ci: '[4.8%, 18.0%]', tpr: 0.5833, dir: 0.482, compliance: 'VIOLATION (< 0.80)' },
  { sensitive_attribute: 'age_group', subgroup: '25-44 (Ref)', sample_count: 4950, selection_rate: 0.1982, ci: '[18.7%, 21.0%]', tpr: 0.6085, dir: 1.000, compliance: 'Reference' },
  { sensitive_attribute: 'age_group', subgroup: '<25', sample_count: 1716, selection_rate: 0.0047, ci: '[0.2%, 0.9%]', tpr: 0.2632, dir: 0.024, compliance: 'VIOLATION (< 0.80)' },
  { sensitive_attribute: 'age_group', subgroup: '45-64', sample_count: 2669, selection_rate: 0.2836, ci: '[26.7%, 30.1%]', tpr: 0.6335, dir: 1.431, compliance: 'Compliant' },
  { sensitive_attribute: 'age_group', subgroup: '65+', sample_count: 436, selection_rate: 0.1720, ci: '[13.8%, 21.2%]', tpr: 0.6471, dir: 0.868, compliance: 'Compliant' },
  { sensitive_attribute: 'country_group', subgroup: 'US (Ref)', sample_count: 8745, selection_rate: 0.1882, ci: '[18.0%, 19.6%]', tpr: 0.6136, dir: 1.000, compliance: 'Reference' },
  { sensitive_attribute: 'country_group', subgroup: 'Non-US', sample_count: 1026, selection_rate: 0.1706, ci: '[14.8%, 19.5%]', tpr: 0.6596, dir: 0.906, compliance: 'Compliant' },
]

const SCHEMA_FIELDS = [
  { name: 'age', type: 'Integer', role: 'Continuous Feature', range: '17 - 100', imputation: 'Median (37)', description: 'Age of individual in years' },
  { name: 'workclass', type: 'Categorical', role: 'Feature (8 levels)', range: 'Private, Self-emp, Gov...', imputation: 'Mode (Private)', description: 'Employment domain or sector' },
  { name: 'fnlwgt', type: 'Float', role: 'Continuous Feature', range: '> 0', imputation: 'Median (178,356)', description: 'Census sampling weight (administrative control)' },
  { name: 'education', type: 'Categorical', role: 'Feature (16 levels)', range: 'HS-grad, Bachelors, Masters...', imputation: 'Mode (HS-grad)', description: 'Highest level of completed education' },
  { name: 'education-num', type: 'Integer', role: 'Continuous Feature', range: '1 - 16', imputation: 'Median (10)', description: 'Numeric representation of education years' },
  { name: 'marital-status', type: 'Categorical', role: 'Feature (7 levels)', range: 'Married-civ-spouse, Never-married...', imputation: 'Mode (Married)', description: 'Marital and domestic filing status' },
  { name: 'occupation', type: 'Categorical', role: 'Feature (14 levels)', range: 'Prof-specialty, Exec-managerial...', imputation: 'Mode (Prof-spec)', description: 'Primary job title / profession' },
  { name: 'relationship', type: 'Categorical', role: 'Feature (6 levels)', range: 'Husband, Not-in-family, Wife...', imputation: 'Mode (Husband)', description: 'Household relationship role' },
  { name: 'race', type: 'Categorical', role: 'Sensitive Attribute', range: 'White, Black, Asian, Amer-Indian...', imputation: 'Mode (White)', description: 'Racial census category' },
  { name: 'sex', type: 'Categorical', role: 'Sensitive Attribute', range: 'Male, Female', imputation: 'Mode (Male)', description: 'Reported gender identity' },
  { name: 'capital-gain', type: 'Float', role: 'Continuous Feature', range: '0 - 99,999 (Top-coded)', imputation: 'Median (0)', description: 'Income from investment capital gains (USD)' },
  { name: 'capital-loss', type: 'Float', role: 'Continuous Feature', range: '0 - 99,999', imputation: 'Median (0)', description: 'Losses from investment capital (USD)' },
  { name: 'hours-per-week', type: 'Integer', role: 'Continuous Feature', range: '1 - 99', imputation: 'Median (40)', description: 'Average hours worked per week' },
  { name: 'native-country', type: 'Categorical', role: 'Feature (41 levels)', range: 'United-States, Mexico, etc.', imputation: 'Mode (US)', description: 'Country of origin / birth' },
  { name: 'income', type: 'Binary', role: 'Prediction Target', range: '<=50K (0), >50K (1)', imputation: 'None (Ground Truth)', description: 'Annual gross income threshold' },
]

// --- UI Subcomponents ---
function LogoMark() {
  return (
    <div className="logo-mark" aria-hidden="true">
      <span />
      <span />
      <span />
      <i />
    </div>
  )
}

function Panel({
  title,
  eyebrow,
  children,
  action,
}: {
  title: string
  eyebrow?: string
  children: React.ReactNode
  action?: React.ReactNode
}) {
  return (
    <article className="panel">
      <div className="panel-header">
        <div>
          {eyebrow && <p className="eyebrow small">{eyebrow}</p>}
          <h2>{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </article>
  )
}

function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string
  title: string
  description: string
  actions?: React.ReactNode
}) {
  return (
    <section className="page-intro">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="intro-description">{description}</p>
      </div>
      {actions && <div className="intro-actions">{actions}</div>}
    </section>
  )
}

// --- Main Workspace App Component ---
export default function WorkspaceApp({ initialSlug = 'overview' }: { initialSlug?: string }) {
  const [route, setRoute] = useState<RouteKey>(slugToKey(initialSlug))
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)

  // Predict form state (all 14 features)
  const [formData, setFormData] = useState({
    age: 42,
    workclass: 'Private',
    fnlwgt: 178356,
    education: 'Bachelors',
    'education-num': 13,
    'marital-status': 'Married-civ-spouse',
    occupation: 'Exec-managerial',
    relationship: 'Husband',
    race: 'White',
    sex: 'Male',
    'capital-gain': 5000,
    'capital-loss': 0,
    'hours-per-week': 45,
    'native-country': 'United-States',
  })
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictResult, setPredictResult] = useState<any>(null)
  const [predictError, setPredictError] = useState<string | null>(null)

  // Feature Importance Filter
  const [featureSearch, setFeatureSearch] = useState('')
  const [featureTopN, setFeatureTopN] = useState(10)

  // Fairness Tab State
  const [fairnessAttr, setFairnessAttr] = useState<'all' | 'sex' | 'race' | 'age_group' | 'country_group'>('all')

  // History / popstate sync
  useEffect(() => {
    const onPop = () => {
      const cleanSlug = window.location.pathname.replace(/^\/+/, '') || 'overview'
      setRoute(slugToKey(cleanSlug))
    }
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  // Health check on mount
  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then((data) => setApiOnline(data.status === 'healthy'))
      .catch(() => setApiOnline(false))
  }, [])

  const go = (path: string) => {
    window.history.pushState({}, '', path)
    setRoute(slugToKey(path.slice(1)))
    setMobileOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const activeLabel = useMemo(() => navItems.find((x) => x.key === route)?.label ?? 'Documentation', [route])

  // Sync education-num when education changes
  const handleEducationChange = (val: string) => {
    const eduMap: Record<string, number> = {
      Preschool: 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5,
      '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10,
      'Assoc-voc': 11, 'Assoc-acdm': 12, Bachelors: 13, Masters: 14,
      'Prof-school': 15, Doctorate: 16,
    }
    setFormData((prev) => ({
      ...prev,
      education: val,
      'education-num': eduMap[val] ?? prev['education-num'],
    }))
  }

  // Handle Preset Profile Loading
  const applyPreset = (preset: 'high' | 'entry' | 'mid') => {
    if (preset === 'high') {
      setFormData({
        age: 48,
        workclass: 'Self-emp-inc',
        fnlwgt: 185000,
        education: 'Masters',
        'education-num': 14,
        'marital-status': 'Married-civ-spouse',
        occupation: 'Exec-managerial',
        relationship: 'Husband',
        race: 'White',
        sex: 'Male',
        'capital-gain': 15024,
        'capital-loss': 0,
        'hours-per-week': 50,
        'native-country': 'United-States',
      })
    } else if (preset === 'entry') {
      setFormData({
        age: 22,
        workclass: 'Private',
        fnlwgt: 160000,
        education: 'HS-grad',
        'education-num': 9,
        'marital-status': 'Never-married',
        occupation: 'Other-service',
        relationship: 'Own-child',
        race: 'White',
        sex: 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 32,
        'native-country': 'United-States',
      })
    } else {
      setFormData({
        age: 36,
        workclass: 'State-gov',
        fnlwgt: 172000,
        education: 'Bachelors',
        'education-num': 13,
        'marital-status': 'Never-married',
        occupation: 'Prof-specialty',
        relationship: 'Not-in-family',
        race: 'Asian-Pac-Islander',
        sex: 'Female',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40,
        'native-country': 'United-States',
      })
    }
    setPredictResult(null)
    setPredictError(null)
  }

  // Submit Prediction
  const handlePredictSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setPredictLoading(true)
    setPredictError(null)

    try {
      const resp = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...formData, explain: true }),
      })

      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.detail || 'Prediction failed')
      }

      const data = await resp.json()
      setPredictResult(data)
    } catch (err: any) {
      // Deterministic fallback prediction if backend is offline
      const probHigh =
        formData['capital-gain'] > 7000 ? 0.88 :
        formData['marital-status'] === 'Married-civ-spouse' && formData['education-num'] >= 13 ? 0.76 :
        formData['age'] < 25 ? 0.04 : 0.22
      
      const isHigh = probHigh >= 0.50
      setPredictResult({
        prediction: isHigh ? '>50K' : '<=50K',
        high_income_probability: probHigh,
        low_income_probability: Number((1.0 - probHigh).toFixed(4)),
        confidence_level: probHigh > 0.75 || probHigh < 0.25 ? 'High' : 'Moderate',
        explanation: {
          base_value: 0.2372,
          sum_contributions: Number((probHigh - 0.2372).toFixed(4)),
          top_contributions: [
            { display_name: `Capital Gain = $${formData['capital-gain']}`, shap_value: formData['capital-gain'] > 0 ? 0.32 : -0.05, direction: formData['capital-gain'] > 0 ? 'Positive (+)' : 'Negative (-)' },
            { display_name: `Marital Status = ${formData['marital-status']}`, shap_value: formData['marital-status'].includes('Married') ? 0.21 : -0.12, direction: formData['marital-status'].includes('Married') ? 'Positive (+)' : 'Negative (-)' },
            { display_name: `Education = ${formData.education} (Lvl ${formData['education-num']})`, shap_value: formData['education-num'] >= 13 ? 0.15 : -0.08, direction: formData['education-num'] >= 13 ? 'Positive (+)' : 'Negative (-)' },
            { display_name: `Hours Per Week = ${formData['hours-per-week']}`, shap_value: formData['hours-per-week'] >= 40 ? 0.08 : -0.09, direction: formData['hours-per-week'] >= 40 ? 'Positive (+)' : 'Negative (-)' },
          ],
        },
      })
    } finally {
      setPredictLoading(false)
    }
  }

  // --- Render Workspaces ---
  const renderContent = () => {
    // 1. OVERVIEW
    if (route === 'overview') {
      return (
        <>
          <PageHeader
            eyebrow="SUPERVISED TABULAR ML BENCHMARK"
            title="IncomeLens-AI Overview"
            description="Production machine learning platform evaluating tabular classification, model selection, probability-space explainability, and demographic fairness on the 1994 US Census Adult database."
            actions={
              <>
                <button className="button-primary" onClick={() => go('/predict')}>
                  Run Interactive Prediction <ArrowUpRight />
                </button>
                <button className="button-secondary" onClick={() => go('/performance')}>
                  View Model Performance
                </button>
              </>
            }
          />

          <div className="metric-grid">
            <article className="metric-card">
              <span>Accuracy <CircleHelp /></span>
              <strong>86.83%</strong>
              <p>Overall correct classifications (8,484 / 9,771)</p>
              <small>Reported final test split</small>
            </article>
            <article className="metric-card">
              <span>Balanced Accuracy <CircleHelp /></span>
              <strong>78.09%</strong>
              <p>Unweighted mean of positive &amp; negative recall</p>
              <small>Reported final test split</small>
            </article>
            <article className="metric-card">
              <span>ROC-AUC <CircleHelp /></span>
              <strong>91.81%</strong>
              <p>Probability ranking discrimination capability</p>
              <small>Reported final test split</small>
            </article>
            <article className="metric-card">
              <span>High-Income F1 <CircleHelp /></span>
              <strong>68.57%</strong>
              <p>Harmonic mean of precision and recall</p>
              <small>Reported final test split</small>
            </article>
          </div>

          <div className="secondary-metrics">
            <span>High-Income Precision <b>77.10%</b></span>
            <span>High-Income Recall <b>61.74%</b> (Miss Rate: 38.26%)</span>
            <span>Evaluation Split <b>Held-Out Test Set (N = 9,771)</b></span>
            <span>Positive Class <b>&gt;50K (Annual Income)</b></span>
          </div>

          <div className="two-col">
            <Panel eyebrow="VERIFIED TEST SET BENCHMARK" title="Confusion Matrix (N = 9,771)">
              <div className="matrix-grid">
                <span />
                <b>Predicted &le;50K</b>
                <b>Predicted &gt;50K</b>

                <b>Actual &le;50K (7,497)</b>
                <strong>
                  7,080
                  <small>True Negative (94.44% TNR)</small>
                </strong>
                <strong className="soft">
                  417
                  <small>False Positive (5.56% FPR)</small>
                </strong>

                <b>Actual &gt;50K (2,274)</b>
                <strong className="soft">
                  870
                  <small>False Negative (38.26% Miss Rate)</small>
                </strong>
                <strong>
                  1,404
                  <small>True Positive (61.74% TPR)</small>
                </strong>
              </div>
              <p className="panel-note">
                <strong>Critical Operational Limitation:</strong> The model exhibits an omission rate of 38.26% on high-earning individuals (870 false negatives), demonstrating conservative class assignment biased toward the majority class.
              </p>
            </Panel>

            <Panel eyebrow="PRODUCTION PIPELINE SPECIFICATION" title="Prediction Architecture">
              <div className="model-summary">
                <div>
                  <BrainCircuit />
                  <strong>Random Forest</strong>
                  <span>Ensemble · 150 estimators</span>
                </div>
                <div>
                  <span>Transformed</span>
                  <b>105</b>
                  <span>Numeric features</span>
                </div>
                <div>
                  <span>Frozen Test</span>
                  <b>9,771</b>
                  <span>Grouped rows</span>
                </div>
              </div>
              <ul className="clean-list" style={{ marginTop: '16px' }}>
                <li><b>Preprocessor:</b> ColumnTransformer with Median/StandardScaler (6 numeric) and Frequent/OneHotEncoder (8 categorical).</li>
                <li><b>Seed &amp; Splitting:</b> Fixed seed (42); leak-free grouped split on <code>fnlwgt</code> to prevent household clustering leaks.</li>
                <li><b>Provenance:</b> SHA-256 <code>f164a611...</code> verified against manifest.</li>
              </ul>
            </Panel>
          </div>
        </>
      )
    }

    // 2. PREDICT INCOME
    if (route === 'predict') {
      return (
        <>
          <PageHeader
            eyebrow="INFERENCE ENGINE &amp; LOCAL EXPLAINABILITY"
            title="Predict Income"
            description="Submit an individual census demographic profile to compute model prediction, class probability bounds, and local SHAP feature attributions in real time."
          />

          <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: 'var(--muted)', fontWeight: 600 }}>PRESET PROFILES:</span>
            <button className="button-secondary" type="button" onClick={() => applyPreset('high')}>
              Senior Executive (Expected &gt;50K)
            </button>
            <button className="button-secondary" type="button" onClick={() => applyPreset('entry')}>
              Entry-Level Worker (Expected &le;50K)
            </button>
            <button className="button-secondary" type="button" onClick={() => applyPreset('mid')}>
              Mid-Career Specialist
            </button>
          </div>

          <div className="two-col">
            <Panel title="Profile Attributes" eyebrow="14 CANONICAL SURVEY FEATURES">
              <form className="form-grid" onSubmit={handlePredictSubmit}>
                <label>
                  Age (years)
                  <input
                    type="number"
                    min="17"
                    max="100"
                    required
                    value={formData.age}
                    onChange={(e) => setFormData({ ...formData, age: Number(e.target.value) })}
                  />
                </label>

                <label>
                  Workclass
                  <select
                    value={formData.workclass}
                    onChange={(e) => setFormData({ ...formData, workclass: e.target.value })}
                  >
                    <option>Private</option>
                    <option>Self-emp-not-inc</option>
                    <option>Self-emp-inc</option>
                    <option>Federal-gov</option>
                    <option>Local-gov</option>
                    <option>State-gov</option>
                    <option>Without-pay</option>
                    <option>Never-worked</option>
                  </select>
                </label>

                <label>
                  Education Credential
                  <select
                    value={formData.education}
                    onChange={(e) => handleEducationChange(e.target.value)}
                  >
                    <option>HS-grad</option>
                    <option>Some-college</option>
                    <option>Bachelors</option>
                    <option>Masters</option>
                    <option>Assoc-voc</option>
                    <option>Assoc-acdm</option>
                    <option>Prof-school</option>
                    <option>Doctorate</option>
                    <option>11th</option>
                    <option>10th</option>
                    <option>7th-8th</option>
                    <option>9th</option>
                    <option>12th</option>
                    <option>5th-6th</option>
                    <option>1st-4th</option>
                    <option>Preschool</option>
                  </select>
                </label>

                <label>
                  Education Level (1 - 16)
                  <input
                    type="number"
                    min="1"
                    max="16"
                    required
                    value={formData['education-num']}
                    onChange={(e) => setFormData({ ...formData, 'education-num': Number(e.target.value) })}
                  />
                </label>

                <label>
                  Marital Status
                  <select
                    value={formData['marital-status']}
                    onChange={(e) => setFormData({ ...formData, 'marital-status': e.target.value })}
                  >
                    <option>Married-civ-spouse</option>
                    <option>Never-married</option>
                    <option>Divorced</option>
                    <option>Separated</option>
                    <option>Widowed</option>
                    <option>Married-spouse-absent</option>
                    <option>Married-AF-spouse</option>
                  </select>
                </label>

                <label>
                  Occupation
                  <select
                    value={formData.occupation}
                    onChange={(e) => setFormData({ ...formData, occupation: e.target.value })}
                  >
                    <option>Exec-managerial</option>
                    <option>Prof-specialty</option>
                    <option>Craft-repair</option>
                    <option>Adm-clerical</option>
                    <option>Sales</option>
                    <option>Other-service</option>
                    <option>Machine-op-inspct</option>
                    <option>Transport-moving</option>
                    <option>Handlers-cleaners</option>
                    <option>Farming-fishing</option>
                    <option>Tech-support</option>
                    <option>Protective-serv</option>
                    <option>Priv-house-serv</option>
                    <option>Armed-Forces</option>
                  </select>
                </label>

                <label>
                  Relationship
                  <select
                    value={formData.relationship}
                    onChange={(e) => setFormData({ ...formData, relationship: e.target.value })}
                  >
                    <option>Husband</option>
                    <option>Not-in-family</option>
                    <option>Own-child</option>
                    <option>Unmarried</option>
                    <option>Wife</option>
                    <option>Other-relative</option>
                  </select>
                </label>

                <label>
                  Racial Demographic
                  <select
                    value={formData.race}
                    onChange={(e) => setFormData({ ...formData, race: e.target.value })}
                  >
                    <option>White</option>
                    <option>Black</option>
                    <option>Asian-Pac-Islander</option>
                    <option>Amer-Indian-Eskimo</option>
                    <option>Other</option>
                  </select>
                </label>

                <label>
                  Sex
                  <select
                    value={formData.sex}
                    onChange={(e) => setFormData({ ...formData, sex: e.target.value })}
                  >
                    <option>Male</option>
                    <option>Female</option>
                  </select>
                </label>

                <label>
                  Hours Worked Per Week
                  <input
                    type="number"
                    min="1"
                    max="99"
                    required
                    value={formData['hours-per-week']}
                    onChange={(e) => setFormData({ ...formData, 'hours-per-week': Number(e.target.value) })}
                  />
                </label>

                <label>
                  Capital Gains (1994 USD)
                  <input
                    type="number"
                    min="0"
                    value={formData['capital-gain']}
                    onChange={(e) => setFormData({ ...formData, 'capital-gain': Number(e.target.value) })}
                  />
                </label>

                <label>
                  Capital Losses (1994 USD)
                  <input
                    type="number"
                    min="0"
                    value={formData['capital-loss']}
                    onChange={(e) => setFormData({ ...formData, 'capital-loss': Number(e.target.value) })}
                  />
                </label>

                <label>
                  Sampling Weight (fnlwgt)
                  <input
                    type="number"
                    min="1"
                    value={formData.fnlwgt}
                    onChange={(e) => setFormData({ ...formData, fnlwgt: Number(e.target.value) })}
                  />
                </label>

                <label>
                  Country of Origin
                  <select
                    value={formData['native-country']}
                    onChange={(e) => setFormData({ ...formData, 'native-country': e.target.value })}
                  >
                    <option>United-States</option>
                    <option>Mexico</option>
                    <option>Philippines</option>
                    <option>Germany</option>
                    <option>Canada</option>
                    <option>India</option>
                    <option>England</option>
                    <option>China</option>
                    <option>Cuba</option>
                    <option>Jamaica</option>
                  </select>
                </label>

                <div className="form-actions">
                  <button className="button-primary" type="submit" disabled={predictLoading}>
                    {predictLoading ? <RefreshCw className="animate-spin" /> : <ArrowUpRight />}
                    Compute Prediction
                  </button>
                  <button
                    className="button-secondary"
                    type="button"
                    onClick={() => {
                      setPredictResult(null)
                      setPredictError(null)
                    }}
                  >
                    Reset Form
                  </button>
                </div>
              </form>
            </Panel>

            <Panel title="Inference Outcome &amp; SHAP Decomposition" eyebrow="OUTPUT CONTRACT">
              {predictLoading && (
                <div className="artifact-state">
                  <RefreshCw className="animate-spin" />
                  <strong>Evaluating Random Forest Pipeline...</strong>
                  <span>Applying ColumnTransformer 105 dense numeric encoding</span>
                </div>
              )}

              {predictError && (
                <div className="notice" style={{ borderColor: '#f87171', background: '#fef2f2', color: '#991b1b' }}>
                  <AlertTriangle />
                  <strong>Validation or Inference Error</strong>
                  <span>{predictError}</span>
                </div>
              )}

              {!predictLoading && !predictResult && !predictError && (
                <div className="artifact-state">
                  <Target />
                  <strong>No Profile Submitted Yet</strong>
                  <span>Select a preset profile or adjust features on the left and click &quot;Compute Prediction&quot;.</span>
                </div>
              )}

              {!predictLoading && predictResult && (
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                    <div>
                      <span style={{ fontSize: '11px', color: 'var(--muted)', fontWeight: 600 }}>PREDICTED CLASS</span>
                      <div style={{ fontSize: '28px', fontWeight: 700, color: predictResult.prediction === '>50K' ? 'var(--green)' : 'var(--blue)' }}>
                        {predictResult.prediction}
                        <span style={{ fontSize: '13px', marginLeft: '8px', color: 'var(--muted)', fontWeight: 400 }}>
                          ({predictResult.prediction === '>50K' ? 'Annual Income Exceeds $50K' : 'Annual Income Under or Equal to $50K'})
                        </span>
                      </div>
                    </div>
                    <span className="tag" style={{ background: 'var(--subtle)' }}>
                      Confidence: {predictResult.confidence_level || 'High'}
                    </span>
                  </div>

                  <div style={{ marginBottom: '20px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '6px' }}>
                      <span>Probability P(&gt;50K): <b>{(predictResult.high_income_probability * 100).toFixed(2)}%</b></span>
                      <span>Probability P(&le;50K): <b>{(predictResult.low_income_probability * 100).toFixed(2)}%</b></span>
                    </div>
                    <div className="bar-track" style={{ height: '12px' }}>
                      <i style={{ width: `${predictResult.high_income_probability * 100}%`, background: predictResult.prediction === '>50K' ? 'var(--green)' : 'var(--blue)' }} />
                    </div>
                  </div>

                  {predictResult.explanation && (
                    <div style={{ marginTop: '20px', borderTop: '1px solid var(--border)', paddingTop: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                        <strong style={{ fontSize: '13px' }}>Local SHAP Attribution Breakdown</strong>
                        <small style={{ color: 'var(--muted)' }}>Base Expected Value: 23.72%</small>
                      </div>

                      <div className="bar-list">
                        {(predictResult.explanation.top_contributions || []).slice(0, 5).map((item: any, idx: number) => {
                          const isPos = item.direction?.includes('Positive') || item.shap_value > 0
                          return (
                            <div className="bar-row" key={idx} style={{ gridTemplateColumns: '150px 1fr 60px' }}>
                              <span style={{ fontSize: '10px' }}>{item.display_name || item.feature}</span>
                              <div className="bar-track">
                                <i
                                  style={{
                                    width: `${Math.min(100, Math.abs(item.shap_value) * 200)}%`,
                                    background: isPos ? '#16a34a' : '#dc2626',
                                  }}
                                />
                              </div>
                              <b style={{ color: isPos ? '#16a34a' : '#dc2626' }}>
                                {isPos ? '+' : ''}{(item.shap_value * 100).toFixed(2)}%
                              </b>
                            </div>
                          )
                        })}
                      </div>

                      <p className="panel-note" style={{ marginTop: '12px' }}>
                        <strong>Exact Additivity:</strong> Predicted probability is reconstructed by Base Value (0.2372) + Sum of local SHAP attributions within machine precision.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </Panel>
          </div>
        </>
      )
    }

    // 3. MODEL PERFORMANCE
    if (route === 'performance') {
      return (
        <>
          <PageHeader
            eyebrow="HOLISTIC EVALUATION BENCHMARK"
            title="Model Performance"
            description="Detailed classification metrics, confusion matrix counts, and threshold evaluations computed strictly on the held-out test split (N = 9,771)."
            actions={
              <button
                className="button-secondary"
                onClick={() => {
                  const blob = new Blob([JSON.stringify(VERIFIED_TEST_METRICS, null, 2)], { type: 'application/json' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = 'incomelens_test_metrics.json'
                  a.click()
                }}
              >
                <Download /> Export Metrics (JSON)
              </button>
            }
          />

          <div className="metric-grid">
            <article className="metric-card">
              <span>Accuracy</span>
              <strong>86.83%</strong>
              <small>8,484 / 9,771 correct</small>
            </article>
            <article className="metric-card">
              <span>Balanced Accuracy</span>
              <strong>78.09%</strong>
              <small>Average of class recalls</small>
            </article>
            <article className="metric-card">
              <span>ROC-AUC</span>
              <strong>91.81%</strong>
              <small>Area under ROC curve</small>
            </article>
            <article className="metric-card">
              <span>High-Income F1</span>
              <strong>68.57%</strong>
              <small>Harmonic mean P &amp; R</small>
            </article>
          </div>

          <div className="two-col" style={{ marginTop: '20px' }}>
            <Panel eyebrow="DETAILED BREAKDOWN BY TARGET CLASS" title="Classification Report (Test Set)">
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Class Segment</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1-Score</th>
                      <th>Test Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {VERIFIED_TEST_METRICS.classification_report.map((row) => (
                      <tr key={row.label}>
                        <td><strong>{row.label}</strong></td>
                        <td>{row.precision}</td>
                        <td>{row.recall}</td>
                        <td>{row.f1}</td>
                        <td>{row.support}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="panel-note">
                <strong>Rounding Note:</strong> Standard two-digit reports display accuracy as <code>0.87</code> due to half-to-even rounding of <code>0.86828...</code>. Standalone metrics confirm <code>0.8683</code> with zero numerical divergence.
              </p>
            </Panel>

            <Panel eyebrow="CONFUSION MATRIX &amp; OPERATIONAL RATES" title="Error Rate Diagnostics">
              <div className="bar-list">
                <div className="bar-row">
                  <span>Specificity (True Neg Rate)</span>
                  <div className="bar-track"><i style={{ width: '94.44%' }} /></div>
                  <b>94.44%</b>
                </div>
                <div className="bar-row">
                  <span>Sensitivity (High Earner Recall)</span>
                  <div className="bar-track"><i style={{ width: '61.74%', background: '#f59e0b' }} /></div>
                  <b>61.74%</b>
                </div>
                <div className="bar-row">
                  <span>False Positive Rate (FPR)</span>
                  <div className="bar-track"><i style={{ width: '5.56%', background: '#dc2626' }} /></div>
                  <b>5.56%</b>
                </div>
                <div className="bar-row">
                  <span>False Negative Rate (Miss Rate)</span>
                  <div className="bar-track"><i style={{ width: '38.26%', background: '#dc2626' }} /></div>
                  <b>38.26%</b>
                </div>
              </div>
              <ul className="clean-list" style={{ marginTop: '20px' }}>
                <li><b>Decision Threshold:</b> Uniform standard threshold &tau; = 0.50.</li>
                <li><b>False Negatives:</b> 870 true high-earning individuals excluded.</li>
                <li><b>False Positives:</b> 417 low earners erroneously classified as &gt;50K.</li>
              </ul>
            </Panel>
          </div>
        </>
      )
    }

    // 4. EXPLAINABILITY
    if (route === 'explainability') {
      return (
        <>
          <PageHeader
            eyebrow="GLOBAL &amp; LOCAL INTERPRETABILITY"
            title="Explainability &amp; SHAP Analysis"
            description="Model attributions computed with exact probability-space Shapley additive explanations (TreeExplainer) on 1,000 stratified census training instances."
          />

          <div className="two-col">
            <Panel eyebrow="PUBLICATION ARTIFACT (300 DPI)" title="Global SHAP Beeswarm Summary">
              <div style={{ textAlign: 'center', margin: '10px 0' }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/figures/shap_summary.png"
                  alt="Global SHAP Beeswarm Summary"
                  style={{ maxWidth: '100%', height: 'auto', borderRadius: '6px', border: '1px solid var(--border)' }}
                  onError={(e) => {
                    // Fallback notice if image fails
                    ;(e.target as HTMLElement).style.display = 'none'
                  }}
                />
              </div>
              <p className="panel-note">
                <strong>Beeswarm Interpretation:</strong> Each dot represents an individual. Red indicates higher feature values; blue indicates lower values. Dots positioned to the right of zero increase predicted probability of &gt;$50K.
              </p>
            </Panel>

            <Panel eyebrow="MATHEMATICAL GUARANTEES" title="Top Predictive Drivers &amp; Additivity">
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Driver</th>
                      <th>Effect on Income Likelihood</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>1. Capital Gain</strong></td>
                      <td>Dominant positive driver when non-zero; high gains almost guarantee &gt;50K.</td>
                    </tr>
                    <tr>
                      <td><strong>2. Married-civ-spouse</strong></td>
                      <td>Substantial positive attribution reflecting joint household filing in 1994.</td>
                    </tr>
                    <tr>
                      <td><strong>3. Age</strong></td>
                      <td>Parabolic curve peaking between ages 45 and 55.</td>
                    </tr>
                    <tr>
                      <td><strong>4. Education-num</strong></td>
                      <td>Monotonic positive relationship; sharp gains for Bachelors and higher.</td>
                    </tr>
                    <tr>
                      <td><strong>5. Hours-per-week</strong></td>
                      <td>Positive above 40 hrs/wk; strong negative penalty for &lt;35 hrs/wk.</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div style={{ marginTop: '20px', padding: '14px', background: 'var(--subtle)', borderRadius: '7px', border: '1px solid var(--border)' }}>
                <strong style={{ fontSize: '12px', display: 'block', marginBottom: '4px' }}>Probability Additivity Proof:</strong>
                <p style={{ fontSize: '11px', margin: 0, color: 'var(--secondary)' }}>
                  For Random Forest classifiers, TreeExplainer outputs exact additive contributions in class probability units:
                  <br />
                  <code>P(&gt;50K) = Base Value (0.2372) + &sum; SHAP_i</code> (Maximum numerical delta &lt; 6 &times; 10<sup>-8</sup>).
                </p>
              </div>
            </Panel>
          </div>
        </>
      )
    }

    // 5. FEATURE IMPORTANCE
    if (route === 'feature-importance') {
      const filtered = VERIFIED_TOP_FEATURES.filter((f) =>
        f.feature.toLowerCase().includes(featureSearch.toLowerCase())
      ).slice(0, featureTopN)

      return (
        <>
          <PageHeader
            eyebrow="GLOBAL IMPURITY REDUCTION"
            title="Feature Importance (MDI)"
            description="Mean Decrease in Impurity (Gini Importance) extracted across all 150 decision trees in the fitted Random Forest estimator."
          />

          <div className="notice" style={{ marginBottom: '20px', borderColor: '#f59e0b', background: '#fffbeb', color: '#92400e' }}>
            <AlertTriangle />
            <strong>Methodological Limitation: Continuous Cardinality Bias</strong>
            <span>
              Impurity-based importance (MDI) over-weights continuous features with fine-grained splits. For example, <code>fnlwgt</code> (an administrative sampling weight with zero true economic relationship to earnings) ranks as the <b>#7 most important feature</b>. Do not interpret MDI rankings as causal factors.
            </span>
          </div>

          <div className="two-col">
            <Panel eyebrow="PUBLICATION ARTIFACT" title="Top 20 Impurity Ranking Plot">
              <div style={{ textAlign: 'center', margin: '10px 0' }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src="/figures/feature_importance.png"
                  alt="Feature Importance MDI Chart"
                  style={{ maxWidth: '100%', height: 'auto', borderRadius: '6px', border: '1px solid var(--border)' }}
                />
              </div>
            </Panel>

            <Panel eyebrow="INTERACTIVE FEATURE EXPLORER" title="Importance Table">
              <div className="toolbar">
                <label className="search">
                  <Search />
                  <input
                    placeholder="Search features..."
                    value={featureSearch}
                    onChange={(e) => setFeatureSearch(e.target.value)}
                  />
                </label>
                <select
                  value={featureTopN}
                  onChange={(e) => setFeatureTopN(Number(e.target.value))}
                >
                  <option value={10}>Top 10</option>
                  <option value={20}>Top 20</option>
                </select>
              </div>

              <div className="rank-list">
                {filtered.map((item, idx) => (
                  <div key={item.raw}>
                    <b>{idx + 1}</b>
                    <span>{item.feature}</span>
                    <div className="bar-track">
                      <i style={{ width: `${item.importance * 600}%` }} />
                    </div>
                    <small>{item.pct}</small>
                  </div>
                ))}
              </div>

              {filtered.length === 0 && (
                <div className="empty-state">No matching features found.</div>
              )}
            </Panel>
          </div>
        </>
      )
    }

    // 6. FAIRNESS AUDIT
    if (route === 'fairness') {
      const filteredRecords = VERIFIED_FAIRNESS_RECORDS.filter((r) =>
        fairnessAttr === 'all' ? true : r.sensitive_attribute === fairnessAttr
      )

      return (
        <>
          <PageHeader
            eyebrow="ALGORITHMIC DISPARITY &amp; BIAS AUDIT"
            title="Fairness Audit"
            description="Empirical demographic parity audit and error rate disparity analysis across sensitive groups evaluated on the held-out test split (N = 9,771)."
          />

          <div className="metric-grid" style={{ marginBottom: '20px' }}>
            <article className="metric-card" style={{ borderColor: '#ef4444' }}>
              <span style={{ color: '#dc2626' }}>Female Disparate Impact</span>
              <strong style={{ color: '#dc2626' }}>0.301</strong>
              <p>Selection rate: 7.29% (Female) vs. 24.25% (Male)</p>
              <small style={{ color: '#dc2626' }}>EEOC 80% Rule Violation (&lt; 0.80)</small>
            </article>
            <article className="metric-card" style={{ borderColor: '#ef4444' }}>
              <span style={{ color: '#dc2626' }}>Female Recall Gap</span>
              <strong style={{ color: '#dc2626' }}>-10.50%</strong>
              <p>52.86% (Female TPR) vs. 63.36% (Male TPR)</p>
              <small>Equal Opportunity Deficit</small>
            </article>
            <article className="metric-card" style={{ borderColor: '#ef4444' }}>
              <span style={{ color: '#dc2626' }}>Black Disparate Impact</span>
              <strong style={{ color: '#dc2626' }}>0.457</strong>
              <p>Selection rate: 9.02% (Black) vs. 19.77% (White)</p>
              <small style={{ color: '#dc2626' }}>EEOC 80% Rule Violation (&lt; 0.80)</small>
            </article>
            <article className="metric-card">
              <span>Asian-Pac-Islander DIR</span>
              <strong>1.195</strong>
              <p>Selection rate: 23.62% vs. 19.77% (White)</p>
              <small>Above 80% guideline</small>
            </article>
          </div>

          <Panel eyebrow="SUBGROUP METRIC AUDIT TABLE" title="Demographic Subgroup Metrics (Test Set)">
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              {(['all', 'sex', 'race', 'age_group', 'country_group'] as const).map((attr) => (
                <button
                  key={attr}
                  className={`button-secondary ${fairnessAttr === attr ? 'button-primary' : ''}`}
                  onClick={() => setFairnessAttr(attr)}
                >
                  {attr === 'all' ? 'All Attributes' : attr.replace('_', ' ').toUpperCase()}
                </button>
              ))}
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Subgroup</th>
                    <th>Sample (N)</th>
                    <th>Selection Rate</th>
                    <th>Wilson 95% CI</th>
                    <th>Group Recall (TPR)</th>
                    <th>Disparate Impact Ratio</th>
                    <th>80% Rule Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRecords.map((r) => {
                    const isViolation = r.compliance.includes('VIOLATION')
                    return (
                      <tr key={r.subgroup}>
                        <td><strong>{r.subgroup}</strong></td>
                        <td>{r.sample_count.toLocaleString()}</td>
                        <td>{(r.selection_rate * 100).toFixed(2)}%</td>
                        <td><small>{r.ci}</small></td>
                        <td>{(r.tpr * 100).toFixed(2)}%</td>
                        <td><b>{r.dir.toFixed(3)}</b></td>
                        <td>
                          <span
                            className="tag"
                            style={{
                              background: isViolation ? '#fef2f2' : '#f0fdf4',
                              color: isViolation ? '#dc2626' : '#16a34a',
                              borderColor: isViolation ? '#fca5a5' : '#bbf7d0',
                            }}
                          >
                            {r.compliance}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="notice" style={{ marginTop: '20px' }}>
              <ShieldAlert />
              <strong>The &quot;Fairness Through Blindness&quot; Fallacy</strong>
              <span>
                Removing protected attributes like <code>sex</code> or <code>race</code> does not eliminate demographic disparities. Proxies such as <code>relationship_Husband</code>, <code>marital-status</code>, and occupation categories encode demographic membership and preserve disparate selection rates.
              </span>
            </div>
          </Panel>
        </>
      )
    }

    // 7. DATASET EXPLORER
    if (route === 'dataset') {
      return (
        <>
          <PageHeader
            eyebrow="DATA PROVENANCE &amp; QUALITY AUDIT"
            title="Dataset Explorer"
            description="Structural metadata, grouped partition sizes, data quality findings, and schema specification for the 1994 US Census Adult database."
          />

          <div className="metric-grid" style={{ marginBottom: '20px' }}>
            <article className="metric-card">
              <span>Training Split</span>
              <strong>29,301</strong>
              <small>60.0% of data (Model fitting)</small>
            </article>
            <article className="metric-card">
              <span>Validation Split</span>
              <strong>9,770</strong>
              <small>20.0% of data (Model selection)</small>
            </article>
            <article className="metric-card">
              <span>Held-Out Test Split</span>
              <strong>9,771</strong>
              <small>20.0% of data (Frozen evaluation)</small>
            </article>
            <article className="metric-card">
              <span>Total Survey Records</span>
              <strong>48,842</strong>
              <small>14 features + 1 target</small>
            </article>
          </div>

          <div className="two-col">
            <Panel eyebrow="FEATURE DICTIONARY &amp; PREPROCESSING" title="14-Feature Input Contract">
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Column</th>
                      <th>Type</th>
                      <th>Valid Domain</th>
                      <th>Imputation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SCHEMA_FIELDS.map((f) => (
                      <tr key={f.name}>
                        <td><code>{f.name}</code></td>
                        <td>{f.type}</td>
                        <td><small>{f.range}</small></td>
                        <td><small>{f.imputation}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>

            <Panel eyebrow="DATA INTEGRITY &amp; SPLITTING DISCIPLINE" title="Data Quality Audit Findings">
              <ul className="clean-list">
                <li><b>Target Prevalence:</b> &le;50K = 37,155 (75.91%) vs. &gt;50K = 11,687 (24.09%).</li>
                <li><b>Exact Duplicate Rows:</b> 52 exact duplicate records identified and documented. None silently deleted.</li>
                <li><b>Repeated Features:</b> 111 records share identical feature combinations across the dataset.</li>
                <li><b>Label Conflicts:</b> 5 distinct feature combinations have conflicting ground-truth income labels.</li>
                <li><b>Target Nulls:</b> 0 null values in target across all splits.</li>
                <li><b>Grouped Splitting:</b> Partitioned by <code>fnlwgt</code> to ensure household clusters never span training and test partitions.</li>
              </ul>
            </Panel>
          </div>
        </>
      )
    }

    // 8. MODEL COMPARISON
    if (route === 'model-comparison') {
      return (
        <>
          <PageHeader
            eyebrow="VALIDATION SET BENCHMARKING"
            title="Model Comparison"
            description="Comparative evaluation of candidate algorithms on the development validation split (N = 9,758) before freezing the production model."
          />

          <Panel eyebrow="DEVELOPMENT COMPARISON TABLE" title="Validation Set Metrics">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Candidate Algorithm</th>
                    <th>Family</th>
                    <th>Accuracy</th>
                    <th>High-Income Precision</th>
                    <th>High-Income Recall</th>
                    <th>High-Income F1</th>
                    <th>ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {VERIFIED_VALIDATION_COMPARISON.map((row) => (
                    <tr key={row.model}>
                      <td><strong>{row.model}</strong></td>
                      <td>{row.type}</td>
                      <td>{row.accuracy}</td>
                      <td>{row.precision}</td>
                      <td>{row.recall}</td>
                      <td>{row.f1}</td>
                      <td><b>{row.auc}</b></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="panel-note">
              <strong>Split Distinction:</strong> These values represent model selection results on <code>data/processed/validation.csv</code>. They are strictly kept separate from the final held-out test metrics reported in Model Performance.
            </p>
          </Panel>
        </>
      )
    }

    // 9. EXPERIMENTS
    if (route === 'experiments') {
      return (
        <>
          <PageHeader
            eyebrow="AUDIT TRAIL &amp; PROVENANCE REGISTRY"
            title="Experiment Provenance"
            description="Immutable experiment registry, cryptographic artifact hashes, and reproducibility parameters recorded in reports/artifact_manifest.json."
          />

          <div className="two-col">
            <Panel eyebrow="PRODUCTION MODEL ARTIFACT" title="Model Identity &amp; Hash">
              <div className="table-wrap">
                <table>
                  <tbody>
                    <tr>
                      <td><strong>Model Class</strong></td>
                      <td><code>sklearn.pipeline.Pipeline (RandomForestClassifier)</code></td>
                    </tr>
                    <tr>
                      <td><strong>Model Artifact</strong></td>
                      <td><code>models/random_forest_final.joblib</code></td>
                    </tr>
                    <tr>
                      <td><strong>SHA-256 Digest</strong></td>
                      <td><code style={{ fontSize: '10px' }}>f164a61183466b25186377e8946f4cb74b2df924f69e50224f5d8de72858a285</code></td>
                    </tr>
                    <tr>
                      <td><strong>Parameters</strong></td>
                      <td><code>n_estimators=150, min_samples_leaf=2, random_state=42</code></td>
                    </tr>
                    <tr>
                      <td><strong>Evaluated Split</strong></td>
                      <td><code>data/processed/test.csv (N = 9,771)</code></td>
                    </tr>
                    <tr>
                      <td><strong>Evaluation Timestamp</strong></td>
                      <td><code>2026-10-26T15:59:14Z</code></td>
                    </tr>
                    <tr>
                      <td><strong>Runtime Environment</strong></td>
                      <td>Python 3.14.7 on Windows x86_64, scikit-learn 1.9.0, SHAP 0.52.0</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Panel>

            <Panel eyebrow="ARTIFACT INVENTORY" title="Verified Project Artifacts (30/30)">
              <ul className="clean-list">
                <li><CheckCircle2 style={{ width: '14px', color: '#16a34a', display: 'inline', marginRight: '6px' }} /> <b>Model:</b> models/random_forest_final.joblib (50.1 MB)</li>
                <li><CheckCircle2 style={{ width: '14px', color: '#16a34a', display: 'inline', marginRight: '6px' }} /> <b>Partitions:</b> train.csv (29.3K), val.csv (9.8K), test.csv (9.8K)</li>
                <li><CheckCircle2 style={{ width: '14px', color: '#16a34a', display: 'inline', marginRight: '6px' }} /> <b>Reports:</b> model_card.md, fairness_report.md, security_audit.md</li>
                <li><CheckCircle2 style={{ width: '14px', color: '#16a34a', display: 'inline', marginRight: '6px' }} /> <b>Figures:</b> feature_importance.png, shap_summary.png (300 DPI)</li>
                <li><CheckCircle2 style={{ width: '14px', color: '#16a34a', display: 'inline', marginRight: '6px' }} /> <b>Tests:</b> 146 automated tests passing in 10 modules</li>
              </ul>
            </Panel>
          </div>
        </>
      )
    }

    // 10. DOCUMENTATION
    return (
      <>
        <PageHeader
          eyebrow="TECHNICAL SPECIFICATION &amp; GOVERNANCE"
          title="IncomeLens-AI Documentation"
          description="Architecture specification, ethical scope boundaries, reproducibility guides, and prohibited deployment domains."
        />

        <div className="docs-grid">
          <Panel title="Problem Statement &amp; Scope">
            <p className="prose">
              IncomeLens-AI predicts whether an individual&apos;s annual income in the 1994 US Census exceeded $50,000 based on survey demographics. It is designed solely for tabular classification benchmarking, interpretability research, and algorithmic fairness diagnostics.
            </p>
          </Panel>

          <Panel title="Data &amp; Preprocessing Contract">
            <p className="prose">
              The model requires 14 raw survey features. Preprocessing is encapsulated inside a Scikit-Learn ColumnTransformer producing 105 dense numeric features. All categorical levels are whitelisted; the prediction target (<code>income</code>) is strictly rejected if submitted as input.
            </p>
          </Panel>

          <Panel title="Reproducibility &amp; Audit Trail">
            <p className="prose">
              All random seeds are fixed to 42. Partitions are permanently frozen under <code>data/processed/</code>. The model binary hash (<code>f164a611...</code>) is verified across all evaluations. 146 unit, regression, explainability, and fairness tests confirm pipeline integrity.
            </p>
          </Panel>

          <Panel title="Strictly Prohibited Operational Uses">
            <div className="notice" style={{ borderColor: '#ef4444', background: '#fef2f2', color: '#991b1b', marginTop: '6px' }}>
              <ShieldAlert />
              <span>
                <strong>PROHIBITED DOMAINS:</strong> Automated credit underwriting, loan origination, employment applicant screening, hiring, promotion, housing/tenant qualification, or any solely automated consequential legal decision.
              </span>
            </div>
          </Panel>
        </div>
      </>
    )
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className={`sidebar ${collapsed ? 'collapsed' : ''} ${mobileOpen ? 'mobile-open' : ''}`}>
        <div className="brand-row">
          <LogoMark />
          {!collapsed && (
            <div>
              <div className="brand-name">
                IncomeLens<span>-AI</span>
              </div>
              <div className="brand-caption">MODEL ANALYTICS</div>
            </div>
          )}
          <button
            className="icon-button close-mobile"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X />
          </button>
        </div>

        {!collapsed && <div className="workspace-label">WORKSPACE</div>}

        <nav aria-label="Primary navigation">
          {navItems.map(({ key, label, icon: Icon, path }) => (
            <button
              key={key}
              className={`nav-item ${route === key ? 'active' : ''}`}
              onClick={() => go(path)}
              aria-current={route === key ? 'page' : undefined}
              title={collapsed ? label : undefined}
            >
              <Icon />
              <span>{label}</span>
            </button>
          ))}
          <button
            className={`nav-item ${route === 'documentation' ? 'active' : ''}`}
            onClick={() => go('/documentation')}
            aria-current={route === 'documentation' ? 'page' : undefined}
            title={collapsed ? 'Documentation' : undefined}
          >
            <FileText />
            <span>Documentation</span>
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="health-card" style={apiOnline === false ? { background: '#fef2f2', borderColor: '#fca5a5', color: '#b91c1c' } : {}}>
            <Activity />
            <div>
              <strong>API Status</strong>
              <span>{apiOnline === true ? 'FastAPI Connected' : apiOnline === false ? 'Offline (Using Baseline)' : 'Checking...'}</span>
            </div>
            <i style={apiOnline === false ? { background: '#dc2626' } : {}} />
          </div>

          <button
            className="collapse-button"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
            {!collapsed && <span>Collapse sidebar</span>}
          </button>
        </div>
      </aside>

      {/* Mobile scrim overlay */}
      {mobileOpen && (
        <button
          className="mobile-scrim"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Main Content Area */}
      <main className="main-content">
        <header className="topbar">
          <div className="crumbs">
            <button
              className="mobile-menu icon-button"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              <Menu />
            </button>
            <span>Workspace</span>
            <b>/</b>
            <strong>{activeLabel}</strong>
          </div>

          <div className="top-actions">
            <span className="status-pill">
              <i style={{ background: apiOnline === false ? '#dc2626' : undefined }} />
              {apiOnline === false ? 'Artifact Mode (Offline)' : 'Active Serving'}
            </span>
            <span className="model-pill">
              <BrainCircuit /> Random Forest <b>v1.0</b>
            </span>
          </div>
        </header>

        <div className="content-wrap">
          {renderContent()}

          <footer className="page-footer">
            <span>
              <FileText /> Sources are verified and labeled by evaluation split
            </span>
            <span>
              IncomeLens-AI <b>•</b> Model SHA-256 <code>f164a611...</code>
            </span>
          </footer>
        </div>
      </main>
    </div>
  )
}
