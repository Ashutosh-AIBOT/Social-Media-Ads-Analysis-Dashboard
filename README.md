# 📊 Social Media Ads Analysis

A full end-to-end data analysis and machine learning project on 300,000 social media ad campaigns across 4 channels, 50 companies, and 5 customer segments — built entirely in Python with an interactive React dashboard.

**Live Dashboard → [social-ads-dashboard.vercel.app](.....)**

---

## 🔍 Project Overview

This project takes raw social media advertising data and runs it through a complete 10-pipeline system — from raw CSV cleaning all the way to machine learning models and an interactive web dashboard.

The data covers **299,815 campaigns** run in 2022 across Instagram, Facebook, Twitter, and Pinterest, with full revenue, profit, ROI, CTR, and engagement metrics.

---

## 🚨 Key Finding

> **Pinterest is losing $165 million.**
> Every other channel delivers ROI ~4.0 with 87%+ profitable campaigns.
> Pinterest delivers ROI 0.72, engagement score of 1.0, and only 30% profitable campaigns.
> The data makes the case to cut Pinterest entirely.

---

## 📁 Project Structure

```
Social-Media-Ads-Analysis/
│
├── notebooks/
│   ├── pipeline_1.py        # Load & clean raw data
│   ├── pipeline_2.py        # Exploratory data analysis
│   ├── pipeline_3.py        # Business insights
│   ├── pipeline_4.py        # Statistical testing
│   ├── pipeline_5.py        # Feature engineering
│   ├── pipeline_6.py        # Model building (with leakage check)
│   ├── pipeline_6b.py       # Model building (clean, no leakage)
│   ├── pipeline_7.py        # XGBoost + AdaBoost
│   ├── pipeline_8.py        # Static matplotlib charts (14 PNGs)
│   ├── pipeline_9.py        # Interactive Plotly dashboards (4 HTML)
│   └── pipeline_10.py       # Master runner — one line runs everything
│
├── charts/
│   ├── p8_01_kpi_summary.png
│   ├── p8_02_channel.png
│   ├── p8_03_pinterest.png
│   ├── p8_04_monthly.png
│   ├── p8_05_audience.png
│   ├── p8_06_campaign_goal.png
│   ├── p8_07_segment.png
│   ├── p8_08_correlation.png
│   ├── p8_09_roc_curves.png
│   ├── p8_10_model_comparison.png
│   ├── p8_11_feature_importance.png
│   ├── p8_12_confusion.png
│   ├── p8_13_duration.png
│   └── p8_14_location.png
│
├── social-ads-dashboard/    # React web dashboard
│   ├── src/
│   │   └── App.js
│   └── package.json
│
└── README.md
```

---

## ⚙️ Pipeline Architecture

| Pipeline | Task | Input | Output |
|----------|------|-------|--------|
| 1 | Load & Clean | `social_media_ads.csv` | `p1_clean.csv` |
| 2 | EDA | `p1_clean.csv` | `p2_eda_summary.csv` |
| 3 | Business Insights | `p1_clean.csv` | `p3_business_insights.csv` |
| 4 | Statistical Tests | `p1_clean.csv` | `p4_statistical_tests.csv` |
| 5 | Feature Engineering | `p1_clean.csv` | `p5_features.csv` |
| 6 | ML Models (leakage check) | `p5_features.csv` | `p6_model_results.csv` |
| 6B | ML Models (clean) | `p5_features.csv` | `p6b_model_results.csv` |
| 7 | XGBoost + AdaBoost | `p5_features.csv` | `p7_boosting_results.csv` |
| 8 | Static Charts | `p1_clean.csv` | 14 PNG charts |
| 9 | Interactive Dashboards | `p1_clean.csv` | 4 HTML dashboards |
| 10 | Master Runner | `social_media_ads.csv` | Everything |

**Run everything in one line:**
```python
p10 = run_pipeline_10('/content/social_media_ads.csv')
```

---

## 📈 Business Insights

| Metric | Value |
|--------|-------|
| Total Revenue | $7.39B |
| Total Profit | $5.06B |
| Total Campaigns | 254,960 |
| Avg ROI | 3.18 |
| Avg CTR | 31.42% |
| Profitable Campaigns | 73.1% |

**Channel Performance:**

| Channel | ROI | Profit | Profitable % |
|---------|-----|--------|--------------|
| Instagram | 4.01 | $1.75B | 87% |
| Twitter | 4.01 | $1.74B | 88% |
| Facebook | 3.99 | $1.74B | 87% |
| Pinterest | 0.72 | **-$165M** | 30% |

**Statistical Tests (key results):**
- Channel significantly affects profitability (Chi² p=0.00) ✅
- Gender ROI difference is NOT significant (p=0.19)
- Location has NO significant impact on ROI (ANOVA p=0.66)
- Duration has NO significant impact on ROI (ANOVA p=0.85)

---

## 🤖 Machine Learning Results

Three prediction targets, five models, no data leakage:

| Target | Best Model | AUC |
|--------|-----------|-----|
| Campaign Success | Gradient Boosting | **0.9888** |
| High Profit | AdaBoost | **0.8571** |
| High ROI | AdaBoost | **0.7552** |

**Top features driving High ROI:**
1. CPC (0.241)
2. Engagement × CTR (0.207)
3. CPM (0.198)
4. Channel Score (0.181)

---

## 🌐 Interactive Dashboard

Built in React with Recharts. Five sections:

- **Overview** — KPI cards, monthly trends, channel revenue
- **Pinterest Problem** — deep dive into the $165M loss
- **Audience** — segment, gender, age, location breakdown
- **ML Models** — AUC comparison, ROC curves, feature importance
- **Campaign Predictor** — input campaign params → get 3 predictions live

---

## 🛠️ Tech Stack

**Analysis**
- Python 3.12, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn, Plotly

**Dashboard**
- React, Recharts
- Hosted on Vercel

---

## 🚀 Run Locally

**Python pipelines (Google Colab or local):**
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost
```

```python
# Run all 10 pipelines at once
p10 = run_pipeline_10('/content/social_media_ads.csv')
```

**React dashboard:**
```bash
cd social-ads-dashboard
npm install
npm start
```

---

## 👤 Author

**Ashutosh**
GitHub → [github.com/Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT)

---
# Social-Media-Ads-Analysis-Dashboard
