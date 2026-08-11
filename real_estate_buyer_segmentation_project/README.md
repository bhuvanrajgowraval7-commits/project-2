# Real Estate Buyer Segmentation & Investment Profiling

This project implements the supplied PRD using the provided `clients(2).csv` and `properties(2).csv`.

## Deliverables
- `buyer_segmentation_pipeline.py` — reproducible preprocessing, feature engineering and K-Means training.
- `app.py` — Streamlit dashboard.
- `research_report.pdf` — research-style report with EDA, methodology, results and recommendations.
- `outputs/` — cleaned analytical outputs and segment assignments.
- `figures/` — generated charts.
- `models/buyer_segmentation_model.joblib` — fitted preprocessing/model artifact.

## Key result
The behavior-focused K-Means model selected **3 clusters** using the highest silhouette score among k=2..8.
Best silhouette score: **0.1981**.

## Run
```bash
pip install -r requirements.txt
python buyer_segmentation_pipeline.py --clients ../clients(2).csv --properties ../properties(2).csv --out outputs
streamlit run app.py
```

## Important data interpretation notes
- The PRD lists `date_of_birth` as an age indicator; this implementation derives age from DOB using 2026-08-11 as the analysis date.
- `sale_price` is cleaned from currency-formatted text to numeric values.
- Property records with `Available` status do not have client-linked purchases in the supplied data; they are used only for rule-based recommendations.
- The supplied data do not contain income or net-worth fields, so the analysis does **not** claim to directly measure income. Segment names are behavioral labels derived from observed purchase frequency, price and investment patterns.


## Dashboard PRD coverage

The Streamlit dashboard explicitly implements all requested modules:
1. Buyer Segmentation Overview — cluster distribution and segment share.
2. Investor Behavior Dashboard — investment patterns, purchase price, purchase frequency, financing and purpose.
3. Geographic Buyer Analysis — interactive world map with region-level buyer segments plus regional table.
4. Segment Insights Panel — descriptive statistics for every ML-derived cluster.
5. User controls — country, region, acquisition purpose, client type and buyer segment filters.
