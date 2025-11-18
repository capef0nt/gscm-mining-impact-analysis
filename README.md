# gscm-mining-impact-analysis

**Green Supply Chain Management (GSCM) – Impact on Operational Efficiency, Safety, Environmental Performance, and Cost in Mining**

`gscm-mining-impact-analysis` is a Python-based analytics toolkit that evaluates and predicts how **Green Supply Chain Management (GSCM)** practices influence:

- ⚙️ Operational Efficiency  
- 🛟 Safety Performance  
- 🌍 Environmental & Resource Efficiency  
- 💸 Cost Competitiveness  
- 📈 Enterprise Performance  

in **mining companies**, using a hybrid **PLS-SEM + Machine Learning** architecture.

This toolkit is built for mining engineers, supply chain analysts, ESG professionals, researchers, and operations teams who want to understand how “green decisions” impact real performance in the field.

---

# 1. 🎯 Project Purpose

Mining companies today must:

- Improve production efficiency  
- Reduce energy, water, and fuel intensity  
- Strengthen safety outcomes  
- Lower cost per ton  
- Meet environmental and ESG expectations  
- Operate reliably in demanding conditions  

Academic research suggests GSCM helps — but real mines need tools that go **beyond survey results**.

This project creates a **scalable, real-data-driven, decision-support toolkit**.

---

# 2. 🧠 What This Toolkit Does

This toolkit ingests:

- GSCM survey data  
- Mediator constructs (maintenance, competence, supplier integration)  
- Operational context  
- Objective KPIs (downtime, MTBF, LTIFR, cost/ton, emissions, etc.)

Then it delivers:

---

## ✔ 1. PLS-SEM Model (Explanatory Layer)

Latent constructs include:

**GSCM dimensions**

- GPUR – Green Purchasing  
- GOPS – Green Operations  
- GLOG – Green Logistics  
- GTRN – Green Training & Awareness  
- GCOL – Green Collaboration  

**Mediators**

- SUPINT – Supplier Integration  
- MAINT – Maintenance Quality  
- COMP – Employee Competence  

**Outcomes**

- OE – Perceived Operational Efficiency  
- EP – Perceived Enterprise Performance  

Outputs:

- Path coefficients  
- AVE, CR, HTMT metrics  
- R² values  
- Bootstrapped significance  
- Construct scores (used in ML models)

---

## ✔ 2. Machine Learning (Predictive Layer)

ML models use:

- Construct scores  
- Site-level metadata  
- Objective KPI values  

to predict:

### ● Operational KPIs
- uptime_percent  
- unplanned_downtime_hours  
- mtbf_hours  
- mttr_hours  
- tons_per_hour  

### ● Environmental KPIs
- energy_kwh_per_ton  
- water_m3_per_ton  
- fuel_l_per_ton  
- co2_kg_per_ton  

### ● Cost KPIs
- cost_per_ton  
- maintenance_cost_per_ton  

### ● Logistics KPIs
- on_time_delivery_percent  
- supplier_defect_percent  

### ● Safety KPIs
- ltifr  
- trifr  
- near_miss_rate  
- safety_incidents_per_year  

Outputs:

- KPI predictions  
- Feature importance  
- SHAP explanations  
- Performance metrics (MAE, RMSE, R²)

---

## ✔ 3. Scenario & Simulation Engine

Example simulations:

- “What if we improve Green Operations by +0.3?”  
- “How much can LTIFR improve with better training?”  
- “How does Supplier Integration influence cost per ton?”  
- “Which GSCM practice gives the highest ROI?”  

The engine produces **forecasted KPI improvements** based on real data.

---

## ✔ 4. Automated Mining Site Reports

Each report includes:

- SEM summary  
- KPI analysis  
- ML prediction insights  
- Scenario results  
- Ranked recommendations  

---

# 3. 📊 Data Requirements

## 3.1 Survey Data (Likert 1–5)

Survey includes:

- GSCM constructs (GPUR, GOPS, GLOG, GTRN, GCOL)  
- Mediators (SUPINT, MAINT, COMP)  
- Perceived OE & EP  
- Respondent details  
- Site-level context  

See: **`docs/questionnaire.md`**

---

## 3.2 Objective KPI Data

Each row = one mining site.

### Operational KPIs
- uptime_percent  
- unplanned_downtime_hours  
- mtbf_hours  
- mttr_hours  
- tons_per_hour  
- cycle_time_hours  
- rework_rate_percent  

### Environmental KPIs
- energy_kwh_per_ton  
- water_m3_per_ton  
- fuel_l_per_ton  
- waste_kg_per_ton  
- recycling_rate_percent  
- co2_kg_per_ton  

### Cost KPIs
- cost_per_ton  
- maintenance_cost_per_ton  
- unit_variable_cost_per_ton  

### Supply Chain KPIs
- on_time_delivery_percent  
- supplier_lead_time_days  
- supplier_defect_percent  

### Safety KPIs
- ltifr  
- trifr  
- near_miss_rate  
- severity_rate_hours  
- safety_incidents_per_year  
- safety_audits_passed_percent  
- employees_competent_percent  

Definitions: **`docs/kpi_definitions.md`**

---

# 4. 🧩 Methodology Summary

## PLS-SEM Layer  
Used for:  
✔ causal relationships  
✔ theory validation  
✔ construct scoring  

## ML Layer  
Used for:  
✔ KPI prediction  
✔ non-linear modelling  
✔ feature importance  
✔ scenario forecasting  

---

# 5. 📁 Project Structure

```text
gscm-mining-impact-analysis/
│
├─ README.md
├─ pyproject.toml
│
├─ docs/
│   ├─ overview.md
│   ├─ questionnaire.md
│   ├─ model_spec.md
│   ├─ kpi_definitions.md
│   └─ usage_guide.md
│
├─ data/
│   ├─ raw/
│   ├─ processed/
│   └─ examples/
│
├─ src/
│   ├─ config/
│   │   └─ model_config.py
│   │
│   ├─ data_ingestion/
│   │   └─ loader.py
│   │
│   ├─ preprocessing/
│   │   ├─ cleaning.py
│   │   ├─ feature_engineering.py
│   │   └─ construct_scores.py
│   │
│   ├─ sem/
│   │   ├─ pls_model.py
│   │   ├─ sem_results.py
│   │   └─ reporting.py
│   │
│   ├─ ml/
│   │   ├─ train_models.py
│   │   ├─ evaluate.py
│   │   ├─ explain.py
│   │   └─ scenarios.py
│   │
│   ├─ reports/
│   │   ├─ generate_site_report.py
│   │   └─ templates/
│   │
│   └─ cli/
│       └─ main.py
│
└─ notebooks/
    ├─ 01_exploration.ipynb
    ├─ 02_sem_development.ipynb
    └─ 03_ml_experiments.ipynb


Inspiration & Academic Reference

This project is inspired by:

Ngcobo, N., Pretorius, J.-H., & van der Merwe, A. (2022).
The Impact of Green Supply Chain Management Practices on Operational Efficiency within the Mining Sector in South Africa.

Their study explored how GSCM practices influence operational efficiency in two mines (Lonmin and Impala), using survey-based PLS-SEM.

7. 🔍 Limitations of the Original Study

The academic research had several constraints:

Sample limited to two mining companies

Reliance on perception-only Likert survey data

No operational KPIs (downtime, MTBF, cost/ton, LTIFR, etc.)

No environmental or safety metrics

No predictive analytics or machine learning

No scenario simulation

No integration with real mining data systems (SCADA, CMMS, ERP)

8. 🚀 How This Toolkit Extends Their Work

This project expands on the research by adding:

✔ Real measured KPIs

Operational, safety, environmental, cost, logistics.

✔ Hybrid SEM + ML architecture

SEM explains.
ML predicts.
Together, they guide decisions.

✔ Scenario simulation engine

Forecast operational improvement from GSCM interventions.

✔ Multi-site scalability

Not limited to two mines — works for entire mining groups.

✔ Integration-ready design

Supports data from SCADA, CMMS, ERP, HSE systems.

✔ Practical decision support

Turns research into real mining operational intelligence.

9. 🙏 Acknowledgement

We acknowledge the foundational contribution of Ngcobo et al. (2022).
Their work set the theoretical baseline for understanding GSCM in mining.
This toolkit aims to evolve that research into a realistic, data-driven, industry-ready platform.

🛠 Roadmap

 Implement SEM model

 Implement ML prediction models

 Data validation & schema enforcement

 Scenario simulation engine

 Automated HTML/PDF report generation

 CLI interface

 Publish v0.1

🤝 Contributions

Mining engineers, ESG specialists, data scientists, SEM researchers — all welcome.

Submit pull requests or open issues to get involved.

Author : B. C Marimbita 
email : bcm637@gmail.com
