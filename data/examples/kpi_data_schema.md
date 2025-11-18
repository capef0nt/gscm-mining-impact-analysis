## KPI Data Schema (Version 1 Core Set)

**Survey Data Schema (v1)**

Each row = one respondent.

We’ll split it into:

Metadata (who/where)

Construct items (Likert 1–5)
### A. Metadata Columns

```
| Column              | Description                                  | Type   | Required |
|---------------------|----------------------------------------------|--------|----------|
| respondent_id       | Unique ID per respondent                     | string | Yes      |
| site_id             | ID linking respondent to a site              | string | Yes      |
| company_id          | Company/group identifier                     | string | Optional |
| job_title           | Respondent job title                         | string | Optional |
| department          | Department (Supply Chain, Ops, etc.)         | string | Optional |
| role_level          | Seniority level (Manager, Supervisor, etc.)  | string | Optional |
| years_at_company    | Years of service (numeric or bracket)        | string | Optional |
| mine_type           | Commodity (platinum, gold, coal, etc.)       | string | Optional |
| automation_level    | Low / Medium / High                          | string | Optional |
| regulatory_pressure | Perceived regulatory pressure (1–5)          | int    | Optional |
| customer_pressure   | Perceived customer pressure (1–5)            | int    | Optional |
| top_mgmt_commitment | Perceived top mgmt commitment (1–5)          | int    | Optional |
```
### B. GSCM Construct Columns (Likert 1–5)

We’ll use the pattern CONSTRUCT_ITEMNUMBER, e.g. GPUR_1.

All Likert items are integer 1–5:

1 = Strongly Disagree

2 = Disagree

3 = Neutral

4 = Agree

5 = Strongly Agree

**Green Purchasing (GPUR)**
Column	Question (short label)
GPUR_1	We prioritise suppliers who meet environmental standards.
GPUR_2	Environmental criteria influence supplier selection.
GPUR_3	We evaluate suppliers’ environmental performance regularly.
GPUR_4	We prefer suppliers with environmental certifications.
**Green Operations (GOPS)**
Column	Question
GOPS_1	Our processes minimise waste and emissions.
GOPS_2	We invest in energy-efficient or cleaner technologies.
GOPS_3	Environmental procedures are clearly defined and followed.
GOPS_4	We monitor resource consumption (water, fuel, electricity).
**Green Logistics (GLOG)**
Column	Question
GLOG_1	Transport routes are optimised to reduce fuel use.
GLOG_2	Logistics partners follow environmental best practices.
GLOG_3	Empty loads or return trips are minimised.
**Green Training & Awareness (GTRN)**
Column	Question
GTRN_1	Employees receive environmental and safety training regularly.
GTRN_2	Management promotes environmental responsibility.
GTRN_3	Employees understand the environmental impact of their tasks.
**Green Collaboration (GCOL)**
Column	Question
GCOL_1	We collaborate with suppliers on environmental improvement initiatives.
GCOL_2	We work with customers to reduce environmental impacts.
GCOL_3	We share environmental best practices with supply chain partners.
### C. Mediator Constructs
**Supplier Integration (SUPINT)**
Column	Question
SUPINT_1	We share operational information/plans with key suppliers.
SUPINT_2	Suppliers participate in solving operational challenges.
SUPINT_3	We maintain long-term relationships with strategic suppliers.
**Maintenance Quality (MAINT)**
Column	Question
MAINT_1	Preventive maintenance is executed on schedule.
MAINT_2	Equipment failures are resolved quickly and effectively.
MAINT_3	Predictive maintenance tools or monitoring systems are used.
**Employee Competence (COMP)**
Column	Question
COMP_1	Employees are properly trained to perform tasks correctly.
COMP_2	Standard operating procedures are followed consistently.
COMP_3	Employees handle operational disruptions effectively.
### D. Perceived Outcomes (OE & EP)
**Perceived Operational Efficiency (OE)**
Column	Question
OE_1	Unplanned downtime has been reduced over time.
OE_2	Production cycles meet planned timeframes.
OE_3	Resources (water, fuel, electricity) are used efficiently.
OE_4	Operational processes run smoothly with minimal disruption.
OE_5	Operational costs have improved due to better practices.
**Perceived Enterprise Performance (EP)**
Column	Question
EP_1	Productivity has improved over the past 2–3 years.
EP_2	Customer satisfaction has improved.
EP_3	We have achieved better cost performance.
EP_4	Our competitive position has improved.
EP_5	Profitability has improved relative to previous years.

All OE_* and EP_* = integers 1–5.

How respondents map to sites (important for SEM/ML)

Multiple respondents can belong to the same site_id.

For the SEM/ML layer that links survey constructs to site-level KPIs, you will typically:

Aggregate construct scores per site, e.g. mean GPUR per site.

Then join that to the KPI table on site_id.

So:

Survey level = individual perceptions

KPI level = objective site performance

Bridge = site_id and aggregation logic in construct_scores.py.