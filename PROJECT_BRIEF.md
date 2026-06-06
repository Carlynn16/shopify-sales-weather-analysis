# Project Brief — Shopify Sales & Weather Analysis

> Project documentation: context, data, methodology, analysis plan, and report structure.

---

## 1. Context

A Danish artisan food chain (ice cream, chocolate, bakery, coffee) operates **7 physical
stores** around the Copenhagen area.
*(Client name and store locations kept out of the public repo for confidentiality.)*

Each store runs its point-of-sale (POS) on **Shopify**; every in-store transaction is recorded
there. Data is pulled from the **Shopify Admin API** over a **14-month window (Jan 2024 →
Mar 2025)** — ~454k transaction lines across ~280k orders after preparation.

This is *physical retail*, not online — so sales (especially ice cream) are strongly
weather-driven. That observation motivates the predictive part of the project.

**Business goals:**
1. Understand sales: what sells, where, when, and surface data-quality issues.
2. Build a weather-driven sales forecast to support staffing, prep, and inventory.

---

## 2. Data dictionary (raw Shopify export — 7 tables)

> Raw data is kept **local only** (`data/`, git-ignored): it contains PII (customer emails,
> names, addresses) and client-confidential figures.

| Table | Grain | Key columns |
|-------|-------|-------------|
| `orders` | one row / order | order_id, order_number, created_at, processed_at, customer_id, total_price, subtotal_price, total_tax, total_discounts, financial_status, fulfillment_status, location_id, cancelled_at, cancel_reason |
| `line_items` | one row / product line | order_id, line_item_id, product_id, variant_id, sku, **name**, quantity, price, total_discount, created_at |
| `products` | one row / variant | product_id, title, product_type, vendor, tags, variant_id, sku, price, compare_at_price, inventory_quantity |
| `customers` | one row / customer | customer_id, email, first/last name, orders_count, total_spent, state, tags, address fields |
| `locations` | one row / store (7) | id -> store_name, city |
| `discounts` | one row / discount code use | order_id, code, amount, type |
| `refunds` | one row / refund | order_id, refund_id, amount, note |

`line_items` is the central fact table; everything joins onto it via `order_id` /
`product_id` / `location_id` to form a single transaction-level table.

---

## 3. Data preparation

- Drop columns that are >95% empty (`sku`, `customer_id`, `email`, `product_type`).
- Normalize inconsistent product labels; fill missing dimensions
  (`store_name`, `title`, `vendor`, `tags`) with explicit "Unknown" markers; standardize tags.
- Keep only completed sales: `financial_status == 'paid'` AND `fulfillment_status == 'fulfilled'`
  AND not cancelled AND excluding a non-production test store.
- Use `name` (the POS product label) as the product identity — it is always present, whereas
  `title` depends on a successful join to the products catalog.
- Derive `revenue = price x quantity`.

### Product categorization (8 families)
Ice Cream - Hot Beverages - Buns & Bakery - Chocolate - Gifts & Cards - Snacks & Nuts -
Christmas - Others. The mapping lives in `src/product_categories.csv` (auditable, single source
of truth) and reflects the client's category definitions (e.g. flodeboller and bars are
classified under Chocolate).

---

## 4. Analysis plan

### Block A — Sales analysis (descriptive + diagnostic)
- Top products by **units** and by **revenue**; Pareto (80/20) concentration.
- Revenue by category (the 8 families).
- Store comparison: top products/categories per store, local champions.
- Seasonality: by month, weekday, hour-of-day.
- Data-quality investigation of the "Unknown Product" / POS-tag pattern (by store and over time).

### Block B — Weather model (the client's custom request)
- Pull daily weather per store-city from the **Open-Meteo archive API**
  (temp max/min, precipitation, rain, snowfall, windspeed, daylight; derive feels-like).
- Correlate weather with revenue — overall, by category, by store, by product.
- Group comparisons: rainy vs dry, warm vs cold, long vs short daylight, summer vs winter.
- **Predictive model:** XGBoost for daily **ice-cream revenue per store**, with a short-horizon
  forecast usable for staffing and prep.

---

## 5. Key results (headline)

- Revenue is highly concentrated: the top ~20 products = **80%** of revenue (clean Pareto).
- **Ice Cream** dominates (~57% of revenue), led by **"2 kugler"** (two scoops);
  Chocolate is #2, Buns & Bakery #3.
- Strong seasonality: peak **May–June**; busiest **Fri/Sat**; busiest hours **12:00–15:00**.
- Ice cream peaks in summer; hot beverages and Christmas items peak in winter.
- **Data quality:** a "Unknown Product" pattern concentrated in one store and tied to a POS
  terminal not synced to Shopify SKUs; it declined sharply after a POS reconfiguration.
- **Weather:** temperature and daylight correlate strongly with daily revenue; ice cream is the
  most weather-sensitive category; hot beverages correlate negatively. Stores differ in
  weather sensitivity.
- **Model:** XGBoost with temporal cross-validation; top drivers (via SHAP): daylight,
  last-week revenue (lag-7), and temperature.

---

## 6. Analytical approach & rigor

The analysis is built to be defensible and reproducible:

1. **Inferential statistics, not just visuals.** Group comparisons (rainy/dry, warm/cold)
   use appropriate tests (t-test / Mann-Whitney) with reported p-values and effect sizes;
   correlations and key estimates come with bootstrap confidence intervals.
2. **Multicollinearity handled explicitly.** Temperature, feels-like, daylight, and season are
   highly correlated; a correlation matrix / VIF is shown and interpreted accordingly.
3. **Model benchmarked and diagnosed.** XGBoost is compared against a naive baseline
   (lag-7 / seasonal-naive) to justify its value; error analysis is reported by store and
   season, with prediction intervals.
4. **Reproducible & honest.** A documented methodology, stated limitations, and a clear
   train/validation protocol (time-series split, no leakage). All credentials are loaded from
   environment variables — never committed.

---

## 7. Final report structure (the PDF deliverable)

1. Executive summary
2. Data & methodology (sources, preparation, categorization, caveats)
3. Sales analysis (Block A) — each figure embedded and interpreted with numbers
4. Data-quality deep-dive (the Unknown Product / POS story)
5. Weather analysis (Block B) — correlations with significance and confidence intervals
6. Predictive model — features, validation, baseline comparison, errors, intervals, SHAP
7. Recommendations (staffing, inventory, POS fixes)
8. Limitations & next steps

---

## 8. Scope

**In scope:** the sales analysis (Block A) and the weather model (Block B), delivered as a
reproducible codebase, a figure set, and a final PDF report.

**Out of scope:** an interactive dashboard.

**Security:** all API tokens and database credentials are loaded from environment variables and
are never stored in the repository.
