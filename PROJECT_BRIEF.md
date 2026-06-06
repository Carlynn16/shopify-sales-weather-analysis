# Project Brief — Shopify Sales & Weather Analysis

> Guiding document for the rebuild. Doubles as a memory aid and the backbone for
> the README and the final PDF report. Keep it in the repo root.

---

## 1. Context

**Client:** a Danish artisan food chain (ice cream, chocolate, bakery, coffee) operating
**7 physical stores** around Copenhagen: [location], [location], [location], [location], [location],
[location], [location]. *(Client name kept out of the public repo for confidentiality.)*

**Platform:** the stores run their point-of-sale (POS) on **Shopify**. Every in-store
transaction is recorded there. Data is pulled from the **Shopify Admin API**.

**Period:** ~14 months, **Jan 2024 → Mar 2025**. After cleaning: ~454k valid transaction
lines across ~280k orders.

**Why it matters:** this is *physical retail*, not online. Sales — especially ice cream —
are highly weather-driven. The client's own observation: a sunny 10°C weekend was their
best ever, while a similar-temperature but windy weekend (lower "feels-like") sold far less.
That anecdote is the seed of the predictive part of the project.

**Business goals:**
1. Understand sales: what sells, where, when, and surface data-quality issues.
2. Build a weather-driven sales forecast to support staffing, prep, and inventory.

---

## 2. Data dictionary (raw Shopify export — 7 tables)

> Real CSVs stay **local only** (`data/`, git-ignored). They contain PII (customer emails,
> names, addresses) and client-confidential figures — never committed.

| Table | Grain | Key columns |
|-------|-------|-------------|
| `orders` | one row / order | order_id, order_number, created_at, processed_at, customer_id, total_price, subtotal_price, total_tax, total_discounts, financial_status, fulfillment_status, location_id, cancelled_at, cancel_reason |
| `line_items` | one row / product line | order_id, line_item_id, product_id, variant_id, sku, **name**, quantity, price, total_discount, created_at |
| `products` | one row / variant | product_id, title, product_type, vendor, tags, variant_id, sku, price, compare_at_price, inventory_quantity |
| `customers` | one row / customer | customer_id, email, first/last name, orders_count, total_spent, state, tags, address fields |
| `locations` | one row / store (7) | id → store_name, city |
| `discounts` | one row / discount code use | order_id, code, amount, type |
| `refunds` | one row / refund | order_id, refund_id, amount, note |

`line_items` is the central fact table; everything joins onto it via `order_id` /
`product_id` / `location_id`.

---

## 3. Cleaning rules (carry over from the original work)

- Drop columns >95% missing: `sku`, `customer_id`, `email`, `product_type`.
- Fill missing: `store_name`→"Unknown Store", `title`→"Unknown Product",
  `vendor`→"Unknown Vendor", `tags`→"Unknown Tag". Remove rows tagged `Indpakning`.
- Keep only **completed sales**: `financial_status == 'paid'` AND
  `fulfillment_status == 'fulfilled'` AND `cancelled_at` is null AND store ≠ "Malte TEST".
- **Use `name` (point-of-sale label), not `title`** for product identity — many rows have a
  missing `product_id`, so the products-table `title` doesn't join. `name` is always present.
  (This is also the root of the "Unknown Product" issue — see findings.)

### Product categorization (8 families, manual mapping)
Ice Cream · Hot Beverages · Buns & Bakery · Chocolate · Gifts & Cards · Snacks & Nuts ·
Christmas · Others. **Client feedback applied:** flødeboller and bars moved from
Buns & Bakery → Chocolate (this changed the ranking — Ice Cream #1, Chocolate #2,
Buns & Bakery #3). Keep the mapping dict; consider externalizing it to a YAML/CSV so it's
auditable rather than a giant hardcoded dict.

---

## 4. Analysis plan

### Block A — Sales analysis (descriptive + diagnostic)
- Top products by **units** and by **revenue**; Pareto 80/20.
- Revenue by category (the 8 families).
- Store comparison: top products/categories per store, local champions.
- Seasonality: by month, weekday, hour-of-day.
- **Data-quality investigation:** the "Unknown Product" / POS-tag problem.

### Block B — Weather model (the client's custom request)
- Pull daily weather per store-city from **Open-Meteo archive API**
  (temp max/min, precipitation, rain, snowfall, windspeed, daylight; derive feels-like).
- Correlations: weather vars × revenue — overall, by category, by store, by product.
- Group comparisons: rainy vs dry, warm vs cold, long vs short daylight, summer vs winter.
- **Predictive model:** XGBoost for daily **ice-cream revenue per store**; output a short-horizon
  forecast usable for staffing/prep.

---

## 5. Key findings to preserve (from the original reports)

- Revenue is highly concentrated: top ~20 products ≈ **80%** of revenue (clean Pareto).
- **"2 kugler"** (two scoops) is the top revenue driver (~5.3M DKK, ~20% of revenue);
  ice cream dominates overall.
- Strong seasonality: peak **May–June**; busiest **Fri/Sat**; busiest hours **12:00–15:00**.
- Ice cream peaks summer; hot beverages & Christmas items peak winter.
- **Data quality:** "Unknown Product" ≈ 11.8% of units, ~73% from **[location]**, tied to a POS
  terminal not synced to Shopify SKUs (missing `product_id`). Issue **declined sharply after
  Oct 2024** — likely a POS reconfiguration fixed it.
- **Weather:** temp-max and daylight correlate ~0.71 with daily revenue; ice cream ~0.76–0.77;
  hot beverages correlate *negatively*. Stores differ in weather sensitivity ([location]/[location]
  most sensitive, [location] least).
- **Model:** XGBoost + TimeSeriesSplit (3 folds), R² ranged 0.42–0.84 across seasons,
  MAE ~966–3,698 DKK. SHAP top drivers: daylight, lag_7 (last week's revenue), temp-max.

---

## 6. What to upgrade (the "data science" rigor the old report lacked)

The original analysis is solid on business framing — keep ~70%. Add the rigor that makes it
a credible *data science* deliverable:

1. **Inferential stats, not just eyeballing.** Where the old report compares groups
   (rainy/dry, warm/cold) or reports a Pearson r, add proper tests and uncertainty:
   t-test / Mann-Whitney for group differences, bootstrap CIs for correlations and medians,
   and report p-values + effect sizes.
2. **Handle multicollinearity honestly.** Temp, feels-like, daylight, and season are nearly
   the same signal. Show a correlation matrix / VIF, and be explicit that the model's SHAP
   importances are entangled.
3. **Strengthen the model section.** Add a **baseline** (naive lag-7 or seasonal-naive) so
   XGBoost's value is justified; add error analysis (residuals by store/season), and
   prediction intervals. Diagnose the weak autumn fold (R²=0.42) instead of just reporting it.
   Replace the dummy-weather "forecast" with either real forecast inputs or a clearly labeled
   scenario analysis.
4. **Fix the discount contradiction.** Cleaning excluded discounts as unreliable, yet the old
   report has discount insights. Either include discounts with a stated caveat or drop them.
5. **Consolidate.** The two overlapping docs (`Shopify.docx` + `Shopify_new.docx`) become
   **one** coherent report. Drop the Streamlit dashboard and the unused SQL/RDS layer.
6. **Add methodology + limitations.** Data collection, cleaning decisions, caveats, and what
   the model can/can't be trusted for.

---

## 7. Final report structure (the PDF showpiece)

1. Executive summary
2. Data & methodology (sources, cleaning, categorization, caveats)
3. Sales analysis (Block A) — each figure embedded + interpreted with numbers
4. Data-quality deep-dive (the Unknown Product / POS story)
5. Weather analysis (Block B) — correlations with significance + CIs
6. Predictive model — features, CV, baseline comparison, errors, intervals, SHAP
7. Recommendations (staffing, inventory, POS fixes)
8. Limitations & next steps

---

## 8. Out of scope / dropped

- Streamlit dashboard (`shopify_store_app.py`) — runs on synthetic data, not the real analysis.
- SQL → PostgreSQL/RDS migration layer — unused by the actual analysis; reads CSVs directly.
- Any committed secrets — the hardcoded Shopify token, DB passwords, and AWS RDS creds in the
  originals must be moved to environment variables and never committed.
