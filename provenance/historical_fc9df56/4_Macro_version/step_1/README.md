# Step 1 — the contract panel and the county universe

An application of the aggregation–disaggregation invariance framework to US defense
procurement: contract dollars at the county level, 1966 to 2018, matched to county
employment, tested going down to county-by-industry cells and up to commuting zones,
states and regions.

`../PLAN_macro_fiscal_multiplier.md` is the work plan — what to build and in what order.
This file is the reference for **step 1**, Part 1 of that plan: what the data is, what the
code does, what was decided, and what has been verified. `INVARIANCE_TEST_PROTOCOL.md` at the
project root governs any test design and has to be read before either test is specified here.
The disaggregation is Part 2; `../step_2/README.md` is its reference and
`4_Macro_version/Results/` holds its five typeset tables.

**`APPENDIX.md` beside this file holds the superseded instrument and the eleven exploratory
exercises built around it.** Nothing there is a current result. Sections swept into it keep
their original numbers, so a citation of `§1.8.5` still resolves — it is now `APPENDIX.md`
§A.4. §0.6 is the crosswalk.

## Contents

| | |
|---|---|
| **§0** | **What is settled** — the design of record, in one place |
| §1.1–1.4 | The DCADS sources: which files are the data, what the records mean, the parsed schema |
| §1.5 | Data-layer decisions: the negative-zero sign byte, FY1989, FY2004, the geography codings |
| §1.6 | The county universe — 3,101 time-invariant CCGs, and §1.6.10 the employment cells |
| §1.7 | The county panel, BEA income, and the shock |
| **§1.8** | **The instrument, and the first stage of record.** §1.8.12 is the live one; §1.8.3–§1.8.11 are in `APPENDIX.md`, so the numbering has a gap |
| §1.9 | Past FY2006: the raw USAspending block and the source audit |
| §1.10 | Recipient identity — the WS3 step 0 re-parse, and the DUNS/UEI coverage table |
| §2 | What each script does, and how to run the two pipelines |
| §3 | Verification, including §3.4 the Pallante test |
| §4 | Status and open |

---

## 0. What is settled

This section is the design of record. It was assembled from §1.9's two settled blocks, which
had been sitting at 75% depth in a file ordered by build date, and from §1.8.12, the only live
first stage. **Everything in §1.8.1 through §1.8.11 measured a different instrument** and is
now in `APPENDIX.md`.

### 0.1 The settings

| | |
|---|---|
| panel | `41_build_symmetric_nonoverlap_panel.py` → `panel.parquet`, re-scaled into Stata inputs by `Candidate for first stage/code/01_build_candidate_panel.py` |
| shock | `shock_ct = future_ct / income_ct` — contracts summed over `[t+1, t+h]` over county real personal income at `t` |
| instrument | `z_ct = s_c^{fixed5} · national_future / national_income`. **No leave-one-out**, and the right-hand side never divides by the focal county's own income |
| exposure | `fixed5` — the county's share of national contract dollars over the five fiscal years ending the year before its era begins. Pre window 1974–78, post window 1993–97 |
| horizon | **`h = 4` headline**, `h = 3` and `h = 2` robustness |
| windows | period-anchored **nonoverlapping start years, spaced by `h`**. At `h = 4` pre is 1979, 1983, 1987, 1991 |
| eras | **pre 1979–1997, post 1998–2018.** 1998 belongs to post and is its first start year. Step 2 estimates `pre` only |
| universe | **3,094 CCGs** strictly positive in employment in every year FY1975–2006, fixed before any period, horizon or rung choice. The geographic crosswalk stays complete at **3,101**. The state rung retains the **48** states with at least two assigned commuting zones — **3,080 counties** |
| cap | `winsor4 … method(iq) level(5)` within start year, on **the shock and the instrument only, never the outcome**, computed at the county and never recomputed after aggregating (A.3.3). The **uncapped arm is reported as its co-equal** |
| floor | **none** — no nominal dollar floor, no population floor |
| fixed effects | two co-equal admissible arms: `unit + state × year` and `unit + year`. §1.8.13. **Step 2 estimates on the second** |
| cluster | the rung's own unit |
| dating | **forward.** At horizon *h*, an observation dated *t* sums contracts over `[t+1, t+h]` |

### 0.2 The instrument of record

```
future_ct = Σ_{s = t+1}^{t+h} D_cs
shock_ct  = future_ct / PI_ct
z_ct      = s_c^{fixed5} × ( Σ_{s = t+1}^{t+h} D^nat_s ) / PI^nat_t
```

`s_c` is the only time-invariant object in the design. Everything else carries a start year,
which is why the aggregation weights are rebuilt within each one rather than frozen at a single
cross-section.

Three properties distinguish this from the instrument `APPENDIX.md` describes, and each is a
reason the numbers there do not transfer: **forward** dating rather than backward over
`[t−h+1, t]`; **no leave-one-out** on either the national numerator or its denominator; and
**national** income in the shifter's denominator, so the right-hand side never divides by the
focal county's own income. That last is the candidate correction.

### 0.3 The first stage of record

`step_1/result/TableF1_ladder_first_stage.tex`, from `51`–`55`. **`h = 4`, exposure `fixed5`,
72 fits, 16 checks, all passing.** β below is the *first-stage* coefficient, the shock on the
instrument — not a 2SLS employment response.

| Era | Rung | Units | Obs | Uncapped, `sxy` | IQR5, `sxy` | Uncapped, `uy` | IQR5, `uy` |
|---|---|---|---|---|---|---|---|
| **1979–1997** | County | 3,080 | 12,320 | 4.90 | **395.32** | 5.01 | **492.48** |
| | Commuting zone | 737 | 2,948 | 4.15 | 111.72 | 4.76 | 174.82 |
| | State | 48 | 192 | 7.48 | 59.79 | 7.48 | 59.79 |
| **1998–2018** | County | 3,080 | 15,400 | 5.95 | **254.88** | 5.53 | **294.37** |
| | Commuting zone | 737 | 3,685 | 5.01 | 205.33 | 4.27 | 217.94 |
| | State | 48 | 240 | 0.66 | 13.53 | 0.66 | 13.53 |

`sxy` absorbs `unit + state × year` at the county and commuting zone; `uy` absorbs
`unit + year`. At the state rung the unit **is** the state, so the two arms are the same
regression and the pipeline asserts they agree at 0. **Step 2's Tables 1–5 estimate on `uy`**,
and its own county first stage is 492.4768 on the same 12,320 rows — §1.8.13.

**Read §0.5 before quoting any cell of this table.** Every number above 9 in it is a capped
number, and the cap is not mild.

### 0.4 What supersedes what

Each row is a convention that was in force, what replaced it, and where the record is. This
table is what replaced the design register §4's blockquote used to point at: the register was a
gate written while the instrument was an open question, and the question is closed.

| retired | replacement | record |
|---|---|---|
| flow difference `ΔG = D_t − D_{t−h}` | cumulative stock, then the candidate `future_ct` | `APPENDIX.md` §A.3 — an abandoned trial, and five claims it voided |
| the `ns` exposure | `fixed5` | §A.1 — invalid by construction, and the attribution to NS14's footnote 32 did not hold |
| exposures `fixed` / `rolling` / `rolling_t5` | `fixed5`, national denominator, no LOO | §A.1, §A.4 reading 6 |
| backward dating over `[t−h+1, t]` | **forward** over `[t+1, t+h]` | §0.2. A dating convention, **not** an identification difference; the two must not be mixed when a first stage feeds a second stage |
| leave-one-out on the national buildup | none | §0.2, §A.0 |
| `14`'s "1998 sits in BOTH sub-samples" | 1998 belongs to post | §A.4's Table I note |
| step_2's `PRE_END` 1997 / `POST_START` 2000 | pre 1979–1997 / post 1998–2018 | §1.9's conventions block |
| `fixed_9398` and its forced 2002–2018 window | the anchored era split | §A.11. The 2002 start was **not** a competing convention: it is the arithmetic cost of `h = 4` giving up leading years so every horizon reports on one sample |
| the \$25,000 dollar floor | none, since 2026-08-10 | §1.7.2. `--floor 25000` recovers it |
| overlapping annual windows | nonoverlapping start years spaced by `h` | §0.1 |
| WLS / employment weighting of the first stage | no regression weights at any rung | §A.6 is the single record of what it was worth; §A.7 declares it out |
| `--industry-support county-own` | `national` — cells absent as zeros rather than missing | §4 item 10 |
| EFSY negative employment cells kept | deleted at four digits, since 2026-08-17 | §1.6.10 item 3, §4 item 19 |

### 0.5 The two caveats that survive the settling

Neither is a legacy artifact. Both qualify the table in §0.3 directly, and a reader who stops
after this section should not stop without them.

**1. No IQR rule is mild on this shock, and the headline is a capped number.** `shock`'s
interquartile range at `h = 2` is **0.0092** while its maximum is **14.07** — 1,529 IQRs above
the median — so `p50 ± L·IQR` binds hard at *any* `L`, and `p50 ± 1000·IQR` still binds. The
consequence is visible in §0.3: the same regression on the same 12,320 rows gives *F* = 4.90
uncapped and **395.32** capped, a factor of 80, and **not one of the 36 uncapped cells in
TableF1 clears 9 while all 36 capped cells do.** That is why the uncapped arm is reported as
co-equal rather than as a robustness check. §4 item 17; `APPENDIX.md` §A.9 has the measurement.

**2. Shift-share inference is untouched, and it applies to this instrument.** `z_ct` is one
common national shifter times a time-invariant county loading, so Adão, Kolesár and Morrison
(2019) applies and clustering on the rung's own unit does not address it. Every interval in
§0.3 and §1.8.12 could be too narrow for that reason. §4 item 5; step_2's README §2.10 carries
the same caveat.

### 0.6 Where the sections went

Nothing was renumbered. Sections moved between two files and were reordered within this one.

| section | file |
|---|---|
| §1.1–§1.7 | this file, §1 |
| §1.8 intro, §1.8.1, §1.8.2 | this file, condensed to their live parts |
| §1.8.3 – §1.8.11 | **`APPENDIX.md`** §A.2 – §A.10 |
| §1.8.12 | this file — **the first stage of record** |
| §1.8.13, §1.8.14 | this file, new |
| §1.9, §1.9.1–§1.9.3, §1.9.5 | this file |
| §1.9.4 | **`APPENDIX.md`** §A.11 |
| §2, §3, §4 | this file |
| the two unnumbered run blocks | folded into §2.1 and §1.8.12 |

`APPENDIX.md`'s entries keep the original numbers as their identity — `§1.8.5` is `§A.4`, and
its heading says so — because those numbers are cited 74 times from 15 files outside this one.

---
## 1. Data

### 1.1 Candidate sources

**DCADS/MPCF at the National Archives, 1965–2006** — the Defense Contract Action Data
System and its predecessor the Military Prime Contract File, one record per contract
action, held as archival records under "Records of Prime Contracts Awarded by the
Military Services and Agencies". Series 606901 (MPCF, FY1966–75) and 578589 (DCADS,
FY1976–2006). Defense-only, but it reaches back to 1965 and carries place of
performance down to the county. **This is the series we downloaded and parsed**, and
everything below is about it.

**USAspending.gov, FY2001 onward** — the public universe of federal prime contracts,
aggregable to county, CBSA or state. Prime contracts are public from FY2001;
subcontracts are not captured before October 2007, so any treatment netting subcontract
flows only exists from late 2007. Park, Zhou and Zubairy (2025, Dallas Fed WP 2535,
`10.24149/wp2535`; revised as NBER WP 34686) build the county-quarter panel from it and
show that assigning every contract dollar to the prime contractor's county misses
subcontract outflows and inflows. Their 8-quarter earnings multiplier is **0.103 using
prime contracts only and 0.126 after correcting subcontract geography**; at 16 quarters,
0.230 and 0.253; 8-quarter employment, 0.078 and 0.095. That gap is a
treatment-construction component, not a spillover inferred from a change in geographic
scale, which is exactly what holding outcome, sample, instrument, controls, fixed effects
and horizon fixed would isolate. Their inputs are public but the establishment analysis
uses proprietary NETS, and no public replication archive was found. The production
county block uses the raw annual Department of Defense archives for FY2007-2018.
Custom FY2001-2006 downloads are retained only as an overlap diagnostic: their
unresolved domestic geography share is 87.747% of absolute award dollars in FY2001
and 28.546% in FY2006, with all six years reported in
`batch_logs/macro_version/usaspending_overlap_verification_report.csv`.

**Pallante (2026, *Journal of Applied Econometrics*, `10.1002/jae.70058`)** — defense
procurement 1966–2019 at the contract, state-year and region-year level, DCADS harmonized
against USAspending into a merged file of roughly 41.1 million observations. Public
replication package, 92 files, held locally and not yet executed. His published geography
stops at the state.

**Muratori, Juarros and Valderrama** ("Heterogeneous Purchases, Heterogeneous
Multipliers") — the county-level intermediate behind their replication package, whose
posted data carries an MSA identifier but no county. Their shared county intermediate is
an external source comparison. `data_govspending_adrien.dta`, 116,511
rows, FY1966–2020, three columns (`year_calendar`, `country_code`, `dollarsobligated`),
sha256 `7898c3a1773cf487…`, 1,772,910 bytes. Provenance and usage terms — cite their paper,
share our code and results — are recorded beside the file in
`4_Macro_version/data/source/README.md`, which is Dropbox-only and gitignored. Ingested by
`18_ingest_muratori.py`; §1.9 is the audit. Its FY1988–2000 block is excluded from
the comparison.

**Reporting thresholds** move across the panel: \$10,000 FY1966–82, \$25,000 FY1983–99,
about \$1 from FY2000. A long panel has to impose one threshold across the whole sample.

### 1.2 Which files are the data

211 objects, 2.8 GB, under the shared Dropbox project's
`1_current_paper/_data/source/DCADS/`. All raw and intermediate data paths are
Dropbox-only. `code/py/dcads_common.py` uses the documented project path by
default; `INVARIANCE_DROPBOX_ROOT` supplies a different mount of the same shared
project when needed. `INVARIANCE_DROPBOX_HOME` supplies the Dropbox account root
for restricted sources stored outside the shared project.

- **42 annual contract-action files** — the data proper. 10 MPCF (`RG330.MPCF.FY66`
  through `FY75`), 29 fixed-width DCADS (`RG330.DEFCON.*`, `RG330.DCADS.*`,
  `DCADS.FY89.PU` through `DCADS.FY03.PU`, plus the transition quarter `TQ76`), and 3
  delimited extracts for FY2004–06 (`web_format_2004.txt`, `web2005.txt`,
  `2006final.txt`), which ship zipped.
- **2 ZARK reference files** — `RG330.DEFCON.ZALF`, the Locator File (naId 1937373,
  1966–83, 2,383,494 bytes of 82-byte records, so 29,067 of them, carrying the FIPS
  equivalent of each pre-1984 state, county and city code) and `RG330.DEFCON.ZACF`, the
  Contractor File (naId 1937378, FY1976–85, 7.2 MB). **ZALF is parsed** — it is the
  pre-1984 geography and §1.5 is about what it says. **ZACF is not**; it is the contractor
  spine for the pre-DUNS era and nothing in Part 1 needs it.
- **9 FY2003 files with no extension** — `DCADS.FY03.DD1057`, `.DD350`, `.MN021` through
  `.MN026`, `.PUROFF`. Despite the names these are PDFs of forms and code lists, not
  data.

Access: MPCF and DCADS through FY1988 are unrestricted; FY1989–2003 are marked
"Restricted – Partly" under FOIA (b)(6), which attaches to taxpayer numbers (a sole
proprietor's is a Social Security number) and not to the business identifiers.
FY2004–06 are unrestricted again.

### 1.3 Which files say the rule of recording

The archive ships its own documentation, and nearly all of the parse rests on it rather
than on inference from the data.

| documentation | what it governs |
|---|---|
| `DCADS_TSS.pdf`, `MPCF_TSS.pdf`, `DCADS_FY2004/05/06_TSS.pdf` | Technical Specifications Summaries: declared record length and record count per file. The input to verification check 1 |
| `125.1DP.pdf` – `125.10DP.pdf` | MPCF per-year record layouts and codebooks |
| `124.1DP.pdf` – `124.28DP.pdf` | DCADS per-year record layouts and codebooks |
| `124.29DP.pdf`, `124.30DP.pdf` | ZARK Locator and Contractor file layouts. 29 is **read** and is what §1.5 rests on; 30, the Contractor file, is unread |
| `124.1-25SD.pdf` | supplementary documentation, including the public-use masking note |
| `DCADS.FY03.LAY`, `DCADS.FY2001.DOC`, `DCADS.FY2002.DOC` | machine-readable layouts |
| `FY2004/05/06_agency_record_layout.pdf`, the Procurement Coding Manual sections, the DD350 and DD1057 form reproductions | the delimited era |
| `FY2004_prime_contract_awards_by_state_and_county.pdf` and siblings | DoD's own published tabulations — an independent benchmark nobody has used yet |

**Our transcription of all of it** is the part that matters day to day:

- `code/layouts/*.json` — 12 layouts, one per era, each declaring a record length and the
  byte offsets, widths, types and labels of every field. A file is parsed only if its
  name appears in some layout's `applies_to`.
- `code/reference/dcads_tss.json` — the declared record counts (24 files), uncompressed
  sizes and record lengths, transcribed from `DCADS_TSS.pdf`.
- `code/reference/zark_fips_supplement.csv` — 146 rows, and the only place a judgement
  call about pre-1984 counties lives. See §1.5.
- `dcads_common.NATIVE_OVERRIDES` — its counterpart for the FIPS eras, and **one row
  long**. Together these two are all the hand-authored geography in the *parse*. See §1.5.
- `code/reference/county_changes_census.csv` — 100 edges transcribed from the Census
  "Substantial Changes to Counties and County Equivalent Entities" pages for the 1970s,
  1980s, 1990s and 2000s, and `code/reference/va_bea_combinations.csv`, the 24 Virginia
  county-city combination groups. These are the *harmonisation* inputs, one level above the
  parse, and they feed `code/reference/county_ccg_crosswalk.csv`. See §1.6.

Still unread and worth reading: `DCADS.FY03.DD1057`, a 131 KB PDF of the DD1057
report-line definitions, which is what a real DD1057 dollar total would need (see §1.5);
and `124.30DP.pdf`, the ZARK Contractor file layout.

### 1.4 What the parsed output looks like

One parquet per fiscal year at `1_current_paper/_data/temp/dcads_year/fy<year>.parquet` —
**42 files, 13,653,309 rows, 622 MB.**

Each file carries every field its layout declares — 50 columns in the earliest years, 97
in the latest — plus derived columns the parser adds:

| column | meaning |
|---|---|
| `fiscal_year` | the year label from the filename, `1976tq` for the transition quarter |
| `fiscal_month` | 1–12, where 1 is the first month of that fiscal year |
| `calendar_yearmonth` | `YYYYMM`, sortable, empty where the reporting period is unreadable |
| `is_lumpsum` | 1 for DD1057 summary rows, which carry no geography |
| `dollars_unit` | `thousands` or `dollars`, per §1.5 |
| `source_file`, `layout_id` | provenance |
| `layout_anomaly`, `month_key_repaired` | FY2004–06 only: which rows were repaired, and how |

Plus the four geography columns `07_add_county_fips.py` writes, which are the reason any
of this can be used as a panel. `county_code` is *not* present in all 42 years — FY2004–06
call it `county_fips` and hold only the last three digits — so **`county_fips5` is the
first county key that exists in every year**:

| column | meaning |
|---|---|
| `county_fips5` | 5-digit **contemporaneous** county FIPS (`SSCCC`), or empty. Not harmonised to one vintage: `fiscal_year` is the vintage |
| `state_fips2` | 2-digit FIPS state, or empty. Equals `county_fips5[:2]` whenever both are present, and is filled for a pre-FY1989 territory even where the county is not |
| `geo_class` | which kind of place the row is — see below |
| `fips_source` | `native` (FY1984–2003, copied) · `concat` (FY2004–06) · `zalf` · `supplement` · `territory_recode` · `manual_override` (one row, §1.5) · `none` |

`geo_class` is the column a county panel filters on:

| value | meaning |
|---|---|
| `county` | the 50 states and DC, county known. **This is the county panel** |
| `territory` | `60` Amer Samoa, `66` Guam, `69` N Mariana, `72` Puerto Rico, `78` Virgin Is. Real places, outside the 50 states and outside most employment sources. `county_fips5` may be empty where the source names no county, which is every pre-FY1989 territory row except Puerto Rico; `state_fips2` still identifies the territory |
| `associated` | `64` Micronesia, `68` Marshall Is, `70` Palau, `74` US Minor Outlying. FIPS-coded but **not US counties** — sovereign since 1986/1994, or uninhabited. Never in a county panel |
| `undistributed` | state `98`, the source's own "somewhere in the United States" |
| `foreign` | a genuinely foreign country or water area |
| `lumpsum` | a DD1057 summary row, which carries no geography by construction |
| `unplaceable` | domestic, in the 50 states or DC, but no county resolves: a blank county field, the county-unknown code `999`, or a malformed code |

30 columns are present in all 42 years, including `dollars`, `state_code`, `county_name`,
`contract_no`, `action_date`, `fsc`, `purchasing_office` and the four above.

### 1.5 Decisions, and what to know before using the data

These are judgement calls the parse required. None is recoverable from the code, and each
would be expensive to work out again. **Append here whenever another call gets made.**

**Dates.** The reporting-period field `month_key` looks like `yyyymm` and the source
labels it that way, but the month counts from the start of the fiscal year: 01 is
October, 12 is September. Checked on all 23 years that carry the field, no exceptions.
Read as a calendar month it shifts everything by a quarter, so the conversion is done
once in `dcads_common.derive_time_columns`. The fiscal year is also not one thing: it ran
**July to June through FY1976**, October to September from FY1977, and July–September 1976
is a one-off bridging quarter, TQ76 — named for 1976, not 1975, because it starts and ends
in the same calendar year. Finally, `month_key` is when an action was *reported*, not
when the money was obligated; the two agree on fiscal year about 99% of the time and less
by month. No column is derived from `action_date` — if you need when the money was
committed, key on it yourself. Eight FY2006 rows carry `073106`, which is 31 July 2006
written day-first; they are real rows, all Navy, \$395,327 between them, so they are kept,
their year taken from `action_date`, their month left empty, and they are flagged in
`month_key_repaired`.

**Negative dollar amounts.** These files came off IBM mainframes, where the last
character of a number carries its sign, so a dollar field often ends in a letter rather
than a digit. Read naively it either fails or silently drops the minus sign, and negative
amounts here are deobligations, so losing the sign inflates every total. Six encodings
are handled; the full character map is the `ZONED` table in `code/py/dcads_common.py`.
One of them, the run `p`–`y` in FY2001–03, was silently failing on **5.16% of FY2001
rows** until it was found. It reads as *negative* because the standard negative encoding,
present in every other year on 4–6% of rows, falls to **exactly zero** in those three
years — deobligations did not stop happening in 2001, they started being written
differently. All 39 fixed-width years now decode every dollar amount, zero failures.

**A blank in that last byte is negative zero, and reading it as positive was our own
error.** Corrected 2026-08-06. In FY1966–75, and in no other fiscal year, the
negative-zero code point is written as a **blank** — 1,495 to 1,921 rows a year, 17,115 in
all, carrying **\$12.35bn** of deobligations. It had been read as *positive* zero, which
inverted the sign of every one of them, so **FY1966–75 contract dollars were overstated by
twice that, \$24.69bn** — 6.7% of the corrected \$368.94bn decade total. Any analysis built
on the pre-2026-08-06 parquet for those ten years has to be rerun. It was found by the Pallante
test (§3.4): it is the
whole of what used to be recorded there as an unexplained "MPCF-era record-universe
difference", and the argument that had ruled decoding out — *FY1977–83 use the same layout
and the same code path and match to the dollar* — was true and beside the point, because
the blank never occurs after FY1975, so those control years never exercise that branch.

Four things fix the reading, none of which depends on his series:

| | |
|---|---|
| **the slot is otherwise empty** | in FY1966–75 no negative-zero code point occurs at all unless the blank is it, while −1 through −9 each occur 700–1,900 times a year |
| **positive zero already has its own byte** | `'+'` in FY1966–67, plain `'0'` from FY1968, 22,000–31,000 rows a year — so the blank cannot also be positive zero |
| **it is continuous across the layout boundary** | FY1975 blank 1,701 rows → FY1976 `'^'` 1,665 → FY1977 `'^'` 1,700, same 272-byte layout; `n(−0)/n(+0)` is a smooth 0.054–0.083 over all nineteen years, and `'}'` takes over from FY1978 |
| **`range_code` settles the magnitude separately** | it is the record's own \|dollar\| band. A zone byte means the field carries a tenth digit, so the value is `head × 10`. In FY1975 **all 1,701** such rows fall inside their declared band at `head × 10` and **0 of 1,701** do at `head` — every one falls below its own floor |

Supporting but not load-bearing: 38% of these rows sit on a contract carrying a positive
row of exactly the same magnitude in the same year — an action being backed out — against
a 22–26% null for ordinary positive rows and 44–51% for the confirmed `'^'` and `'}'`
negatives. Some are unmistakable: contract `N000194C0041`, McDonnell Douglas, St Louis,
carries `'000030000'` and `'00003000 '` in the same month with the same FSC and the same
`range_code`. Read our old way that is two identical +\$30m obligations; read correctly the
second backs out the first.

**The fix is provably local.** Across all 42 fiscal years the only int-typed field values
ending in a blank with a non-blank head are these 17,115 FY1966–75 dollar amounts — every
other year and every other numeric field has none, the nine zipped years included. So
FY1976–2006 are untouched, and rerunning `06` and `08` afterwards returns their previous
results unchanged.

**Dollars change unit at FY1989.** Thousands of dollars through FY1988, dollars from
FY1989. The `dollars_unit` column says which for every year. Any total spanning that line
has to rescale.

**Part of FY2004 is corrupted, and the damage misses everything that matters.** 290 rows
depart from the standard record pattern, in four signatures, all declared in
`code/layouts/dcads_delim_2004.json` and remapped field by field; an undeclared signature
fails the file rather than being skipped. **Dollars, geography and dates survive on every
one of them** — the first byte position that moves in a damaged row is 396, and the amount,
state and county all sit before it. Of the 1,713 field values those rows leave empty,
**1,207 are recovered** from the rest of the same contract, guarded on contract number; the
other **506 are lost**, being blank in the source anyway. All 290 rows are flagged in
`layout_anomaly`, so a strict parse can drop them. What holds the damage to 290 rows is that
fields are read by **byte position, not by splitting on the `!` delimiter** — these files
are delimited and perfectly column-aligned at the same time, every row 601 characters and
603 in FY2005, and splitting instead discarded 238 rows carrying **\$6.23bn** and silently
shifted the tail fields of 52 more.

**Geography.** Place of performance is coded three incompatible ways across the panel:
ZARK through FY1983, a 5-digit SSCCC FIPS county FY1984–2003, and state and county in
separate fields FY2004–06. All three now resolve to `county_fips5`.

- **The filter any county-level work must use is `geo_class == 'county'`.** That replaces
  the old compound condition — `is_lumpsum = 0`, numeric `state_code` other than `98`,
  non-blank county — which is still correct but misses the FY1984–85 zero-fill trap below.
  County sums to state under this filter, and neither equals the national total.
- Pre-1984 codes are **ZARK**, the Pentagon's own scheme: the state is an alphabetical
  sequence (Virginia is 47, not 51; Wisconsin 50, not 55) and the county a *within-state*
  alphabetical sequence that repeats across states, so a ZARK code only means anything as
  a `(state, county)` pair. Concatenating the two and reading it as FIPS gives garbage.

**`RG330.DEFCON.ZALF` resolves the ZARK era exactly, and it was worth reading the file
rather than matching county names.** Its layout is documented in `124.29DP.pdf` p.4 and
confirmed by a column profile of all 29,067 records: 80 characters plus CRLF, three record
types discriminated by the last character — 241 state/country (FIPS state at cols 25–26),
3,278 county (FIPS county at 50–52), 25,548 city (FIPS place at 75–79). The documentation
calls those columns "GSA", but for the 50 states and DC they are FIPS 5-2 and FIPS 6-4
exactly, gaps included: Hawaii 12→15, Rhode Island 40→44, Washington 48→53. **It is not a
formula** — 245 of the 3,278 county records break the naive odd-number pattern and are
still right (New Mexico `015 LOS ALAMOS → 028`, Georgia `021 CANDLER → 043`), so the file
is required and the mapping cannot be reconstructed without it.

Three things about the mapping that decide how it has to be used:

- **It is many-to-one, not one-to-one.** Nevada `(29,013)` Ormsby and `(29,510)` Carson
  City both give 32510; Maryland `(21,004)` and `(21,510)` both give 24510. Fine as a
  function — but there is no inverse, and nothing should try to build one.
- **ZALF is a single undated cross-section** — no effective-date field exists anywhere in
  the record, and it is cumulative as-of-latest (it contains Virginia Beach, Carson City,
  and Nansemond city, which existed only 1972–74).
- **Some ZARK codes were redefined mid-panel**, so the join key is
  `(fiscal_year, state_code, county_code)`, never the pair alone. Alaska recycled its
  pre-1968 judicial-division codes onto census-division names — `(02,019)` is Fairbanks in
  FY1966–67 and Ketchikan from FY1968 — and Puerto Rico renumbered at FY1972, where
  `(PR,050)` is Mayagüez under one regime and Humacao under the other.

**What ZALF cannot say is in `code/reference/zark_fips_supplement.csv`, 146 rows.** It is
scoped by fiscal year and, for one code, by the recorded county name; a supplement row
always beats ZALF. Every row carries its reason. Four groups:

| kind | rows | what |
|---|---|---|
| `puerto_rico` | 86 | ZALF's FIPS column **echoes the ZARK code** for 77 of 79 municipios (Aguada `002 → 002` where the FIPS is `003`), so the whole island is overridden by name. 78 municipios for FY1972–83 plus the eight coarse FY1966–71 codes |
| `alaska` | 47 | The FY1966–67 judicial divisions and the FY1968–73 census divisions, split at the FY1967/68 break, mapped onto the 1970-census census-division FIPS that ZALF itself uses from FY1973. Plus three 1980-census borough codes that appear in FY1983 |
| `renumbering` | 8 | Virginia and Maryland renumbered at **FY1968**. Each old code appears only in FY1966–67 and its successor only from FY1968, at the same annual volume — which is what identifies them, not the name: `47/095 WARWICK` → Newport News 51700 (685/540 actions before, 400–516 after), `21/004` → Baltimore city 24510, and Norfolk, Alexandria, Hampton, Virginia Beach, York likewise. Nansemond County keeps its own 51123 rather than being folded into Suffolk |
| `zalf_patch`, `typo` | 5 | ZALF leaves Oregon Baker's FIPS blank and gives Tennessee Jackson an invalid `098`; Idaho `402` and New Jersey `031` are outside their states' ranges |

Two further ZALF records are **malformed rather than wrong** and are dropped by a guard
taken from the documented layout — position 3 is declared blank in all three record types.
One of them is a duplicate California state record carrying `YM` instead of `06`, which
**silently overwrote the correct one** when the file was first read without the guard.

**One record is corrected by hand, and it is the only one in the project outside the ZARK
supplement.** One FY1984 row has a damaged geography block: its state says California and
its county names a county in New York. The derived geography now reads Los Angeles County,
California. The override lives in `dcads_common.NATIVE_OVERRIDES`, keyed
`(fiscal_year, state_code, county_code)` exactly as `zark_fips_supplement.csv` keys the
ZARK era, and it is read by both `07` (which applies it) and `08` (which must not report
it as a hole in the crosswalk).

**The two rows of contract `F3460184C0986` as the source wrote them**, so that the change
is reversible and the original is never lost. They are the same contract, the same
contractor, the same day, the same supply class and the same purchasing office. Only the
geography block differs, and only one of the two is coherent:

| field | the intact row | **the damaged row** |
|---|---|---|
| `dollars` | `52` (thousands, \$52,000) | **`78`** (thousands, **\$78,000**) |
| `action_date` | `840322` | `840322` — same |
| `month_key` | `8406` | `8407` |
| `est_duns` / `hq_duns` | `066643735` | `066643735` — same |
| `contractor_name` | `A D E MANUFACTURING CORP` | same |
| `fsc` | `1650`, aircraft hydraulic vacuum de-icing components, system `E-3A` | same |
| `purchasing_office` | `3FD04`, `OKLA CITY ALC/PMZA ACCS BR` | same |
| `state_code` | `06` California | **`06` California** |
| `state_name` | `California` | **`California`** |
| `city_name` | `Irwindale` | **blank** |
| `city_code` | `0636826` | **`0633961`** |
| `location_code` | `0636826` — agrees with `city_code` | **`0636812`** — disagrees with it |
| `county_code` | `06037` | **`36103`** — Suffolk County, **New York** |
| `county_name` | `Los Angeles` | **`Suffolk`** |

**What was changed, and only this.** On the damaged row: `county_fips5` `36103` → `06037`,
`state_fips2` `36` → `06`, `fips_source` `native` → `manual_override`. `geo_class` stays
`county`. **No raw parsed field is touched.** `county_code` still reads `36103` and
`county_name` still reads `Suffolk`, because those are what the source says and falsifying
them would break the reconciliation against the raw bytes that checks 6 and 9 of `06` rest
on. The correction lives in the derived columns, which is where a judgement belongs.

**Why California, and why Los Angeles.** The state is not in doubt: `state_code` and
`state_name` both say California, and Pallante's independent extract of the same record
also places it in California (§3.4), so the county field is the error rather than the state
field. The county is then filled from the rest of the same contract, guarded on contract
number — **the method §1.5 already uses for 1,207 damaged FY2004 field values**. The
evidence is unusually tight for a within-contract fill: same contract number, same DUNS,
same action date, same FSC, same purchasing office, and the other action performed in
Irwindale, Los Angeles County. Nothing competes with it, because the damaged row's own
city fields are corrupt — the city name is blank and its two place codes disagree with each
other, where on the intact row they agree.

**Why do it at all, rather than leave it and note it.** Left alone, the row put \$78,000 of
California money into New York in any county or state panel built off `state_fips2` or
`county_fips5` — the columns the whole build exists to provide. Anyone aggregating by state
or county would have silently inherited a known error, and the note explaining it sits in a
README they may never read. **The modification is here to prevent future misuse of the
data**, not to improve a statistic: \$78,000 out of \$124bn changes no result, and correcting
it removed the only cross-state deviation in the Pallante test's FY1984 (§3.4).

**The guard that makes this safe.** The override declares how many rows it expects to
match — one — and `07` counts what it actually matched and fails the run on any other
number. An override that quietly matched nothing, or matched a hundred rows after a parse
change, would be worse than no override at all.

**Territories follow the FY1989–2006 convention, which the pre-1989 years are recoded to.**
The source is not consistent with itself: FY1984–88 files territories as foreign using
FIPS 10-4 alpha codes (`RQ`, `GQ`, `VQ`/`VI`, `AQ`, `RM`), while FY1989–2006 files them as
domestic with numeric FIPS and real counties. The numeric convention is what 18 of the 23
post-1984 years use and the only one whose codes are genuine FIPS, so it is the target for
all 42 years: `RQ005 → 72005` (the three digits are already FIPS), `GQ → 66`, `VQ`/`VI` →
`78`, `AQ → 60`, `RM → 68`, `PW → 70`, `CQ → 69`, `FM → 64`, and ZARK `PR → 72`. `WS`,
Western Samoa, is deliberately **not** recoded — it is a sovereign country, not American
Samoa. Where the source names no county the row keeps `state_fips2` and loses only the
county. The `associated` class exists so that Micronesia, the Marshall Islands, Palau and
the minor outlying islands — FIPS-coded but not US counties — can never leak into a county
panel.

Everything else about pre-1984 geography that was already known still holds:

- `location_codes` in the 272-byte layout is `state_code + city_code + county_code`
  concatenated and is **fully redundant** with those three columns, which are parsed
  separately in all 19 years. It is not a route to FIPS.
- State code **98** is the source's own catch-all for "somewhere in the United States,
  county unknown". It is not a real state, never reaches `state_fips2`, and classes as
  `undistributed`. The county code **`999`** is the same catch-all one level down — the
  state is known, the county is not — and appears on 635 rows across 50 states; those keep
  `state_fips2` and class as `unplaceable`. Reading `999` as a county would attach \$3.5bn
  to counties that do not exist.
- **Foreign work is kept** and reported on, never required to satisfy the county
  invariant, because the source breaks its own rule — Cuba arrives as state `CU` with
  county `00000` every year from FY1987 to FY1993.
- Rows are classified by **state code, never by the county string**. The county string
  used to be the test, and because FY2004–06 store the county separately and leave it
  blank for foreign work, every overseas row in those three years was invisible:
  unplaceable money was understated by **\$18–36bn a year**.
- **The FY1984–85 zero-fill trap.** Lump-sum rows in those two years write `state_code`
  `00` and `county_code` `00000` rather than blanks — 3,640 and 3,744 rows, about 31% of
  dollars. A `!= ''` test attaches them to a county that does not exist. Filter on
  `is_lumpsum` or on `geo_class`, never on non-blankness.

**The DD1057 figure is not a dollar total.** The DD1057 summary rows are a nested,
cross-tabulated report keyed by report number — B01, B02, B02A, B02B, B03, B03A–E, C01…,
D…, E…, F… — in which a parent line equals the sum of its lettered children exactly and
the blocks re-cut the same money different ways. Adding every row double- and
triple-counts: FY1995 gives **\$99.11bn** over all rows against **\$53.41bn** over parent
rows only. The column is named `lump_report_sum_nonadditive` and **must never be added to
a national total**. A real DD1057 figure needs the report-line definitions in the unread
`DCADS.FY03.DD1057`.

**The reporting threshold moves, and so does what a county can see.** \$10,000 FY1966–82,
\$25,000 FY1983–99, about \$1 FY2000–06. Below-threshold money sits inside DD350 before
FY1983, in the geographyless DD1057 block FY1984–2003 (5–12% of DD350), and back inside
DD350 from FY2000. **FY1983 has it in neither.** This moves the panel's coverage on DoD
reporting dates, not defense-policy dates, and FY1983–84 sits inside the Reagan buildup —
worth a robustness split at FY1984.

**Housekeeping — three rules, each preventing a specific silent error.**

1. **The fiscal year comes from the filename, never from `manifest.csv`.** The manifest
   reads the start of a title's date range and so labels `RG330.MPCF.FY66` as 1965. Using
   it would date every MPCF year one year early. The manifest's year is used only for
   download folders and log lines.
2. **The layout is chosen by filename, never by record length.** FY1989 and FY1990–95 are
   both 464 bytes, but byte 408 means different things in each — hence a separate
   `dcads_464_fy89.json`. An unrecognised file is reported as skipped, never guessed at.
3. **Ten files carry no layout and are not parsed by 04/05**, and are listed as such
   rather than ignored: the nine extensionless FY2003 PDFs, plus `ZACF`. `ZALF` is also
   claimed by no layout — it holds no contract actions — but it is not unparsed: 07 reads
   it, against the record layout in `124.29DP.pdf` rather than a `code/layouts/*.json`.

### 1.6 The county universe: every boundary change, and what each one is fixed with

`county_fips5` is **contemporaneous** — each fiscal year carries whatever county code the
Pentagon used that year. A panel cannot key on it. This section is the catalogue of what
moves across the 42 years and the harmonisation each one gets;
`code/py/10_build_county_universe.py` implements it and
`code/reference/county_ccg_crosswalk.csv` is the result. Everything below was found by
querying the parsed files, not read off a list: 3,103 distinct county codes, 86,587
county-year cells, 12,547,161 actions under `geo_class == 'county'`.

**The answer, first.** A **consistent county group** (CCG) set: **3,101 units — 3,056
single counties and 45 multi-county groups holding 114 codes.** Labelled on the 1990 county
vintage, which is the vintage of the commuting-zone bridge already in the repo
(`2_illustration_aggregation/.../county_to_commuting_zone.parquet`, 3,141 counties → 741
zones), so the CZ rung costs no new reference file.

Two properties do all the work.

**The map is time-invariant.** One `county_fips5 → ccg_id` map applied to all 42 years, with
no year key. This is not tidiness. The source keeps using retired codes for years after the
successor exists, often in the same year, and a date-triggered map would put one place in
two units:

| retired code | successor live from | dollars booked to the retired code afterwards | last used |
|---|---|---|---|
| 12025 Dade FL | 12086, FY2004 | **\$1,666m**, 16,739 actions | FY2006 |
| 02010 Aleutian Islands | 02013/02016, FY1992 | \$288m, 805 actions | FY2006 |
| 02290 Yukon-Koyukuk | 02068, FY1992 | \$244m, 631 actions | FY2006 |
| 02060 Bristol Bay | 02164, FY1994 | \$109m, 440 actions | FY2006 |
| 02140 Kobuk | 02188, FY1998 | used in every year FY1993–2006 | FY2006 |
| 02231 Skagway-Yakutat-Angoon | 02282, FY2003 | \$26m | FY2006 |
| 51780 South Boston city | reverted to a town 1995 | \$0.12m, including FY2004–06 | FY2006 |
| 30113 Yellowstone Natl Park | abolished 1997 | \$0.25m, **in FY2003** | FY2003 |

Note the lags run the other way too: **Aleutians East and West were created in October
1987 and the source does not use them until FY1992–93.** Nothing about the change dates
predicts when the codes move.

**Every map is a merge, so nothing needs a weight.** Predecessor and successor go in one
group; so do parent and daughter where a county was created inside the panel. That is
exactly additive. The alternative — keeping the 1990 counties apart and splitting a
parent's pre-creation dollars into its daughters — would invent dollars the source never
recorded, because the source cannot say what Cibola County's dollars were in 1979: Cibola
did not exist. The cost is losing 69 distinctions, and it is paid openly below.

**The edge rules, stated once.**

- `new`, `deleted` and `code_change` **always** form a group, **unless** the source
  explicitly records that the population moved was zero. A unit that appears or vanishes is
  not survivable in a fixed universe; a boundary that moves no people moves no employment.
  That one exemption is what keeps Gallatin MT out of the Yellowstone group (the Gallatin
  side was unpopulated, the Park side had 52 people) and Southeast Fairbanks AK out of the
  Denali group.
- `boundary` edges **never** form a group by themselves. They are reported with their
  populations, and check 7 sweeps county employment for level breaks so an undocumented one
  still surfaces. Virginia is the exception and §1.6.3 is why.

#### 1.6.1 The authority, and where it is silent

`code/reference/county_changes_census.csv`, 100 edges transcribed from the Census
"Substantial Changes to Counties and County Equivalent Entities" pages for the 1970s,
1980s, 1990s and 2000s. 90 are effective on or before the end of FY2006 and form groups; 10
are later and are reported only.

**The cut-off is a decision, not an accident.** The Census records York County VA and
Newport News exchanging territory on 1 July 2007. Honouring it would fold Newport News —
\$50.9bn, the largest defence county in the country — into York. The panel ends at FY2006
and cannot see the exchange, so the edge is transcribed, flagged `in_window=0`, and not
used. The same rule parks the 2007–08 Alaska reorganisation (02105, 02195, 02198, 02230,
02275).

Three changes are **not in the Census list at all** and rest on this project's own
evidence. They are marked `proj_` in the file so the provenance is never lost:

| change | evidence |
|---|---|
| 13510 Columbus city GA → 13215 Muscogee, 1971 | Census does not treat Columbus city as a county equivalent. 13510 appears in DCADS FY1966–83 and in EFSY 1975–79 and never after; 13215 appears from FY1984. \$168m |
| 29193 → 29186 Ste. Genevieve MO, 1980 census | a re-alphabetisation of `Ste.` against `St.`. Both DCADS and EFSY carry 29193 early and 29186 late and the two never coexist. **29193 is not a valid Missouri county FIPS in any modern vintage** — it will look like a crosswalk error to the next reader, and it is not |
| 51695 Nansemond city VA → 51800 Suffolk | Census records only Nansemond *County* 51123 combining into Suffolk on 1972-07-01 and never mentions 51695. DCADS carries it in FY1974, FY1976 and FY1981; EFSY 1975–79 |

The Census is also silent on **what each of the ten deleted Alaska divisions became**: it
records the 1970s reorganisation and names no successor for any of them. Those successors
are `dcads_common.SUPERSEDED`, which is this project's research, moved there from `08` so
that `10` can make it fail rather than merely print it. That is the accountability gap the
old arrangement had — nothing failed if one of the thirteen stopped appearing.

#### 1.6.2 The 22 contract codes the 1990 vintage does not contain

These are the codes a panel keyed on a modern county list would silently lose. All 22 map
forward; none is dropped.

| code | name in the data | dollars | actions | → |
|---|---|---:|---:|---|
| 02010 | Aleutian Islands | \$1,437.9m | 3,049 | 02013 |
| 02068 | Denali Borough | \$631.4m | 421 | 02290 |
| 13510 | Columbus | \$167.9m | 2,044 | 13215 |
| 02120 | Kenai-Cook Inlet | \$147.8m | 227 | 02122 |
| 02260 | Valdez-Chitina-Whittier | \$115.7m | 112 | 02261 |
| 02040 | Barrow / North Slope | \$80.0m | 240 | 02185 |
| 02160 | Kuskokwim | \$37.0m | 89 | 02050 |
| 02140 | Kobuk | \$31.8m | 152 | 02188 |
| 02080 | Cordova-McCarthy | \$11.4m | 75 | 02261 |
| 29193 | Ste. Genevieve | \$10.6m | 71 | 29186 |
| 12086 | Miami-Dade | \$8.8m | 47 | 12025 |
| 02210 | Seward | \$4.1m | 92 | 02122 |
| 51123 | Nansemond County | \$2.7m | 71 | 51800 |
| 02250 | Upper Yukon | \$2.4m | 13 | 02290 |
| 02230 | Skagway-Yakutat | \$0.8m | 8 | 02231 |
| 02282 | Yakutat | \$0.6m | 9 | 02231 |
| 02200 | Prince of Wales | \$0.4m | 3 | 02201 |
| **48210** | Harris | \$0.3m | 11 | **48201, a source typo** |
| 02190 | Outer Ketchikan | \$0.2m | 4 | 02201 |
| 08014 | Broomfield | \$0.1m | 16 | **08013, a judgement — see below** |
| 51695 | Nansemond city | \$0.1m | 5 | 51800 |
| **36173** | *(no name)* | \$0.1m | 1 | **unplaceable** |

**Two of them are not counties at all.** `07_add_county_fips.py:94` accepts any 5-digit
numeric county whose last three digits are not `999`, so source typos survive into
`geo_class == 'county'`. **48210** is not a Texas FIPS — Harris County is 48201 — and the
rows carry the county name Harris, so it is corrected. **36173** is not a New York FIPS (NY
runs 001–123) and its single row carries no county name, so nothing can identify which
county was meant; it goes to `unplaceable`, the only \$99,000 in the whole build
deliberately kept out of every rung.

Those two were found by cross-checking all 3,103 contract codes against EFSY's 3,219, which
leaves exactly four unmatched. The other two are not in the table above because both are
genuine 1990-vintage counties: **30113** Yellowstone National Park, real but abolished in
1997 and carrying FY2003 rows anyway, and **51123** Nansemond, a real county abolished in
1974 — which is why EFSY, starting in 1975, has never heard of it.

**Broomfield is the only judgement in the file that is not a union of whole counties.**
Broomfield CO was created on 15 November 2001 from Adams 08001 (15,870 people), Boulder
08013 (21,512), Jefferson 08059 (1,726) and Weld 08123 (69). The rule would merge all five.
That would destroy four major Front Range counties carrying \$7.1bn of contracts in order
to absorb \$0.14m, so **08014 is assigned to Boulder alone, its plurality parent at 55% of
the transferred population**. It is the only place the crosswalk knowingly puts money in a
unit that did not contain it, and `--broomfield-merge-all` runs the four-parent alternative.

#### 1.6.3 Virginia, which is the hard case

42 independent cities, **\$170.14bn — 53% of all Virginia contract dollars**, against 96
counties holding \$149.36bn. And the coding granularity is not constant: **only 5 cities are
coded in FY1966–67 and 31 appear in FY1968**, so the city share of Virginia's dollars jumps
from 36.0% and 46.9% to 74.8% and 78.3%. In those first two years most city money sits
inside the surrounding county. Maryland, which has only 24510 Baltimore city, is fine.

Virginia gets the **BEA county-city combination convention**, and the reason is a finding
rather than a preference: **every annexing city in the Census record from 1970 to 1995 is a
BEA combination-area city.** The Census documents 21 Virginia annexations in that window,
moving between 200 and 13,500 people, and the BEA pairs absorb all of them. A convention
built for time-consistency in the national accounts turns out to cover exactly the boundary
history the archive contains.

The file is `code/reference/va_bea_combinations.csv`, 60 rows in 24 groups, each row
flagged for whether a Census annexation also backs it. **47 of 60 are Census-backed; 13 rest
on the BEA convention alone** and are printed by name at every run, because they are where
the convention costs something the data could otherwise resolve:

| BEA-only code | \$ | folded into |
|---|---:|---|
| 51610 Falls Church city | **\$15.399bn** | 51059 Fairfax |
| 51570 Colonial Heights city | \$0.926bn | 51053 Dinwiddie |
| 51670 Hopewell city | \$0.709bn | 51053 Dinwiddie |
| 51690 Martinsville, 51720 Norton, 51775 Salem, 51678 Lexington, 51580 Covington, 51640 Galax, 51089 Henry, 51195 Wise, 51035 Carroll, 51695 Nansemond city | \$0.001–0.105bn each | their counties |

`--census-only-va` drops all 13 and keeps them separate. **The largest consequence of the
convention is the Fairfax group — 51059 + 51600 + 51610, \$114.7bn in one unit**, the
biggest CCG in the panel by a factor of seven. Fairfax city is Census-backed (annexations in
1980, 1991 and 1994) so Fairfax County and Fairfax city are forced together whatever is
decided; Falls Church is the discretionary part.

Two Virginia groups are **not** BEA combination areas and exist because the Census forces
them: Halifax + South Boston (the 1995 reversion) and **Suffolk + Nansemond County +
Nansemond city**. The second is the subtlest case in the whole catalogue — 51800 runs
FY1968–2006 with no code change at all, but in 1972 it absorbed the entirety of Nansemond
County, so pre-1972 51800 and post-1972 51800 are different places. **No code set can show
that.** It is here only because the Census change list says so.

#### 1.6.4 The 42 groups, by contract dollars

| ccg | \$bn | members |
|---|---:|---|
| 51059 Fairfax +2 | 114.714 | 51059 51600 51610 |
| 51153 Prince William +2 | 16.908 | 51153 51683 51685 |
| 12025 Dade +1 | 5.779 | 12025 12086 |
| 51121 Montgomery +1 | 5.253 | 51121 51750 |
| 51053 Dinwiddie +4 | 3.840 | 51053 51149 51570 51670 51730 |
| 08013 Boulder +1 | 3.104 | 08013 08014 |
| 13215 Muscogee +1 | 2.882 | 13215 13510 |
| 51161 Roanoke +2 | 2.795 | 51161 51770 51775 |
| 04012 La Paz +1 | 2.109 | 04012 04027 |
| 02013 Aleutians East +2 | 1.964 | 02010 02013 02016 |
| 51003 Albemarle +1 | 1.637 | 51003 51540 |
| 02290 Yukon-Koyukuk +2 | 1.564 | 02068 02250 02290 |
| 51800 Suffolk +2 | 1.185 | 51123 51695 51800 |
| 51191 Washington +1 | 0.862 | 51191 51520 |
| 51019 Bedford +3 | 0.823 | 51019 51031 51515 51680 |
| 51199 York +1 | 0.754 | 51199 51735 |
| 02261 Valdez-Cordova +2 | 0.702 | 02080 02260 02261 |
| 02122 Kenai Peninsula +2 | 0.334 | 02120 02122 02210 |
| 35006 Cibola +1 | 0.298 | 35006 35061 |
| 02185 North Slope +1 | 0.250 | 02040 02185 |
| 02050 Bethel +1 | 0.181 | 02050 02160 |
| 30031 Gallatin +2 | 0.084 | 30031 30067 30113 |
| 29186 Ste. Genevieve +1 | 0.047 | 29186 29193 |
| 02231 Skagway-Yakutat-Angoon +4 | 0.034 | 02030 02230 02231 02232 02282 |
| 02188 Northwest Arctic +1 | 0.033 | 02140 02188 |
| 02201 Prince of Wales-Outer Ketchikan +2 | 0.031 | 02190 02200 02201 |
| 02070 Dillingham +1 | 0.029 | 02070 02164 |

The remaining 15 are Virginia pairs under \$0.51bn. The full list with members and dollar
weights is `$temp/dcads_year/county_universe_report.csv`.

#### 1.6.5 Boundary changes that no code set can show, and what catches them

Three documented cases move territory without moving a code, and none is merged, because
each would cost far more than it fixes:

- **Takoma Park, 1 July 1997** — 5,156 people moved from Prince George's MD 24033 to
  Montgomery MD 24031. Both codes unchanged, both counties above 700,000 people.
- **Denver / Adams / Arapahoe CO**, 18 October 1980 (2,500 people) and **17 May 1988, 43.31
  square miles with no population** — which is the Denver International Airport site.
  DIA opened in 1995, so Denver's employment steps up years after the boundary moved. That
  is a level break with no boundary change at its date, and it is the reason a level-break
  detector needs a documented change list beside it.
- **Caddo Parish LA 22017** goes from 212 actions in FY1982 and 93 in FY1983 to zero for
  FY1984–89 while adjacent Bossier Parish 22015 (Barksdale AFB) goes 89 → 203. Not a
  boundary change: the source reassigned its Shreveport-area work across the parish line.

What catches an undocumented one is check 7, and the instrument matters. **Run on contract
dollars the sweep is useless** — a 10× step in the three-year mean above \$1m fires 2,238
times, because a single large award genuinely moves a county tenfold. **Run on county
employment it works**: a 2× step in the three-year mean above 1,000 employees fires 82
times, 7 of them at a documented change date and every one of the largest "undocumented"
hits already inside a group (12025/12086, 13510/13215, 08014, 51683, 51095, 51177). The
residual is 75 county-years, and they are industrial history rather than geography.

#### 1.6.6 What the crosswalk does not fix

- **The reporting threshold**, §1.5, is a bigger distortion of the county *count* than every
  boundary change combined, and this build deliberately does not filter a single row.
  Sub-\$25k actions carry 0.18–3.6% of the money and decide whether a county appears at all:
  **423 of FY2005's 2,638 counties exist only because of actions under \$25,000.** Check 8
  prints the full per-year accounting; the floor is Part 1.4's to choose, and the two
  candidates are \$25,000 nominal in all 42 years (cheap, feasible in both directions, and
  it flattens the FY2000–06 break to 2,215) and a constant real floor anchored at the most
  binding year the source allows, about \$10k in 1975 dollars. **Settled 2026-08-07 as a
  \$25,000 nominal floor; reopened and reversed 2026-08-10 — the default is now NO floor**,
  with `--floor 25000` to restore the old one, because the floor guards against a reporting
  threshold rather than against anything in the design and one convention across every step
  beats two. The reasoning above did not become wrong, it was overruled; §1.7.2 records both
  sides and what the reversal costs.
- **The panel must be rectangularised.** Only 864 of 3,103 codes appear in all 41 fiscal
  years, carrying 90.4% of dollars. A county absent from a year has \$0 of contracts, not
  missing data. Build `CCG × year` from the universe, left-join, fill zero — otherwise
  zero-versus-positive stops being economic entry and exit. **Done on both sides now**:
  `11_build_employment_cells.py` for employment and `dcads_common.build_contract_panel`,
  called by `13_build_county_panel.py`, for contracts.
- **906 county-year cells are negative**, totalling **−\$1.25bn**, and 77 are exactly zero.
  Negatives must stay in levels or the county-sums-to-state invariant breaks. **Settled by
  the denominator, 2026-08-07**: the shock is a contract *difference* over lagged county
  personal income, so a negative or zero contract level never reaches a denominator and no
  special rule is needed. See §1.7.
- **Commuting zones are not free after all, but nearly.** Two groups straddle a CZ boundary
  — 02010/02013/02016 across CZ 34101 and 34114, and 04012/04027 across 38100 and 38300 —
  so the zones are merged rather than the groups split, and **741 CZs become 739**.
- **FY1966–74 has no employment.** EFSY's NAICS panel starts in 1975, so the county panel
  starts there. That also disposes of §1.6.3's Virginia problem, the FY1966–67 Alaska
  judicial divisions and the FY1966–71 Puerto Rico codes, all of which are pre-1975.

#### 1.6.7 The District of Columbia, which the employment residual was hiding

**EFSY files DC's employment under the statewide residual, and dropping the residual
deleted DC from 28 of the panel's 32 years.** Corrected 2026-08-06.

CBP, and EFSY after it, put employment the Census cannot place in a county into a
`fipscty = 999` row — a statewide residual, which must never be attributed to a county.
Check 6 dropped every `999` on that reasoning, which is right for fifty states and wrong
for one. **DC is a single county-equivalent, so `11999` *is* `11001`**, and EFSY writes it
that way in 1976–2003:

| years | how EFSY codes DC |
|---|---|
| 1975 | `11001`, 307,315 employees |
| **1976–2003** | **`11999`, 298,274–422,730 a year** |
| 2004–2006 | `11001`, 436,507–446,345 |

So DC had no employment denominator in 28 of 32 years while carrying **\$35.04bn of
contracts** in exactly those years — and DC is a top-tier defence county, not a tail unit.

**Why the old check 6 passed anyway, which is the more useful lesson.** It compared
*pooled* sets: every CCG appearing anywhere in employment against every CCG appearing
anywhere in contracts, over 1975–2006 as a whole. DC has real `11001` rows in 1975 and
2004–06, so it entered the employment universe and the set difference came out empty. A
unit present in one year passed a test the other 31 years failed. Check 7 could not catch it
either — a series that is *absent* rather than stepped never enters a three-year mean.

**The recode is discovered, not asserted.** A `999` residual becomes the county only where
the state has **no non-999 county rows at all in that year**, so the whole of the state's
employment is in the residual and a double count is impossible. Sweeping all 51 state
equivalents over 1975–2006, **state 11 in 1976–2003 is the only case**; every other `999`
sits beside real counties and is a genuine unplaceable residual, which stays dropped —
13.19m employee-years, 0.45% of EFSY. The 10.69m employee-years recoded are all DC.

What is hand-authored is only the *answer* — which state-years that sweep is expected to
return — and it lives in `10_build_county_universe.EFSY_RESIDUAL_AS_COUNTY` under the same
rule as every other judgement in the file: **it fails the run both on an undeclared
state-year and on a declared one that stops occurring**, and when it fails it moves no
employment at all, so a failing run cannot also half-apply its own declaration.

**Check 6 now compares unit-*years*, not pooled sets**, which is what makes this class of
hole fail rather than pass. Two CCG-years survive it and are declared in `EMPLOYMENT_GAPS`:
02188 Northwest Arctic in 1987 (\$0.68m) and 48269 King TX in 2006 (\$19,899). Both are
counties genuinely absent from CBP that year — no covered establishment, an economic zero
rather than a defect — but a zero the panel has to know it is carrying, so each is named
with its dollars and anything else fails.

Nothing in the data changed: `10` writes reference files only, the crosswalk is unaffected
at 3,172 rows, and no parquet was touched.

#### 1.6.8 What was verified

`10_build_county_universe.py`, eleven checks, **exit 0**. Nine can fail; 7 and 8 are reports,
because any threshold on them would be a number invented in order to be passed.

> - **0 unmapped contract codes**, 0 codes in more than one group, 0 groups without a
>   1990-base county.
> - **Dollar conservation exact in all 42 fiscal years.** \$99,000 routed to `unplaceable`
>   and excluded from every rung — the single 36173 row, and nothing else.
> - **85,523 group-years invariant** to which member code the source wrote.
> - **15 predecessor/successor pairs checked, 0 split.**
> - **741 commuting zones → 739**, 2 groups straddled.
> - **EFSY 1975–2018: 3,168 codes → 3,101 CCGs, 0 unmapped.** Contracts reach 3,024 CCGs,
>   all 3,024 inside the employment set; 77 CCGs have employment and never a contract. The
>   comparison is on CCG-**years**, not pooled sets (§1.6.7): of 66,922 contract unit-years,
>   **2 have no employment that year and both are declared**, and the DC statewide residual
>   is recoded in the 28 years EFSY files it that way. So every unit-year the contract panel
>   can see has an employment denominator, bar two cells holding \$0.70m between them.
> - **All 13 `SUPERSEDED` codes sit with their successor** in a multi-county group.
> - **4 hand-authored judgements, each declaring its expected contract-action count and each
>   matching**: 08014 → 08013 (16), 02030 → 02231 (0, deliberately — it is in EFSY but never
>   in the contracts), 48210 → 48201 (11), 36173 → unplaceable (1).
> - Then `08` and `06`, both unmodified in behaviour: **42 of 42, 2,290 state-years exact,
>   13,653,309 records.** The crosswalk build writes no parquet, so this is true by
>   construction and checked anyway.

#### 1.6.9 Extending the universe to 2018, which the employment panel needs

Added 2026-08-06. The contract panel stops at FY2006 and the employment panel runs to
2018, so **"the window" is two windows** and `county_changes_census.csv` now carries a
flag for each: `in_window` keeps its meaning, the contract window, and `in_emp_window`
sits beside it. Both are asserted against `effective_date` on every run, so a
mis-transcribed flag fails rather than quietly grouping the wrong era.

**Group formation now uses the longer window.** One CCG set has to serve every rung of
both ladders — `INVARIANCE_TEST_PROTOCOL.md` A.1.6 requires an identical set of baseline
units at every level — so an edge effective any time through 2018 forms a group. The
cost is losing six more distinctions in the contract years, which is the same trade this
file makes everywhere else.

Two things made this smaller than it looked and one made it larger.

**The 2007–08 Alaska edges were already transcribed.** `2000_new_02105`, `02195`,
`02198`, `02230`, `02275` and their three `deleted` partners had been in the file all
along with `in_window=0`, and the loader skipped them. Nothing needed researching; the
flag was the whole problem.

**The 2010s decade was missing entirely.** Transcribed from the Census "Substantial
Changes to Counties" 2010s page. EFSY lags the Census by one to four years, so each row
records the year EFSY actually switches:

| change | Census | EFSY switches |
|---|---|---|
| `02270` Wade Hampton → `02158` Kusilvak | 2015-07-01 | 02270 ends 2016, 02158 begins 2017 |
| `46113` Shannon → `46102` Oglala Lakota | 2015-05-01 | 46113 ends 2016, 46102 begins 2017 |
| **`51515` Bedford city → `51019` Bedford County** | 2013-07-01 | 51515 ends 2016 |
| `02195` Petersburg Borough absorbs part of `02105` | 2013-01-03 | same code, `boundary`, no group |
| `02261` → `02063` + `02066` | 2019-01-02 | outside both windows; 02261 runs to 2018 |

**`51515` is the one a code-set sweep cannot find, and the reason this section exists.**
Sweeping EFSY for county codes the crosswalk does not contain returns six — the Alaska
codes and the two renames — and returns nothing for Bedford, because `51515` and `51019`
are both perfectly good 1990-vintage codes that were already in the universe. Without the
edge, Bedford city's employment simply stops after 2016 and Bedford County's steps up: a
fabricated exit and a fabricated jump, of exactly the kind §1.6.5 says check 7 exists to
catch. Only the Census list finds it.

**Newport News is untouched**, which was the worry that put the window rule here in the
first place. The July 2007 York exchange is a `boundary` row, and boundary rows are
reported and never merged, so widening the window cannot fold a \$50.9bn county into
York. Verified in the rebuilt crosswalk: `51700` is still `identity`, `n_members=1`.

**After the extension:** 3,172 crosswalk rows, still **3,101 CCGs**, 0 unmapped EFSY
codes over 1975–2018, dollar conservation still exact in all 42 fiscal years, 85,523
group-years still invariant, and all four hand-authored guards still matching.

#### 1.6.10 The employment cells and the allocation ladder — Part 1.3

Written 2026-08-10, from the code and `batch_logs/macro_version/11_build_employment_cells.log`,
replacing what §4 item 3 records as lost. It sits under §1.6 because the cells are rectangularised
on §1.6's CCG universe and §1.6.9's 2018 extension exists for them. `11_build_employment_cells.py`
writes `naics4_alloc`, `naics4_raw` and `naics2`, one parquet per year, 1975–2018, on a support
that is **identical in every year and in the same row order**, so a lag is an array shift and not a
join. Downstream code asserts that rather than trusting it.

**Three properties of EFSY that bite the estimates, not just the build.**

1. **Not everything reaches four digits.** EFSY assigns employment to the most detailed NAICS code
   it can and parks the rest on a shorter *root*: `237///` is the part of sector 237 that could not
   be pushed further down, not the sector total. Roots come in two forms and EFSY marks them
   differently — three-digit roots are slash-padded and documented, **two-digit roots are
   dash-padded and are not** — so any code that finds roots by looking for a `/` silently reads
   every sector root as a resolved six-digit code. That is 13.0m employee-years misread, and it is
   why `00_efsy_granularity_check.R` is superseded. Together the roots are **63.6m employee-years,
   all of it before 1998**: the pre-1998 share of employment below four digits runs 2.2–5.0% a
   year and is exactly **0.00% from 1998**, when CBP goes native NAICS.
2. **Empty cells are absent, not zero.** EFSY stores only positive rows, so a county that lost an
   industry has no row rather than a zero, and entry and exit — defined on a cell being zero at
   `t−h` or `t` — cannot be computed at all. The panel is rectangular here: every CCG in every
   year, and within a CCG every industry it ever reported, as a zero where it is absent. A county
   absent from a year had no covered establishment, which is an economic zero and not missing
   data; dropping such a cell is what a log specification does by accident and what A.3.1 forbids
   doing on purpose.
3. **Negative employment. Deleted at four digits since 2026-08-17** — this reverses what this item
   said before, and the old behaviour is `--negative-cells keep`. Raw EFSY carries 4,661 negative
   rows totalling **−1,276 employee-years**, which become **1,674 negative cells** at
   `naics4_alloc` totalling **−134.44 employee-years**, across 191 industry codes and confined to
   1981–1997. They are rounding artifacts of EFSY's desuppression. They are now deleted, and the
   two-digit rung and the county rung are summed from the cleaned four-digit cells, so **no rung
   can carry a negative** — asserted, not assumed.

   *Why deleting is safe here.* The deletion happens on the allocated frame **before the support is
   built**, so rectangularisation puts every deleted cell back as a zero: `emp`, `emp_observed` and
   `emp_alloc` all 0, `alloc_rule` −1, `filled_zero` true, `in_efsy` false — indistinguishable from
   a county-industry EFSY never reported. Verified against the previous build: **939,603 rows and
   303 codes a year, unchanged**, which is what `22` and `28` rely on when they treat a `t−h` lag as
   an array shift. All 191 affected codes are positive in some other county-year, so no code leaves
   the support; `11` fails rather than proceeds if that ever stops being true.

   *What it moves.* County employment rises in **826 of 136,444 county-years**, 659 CCGs, 1981–1997,
   by **+134.44 employee-years** in total — median 4.9e-6 of the county total, at most 1.0e-2.
   **No county-year crosses zero**, so the `emp_lag > 0` estimation sample is the same one and every
   `n_obs` downstream is unchanged.

   > **Measured on the 3,101-CCG universe** — 136,444 is `3,101 × 44`, so this whole paragraph
   > predates the 3,094 restriction merged on 2026-08-17 and the counts are the pre-restriction
   > ones. The *qualitative* claims survive the restriction, because it only removes counties:
   > no county-year can start crossing zero by dropping other counties. The *numbers* have to be
   > re-cut. See the register, item 12.

   The county first stage moves in the sixth decimal: `h = 2`,
   `fixed`, 1975–2006 goes *F* 72.480000 → **72.479999** and β +0.076299 → **+0.076296**, se
   0.027232 unchanged. Nothing in §1.8 reads differently at the precision it is reported to.

   *What it costs, and it is a real cost.* Conservation against raw EFSY is no longer exact. The
   panel now runs **+134.44 employee-years above** raw EFSY, 3.1e-8 of it, all pre-1998, where it
   used to be asserted at *max deviation 0.000000e+00 in every year*. `11`'s section [4] keeps that
   assertion on the **allocation** — evaluated before the deletion, so a ladder bug cannot hide
   inside the deletion's tolerance — and reports the deletion on its own line beneath it. On A.3.1:
   the letter holds, because the cell stays in the panel as a zero and no baseline unit enters or
   leaves; what changed is a *value*, by hand, and that is a declared departure rather than a
   neutral cleanup. `naics4_raw` is deliberately **not** cleaned — it is built from the raw cells,
   is not a rung, and is the record the ladder is priced against.

   *One residual, reported not fixed.* 31 cells have `emp_observed < 0` while `emp ≥ 0` (−0.76
   observed employee-years): a negative observation offset by a positive allocation. Their
   employment is non-negative, so they are not negative cells and they stay.

**The ladder.** Each root is split across the genuine four-digit children sharing its prefix, in
proportion to their employment, three-digit roots first and sector roots second — the cascade
matters, because a sector root must be spread using weights that already carry what the
three-digit roots inside that sector delivered. Weights are `emp.clip(lower=0)`, so a negative
cell can never become a negative share, and a group whose whole weight clips to zero falls through
to the next rule rather than dividing by zero. That clip is still live and still needed under item
3's deletion, because **the ladder runs before the deletion**: negatives are data on the way in and
a measured deletion on the way out, which is also why a negative root is distributed rather than
discarded — the allocated total has to match the root total. Five rules, tried in order, with their
measured shares of the 63.6m root employee-years:

| rule | scope | 3-digit roots | 2-digit roots |
|---|---|---|---|
| 0 | same CCG, same year | 95.51% | 64.20% |
| 1 | same CCG, pooled over all years | 3.88% | 1.08% |
| 2 | same state, same year | 0.05% | 0.02% |
| 3 | national, same year | 0.00% | 0.00% |
| 4 | no genuine child anywhere — kept as its own cell | 0.56% | 34.70% |

Rule 4 is where sectors 95 (Auxiliaries) and 99 (Unclassified) land, because neither has any
four-digit code in NAICS at all, and it is a third of sector-root employment. **Rules 2 and 3 are
the only ones that can put employment into an industry the county never reports**, they are
**0.0006% of all employment**, and the run fails outright above 0.05% rather than letting it pass
unnoticed. `alloc_rule` on the output is the *coarsest* rule that contributed to a cell, not the
only one: a cell can receive from several, so the per-rule totals above are what to read.

**The one number to carry into any cell-level result.** Rules 1, 2 and 3 can give a positive
lagged base to a cell that EFSY reported as nothing that year, which moves it out of the entry
term and into the weighted-growth term. That is **979,542 cells, 5.389% of the 18.2m genuine
cells, carrying 4,591,153 employee-years or 0.1055% of national employment.** It is small in
employment and not small in cell count, and it lands in exactly the small cells where an unbounded
growth rate is largest — so any statistic computed on the tail of cell growth has to be shown at
`naics2` as well as `naics4_alloc`. **`naics2` is the control, and it is an exact one:** every
code, root or not, has a two-digit sector, so that rung needs no imputation in either era and the
ladder never runs on it. **`../step_2/README.md` §2.7 does exactly this**, and the cutoff collapse
across 1998 **appears at `naics2` too** — 185.5 → 5.5 there against 465.8 → 8.4 at `naics4_alloc`.
Since `naics2` needs no imputation at all, the break is **EFSY's vintage break and not the
allocation ladder**, which is what the control was built to establish.

> **Corrected 2026-08-18, and one claim withdrawn.** This paragraph used to cite
> `../step_2/README.md` **§2.0 Finding 3** — a section that does not exist in that file, whose §2
> runs §2.1–§2.12 — and the pair **28.1× at `naics2` against 25.2× at `naics4_alloc`**, measured on
> the `z_fixed_7074` build that step_2 superseded on 2026-08-17. On the current build the two
> factors are **34× at `naics2` against 55× at `naics4_alloc`**, so the ordering has **reversed**.
>
> **The conclusion the control exists for survives; a second claim attached to it does not.** That
> the break appears at the unimputed rung still shows it is EFSY's and not the ladder's. But this
> paragraph also used to conclude that "if anything the imputation attenuates the tail rather than
> creating it", and that rested on `naics2`'s jump being the *larger* of the two. It is now the
> smaller. **Do not carry the attenuation claim forward** — on the current build the imputed rung
> has the bigger break, and whether the ladder contributes to it is untested either way.

**`naics4_raw` is written and is not an outcome. Do not estimate on it.** It looks like "four
digits without the ladder" and is not: pre-1998 its cells mix genuine industries with the root
buckets themselves (`237///` → `237R`, `92----` → `92R`). A root is the part of a sector CBP could
not push further down, so cell growth on `237R` is the growth of what could not be classified, its
entry and exit are reclassification events, and the cell set is not the same object either side of
1998, when every root disappears. It is a worse outcome variable, not a control for the ladder;
`naics2` is the control. The file exists to make the ladder's effect on the *totals* auditable,
which is what §1.6.9's conservation check uses it for.

**One more property of the source, which is not EFSY's fault and grows.** CBP places less and less
employment in a named county over time, so the statewide `fipscty=999` residual this panel cannot
see runs 0.15% of employment in 1975, 2.05% in 2006 and 4.62% in 2018 — 1.26% pooled. The DC case
is recoded by declaration (`state 11 fipscty=999 -> 11001`, 28 years, 10.7m employee-years,
§1.6.7); the rest is reported and left. The panel keeps 98.74% of raw EFSY employment.

### 1.7 The county panel and the shock — Part 1.4

> **The counts in §1.7 predate the 3,094-CCG analysis universe**, as §1.8's banner explains at
> length. `13` now builds on `dc.load_analysis_ccg_map`, so the shape below reads **136,136 rows,
> 3,094 CCGs × 44 years, 1975–2018** — the panel was extended past FY2006 *and* restricted to
> counties strictly positive in employment throughout FY1975–2006. Check 1's own label was
> changed from "row count is 3,101 x n_years" to "row count is analysis CCGs x n_years after
> every join" to stop the figure being retyped. §1.9 has the current numbers.

Built 2026-08-07, rebuilt on the cumulative ΔG 2026-08-10, by `12_build_county_income.py` and
`13_build_county_panel.py`. Output under `1_current_paper/_data/temp/county_panel/`:
`income_ccg_year.parquet`, `contracts_ccg_year.parquet` and `county_panel.parquet`, the last
being **99,232 rows, 3,101 CCGs × 32 years, 1975–2006** *(pre-restriction, pre-extension)*.

```
G_ct     = Σ_{s ≤ t} D_cs                                      the cumulative stock
Shock_ct = ( G_ct − G_c,t−h ) / Y_c,t−h  =  ( Σ_{s=t−h+1}^{t} D_cs ) / Y_c,t−h
```

`D` real defense contract dollars, `Y` real BEA county personal income, both deflated to
2006 dollars by the CPI table in `dcads_common.CPI`, and `Y` **lagged** so the regressor is
predetermined. Income rather than employment because §3.3 of the plan needs the aggregate
shock — unit contract flow over unit income — to be the income-weighted average of its
counties' shocks, which is the arithmetic Part 3 rests on, and accumulating the numerator
leaves that untouched because a window sum of sums is a sum of window sums.

**`G` is cumulated because DCADS records obligations, not spending.** An obligation is a lumpy
one-time award covering years of work, so `D_ct − D_c,t−h` measures award *timing* and enters a
single award twice, once positive and once spuriously negative `h` years later. §1.8.4 carries
the record of that design, the first stage it produced, and `--delta flow` to reproduce it. The
cumulation's start year cancels in the difference, which `13`'s checks 4b and 4c assert.

At `h = 1` the window is one year, so `ΔG_ct = D_ct` and the shock is bare contract intensity;
the shock exists in 96,100 CCG-years, `sd 0.0704`, `median +0.0005`, `p99 +0.208`, `min −0.657`,
`max +11.08`. At `h = 2` it is `sd 0.1288`, `p99 +0.406`, `min −3.386`, `max +14.07`. It is not
winsorized: A.3.2 governs growth rates, and the tails here are small counties landing one
large contract.

**764 negative shocks — 0.8% of the estimation sample — are not a problem here, and that is the
point of the denominator.** Under the accumulation a negative shock needs *net* deobligations
across the whole window, so it is rarer than the 780 negative raw county-years it comes from.
They sit in a numerator over an income denominator and nothing ever divides by them. A shock
scaled by lagged contracts could not survive them.

#### 1.7.1 County personal income — BEA CAINC1

Nothing in this project had a county income series. CAINC1 is a static zip, no API key,
`LineCode 1`, thousands of current dollars, 1969–2024; the download is verified by content
and its sha256 recorded, following `03_verify_dcads.py`'s rule that a served size
disagreeing with a published one is not evidence of a damaged file.

**The work is not the download, it is that BEA's county set is not ours** — 3,149 areas
against our 3,101 groups, disagreeing in four ways. Each is declared in code and each
declaration fails the run if it stops being true.

| | what BEA does | how it resolves |
|---|---|---|
| Virginia | combines counties with their independent cities | `va_bea_combinations.csv` already carries `bea_code`, because the CCG set uses the same convention (§1.6.3). Our groups are sometimes **wider** — BEA keeps Roanoke city `51770` out of its `51944` and reports it separately — and both rows land on one CCG and are summed, which is correct because BEA's areas partition the country |
| Alaska | **time-splits at 1979**: the `029xx` historical divisions carry 1969–78, the modern boroughs 1979 on, with no overlap | `BEA_ALASKA_DIVISIONS`, 19 codes onto the 1970-census FIPS the crosswalk already knows — the same vintage `zark_fips_supplement.csv` resolves the pre-1984 contract records onto |
| two straddles | `55901` Shawano incl. Menominee (1969–88) and `15901` Maui + Kalawao (always) each span two CCGs | opposite rules, below |
| out of window | the 2024 Connecticut planning regions, Chugach and Copper River from 2020 | dropped by the window filter, not by name |

**The two straddles get opposite treatments because the data allows one and not the other.**
`55901` splits from 1989, so the split is taken from **BEA's own 1989 shares** — 93.084%
Shawano, 6.916% Menominee — and applied to 1969–88. Menominee is a real unit here, 740
workers on average and \$2.7m of contracts, so assigning it nothing for twenty years would
be worse. `15901` has never been split by BEA in any year, so there is no year to take a
share from: it goes whole to Maui, and **Kalawao `15005` is declared to have no income
denominator at all**. That is checked rather than waved through — Kalawao draws **\$0 of
contracts in every year of the panel** and averages 0.8 covered workers, and the run fails
if either stops being true. The county universe for the shock is therefore **3,100 by
declaration**, not 3,100 by accident.

**The check that catches any of this going wrong is the state rollup.** BEA's areas
partition each state, so a combination dropped, an Alaska division missed or a straddle
double-counted shows up immediately as a state that no longer adds up. **1,938 state-years,
maximum relative deviation 0.00e+00.**

> **A second BEA series was added on 2026-08-13, on the same geography.**
> `25_ingest_bea_employment.py` reads **`CAEMP25S`**, county total employment on the SIC basis,
> **1969–2000**, into `bea_employment_ccg_year.parquet` — 99,200 CCG-years, 3,100 CCGs. It exists
> only to supply §1.8.10's employment flag, which has to be measured on the era's exposure window
> and therefore needs 1969–74, where EFSY does not reach. Every declaration on this page applies to
> it unchanged, because it is the same BEA county set: the resolver, the Virginia combinations, the
> Alaska time-split, both straddle rules and the state rollup are **imported from `12` rather than
> restated**, and the rollup reproduces at **max relative deviation 0.00e+00** on employment as it
> does on income. Kalawao is absent for the same declared reason, so the support is the same 3,100.
>
> **It is not a general-purpose employment panel and must not be used as one.** EFSY is that, and it
> is what every *outcome* in this project is built on. BEA total employment counts jobs by place of
> work including proprietors, farm and government; EFSY counts paid employees of private covered
> establishments at a March reference date. The column is named `bea_emp` and not `emp` for exactly
> that reason. Against Dupor and McCrory's 2003–10 extract —
reported, not asserted, since it is an older BEA vintage on a modern county list — the
median county gap is 2–3% and the national gap runs −0.02% in 2003 to −0.71% in 2006, which
is BEA's own annual revision.

#### 1.7.2 Three construction decisions, and what each is worth

| decision | default | why | flag |
|---|---|---|---|
| dollar floor | **none, every action counts** — changed 2026-08-10, was \$25,000 nominal | the floor guarded against a DoD *reporting* threshold rather than against anything in the design, and one convention across every step beats two. Nothing downstream reimposes it. **What the floor was guarding against has not gone away** — see below | `--floor` |
| ΔG | **cumulative** — `G` is a stock and `ΔG` is the `h`-year window sum. Changed 2026-08-10, was the flow difference `D_t − D_t−h` | obligations are not a spending flow, so differencing them measures award timing; §1.8.4 has the record and the first stage it cost | `--delta` |
| TQ76 | **folded into FY1976** | the convention Pallante's published series uses, which is what §3.4 verified FY1976 against | `--drop-tq76` |
| fiscal → calendar | **FY *t* is calendar *t*** | FY *t* runs October *t*−1 to September *t* from FY1977, and CBP's March-12 reference date for year *t* sits inside it | `--fy-offset` |

At no floor the panel keeps **10,859,971 actions and \$5,968.5bn** of 2006 dollars across
66,859 nonzero CCG-years, against 7,481,956 actions, \$5,912.0bn and 63,390 nonzero CCG-years
under the old \$25,000 default. So dropping the floor adds **45% more actions for 0.96% more
dollars** — which is the shape of the thing: a very large number of very small awards.

**What the floor was doing, and what now has to be watched instead.** The source's own
reporting threshold moves — \$10,000 FY1966–82, \$25,000 FY1983–99, about \$1 FY2000–06
(§1.5). A common floor above all three flattened that. Without one, two things follow and
neither is hypothetical:

1. **The national buildup carries breaks on DoD reporting dates.** `ΔG^nat` now jumps at
   FY1983 and FY2000 for administrative reasons, and the buildup *is* the identifying
   variation. FY1983–84 sits inside the Reagan buildup, which is why §1.5 already called for a
   robustness split at FY1984 — that call is now load-bearing rather than optional.
2. **The county support moves with the threshold.** 423 of FY2005's 2,638 counties exist only
   because of actions under \$25,000, so which counties are non-zero is now partly a reporting
   artifact. The estimation *universe* is unaffected — it stays frozen at the 3,100 CCGs with
   an income denominator, §1.8.4 — but the set drawing non-zero contracts is not.

`--floor 25000` recovers the old default at every step, and §1.8.4's tables report what the
change cost the first stage. **The fiscal calendar itself changed at FY1977** — July–June
before, October–September after — so 1975 and 1976 are aligned on a different convention from
every later year, and `--first-year 1978` drops them.

#### 1.7.3 What was verified

`13_build_county_panel.py`, ten checks, **exit 0**.

> - **Row count fixed at 99,232 before and after every join**, and the row *order* is
>   byte-identical to the `(ccg, year)` spine. This is what makes the panel an aligned-array
>   object rather than something a join has to be trusted with, and it is asserted three
>   times, once per join, so a left-join fan-out or a dropped county fails on the spot.
> - **Contract dollars conserved by the collapse, max deviation \$0.00** in all 32 years,
>   against a total computed without the groupby or the reindex.
> - **CCG-year set equality**, not pooled sets — §1.6.7 is the record of what pooling hides.
>   Contracts and employment both cover all 99,232; income is short exactly the 32 declared
>   Kalawao years and nothing else.
> - **Cells sum to the county total**: 4.7e-10 at two digits, 9.1e-13 at four.
> - **The h-year lag is a true h-year lag**, asserted rather than assumed of `shift`.

---

### 1.8 The instrument, and the first stage of record — Part 1.5

The plan's Part 1.5 asks for the Nakamura–Steinsson instrument one level below their states, and
its check 5 asks for the first-stage *F* at every rung with any weak rung named in the text
rather than quietly dropped. **§1.8.12 is the answer**, and it is the only subsection here estimated
on the instrument the application uses. §1.8.13 reports it under both admissible absorptions.

**The numbering ascends and has a gap in it.** §1.8.1 and §1.8.2 come first because they are
background the rest assumes — what the exposure choices were and what the national buildup looks
like. §1.8.3 through §1.8.11 are in `APPENDIX.md`; the gap is where they were, and §0.6 is the
crosswalk. Nothing was renumbered to close it, because those numbers are cited from outside this
file.

> **§1.8.3 through §1.8.11 are in `APPENDIX.md`, as §A.2 through §A.10.** They measured the
> legacy `14_build_instrument.py` design — backward-dated, leave-one-out on both the national
> numerator and its denominator, the county's own income in the denominator, exposures
> `fixed`/`rolling`/`rolling_t5`, overlapping annual windows, `h = 2`, 3,101 CCGs — and every
> count and fit in them predates the 3,094-CCG analysis universe. They are a record of how the
> design was arrived at, not a source of numbers. §1.8.14 promotes the four findings from them
> that still carry an argument.

**One structural idea carries forward from the legacy build, and it is the reason one functional
form suffices.** At a higher rung the unit instrument is the ω-weighted average of the county
instruments,

```
Z_jt = Σ_c ω_cj Z_ct
```

with ω the same weight vector that aggregates the outcome. That is what protocol A.2.2 demands,
and it means the aggregation property belongs to how the unit instrument is *built* rather than
to the county instrument's algebra. The settled design uses income weights,
`ω_ct = income_ct / income_jt`, rebuilt inside every start year rather than frozen at a single
cross-section, and the **same** ω aggregates the shock and the instrument. The alternative
construction — build exposure at the unit and interact it there — was computed alongside the
legacy build and reported as a construction error; `APPENDIX.md` §A.4 has it, correlating at
0.98–1.00 with gaps up to 7.2 standard deviations in the tail.

The instrument is **just identified**, so rescaling `Z` leaves the 2SLS coefficient untouched
and moves only the first-stage coefficient and its *F*. The units are not worth arguing about
and nobody should later "fix" them — which is also why §1.8.12 leads on *F* and reports β
underneath rather than the other way round.

#### 1.8.1 The three exposures, and what each is worth

The plan left this open — "whether from one fixed cross-section or a rolling window. County
exposure is lumpy enough that the choice matters more than it would at the state." All are
built and reported side by side, because the answer turns out to depend on the rung.

| `--exposure` | definition | what it costs |
|---|---|---|
| `fixed` | Σ 1969–1974 real contracts / Σ 1969–1974 real income | **Strictly pre-sample.** BEA county income starts in 1969 and the estimation panel starts in 1975, so the window touches no estimation year: predetermined by construction rather than by argument. **611 counties come out at zero**, and their instrument is zero in all 32 years |
| `rolling` | the same ratio of sums over the trailing 5 years ending at `t−h−1`, needing at least 3 usable years | Tracks a defense base that moves over thirty-two years, and cuts the zero-instrument county-years from 19,552 to 18,068 at `h = 2`. The loading now varies with `t`, which puts it closer to the endogenous shock — and §1.8.5 Table II shows almost all of its first-stage power is cross-sectional, so it dies at the county rung |
| **`rolling_t5`** | the same window, **ending at `t−5` whatever `h` is** — added 2026-08-13 | Holds the loading at a **fixed distance from the sample at every horizon**, which `rolling` does not. Defined in 133,300 of 136,444 CCG-years against `rolling`'s 136,400: it pays three leading years for that. **It is the rolling exposure every table from §1.8.6 on now reports**; `rolling` survives only as an anchor |

##### Why `rolling_t5` exists, and why its window ends at `t−5`

`rolling` ends at `t−h−1`, so its clearance from the sample is one year at **every** horizon and the
window itself slides as `h` changes. A change in β across horizons therefore mixes a change in the
horizon with a change in where the loading was measured — which is the same class of defect §1.8.6
was written to remove from the *fixed* windows, left in place on the rolling one.

`rolling_t5` fixes the window at `[t−9, t−5]`. The choice of `t−5` is not free: the shock at `t` sums
`D` over `{t−h+1 … t}` and divides by income at `t−h`, so the earliest year the design touches is
`t−h`, which at `h ≤ 4` is `t−4`. A window ending `t−5` therefore clears every estimated year by at
least one year, and clears it by `5−h` years — the *same* amount at a given horizon for both eras.

| | window | first year the shock touches | clearance |
|---|---|---|---|
| `rolling`, `h = 2` | `t−7 … t−3` | `t−2` | 1 year |
| `rolling`, `h = 3` | `t−8 … t−4` | `t−3` | 1 year |
| `rolling`, `h = 4` | `t−9 … t−5` | `t−4` | 1 year |
| `rolling_t5`, any `h ≤ 4` | `t−9 … t−5` | `t−h` | `5−h` years |

**At `h = 4` the two coincide exactly**, because `h + 1 = 5` there. That is a free bug detector for a
shift-by-one, which is otherwise invisible in the output and fatal to the design, and it is asserted
in `14` (check 8b) and again in `19`, `23` and `26` (check 3b) in **both** directions — identical at
`h = 4`, and *not* identical at `h = 2` and `h = 3`. A one-sided check would pass an `end_lag` that
was silently ignored, since that would make the two series identical everywhere.

The construction is one function with one parameter, `roll_lag(h, end_lag)`, so the two exposures
differ in a single number and cannot drift apart; `slow_rolling_exposure` recomputes both by an
explicit loop over the window years, from the window's *definition* rather than from a shift.

Both are ratios of sums, so one low-income year cannot dominate them. Both exist so that the
exposure choice can be seen rather than asserted. Cross-sectionally `fixed` has mean 0.01471 and
sd 0.04896: the exposure is extremely lumpy, and its 90th percentile sits at 3.3% of income.

#### 1.8.2 What the national buildup looks like

The design needs national swings, and under the cumulative ΔG the series traces them cleanly. At
`h = 2` the buildup rate is **non-negative in every year**, rises from 0.0548 in 1975 to a peak
of **0.0837 in 1985** across the Carter–Reagan buildup, falls to a trough of **0.0302 in 2000**
through the post-Cold-War drawdown, and recovers to 0.0470 by 2006. At `h = 1` it runs 0.0148 to
0.0414 with a peak in 1983, at `h = 3` it runs 0.0466 to 0.1281 with a peak in 1985.
`instrument_report.csv` carries the year-by-year series at all three horizons.

The whole of the identifying variation is in that hump. §1.8.4 records what the flow difference
gave instead: a series that never exceeded ±0.009, flipped sign in 14 of 30 years, and was two
orders of magnitude smaller than the level of defense intensity it was supposed to be measuring.

**Two things to read alongside it.** First, **1998–2006 contains almost no swing** — 0.0314 down
to 0.0302 and back to 0.0470, one fifth of the pre-1998 range across nine years — and §1.8.5
Panel C shows that no rung identifies anything there. Second, **§1.7.2's warning still applies**:
without a floor the national aggregate moves when DoD's reporting threshold moves, at FY1983 and
FY2000, and both dates sit inside swings the design is trying to use. FY2000 sits inside the
post-1998 flat stretch, so the reporting break and the absence of a buildup cannot be separated
there. The FY1984 robustness split is the check for the other one and has not been run — open
item in §4.

#### 1.8.12 The ladder on the candidate instrument — `51`–`55`

Added 2026-08-18. **Everything in this section was re-estimated.** §1.8.11 is a view of two frozen
CSVs and created no script; this section is the opposite, and the distinction matters because every
number below sits on an instrument that none of §1.8.5 through §1.8.11 was measured on.

**Why the old ladder no longer answers check 5.** Tables I–V report the F at every rung on the
legacy `14_build_instrument.py` design: backward-dated `ΔG` over `t−h → t`, leave-one-out on both the
national numerator and the national denominator, exposure measured as a contracts-over-income ratio,
overlapping windows. Everything downstream has since moved to the **candidate income instrument**,
which is what step_2's Tables 1–5 and its winsorization exhibit run on:

```
future_ct = sum_{s = t+1}^{t+h} D_cs
shock_ct  = future_ct / PI_ct
z_ct      = s_c^{fixed5} * ( sum_{s = t+1}^{t+h} D^nat_s ) / PI^nat_t
```

Forward-dated, **no leave-one-out**, national income in the shifter's denominator, and the right-hand
side never divided by the focal county's own income. `41`–`50` and the candidate package are all
county-rung only — every one of them hard-asserts `G == 3080` and carries no geography above the
county — so until now there was **no first stage at the commuting zone or the state on the instrument
the application actually uses**. That is the gap this section closes.

##### The design

| Item | Setting |
|---|---|
| Rungs | county (3,080) → commuting zone (737) → state (48). No Census division. |
| Endogenous | `shock_ct`, the county contract shock above |
| Instrument | `z_ct`, exposure `fixed5` — pre window 1974–78, post window 1993–97 |
| Horizon | **`h = 4` headline**, `h = 3` and `h = 2` robustness |
| Windows | period-anchored nonoverlapping start years, spaced by `h` |
| Winsorization | two co-equal arms: **uncapped**, and `winsor4 … method(iq) level(5)` within start year, applied to **both** the shock and the instrument |
| Where the cap is computed | at the county **only**, never recomputed after aggregating (A.3.3) |
| Aggregation | income weights on both sides, `x_jt = Σ_{c∈j} ω_ct x_ct` with `ω_ct = income_ct / income_jt`, rebuilt inside every start year |
| Fixed effects | **two co-equal arms**, `sxy` = county and CZ absorb unit + state × year, and `uy` = unit + year at every rung. Both admissible under A.1.4; step_2 estimates on `uy`. §1.8.13 |
| Weights in the regression | none, at any rung |
| Cluster | the rung's own unit |
| Floor | none — no nominal dollar floor, no population floor |
| Eras | pre 1979–1997 and post 1998–2018, both estimated |

`s_c` is the only time-invariant object in the design: the county's share of national contract
dollars over the five fiscal years ending the year before its era begins. Everything else carries a
start year, which is why the aggregation weights are rebuilt within each one rather than frozen at a
single cross-section.

**β below is the first-stage coefficient, the shock on the instrument.** It is *not* the 2SLS
employment response that Tables I and V report beside their *F*. The employment ladder is Part 3's
object and needs the outcome; this section answers check 5 and nothing else.

##### Table XX. First-stage *F* at every rung, `h = 4`

Each cell is `F | β (se)`, on the `sxy` arm; §1.8.13 has `uy`. 72 fits in `ladder_first_stage.csv`, keyed on
`rung × era × h × winsorization`.

| Era | Rung | Units | Obs | Uncapped | IQR5 within start year |
|---|---|---|---|---|---|
| **1979–1997** | County | 3,080 | 12,320 | **4.90** \| +153.6 (69.4) | **395.32** \| +2666.6 (134.1) |
| | Commuting zone | 737 | 2,948 | 4.15 \| +87.6 (43.0) | 111.72 \| +2104.8 (199.1) |
| | State | 48 | 192 | 7.48 \| +50.7 (18.5) | 59.79 \| +4036.4 (522.0) |
| **1998–2018** | County | 3,080 | 15,400 | **5.95** \| +175.1 (71.8) | **254.88** \| +3541.0 (221.8) |
| | Commuting zone | 737 | 3,685 | 5.01 \| +115.0 (51.3) | 205.33 \| +4368.0 (304.8) |
| | State | 48 | 240 | 0.66 \| +33.1 (40.7) | 13.53 \| +2984.5 (811.5) |

##### Table XXI. The same ladder at `h = 3` and `h = 2`

| Era | Rung | h=3 uncapped | h=3 IQR5 | h=2 uncapped | h=2 IQR5 |
|---|---|---|---|---|---|
| **1979–1997** | County | 4.75 | 535.12 | 4.89 | **589.39** |
| | Commuting zone | 4.20 | 174.58 | 4.30 | 193.02 |
| | State | **8.43** | 77.81 | 8.33 | 75.73 |
| **1998–2018** | County | 6.64 | 288.83 | 6.58 | 344.44 |
| | Commuting zone | 5.22 | 204.10 | 5.21 | 242.63 |
| | State | 0.63 | 13.91 | **0.83** | 18.82 |

> *Notes to Tables XX–XXI.* **72 fits**, all in `batch_logs/macro_version/ladder_first_stage.csv`,
> none missing and all estimated for this section. Endogenous regressor the unit contract shock, the
> income-share-weighted average of its counties' `shock_ct`; excluded instrument the same weighted
> average of `z_ct`, one ω vector for both sides per protocol A.2.2. Just identified, so *F* is the
> single-instrument Wald `(b/se)²` under clustering on the rung's unit — the statistic every step_1
> script reports, and **not** a Kleibergen–Paap or effective *F*; register item 9 stays open and this
> section does not close it. Real 2006 dollars, **no contract floor**. Start years: 4 / 6 / 9 pre and
> 5 / 6 / 10 post at `h = 4 / 3 / 2`. The two winsorization arms are estimated on identical rows,
> `e(sample)` being fixed once on the uncapped shock before either arm is built, and **neither arm is
> the robustness check of the other**.

##### What it says

**1. The winsorization, not the ladder, is what decides whether this instrument has a first stage.**
**Not one of the eighteen uncapped cells clears 9** — the whole uncapped grid runs from 0.63 to 8.43,
across three rungs, two eras and three horizons. **All eighteen capped cells clear it**, from 13.53 to
589.39. That is a factor of forty to eighty on the same rows, the same instrument and the same
estimator, with one rule changed. Register item 1 is exactly this reading, and it is now visible at
every rung rather than only at the county. Nothing in this pipeline is selected on *F*, and the two
arms are reported side by side for that reason; but a table that quoted only the capped column would
be describing a property of the cap.

The mechanism is not mysterious. The exposure share is severely right-skewed — a handful of counties
hold very large shares of national defence contracting — so five interquartile ranges is a tight
bound, and the cap binds on 11.3–12.8% of county-years on the instrument and 8.5–10.4% on the shock,
per block, all six of them in `ladder_first_stage_share_audit.csv`. Capping
removes most of the instrument's variance, which raises β by more than an order of magnitude (+153.6
against +2666.6 at the county, pre, `h = 4`) and raises the *t* with it.

**2. First-stage strength falls up the ladder, in both eras and under both arms.** Capped, `h = 4`:
**395.32 → 111.72 → 59.79** before 1998 and **254.88 → 205.33 → 13.53** after. This is the ordinary
direction, and the same direction §1.8.5 reading 4 found under the cumulative ΔG once the county
stopped being dominated by lumpy award timing. It is the reverse of the flow-ΔG design.

**3. The post-1998 era identifies at the county and the commuting zone, which is new — under the
cap.** §1.8.5 reading 7 and §1.8.6 reading 2 both concluded that this is a pre-1998 design: county
*F* was 1.8 and then 6.2 there, and β was negative at three rungs. Here the post era gives county
254.88 and commuting zone 205.33, both with positive β. **The conclusion is arm-dependent**: the same
cells uncapped are 5.95 and 5.01, which is the old verdict. So the honest statement is that the post
era identifies under a rule that removes an eighth of the instrument's mass, and does not otherwise —
not that the earlier reading has been overturned.

**4. The state rung is where the two eras part.** Capped `h = 4`, 59.79 before 1998 against 13.53
after; uncapped, 7.48 against **0.66**, the weakest cell in the entire grid. Forty-eight units and
four or five start years is 192–240 observations, so this rung was never going to carry much, and the
post-1998 half of it carries nothing at all without the cap.

**5. Stable in the horizon, and `h = 4` is not the flattering choice.** The rung ordering is identical
at all three horizons in both eras, and within the capped arm the county *F* falls monotonically as
`h` lengthens — 589.39 → 535.12 → 395.32 before 1998, 344.44 → 288.83 → 254.88 after — because each
step costs start years. `h = 4` is the headline on step_2's convention, not on its *F*, and it is the
weakest of the three at five of the six rung-era blocks.

**6. The commuting zone is not nested in a state, and one zone needed repairing.** 135 of the 737
zones cross a state line, so the state-by-year effect at that rung uses `27`'s predetermined
largest-1969-population assignment. That assignment was computed on the full 3,101-county universe,
and it points one zone — CZ 20401 — at Rhode Island, which the 48-state rule excludes; the only
county of it that survives into the support is in Massachusetts. Left alone the zone would have sat
alone in each of its state-by-year cells, been absorbed exactly, and dropped, so the commuting-zone
rung would have covered 3,079 counties against the county rung's 3,080. `51` reassigns it on its
retained counties under `27`'s own tie rule and names it in the log. It is a support artefact
arriving through the geography labels rather than through the data, and it is the smallest possible
instance of the thing the aggregation exercise exists to measure.

##### Why these numbers can be trusted

Fourteen checks in `ladder_first_stage_verification.csv`, all PASS. The order is deliberate: the
county rung is pinned before any aggregate is looked at, because a correct aggregation of a wrong
county panel would satisfy every identity below it.

- **The pin, check 9.** The county rung of this ladder **is** the candidate first stage, so it is
  merged onto the certified 252-row `candidate_first_stage_all.csv` on the specification and required
  to reproduce it. Twelve cells, `n_obs` / `n_units` / `df_absorbed` equal **as integers**, and
  `fs_coef` / `fs_se` / `fs_F` to **3.3 × 10⁻¹⁴ relative**. Relative rather than absolute, for the
  demeaner reason §1.8.6 records. If it fails, `53` prints that the aggregate rungs were not evaluated
  and reports nothing else.
- **The build, check 1.** 400 sampled county-years have their shock and instrument rebuilt from
  `contracts_harmonized_ccg_year.parquet` and `county_panel.parquet` and compared: **0.0** on the
  shock and **2.2 × 10⁻¹⁸** on the instrument. The two properties that define the candidate correction
  — not leave-one-out, and never divided by own income — are asserted from the design contract as
  well, rather than inferred from a value that happens to match.
- **The preflight, checks 2–8.** One check per axis the specification moved on, so a script that
  quietly kept an old setting fails here rather than producing a plausible table: the harmonized
  FY1966–2018 panel on 3,080 counties and 48 states; horizons 2, 3, 4 with 4 declared the headline;
  `fixed5` windows ending the year before each era with shares summing to 1 within 10⁻¹⁵;
  nonoverlapping start years **and** pairwise-disjoint contract windows, since spacing alone is
  necessary and not sufficient; the IQR5-within-start-year cap reproduced from the stored raw columns
  at **0.0** on 26,633 capped values; no dollar floor and no population floor; and both eras present
  at all three horizons.
- **The aggregation, checks 10–12.** Unit income, employment and contracts are exact county sums at
  every rung and every start year (**2.1 × 10⁻¹⁶** relative); `Σ_{c∈j} ω_ct = 1` inside every
  unit-year, asserted year by year rather than pooled (**2.2 × 10⁻¹⁶**); and 3,080 counties sit under
  every rung-year, which is the frozen-support rule made numerical. `52` asserts the same identities
  independently inside Stata before it estimates, and refuses to proceed if `reghdfe` absorbs away a
  single unit-year.
- **Independent re-estimation, check 13.** All 72 cells refit in `pyfixest` with `CRV1` on the unit
  and `MapDemeaner(fixef_tol=1e-12)`, agreeing to **5.7 × 10⁻¹⁴ relative**. This is a genuine second
  opinion on the standard errors, not only the point estimates: `52` builds its *se* by
  Frisch–Waugh–Lovell plus a manual `sqrt((N−1)/(N−df_a−2))` factor, inherited unchanged from the
  candidate estimator, while `pyfixest` is handed the model in its natural fixed-effect form with no
  manual correction at all. The two paths agree to fourteen digits.
- **One implementation note that is not a result.** `pyfixest` screens for collinearity on an absolute
  tolerance and rejects the capped instrument at the aggregate rungs, which lives around 10⁻⁵. `53`
  standardises the instrument before fitting and undoes the scaling exactly — `b = k b'`,
  `se = k se'`, `F` unchanged — so check 13 compares on the original scale and no tolerance was
  relaxed to obtain it. Stata's `regress` on residuals has no such screen, which is why `52` estimates
  those cells without complaint.

##### Where the outputs are

| Output | Contents |
|---|---|
| `step_1/result/TableF1_ladder_first_stage.tex` | Tables XX–XXI as one typeset exhibit: **two tabulars, one per absorption arm**, each with both eras as panels. A tabular cannot break across a page, so all four panels in one ran 251pt off the body |
| `step_1/result/_preview.tex`, `_preview.pdf` | Throwaway wrapper so the table compiles on its own |
| `batch_logs/macro_version/ladder_first_stage.csv` | The 72 fits — 3 rungs × 2 eras × 3 horizons × 2 winsorization arms × 2 absorption arms |
| `batch_logs/macro_version/ladder_first_stage_headline.csv` | The `h = 4` block, wide over the winsorization arm — what a later run reads instead of re-estimating |
| `batch_logs/macro_version/ladder_first_stage_verification.csv` | The 16 checks |
| `batch_logs/macro_version/ladder_first_stage_design.json` | The design contract, including the CZ-state repair |
| `batch_logs/macro_version/ladder_first_stage_panel_audit.csv` | Rows, units, windows and start years per era-horizon block |
| `batch_logs/macro_version/ladder_first_stage_share_audit.csv` | The cap's cutoffs by start year, and how many values bound at each end |
| `batch_logs/macro_version/ladder_first_stage_collapse.csv` | Stata's own unit counts and county coverage per rung |
| `code/reference/ladder_first_stage_sxy_v1.csv` | **Git-tracked**, and the only output here that a run cannot overwrite: the 36-cell `sxy` grid as it stood before the `uy` arm existed. Check 15 pins against it. Same reasoning as `50`'s `V1_BASELINE` — the live CSV is rewritten every run |
| Dropbox project: `_data/temp/ladder_first_stage/panel.parquet` | The county panel, **already winsorized** |
| Dropbox project: `_data/temp/ladder_first_stage/<era>_h<h>.dta` | The six Stata inputs |
| Dropbox project: `4_Macro_version/data/temp/ladder_first_stage*.csv` | Mirrors of the grid and the headline |

**The stored panel is county-level and already capped, deliberately.** The capped shock and capped
instrument are what later analysis should use, so they are stored rather than the recipe. The rungs
above the county are *not* stored, because climbing needs a weight vector and that choice is not
frozen — income weights are what this section uses and PLAN §3.3's invariance version wants
employment weights. County rows plus both denominators leave every rung one collapse away; storing
the collapsed rows would freeze a choice that is still live.

##### How to run it

`55_run_ladder_first_stage.py` orchestrates `51` → `52` → `53` → `54`, with the same two modes and
the same Dropbox path resolution as `44` (§2.1).

```bash
# Verify and reuse the certified raw-derived county caches, then rebuild and estimate.
4_Macro_version/.venv/bin/python \
  4_Macro_version/step_1/code/py/55_run_ladder_first_stage.py \
  --certified-cache

# Execute the authoritative raw-data producers before rebuilding and estimating.
4_Macro_version/.venv/bin/python \
  4_Macro_version/step_1/code/py/55_run_ladder_first_stage.py \
  --from-raw
```

`55` requires the candidate package's certified result to pin against and rebuilds its gitignored
Stata inputs if they are missing, so a fresh checkout fails at a missing input rather than three
steps later inside a pin. **Rerunning `44` and `50` afterwards must still pass**: this pipeline
reads `41`–`50` and the frozen reference CSVs and writes to neither.

---

#### 1.8.13 Both absorptions, and why the *F* differs between step 1 and step 2

Added 2026-08-18. Until then `TableF1` reported one absorption, `unit + state × year`, and step
2's Tables 1–5 estimated on another, `ccg + year`. **On the same rows those give a different
*F*, so step 1's only typeset exhibit could not be compared with the application built on top of
it.** This subsection is that gap closed, and the arm is now in `52` rather than in prose.

##### The two arms

| `fe_arm` | county | commuting zone | state |
|---|---|---|---|
| `sxy` | `unit + state × year` | `unit + czstate × year` | `unit + year` |
| `uy` | `unit + year` | `unit + year` | `unit + year` |

**Both are admissible under protocol A.1.4, and the argument is one line**: the treatment varies
at county × start year, and each arm is strictly coarser than that at every rung it is estimated
on. Neither operates at the level where the treatment varies, which is the protocol's
admissibility question 5. `sxy` is the arm this ladder was frozen on and clones the candidate
package's `absorb(unit state_year)`; `uy` is the arm step 2 estimates on. Neither is a headline:
both are reported, as the two winsorization arms are.

**At the state rung the two arms are the same regression.** The unit *is* the state, so
`state × year` is not defined there and `51` has always set that rung to `unit + year`. That
makes the state rung a free falsification check on the arm plumbing, and it is asserted rather
than left to be noticed: check 14 requires all twelve state-rung pairs to agree at **exactly
0.0** on β, on the standard error, on *F* and on the absorbed degrees of freedom. A wrong column
in `52`'s `fe_arm` loop separates them immediately.

##### What the absorption is worth, `h = 4`, IQR5

| Era | Rung | `sxy` | `uy` | df absorbed, `sxy` / `uy` |
|---|---|---|---|---|
| **1979–1997** | County | 395.32 | **492.48** | 191 / 3 |
| | Commuting zone | 111.72 | 174.82 | 191 / 3 |
| | State | 59.79 | 59.79 | 3 / 3 |
| **1998–2018** | County | 254.88 | **294.37** | 239 / 4 |
| | Commuting zone | 205.33 | 217.94 | 239 / 4 |
| | State | 13.53 | 13.53 | 4 / 4 |

The coarser arm is stronger everywhere it differs, which is the ordinary direction — 188 extra
state-by-year effects on four pre-era start years absorb variation the instrument would
otherwise use. **Nothing about the design's conclusions turns on the choice.** Both arms put all
eighteen uncapped cells below 9 and all eighteen capped cells above it; the ladder's shape and
the cap's factor-of-eighty effect are the same under either.

##### The external pin, which is what makes this arm evidence

Step 2's county first stage at pre, `h = 4`, `fixed5`, IQR5-by-start-year is written to
`step_2/result/coeff_candidate_iv_mpgr_pre_h4_fixed5_naics4_alloc.csv` by a pipeline that
**shares no estimator code with `52`** — hand-rolled two-stage least squares in Python against
`reghdfe`-as-demeaner in Stata. On the same 12,320 rows:

| | first-stage *F* | N |
|---|---|---|
| this file, county / pre / `h = 4` / iq5 / `uy` | 492.476800 | 12,320 |
| step 2, `coeff_candidate_iv_mpgr_pre_h4_fixed5_naics4_alloc.csv` | 492.476800 | 12,320 |

Absolute gap **3.98e-12**, relative **8.08e-15** — floating-point identity. This was not
guaranteed: `52` rescales its second-step standard error by `sqrt((N−1)/(N−df_a−2))` because
`regress` on residuals does not know the absorbed degrees of freedom, and step 2's estimator
could have treated them differently. It does not, so the two now agree by construction of the
same statistic rather than by coincidence. `53` reports this comparison and **does not fail the
run on it**, because a df-convention gap would need a decision rather than a tolerance.

##### Why adding the arm did not revise the exhibit

Three things separate an addition from a silent revision, and all three are asserted:

- **The 36 pre-existing `sxy` fits reproduce at exactly 0.0** — β, standard error, *F*, and the
  integer counts — against `code/reference/ladder_first_stage_sxy_v1.csv`, a git-tracked copy of
  the grid as it stood before the arm existed. `52` rewrites the live CSV on every run, so the
  pin has to be against a file this pipeline cannot write; that is `50`'s convention for its own
  frozen baseline. Check 15.
- **All 72 cells are re-fitted independently in pyfixest**, both arms, demeaning the `uy` cells
  on `year` with no reference to what `52` did — a different language, a different estimator, max
  **5.74e-14** relative. Check 13. This is what a verifier co-authored with its builder cannot
  give, and it is the reason the arm is not merely asserted.
- **The county-rung pin to the certified candidate result is fitted on `sxy` only**, because the
  candidate package absorbs `unit state_year`. Reproducing it under `uy` would mean the arms were
  not distinct. Check 9, 12 cells, 3.27e-14 relative.

`e(sample)` is fixed once **per absorption arm** on the unwinsorized shock, so within an arm the
two winsorization arms see identical rows and any difference between them is the cap; a different
absorption is a different singleton problem and gets its own sample. Both arms retain every
unit-year — `N == G × W` is asserted per arm, and the coarser arm cannot drop more than the finer
one.

##### One thing this does not resolve

Step 2 uses `state_year` too, but only to reproduce `reghdfe`'s iterative singleton removal
before estimating, and then absorbs `ccg + year`. So **step 2's sample is step 1's first-stage
sample while its fixed effects are not**, and it imposes an additional A.1.6 county-year
exclusion that this ladder does not. The county first stage above is unaffected — the pin lands
at 8.08e-15 — but a cell-level comparison between the two files is not a like-for-like one, and
step_2's README is where that is documented.

#### 1.8.14 What the exploratory work established, in one table

`APPENDIX.md` §A.2–§A.10 is 2,100 lines of screening, reweighting and winsorizing on an
instrument nothing now uses. Four of its findings are methodological rather than numerical, and
they survive the instrument change because they are about what a screen or a cap *does* rather
than about what this particular first stage was. They are promoted here so the main body carries
the conclusions without the tables.

| finding | where it was measured | why it still holds |
|---|---|---|
| **Capping beats deleting, above the county.** `fixed` × pre at the commuting zone: 31.27 unscreened, **70.96** dropping the 1,363 non-CBSA counties, **104.02** capping them instead — and β moves *with* it, +0.225 (0.132) → +0.348 (0.138), where every other intervention that raised an *F* did it by making a coefficient more precise | §A.9 reading 1 | The mechanism is the aggregation step, not the tail: **capped counties remain inside their commuting zone's ω-weighted regressor and dropped ones do not.** That is a property of any ladder built by ω-weighting, which is A.2.2's requirement and unchanged here. The obvious story — rural counties carry the big `x` values — was checked and is **wrong**: rural county-years are *less* outlier-prone (5.6% outside the `L = 4` band against 14.2% urban) |
| **A percentile cut on a ratio removes the big denominators before it removes the outliers.** `s_c` is contracts over income, so its rank order is "defense per dollar of income", and ranks below about p98 are large defense-industrial metros rather than small base towns. p95 was priced and rejected: it deletes **55.7% of the pre-era defense obligations and 17.4% of the pre-era economy** | §A.8 | The settled shock is still a ratio, `future_ct / income_ct`, and the settled cap is still `method(iq)` on it. Anything later reaching for a percentile or IQR rule on this variable inherits the arithmetic |
| **A screen that ranks on the instrument's own loading cannot be read as evidence about that instrument.** Under an `own`-window screen the pre-era `rolling` county cell goes *F* = 0.30 → **93.16**; under a `fix`-window screen the *same cell* goes 0.30 → **0.07**. Same counties dropped, comparable dollar cost, opposite result | §A.8 reading 3 | `z_ct = s_c × buildup_t`, so with a year effect absorbing `buildup_t` the cross-sectional spread of `z` **is** the spread of `s_c`. Trimming the top of `s` is not a sample restriction with respect to that instrument; it is a transformation of it. The settled instrument has the same product form. It is also why no contract-dollar winsorization panel was ever run |
| **Weighting is out, and the record of what it was worth is kept rather than deleted.** The post-1998 county *F* goes 6.18 → 15.64 under employment weights **with the coefficient staying negative** — precision on the wrong sign — and the commuting zone's negative coefficient becomes more than three times as precise | §A.6, §A.7's opening | The settled ladder carries **no regression weights at any rung**. §A.6's Table VI is the single record that the choice was priced, so it cannot be reopened as though it never was |

Two further results from that block are worth knowing about because they qualify things a reader
might otherwise assume, and both are *negative*:

- **The legacy design's post-1998 sign was not a feature of the data.** Two independent screens
  push it in opposite directions — the county β rises monotonically from −0.087 to +0.130 as
  small counties are removed (§A.7 reading 4) and falls monotonically to −0.973 as high-exposure
  counties are removed (§A.8 reading 2). The conclusion was to quote the era on its *F* and never
  on its sign. The settled instrument's post era is a separate question and §1.8.12 reading 3 is
  where it is answered.
- **Every published post-era number in §A.4–§A.10 contains EFSY's 2017–18 vintage straddle.** The
  design refuses the straddle at the 1998 break and does not refuse it at the 2017 one. §A.6
  reading 7 shows what those two years were worth on one loading: *F* barely moves while **four
  of six β change sign**, and the standard errors *fall* as data is removed. Whether the same is
  true of the other loadings is untested. The settled design's post era ends at 2018 and inherits
  the question.

### 1.9 Past FY2006: the raw USAspending block and source audit

#### The production contract rule, settled

Signed action obligations, both eras. **Deobligations keep their sign** and are never dropped
or absolute-valued — a −\$50,000 deobligation is a large action, not a small one, and dropping
it while keeping the obligation it reverses would inflate the county. The fiscal year of a
dollar is its **action date**, asserted row by row against the archive year (`31` aborts the
parse otherwise). Any dollar floor is a threshold on `abs()` of an **individual action**,
applied before collapse, on **both** sides of the seam; production runs at **no floor**. DCADS
supplies FY1966–2006 and USAspending FY2007–2018, **one source per fiscal year** — `33` asserts
it. Two flow definitions are built from the same transactions: `award` (action fiscal year) and
`duration` (day-weighted across the period of performance).

The estimation universe is the **3,094 CCGs** strictly positive in employment in every year
FY1975–2006, fixed before any period, horizon or rung choice; the **geographic crosswalk stays
complete at 3,101**. The state rung retains the **48** states with at least two assigned
commuting zones (3,080 counties).

#### Conventions, settled 2026-08-17

Dating and scaling rather than data cleaning, so they are stated separately and their
superseded predecessors are named. Bringing `14`, `28` and `../step_2/` into line with them is
downstream work that has not been done.

- **Shock dating is forward.** At horizon *h*, an observation dated *t* sums contracts over
  `[t+1, t+h]`. `14`'s backward `[t−h+1, t]` is the superseded predecessor, and the two must
  not be mixed when a first stage feeds a second stage. At `h = 3, t = 1990` the forward window
  is `{1991, 1992, 1993}` and the backward one `{1988, 1989, 1990}` — disjoint, sharing no
  contract year. This is a dating convention, **not** an identification difference. **What the
  window is divided by is a separate and still-open question** — §4's register, item 2.
- **The era split is pre 1979–1997, post 1998–2018.** 1998 belongs to post and is its first
  start year; fixed share windows end the year before each period begins (`41`'s
  `share_windows()` sets `end = first − 1`), so each period is clear of its own loading by
  exactly one year. `14`'s "1998 sits in BOTH sub-samples by request" and `../step_2/`'s
  `PRE_END` 1997 / `POST_START` 2000 are superseded. The **2002** start in `28` and `14` is not
  a competing convention: it is the arithmetic cost of `h = 4` giving up leading years so every
  horizon reports on one sample.

#### The panel

The production contract panel runs **1975–2018**. FY1975–2006 comes from parsed
DCADS actions and FY2007–2018 comes from parsed USAspending Department of Defense
prime-contract actions. `dc.SEAM_YEAR = 2007`, and the source rule lives inside
`dc.build_contract_panel` so that `13` and `14` cannot disagree about it. `14`
builds its own contract panel and asserts in checks 4 and 6 that it equals `13`'s.

The raw-source archive is reproducible from scripts `29`–`34`. It retains transaction
identifiers, signed obligations, action and performance dates, award type, agency,
place-of-performance geography, and archive-row provenance. The verification report
compares every FY2007–2018 county total with Pallante and records every excluded
geography class in actions, signed dollars, and absolute dollars.

`33` writes **164,353 rows = 3,101 CCGs × 53 fiscal years**. The USAspending
block contains **24,291,068 mapped award actions** and **$3,636.777bn nominal**;
its twelve county-to-Pallante ratios lie between **0.9977 and 1.0000**, with zero
duplicate transaction keys. Exact annual values are in
`1_current_paper/_data/temp/county_panel/contracts_harmonized_report.csv` and
`_data/temp/usaspending_year/usaspending_verification_report.csv`. The fixed positive-
employment analysis sample written by `13` has **136,136 rows = 3,094 CCGs × 44
years**, as recorded in `batch_logs/macro_version/county_panel_report.csv`.

**Nothing before the seam moved.** `13 --last-year 2006` reproduced the frozen 99,232-row
panel with **maximum deviation 0.0 on every column and an identical NaN pattern**, and
`22`'s β₀ still reproduced §1.8.5 exactly at +0.0887, F 80.6.

> **That check was run before the universe was restricted, and its row count says so.**
> 99,232 is `3,101 × 32`; on the 3,094-CCG analysis universe the same command returns
> **99,008**, so the two figures in this section — 99,232 here and `3,094 × 44 = 136,136`
> above — cannot both describe the same build. The claim the check actually established is
> the one that still holds: **the pre-seam DCADS dollars themselves did not move**, only the
> set of counties carried into estimation did. The seam splice is not what changed them.
> Re-run `13 --last-year 2006` to restate the count, and note that `22`'s β₀ is additionally
> stale against `11`'s negative-cell deletion (§4 item 19).

#### 1.9.1 Source boundary and overlap

Official USAspending custom downloads cover FY2001–2006, but their historical
transactions lack usable county geography for 87.747% of absolute award dollars in
FY2001 and 28.546% in FY2006. Their county-to-Pallante ratios are 0.0730 and 0.3448,
respectively; all six annual results are in
`batch_logs/macro_version/usaspending_overlap_verification_report.csv`. DCADS therefore
supplies the production county series through FY2006, and USAspending begins in FY2007.

The Muratori file is a source-comparison benchmark produced by `18`, not an input to
`dc.build_contract_panel`.

His FY1988–2000 block is **3.0×–3.5× the historical record** — $321–401bn a year against
our $106–124bn, FY1988 at 1.84×. Ours matches the record and is verified to the dollar
against Pallante. `18` writes only `year >= SEAM_YEAR` to
`contracts_muratori_ccg_year.parquet` for the FY2007–2018 comparison.

#### 1.9.2 The audit — `18_ingest_muratori.py`

| what | result |
|---|---|
| fiscal alignment, **first differences** | lag 0 **0.3101**, lag −1 0.0351, lag +1 −0.1533. In levels the three are 0.798/0.827/0.819 — a level panel is dominated by county size and cannot tell them apart, which is why this is run differenced |
| `dollarsobligated` is nominal | ratio-to-ours uncorrelated with CPI (−0.16) and with the deflator (+0.12), so `dc.deflate` is right |
| geography | 260 rows / \$41,106,000 `SS999`, 34 rows / **\$1,807,260,620** bare 2-char state codes, 1 row / \$12,000 `47098` (declared, not a valid FIPS). 0.016% of national, gated on materiality not equality so a re-share does not fail on rounding |
| **the imported block, against Pallante FY2007–2018** | ratio **0.879–1.054, spread 0.174** — *tighter* than the FY2001–06 years we already accept (0.931–1.188, spread 0.257). Our own DCADS against the same benchmark is 1.000 in every year |
| λ_c, the county-specific step | corr(log λ_c, exposure percentile) **+0.0505, t = +2.24** on 1,963 CCGs; log ratio p10 −0.031, median +0.011, p90 +0.119 |
| contractor-HQ tilt | HQ mean **+0.103** against non-HQ **+0.037** — Arlington +0.086, Fairfax +0.082, DC +0.059, LA +0.097, St Louis county +0.195 |

The Pallante gate is **self-calibrating**: two independent constructions never give a ratio
of 1, so a hand-set band means nothing. It requires the imported years to agree with the
benchmark as well as the years our own panel independently confirms.

**Foreign work is the open question.** Our panel is US-county place-of-performance only;
the block `geo_class == "county"` excludes is **\$124.4bn over FY2001–06, 10.5%** of what we
keep. His file carries no non-US code at all, so he either dropped foreign work as we do or
redistributed it into US counties, and three columns cannot say which. The HQ tilt above is
a small positive signal for partial redistribution. Two things make this worse than it
looks: the 10.5% is measured on data ending in 2006, and the Iraq/Afghanistan increment is
heavily in-theatre, so the foreign share very likely rises over exactly the years added.
**Unresolved. Ask the coauthors for a place-of-performance-country flag and an FMS flag.**

#### 1.9.3 The seam, and which years it costs

ΔG at *t* sums D over `{t-h+1..t}`, so it mixes sources only where that window straddles
FY2007 — exactly `SEAM_YEAR .. SEAM_YEAR+h−2`. Those years are **masked, not deleted**:
`dG` and `shock` go NaN, the row stays, and the spine stays rectangular because checks 1,
1b and 1c assert it and Part 3's arithmetic rests on it.

| h | window | masked | rows |
|---|---|---|---|
| 1 | `{t}` | none | 0 |
| 2 | `{t−1, t}` | FY2007 | 3,101 |
| 3 | `{t−2, t−1, t}` | FY2007–08 | 6,202 |

No *other* year is touched: checks 4b and 4c prove the window sum does not depend on where
the cumulation starts. `14` applies the same mask to the national series too, since
`buildup` is what every `z` is built from.

**`n_actions` is 0-filled on both sides of the seam**, and the masked-row counts in the
table above are the pre-restriction ones — `3,101` and `6,202` read `3,094` and `6,188` on
the analysis universe. Both statements changed when FY2007+ became transactions rather than
a pre-aggregated county-year block. Until then `n_actions` was `<NA>` from FY2007 because
the source carried no action count and a zero there would have been summed as one; both raw
sources now carry individual actions, so an absent county-year means a genuine zero action
count on either side (`dcads_common.build_contract_panel`). The **dollar floor** followed the
same path: it is a threshold on `abs()` of one action, it could not be applied to a sum, and
it now applies action by action on both sides. Production still runs at no floor.

Check 4c's tolerance became **relative**. Pandas slides a rolling accumulator rather than
re-adding each window, so its rounding depends on how many earlier elements it saw; the
probe sees five fewer years. On 1975–2006 the two agreed exactly, and adding FY2007–18
raises the largest window to ~\$55bn where one float64 ulp — 1.9e-06 dollars, 3e-17
relative, on 7,332 of 117,838 rows — becomes visible. Check 2 also **used to fail open**:
post-seam years had no source file, the loop skipped them, and a 44-year panel would have
reported PASS on 32 years. New check 2a asserts every year has a reference.

#### 1.9.4 The post-1998 exposure and the window it forces — moved

**Superseded by §1.9's conventions block above, and swept to `APPENDIX.md` §A.11.** It documented
`fixed_9398` = 1993–98 and the 2002–2018 window that exposure forces; the settled era split is pre
1979–1997 / post 1998–2018 with fixed-share windows ending the year before each period begins. The
number worth carrying forward is in §A.11: the same predeterminedness guard found a **pre-existing
leading edge in the frozen design** — on 1975–2006 with `fixed`, `h = 1/2/3` contaminate 1975 /
1975–76 / 1975–77 — which was reported as a WARNING and **not changed**, because the frozen numbers
are what §A.4 published.

The numbering skips rather than closing up, so that the four existing citations of `§1.9.4` still
find this heading and are told where it went. §0.6 is the crosswalk.

#### 1.9.5 What the extension bought — the honest answer

**The national buildup barely widens.** At h=2, `ΔG^nat/Y^nat`:

| window | min | max | range | sd |
|---|---|---|---|---|
| 1975–1998 | 0.0314 | 0.0837 | **0.0523** | 0.0162 |
| 1998–2006 (F = 1.8) | 0.0302 | 0.0470 | 0.0168 | 0.0065 |
| 1999–2018 | 0.0302 | 0.0524 | 0.0222 | 0.0076 |

The Iraq/Afghanistan hump is real — 0.030 (2000) → 0.052 (2008–09) → 0.035 (2014–16) →
0.039 — so it is not the flat line 1998–2006 was. But it adds only **17% to the sd and 32%
to the range**, and post-1998 still has under half the pre-1998 variation.

**And the first stage does not recover.** h=2, `fixed_9398`, unit and year FE, employment
weights, 2002–2018:

| rung | units | obs | first-stage F | 2SLS β | (se) |
|---|---|---|---|---|---|
| county | 3,100 | 49,577 | **6.3** | **−0.1677** | 0.0880 |
| commuting zone | 739 | 11,824 | 7.9 | −0.6916 | 0.4880 |
| state | 51 | 816 | 48.1 | −0.2687 | 0.2838 |
| Census division | 9 | 144 | 2.9 | +1.5919 | 2.5577 |
| county × NAICS2 | 3,100 | 1,276,703 | 6.3 | — | — |
| county × NAICS4 | 3,100 | 11,921,023 | 6.3 | — | — |

County F = 6.3 against 72.5 pre-1998. The sign flips negative at three rungs and the ladder
does not behave — county −0.17 → CZ −0.69 → state −0.27 → division +1.59. **This is a
usable panel and an unusable post-1998 first stage**, which is what §4 item 6 predicted and
what the buildup range above already implied. Named here rather than dropped, per plan §1.5
check 5. The FY2000 DoD reporting break also sits just before this window, and the
post-2007 identifying variation is close to collinear with the Great Recession and recovery
— neither is addressed.

### 1.10 Recipient identity: who the contract went to — `pipeline/09`

The parsed transaction store carries **no recipient identifier at all**.
`31_parse_usaspending.py` selects 54 of the archive's 297 columns, and not one
of them is a DUNS, a UEI, a CAGE code, a name, or a recipient address. WS3's
contract-to-NETS establishment match therefore has nothing to join on, which is
why WS3 lists this re-parse as its step 0.

`pipeline/09_usaspending_recipient.py` re-reads the same archives in the same
row order and writes 22 identity columns to a **sidecar** — never into the
locked store, which sits upstream of stage 5's tie-out and stage 23's industry
axis. The join key is producer 31's own
`(source_fiscal_year, source_member, source_row_number)`, recomputed by the same
rule, with `contract_transaction_unique_key` carried so the join can be checked
independently of row order. Every run asserts row-for-row agreement with
`usaspending_year/fy<year>.parquet` and fails loud otherwise.

    V=4_Macro_version/.venv/bin/python
    $V step_1/code/pipeline/09_usaspending_recipient.py --years 2001-2018 --include-overlap \
        > batch_logs/macro_version/09_usaspending_recipient.log 2>&1

Roughly 25 minutes for the 18 years, 1.6 GB.

| Output | Contents |
|---|---|
| `4_Macro_version/data/temp/usaspending_recipient/fy<year>.parquet` | LOCAL, gitignored; 4 join keys + 22 identity columns. See the README in that directory |
| `batch_logs/macro_version/usaspending_recipient_fill_report.csv` | fiscal year × column: rows, non-blank count and share, sentinel count, distinct values, absolute dollars covered and share |
| `batch_logs/macro_version/usaspending_identifier_coverage.csv` | the same, for the five join-candidate identifiers |
| `batch_logs/macro_version/usaspending_identifier_coverage_summary.csv` | per identifier: first and last covered year, contiguity, min/max row share |
| `batch_logs/macro_version/09_usaspending_recipient.log` | the run, with both coverage matrices printed |

**The time coverage of the DUNS code, FY2001-2018** — the question WS3 step 0
was asked to settle. Run 2026-08-31, contiguous, no gap years:

| Identifier | Rows non-blank | \|Dollars\| non-blank | Span |
|---|---|---|---|
| `recipient_uei` | 1.000 every year | 1.000 every year | FY2001-2018, 18/18 |
| `recipient_duns` | 0.921-0.959 | 0.712-0.958 | FY2001-2018, 18/18 |
| `recipient_parent_uei` | 0.999-1.000 | 0.999-1.000 | FY2001-2018, 18/18 |
| `recipient_parent_duns` | 0.921-0.959 | 0.712-0.958 | FY2001-2018, 18/18 |
| `cage_code` | 0.014-0.821 | 0.006-0.888 | FY2001-2018, but 0.014-0.026 before FY2017 |

Three things follow, and they are the input to the step 1 match memo rather than
settled decisions:

1. **DUNS did not disappear, and UEI is not empty.** These are 2026-vintage
   re-publications, and USAspending has back-filled UEI onto pre-2022
   transactions: FY2001 rows carry a real UEI. It is not a placeholder — zero
   sentinel values, and 37,353 distinct UEIs in FY2001 against 36,418 distinct
   DUNS.
2. **UEI strictly dominates DUNS as a USAspending-side key**, in every year, on
   both rows and dollars, and on distinct-entity count. That does **not** make
   it the join key: NETS `estab_xwalk` is keyed on DUNS, so the choice is
   between joining on the sparser field that matches the target directly and
   joining on the denser field that then needs a UEI→DUNS bridge. The memo
   decides that; this table only says what each side costs.
3. **The DUNS dollar share decays late** — 0.930 in FY2007 to 0.712 in FY2018,
   while the row share barely moves. The transactions missing a DUNS are
   disproportionately the large ones, so a DUNS-only match loses more of the
   contract dollars than of the contract count, exactly in the years the
   post-2007 window leans on. `cage_code` moves the other way, near-useless
   before FY2017 and then 0.821 of rows in FY2018; it is a late-period
   supplement at best, not a spine.

No fill rate is asserted anywhere. The plan asked us to *check* whether DUNS is
still populated, not to require it, and a thin identifier is a finding that sets
the era tiers rather than a failure.

---

## 2. What each script does

```
01 manifest → 02 download → 03 verify download → 04 parse fixed-width
           → 05 parse delimited → 06 verify parse
           → 07 add county FIPS → 08 verify geography → 06 again
           → 09 the Pallante test
           → 10 build the county universe → 08 and 06 again
                                          → 11 build the employment cells   (Part 1.3)
           18 ingest the post-FY2006 block (§1.9)
                                          → 12 build county income          (Part 1.4)
                                          → 13 build the county panel       (Part 1.4)
                                             → 14 build the instrument      (SUPERSEDED, APPENDIX)
                                                → 15 the NS14 comparison    (stale, APPENDIX §A.3)
                                                → 16 the floor comparison   (stale, APPENDIX §A.3)
                                                → 17 AR sets and OLS        (stale, APPENDIX §A.3)

           29 → 30 → 31 → 32  acquire, parse and verify raw USAspending  (§1.9)
                        → 33 harmonize the two contract sources          (§1.9)
                        → 34 the FY2001-06 overlap download              (§1.9.1)

  THE LIVE PIPELINE, and everything the settled design reads:

           33 + 12 + 13 → 40 the pre-era nonoverlap probe
                        → 41 BUILD THE PANEL OF RECORD  ---------------- (§2.1)
                             → 42 county first stages, Stata
                             → 43 verify → 45 export
                             → 44 ORCHESTRATOR for 41-43, 45
                        → 46 → 47 → 48 → 49 the post-anchor sweep
                             → 50 ORCHESTRATOR for 46-49, pins 48 to the frozen v1 grid
                        → 51 build the ladder panel, pinned to the candidate .dta at 0.0
                             → 52 the ladder first stages, Stata, 72 fits  (§1.8.12, §1.8.13)
                             → 53 verify, 16 checks → 54 export TableF1
                             → 55 ORCHESTRATOR for 51-54

  41's panel.parquet is re-scaled by `Candidate for first stage/code/01_build_candidate_panel.py`
  into the {pre,post}_h{2,3,4}.dta that BOTH 51 and step_2's 29-44 read. That is the join
  between the two steps; nothing else crosses.

           19, 20, 21, 23, 24, 25, 26, 28, 35, 38, 39  the exploratory grids  (APPENDIX §A.4-A.10)
```

**The live pipeline is `33`/`12`/`13` → `41` → the candidate builder → `51`–`55` and step 2.**
`14` through `28` and `35`–`39` are the superseded instrument and the exercises around it;
`APPENDIX.md` documents them and §1.8.14 promotes what they established. Nothing in `40`–`55`
reads `14`'s `instrument_ccg_year.parquet`.

**`18` runs before `12`**, not after `13`. Since the splice, `12`'s declared-gap check and
`13`'s and `14`'s contract panels all call `dc.build_contract_panel` for years past FY2006,
and that function raises if `contracts_muratori_ccg_year.parquet` is not there yet. `18`
itself reads `contracts_ccg_year.parquet` for its FY2001–06 overlap audit, so a cold build
runs `13` once at `--last-year 2006` first; on a warm tree the existing file serves.

**`15`, `16` and `17` now take an explicit window.** They read
`instrument_ccg_year.parquet` whole, which used to mean 1975–2006 by accident and would
have silently become 1975–2018. `15` and `17` gained `--period`, defaulting to the frozen
`1975-2006`; `16` is pinned to FY2006 in its own file, because a dollar floor is a threshold
on an individual action and the post-seam block arrives aggregated to the county-year, so
there is nothing there to floor.

`12` and `13` are documented in §1.7 above and `14` in §1.8. **`15`, `16` and `17` are not
documented any more**: they had three sections of their own, all measuring the flow ΔG, and those
went with it when §1.8.4 became the trial record — the §1.8.5 that exists now is the new first
stage and has nothing to do with them. The scripts still run, and they now default to the
cumulative definition, so rerunning any of them measures the current design and needs a fresh
write-up rather than the old one. `11` **is** documented, as of 2026-08-10 — §1.6.10, rewritten
from the code and `batch_logs/macro_version/11_build_employment_cells.log` after its original
write-up was lost with a deleted Part 2 reference.

The scripts are independent and rerunnable; each reports what it did and exits non-zero
on failure. Output goes to `1_current_paper/_data/temp/dcads_year/` and logs to
`batch_logs/macro_version/`. Output must stay under `_data/`, which is gitignored —
`4_Macro_version/step_1/data/` is not, and multi-GB parquet must never be stageable.

06 runs twice on purpose. It is the parse gate before 07 and, unmodified, the proof
afterwards that adding four columns left the dollars, the row counts and the
county-sums-to-state invariant exactly as they were.

```bash
python3 4_Macro_version/step_1/code/py/07_add_county_fips.py      # ~35 s, rewrites 42 files
python3 4_Macro_version/step_1/code/py/08_verify_county_fips.py   # ~1 min, exit 0 = all clear
python3 4_Macro_version/step_1/code/py/06_verify_dcads_parse.py   # ~10 min, exit 0 = all clear
python3 4_Macro_version/step_1/code/py/09_pallante_test.py        # ~5 min, exit 0 = all clear
python3 4_Macro_version/step_1/code/py/10_build_county_universe.py  # ~4 min, exit 0 = all clear

V=4_Macro_version/.venv/bin/python
$V 4_Macro_version/step_1/code/py/11_build_employment_cells.py    # ~1 min, exit 0 = all clear
$V 4_Macro_version/step_1/code/py/18_ingest_muratori.py           # ~1 min, the post-FY2006 block
$V 4_Macro_version/step_1/code/py/12_build_county_income.py       # ~2 min, CACHED ZIP - do NOT
                                                                  #   pass --refresh: BEA revises
                                                                  #   county income annually and a
                                                                  #   new vintage moves pre-2006
                                                                  #   income under the frozen
                                                                  #   F = 72.5 / beta = +0.076
$V 4_Macro_version/step_1/code/py/13_build_county_panel.py        # ~4 min, exit 0 = all clear
$V 4_Macro_version/step_1/code/py/14_build_instrument.py          # ~2 min, exit 0 = all clear
$V 4_Macro_version/step_1/code/py/15_ns14_f_comparison.py         # ~3 min, NOT rerun since the
$V 4_Macro_version/step_1/code/py/16_floor_comparison.py           # ~6 min,   delta change, and
$V 4_Macro_version/step_1/code/py/17_ar_ols_ladder.py              # ~2 min,   undocumented
```

`14`'s build takes seconds because everything expensive was done upstream; the time is the sweep,
486 regressions across three horizons, three periods, two exposures and three FE settings.
`--skip-sweep` leaves only the build. `--exposure`, `--horizon`, `--periods`, `--delta`,
`--no-loo`, `--roll-k` and `--roll-min` are the switches behind the choices §1.8.1 and §1.8.4
record.

**`--floor` and `--delta` default to `0` and `cumulative` in both `13` and `14`, and they must
always agree.** The two read the same `build_contract_panel`, and §1.8's leave-one-out identity
requires the shock and the national aggregate to be built from the same retained rows and the
same operator; running one floored and the other not would break check 5 rather than fail loudly
at the boundary. `14`'s check 3 catches a `--delta` mismatch and says which flag to rerun with.
To rebuild the old floored panel or the old flow ΔG, pass the flag to **both**, in order. Note
that every `14` variant writes to the same fixed paths as the default run, so it overwrites the
production parquet and `instrument_first_stage.csv` — back them up first, as §1.8.4's flow table
did.

`15` reads `14`'s parquet and writes no data. It imports `14` by file path rather than restating
the rung map and the collapse, so the two cannot drift into describing different ladders; its
exit code is the rank-one Cragg–Donald check against pyfixest, not a weak-instrument verdict.
`--skip-baseline` drops the slow half.

`12` is the only step in the whole pipeline that reaches the internet after the DCADS
download. It caches `CAINC1.zip` under `_data/source/BEA/CAINC1/` with a sha256 receipt and
reuses it; `--refresh` re-downloads.

`10` reads the parquet and writes only reference files, never data, so `08` and `06` stay
valid afterwards by construction — and are rerun anyway. Two robustness switches,
`--broomfield-merge-all` and `--census-only-va` (§1.6.2, §1.6.3), and `--skip-employment`,
which drops check 6 and the employment half of check 7 — the only part that reads the
227 MB EFSY panel.

| script | what it does | writes |
|---|---|---|
| `code/r/00_efsy_granularity_check.R` | separate, for Part 1.3: how much EFSY employment resolves to four NAICS digits rather than sitting on a root. **Superseded, and it undercounts.** It finds roots by looking for `/`, which misses the dash-padded two-digit roots entirely — 13.0m employee-years read as fully resolved six-digit codes. `11_build_employment_cells.py` counts leading digits instead and writes the same table; keep this one only as the record of how the question was first asked | log only |
| `code/py/01_build_dcads_manifest.py` | queries the NARA catalog for both series and writes one row per digital object, tagging documentation apart from data | `manifest.csv` |
| `code/py/02_download_dcads.py` | downloads every object. Resumable, aborts on a stalled connection and retries, and orders the queue so an interrupted run leaves whole parseable fiscal years rather than data files with no layout | `download_receipt.csv`, with a sha256 per file |
| `code/py/03_verify_dcads.py` | verifies the download **by content, not byte count** — see §3.1 | `verification_report.csv` |
| `code/py/04_parse_dcads.py` | parses the 39 fixed-width years into one parquet each, streaming chunk by chunk because the later years reach 303 MB uncompressed. Extracts zipped members to scratch and deletes them again | `fy<year>.parquet` |
| `code/py/05_parse_dcads_delimited.py` | parses FY2004–06, addressing fields by byte position and remapping the four declared damage signatures | the same per-year parquet |
| `code/py/06_verify_dcads_parse.py` | nine checks over the parsed output. Must pass before anything downstream runs | `parse_verification_report.csv` |
| `code/py/07_add_county_fips.py` | reads ZALF, the supplement and `NATIVE_OVERRIDES`, writes `county_fips5`, `state_fips2`, `geo_class` and `fips_source` into all 42 files. Each file goes to `.tmp` and is renamed, so an interrupted run leaves every file wholly old or wholly new; rerunning rebuilds the four columns rather than duplicating them | the same per-year parquet |
| `code/py/08_verify_county_fips.py` | four geography criteria and two reports. Deliberately checks nothing 06 already does | `county_fips_verification_report.csv` |
| `code/py/09_pallante_test.py` | the external benchmark: our state-year dollars against Pallante's published series, a diagnosis of anything that deviates, and the same comparison with the §1.5 sign byte repaired on his side. Re-reads the raw source alongside the parse, and fails if the two are not row-aligned. Supersedes 06's informational checks 7 and 8 | `pallante_test_report.csv`, `pallante_test_diagnosis.csv`, `pallante_test_corrected.csv` |
| `code/py/10_build_county_universe.py` | Part 1.4's county harmonisation: reads the Census change list, the Virginia BEA combinations, `SUPERSEDED` and all 42 parquet files, builds the time-invariant `county_fips5 → ccg_id` map, and runs eleven checks over it. Writes reference files only, never data — see §1.6 | `code/reference/county_ccg_crosswalk.csv`, `county_universe_report.csv` |
| `code/py/11_build_employment_cells.py` | Part 1.3: builds the county-by-industry employment cells, allocates the NAICS roots down to four digits, deletes EFSY's negative cells at four digits (`--negative-cells`, default `drop` since 2026-08-17; `keep` restores the conserving build), and rectangularises the panel on the full CCG universe. The two-digit and county rungs are summed from the cleaned four-digit cells. Moved here from the Part 2 folder on 2026-08-07 because it closes a Part 1 item and Part 1.4 runs after it. Written up in §1.6.10 | `_data/temp/employment/`, `efsy_granularity_by_year.csv` |
| `code/py/12_build_county_income.py` | Part 1.4: downloads BEA CAINC1, maps its 3,149 areas onto the 3,101 CCGs through four declared reconciliations, and checks the result against BEA's own state totals — see §1.7.1. Since 2026-08-12 it also reads **LineCode 2, population**, through the same four reconciliations and the same state-total check, for §1.8.8's rule (b). That goes to its own parquet: `income_ccg_year.parquet` is read by `13`, `14`, `19` and `20` and transitively pins step_2's `beta_0`, so it is not rewritten to gain a column none of them reads, and it is byte-identical after the change | `_data/source/BEA/CAINC1/`, `_data/temp/county_panel/income_ccg_year.parquet`, `population_ccg_year.parquet`, `county_income_report.csv` |
| `code/py/13_build_county_panel.py` | Part 1.4: collapses contracts to CCG × year under a declared dollar floor, joins income and employment on one spine, builds the shock, and runs the ten checks of §1.7.3 | `_data/temp/county_panel/contracts_ccg_year.parquet`, `county_panel.parquet`, `county_panel_report.csv` |
| `code/py/14_build_instrument.py` | Part 1.5: builds the two pre-period exposures, the cumulative-ΔG leave-one-out national buildup and the instrument at three horizons, runs ten checks, and sweeps the clustered first-stage F over four aggregation rungs, two cell rungs, two exposures, three horizons, three estimation windows and three fixed-effect settings — the plan's check 5. See §1.8 | `_data/temp/county_panel/instrument_ccg_year.parquet`, `instrument_report.csv`, `instrument_first_stage.csv` |
| `code/py/15_ns14_f_comparison.py` | Part 1.5: recomputes the first stage as **Cragg–Donald**, homoskedastic and unclustered, so it can be laid beside Nakamura and Steinsson's own footnote 30; and runs their baseline instrument set on our panel as the test of the mapping. Reads `14`'s parquet, writes no data. **Not rerun since the ΔG change; undocumented** (§1.8.4) | `ns14_f_comparison.csv`, `ns14_baseline_analogue.csv` |
| `code/py/16_floor_comparison.py` | Part 1.5: rebuilds the instrument at both the \$25,000 floor and no floor, in memory, and reports the clustered *F*, the Cragg–Donald *F* and the 2SLS coefficient side by side for every rung, exposure and horizon. Exists as a separate file because `14 --floor 25000` would overwrite the production panel. **Not rerun since the ΔG change; undocumented** (§1.8.4) | `floor_comparison.csv` |
| `code/py/17_ar_ols_ladder.py` | Part 1.5: the Anderson–Rubin confidence set at every rung, solved as a quadratic rather than on a grid, plus OLS beside 2SLS so that a rising coefficient ladder can be separated from a falling weak-instrument bias. Exits non-zero unless the AR scale reproduces the reported first-stage *F* and a bounded set is equivalent to *F* > 3.84. **Not rerun since the ΔG change; undocumented** (§1.8.4) | `ar_ols_ladder.csv` |
| `code/py/18_ingest_muratori.py` | Imports the shared Muratori, Juarros and Valderrama county intermediate onto the CCG geography for an external FY2007–2018 source comparison | `contracts_muratori_ccg_year.parquet`, `muratori_ingest_report.csv` |
| `code/py/19_first_stage_table.py` | Part 1.5 (§1.8.6): the clustered first stage on the **symmetric two-era design** — five-year exposure windows 1970–74 and 1993–97, one per era, with the eras taken from step_2's `periods_for(h)` so the split follows the employment data rather than the contracts. Imports `14` for the build and the estimator and **writes nothing any other script reads** (§4 item 9's hazard). Registers its windows into `14`'s dicts so `contaminated_years` is reused verbatim, and asserts the registration only added. Exits non-zero unless all nine checks pass, among them: the rebuilt `shock` and `z_rolling` equal the production parquet, and `rolling` × the frozen 1975–1998 window reproduces `instrument_first_stage.csv` at all twelve cells — the anchor that pins a table in which no reported cell has a published counterpart | `first_stage_table.csv` |
| `code/py/20_first_stage_grid.py` | Part 1.5 (§1.8.7): the same first stage across **eight era × exposure blocks × three rungs, OLS and WLS on lagged employment** at `h = 2` — the original `fixed` 1969–74 on the pre era, **on 2000–2018 and on its 2016 truncation**, `fixed_9397` 1993–97 on the post eras, `rolling_t5` on all three, and a third era stopping at `dc.SEAM_YEAR − 1` that isolates the splice's contribution. Imports `14` for the build and the estimator and `19` for the eras, the 1993–97 window and the logging, so neither is restated; **writes nothing any other script reads**. Pins `pf.MapDemeaner(fixef_tol=1e-12, ...)` through `14`'s `feols_kw` passthrough, which is what lets check 10 assert an **exact** refit rather than a tolerance — within one process; across processes WLS cells wobble at 2 × 10⁻¹² relative (§1.8.7). Loads `11` for one constant, `VINTAGE_BREAKS`, so the 2016 bound is `VINTAGE_BREAKS[-1] − 1` and is never typed. Exits non-zero unless all eleven checks pass, among them three anchors — the production parquet, `instrument_first_stage.csv`, and the nine cells §1.8.6 Table V already publishes — and check 1b, that the 2017 straddle is in the post era and absent from the truncation. Check 6 asserts all 48 *F* values against the table it was asked to produce | `first_stage_grid.csv` |
| `code/py/21_ingest_cbsa.py` | Part 1.5 (§1.8.8): downloads the **June 2003 OMB CBSA delineation** and parses its fixed columns onto the CCG universe, so that "purely rural" has a definition. Asserts the vintage by its own counts — 1,838 county FIPS, 1,159 of them metropolitan, 370 + 565 areas — and declares the 73 Puerto Rico FIPS the crosswalk does not contain rather than dropping them silently. Falls back from `urllib` to `curl` on a certificate failure and records which transport was used | `_data/source/CBSA/03mfips.txt`, `download_receipt.csv`, `code/reference/county_cbsa_2003.csv` |
| `code/py/23_sample_screen_grid.py` | Part 1.5 (§1.8.8): Table VI's grid **less the 1999–2006 era and less the weighted column** — six exposure × era pairs, both `fixed` 1969–74 post blocks included — run under five restrictions of the county set — the smallest 500 and 1,000 counties by 1969 population, the two 2003 CBSA screens, and the zero-contract screen — at `h = 2, 3` and **4**. OLS throughout. Takes `14` and `19` **from `20`** rather than loading them again, because `importlib` builds a fresh module per call and a private copy of `14` would be registered into by nobody. Exits non-zero unless all eleven checks pass, among them that the unscreened `h = 2` grid *is* Table VI's unweighted column on all six pairs, that the `h = 3` cells reproduce Table V, and that each screen is a subset of exactly the declared size | `sample_screen_grid.csv` |
| `code/py/24_exposure_screen_grid.py` | Part 1.5 (§1.8.9): the same grid as `23`, at `h = 1, 2, 3` and screened on the **exposure's own right tail** — the top 0.5% and top 1% of `s_c`, measured over the window that design row's exposure uses. Two families: `own` ranks on the row's own exposure, `fix` ranks on the era's fixed window so the screened sample matches the paired fixed-exposure cell, which is the only way to tell a sample effect from an instrument effect. The `fix` family runs on the `rolling` cells only, because on the fixed exposures it *is* `own` — asserted at set level (check 9) rather than fitted twice. Takes `20`, `19` and `14` **from `23`**, for the module-identity reason `23` records. Reports what each screen drops in three currencies, defense dollars, income and employment, because a screen on a ratio can be cheap in counties and ruinous in treatment. Exits non-zero unless all twelve checks pass, among them that the unscreened `h = 1` grid *is* Table VI's unweighted column and that the unscreened cells reproduce Table V at **all three** horizons | `exposure_screen_grid.csv` |
| `code/py/25_ingest_bea_employment.py` | Part 1.5 (§1.8.10): downloads BEA **`CAEMP25S`**, county total employment on the SIC basis, **1969–2000**, and puts it on the CCG spine. Exists because §1.8.10's employment flag has to be measured on the era's exposure window and EFSY starts in 1975, so it cannot supply a 1969–74 growth rate. **Four BEA tables were read before this one was chosen** and the rejected three are named in the docstring — `CAINC4` and `CAINC30` carry no employment at all despite their titles, and `CAEMP25N` is 2001–2022 and covers neither flag window. Both windows sit inside `CAEMP25S`, so no SIC-to-NAICS splice enters the flag. The line code is resolved **by its `Description` text** rather than hardcoded, and the run fails if the match is not unique; the reader, the four county reconciliations and the state-rollup check all come from `12` rather than being restated. Seven checks, and the rollup reproduces at **max relative deviation 0.00e+00**, the same figure it gave on income | `_data/source/BEA/CAEMP25S/`, `bea_employment_ccg_year.parquet` |
| `code/py/26_winsor_screen_grid.py` | Part 1.5 (§1.8.10): the same grid as `23` **less the two `fixed` 1969–74 post blocks, pinned by era on 2026-08-13 so that widening `23` did not widen Tables XI–XIII and XVII–XIX uninvited**, at `h = 2, 3, 4`, **capping** the flagged counties instead of dropping them, and reporting both treatments of one flag set side by side. `winsor4 … method(iq)`'s rule is applied twice with one `L`: the characteristic's distribution picks the counties, and their `x` and `z` are capped at `p50 ± L·IQR` of `x` and of `z`. Three flags — population, CBSA rurality, BEA employment growth — each measured on the era's own exposure window, which is asserted against `20`'s dict rather than typed. **`y` is never capped** (A.3.2) and is asserted identical to the baseline. Sixteen checks, among them that the `all` column *is* §1.8.8's unscreened column, that the rural `drop` columns *are* Table VIII's CBSA screens, and that the cap is confined to the flagged rows | `winsor_screen_grid.csv` |
| `code/py/27_build_cz_state_assignment.py` | Assigns each commuting zone to **one** state, so that `state × year` fixed effects and state clustering are defined at the CZ rung. The rule is the state holding the largest share of the CZ's **1969** population, ties to the lowest state FIPS — population rather than contracts, so the assignment cannot be a function of the instrumented variable, and 1969 because it is the first BEA year and so precedes every estimation sample. Carries a 1970–74 average-population sensitivity, which differs for **1 of 739** CZs and is therefore near-vacuous. Audits 136 cross-state CZs, 256 CCGs assigned outside their actual state, and the **3 CZs whose plurality state holds under half the population** (11304 → MD 0.441, 17100 → WV 0.442, 17400 → PA 0.392) — counted, with no handling rule. **Design choices open**, §4's register items 9–10 | `code/reference/cz_state_1969_population.csv` |
| `code/py/28_post_first_stage_alignment_grid.py` | First-stage alignment grid at the county and CZ rungs with `unit + state-by-year` fixed effects, counties on their actual state and CZs on `27`'s assignment. Main design: pre `fixed_7074` 1979–1997, post `fixed_9397` **2002–2018**, `h = 2, 3, 4`, unweighted and lagged-employment weights, clustered by unit and by state — 24 cells. Diagnostics at `h = 2`: post samples ending 2006 or 2016, the 1970–74 assignment, exclusion of all cross-state CZs, and leave-one-state-out. Introduces `MIN_CZS_PER_STATE = 2`, which drops DC, CT and RI and is a bare constant with no stated justification. Purges singleton fixed effects to a fixed point so check 10 can assert the estimator dropped nothing. **Not the settled design** — its 2002 post start and its cluster axis both differ from `41` | `post_first_stage_alignment_grid.csv` |
| `code/py/29_build_usaspending_manifest.py` | Queries the official annual archive index for Department of Defense prime contracts and pins FY2007–2018 archive metadata | `_data/source/USAspending/manifest.csv`, `usaspending_manifest.csv` |
| `code/py/30_download_usaspending.py` | Range-downloads the twelve annual archives, verifies each zip, and records bytes and sha256 | `_data/source/USAspending/download_receipt.csv`, `usaspending_download_receipt.csv` |
| `code/py/31_parse_usaspending.py` | Streams the raw CSV members to annual transaction parquets, retains audit fields and signed obligations, and resolves eligible domestic place-of-performance counties onto CCGs | `_data/temp/usaspending_year/fy<year>.parquet`, `usaspending_parse_report.csv` |
| `code/py/32_verify_usaspending.py` | Verifies transaction identity, fiscal timing, agency and award type, county mapping, excluded geography shares, and annual county totals against Pallante | `usaspending_verification_report.csv`, `usaspending_overlap_verification_report.csv` |
| `code/py/33_harmonize_contract_sources.py` | Builds separate raw-derived DCADS and USAspending county-year blocks plus the rectangular FY1966–2018 union | `contracts_dcads_ccg_year.parquet`, `contracts_usaspending_ccg_year.parquet`, `contracts_harmonized_ccg_year.parquet`, `contracts_harmonized_report.csv` |
| `code/py/34_download_usaspending_overlap.py` | Uses official custom annual downloads to acquire FY2001–2006 for the county-geography overlap diagnostic | `_data/source/USAspending/overlap/`, `usaspending_overlap_download_receipt.json` |
| `code/py/35_build_post_first_stage_full_grid.py` | Builds the grid-ready county and CZ panels for the post-1998 design space, from `33`'s harmonized transactions. Axes: flow (`award`, `duration`) × window (`overlapping`, `nonoverlap`) × obligation treatment (`net`/`positive` × \$0/\$25,000 floor) × `h = 2, 3, 4` × rung — **96 Stata input files**. Twelve allocation shares `s01`–`s12` (3 fixed × 4/5/10 years, 9 rolling × lag 3/4/5 × length 4/5/10). At horizon *h* the local shock is contracts `t+1..t+h` over **real personal income at `t`**, and the instrument is the share times the leave-one-unit-out national sum over **the same local denominator** — §4's register item 2. Obligation treatments are applied at the **transaction** level and zeroed in place, not dropped, so the duration spread still sees the row. Every duration-branch date failure is audited rather than silently excluded | `_data/temp/post_first_stage_full_grid/*.dta` |
| `code/stata/36_post_first_stage_full_grid.do` | Estimates the grid: **11,520 regressions** = 96 inputs × 12 shares × 10 winsorization cells (`none`/`iq4`/`iq5`/`tail1` × `pooled`/`year`/`state`). Not a canned IV command — a hand-rolled Frisch–Waugh–Lovell two-step that fixes `e(sample)` once on the **unwinsorized** outcome, residualizes each variant on `unit + state_year`, then regresses with `vce(cluster unit)` and a manual `sqrt((N−1)/(N−df_a−2))` correction because the second-step `regress` does not know the absorbed degrees of freedom. Winsorizes the **LHS only**; the instruments are never winsorized. Clusters on unit only — no state cluster, and no HAC despite the overlapping windows | `post_first_stage_full_grid.dta` |
| `code/stata/37_verify_post_first_stage_estimator.do` | The equivalence proof for `36`'s hand-rolled estimator. On one reference cell runs the full `reghdfe … absorb(unit state_year) vce(cluster unit)` beside the fast FWL path and asserts agreement to **1e-8** on β and on the clustered SE, then asserts the value `36` stored for that key matches. This is what licenses the `se_factor` step | log only |
| `code/py/38_verify_post_first_stage_full_grid.py` | Independent verification of `35`/`36`: declared share windows, non-overlap year grids, the duration date audit, and — the load-bearing one — that the `year` and `state` winsorization columns actually **differ from `pooled`**, which is the exact failure mode of the pre-2026-08-15 `byable(recall)` `winsor4`. Also asserts one sample per `(rung, flow, window, treatment, horizon)` block across every share and winsor cell. Note it writes a second `maximum_F_within_rung_flow` selector, which compounds `39`'s | `post_first_stage_full_grid_summary.csv` |
| `code/py/39_export_award_first_stages_above9.py` | Exports the award-flow specifications whose first-stage *F* exceeds **9**, collapsed to structural families, reporting `best_F` with its winsorization rule alongside `F_min_all`, `F_median_all`, `F_no_winsor` and a count of winsor cells clearing 9. **Read §4's register item 1 before using this.** It selects on first-stage strength over 11,520 specifications that include transformations of the dependent variable; nothing in the file mentions pre-testing or multiplicity, and the threshold is 9 rather than the conventional 10 with no stated reason | `post_first_stage_award_structural_above9.csv` |
| `code/py/40_pre_first_stage_nonoverlap.py` | The pre-era probe that established the nonoverlapping schedule: one start year every `h` rather than every year, so no two observations share a contract window. Defaults `--share fixed5`, `--period pre` | `batch_logs/macro_version/` CSVs |
| `code/py/41_build_symmetric_nonoverlap_panel.py` | **THE PANEL OF RECORD.** Builds the symmetric nonoverlapping county panel from `33`'s harmonized contracts, `12`'s income and `13`'s panel: forward-dated `future` over `[t+1, t+h]`, income and employment at `t`, fixed shares `fixed4`/`fixed5`/`fixed10` on windows ending the year *before* each era begins (`share_windows()` sets `end = first − 1`), start years spaced by `h`, and both national denominators. Screens to the **3,094** positive-employment CCGs and then to the **48** states with at least two assigned commuting zones — **3,080 counties**. Everything the settled design reads descends from its `panel.parquet` | `_data/temp/symmetric_nonoverlap_first_stage/panel.parquet`, `<period>_h<h>.dta`, `symmetric_nonoverlap_design.json` |
| `code/stata/42_symmetric_nonoverlap_first_stages.do` | County first stages on `41`'s panel: three shares × two denominators (income and employment) × seven winsorization arms (`n`, IQR4/IQR5 pooled and by year, tail-1 pooled and by year) × `h = 2, 3, 4` × two eras = **252 fits**. FWL two-step, `absorb(unit state_year)`, `vce(cluster unit)` | `symmetric_nonoverlap_first_stage.csv` |
| `code/py/43_verify_symmetric_nonoverlap_pipeline.py` | Verifies `41`+`42`. Its check 20 pins the grid against the git-tracked `code/reference/symmetric_nonoverlap_first_stage_v1.csv` | `symmetric_nonoverlap_verification.csv` |
| `code/py/44_run_symmetric_nonoverlap_pipeline.py` | **Orchestrator** for `41` → `42` → `43` → `45`, with the certified-cache gate and the step runner that `50` and `55` both reuse. §2.1 | `44_*.log`, run receipt |
| `code/py/45_export_symmetric_first_stages.py` | Exports the specifications clearing *F* > 9, the structural families, and the income-versus-employment denominator comparison. **CSV only — it writes no `.tex`** | `symmetric_first_stage_above9.csv`, `..._structural.csv`, `..._denominator_comparison.csv` |
| `code/py/46_compile_first_stages_above9.py` | A compilation helper over `39`/`45`'s above-9 exports. **Number collision with `46_build_post_anchor_sweep_panel.py`** — the counter was advanced twice; the sweep builder is the one `50` calls | CSV |
| `code/py/46_build_post_anchor_sweep_panel.py`, `code/stata/47_post_anchor_sweep_first_stages.do`, `code/py/48_verify_post_anchor_sweep.py`, `code/py/49_export_post_anchor_sweep.py` | The post-anchor sweep: which post-era anchor year the post-1998 first stage rests on. `48`'s check 16 pins the coupled anchor-1998 diagonal to the eighteen post rows of the **frozen v1 grid**, which is git-tracked precisely because `42` rewrites the live CSV on every `44` run. **CSV only, no `.tex`** | `post_anchor_sweep_{headline,decomposition,verdict}.csv` |
| `code/py/50_run_post_anchor_sweep.py` | **Orchestrator** for `46`–`49`. Reads `44`'s gate and the frozen reference and writes to neither, so rerunning `44` afterwards must still pass | `50_*.log`, run receipt |
| `code/py/51_build_ladder_first_stage_panel.py` | §1.8.12's panel. Rebuilds the candidate shock and instrument from `41`'s symmetric panel — restating `01_build_candidate_panel.py`'s arithmetic rather than importing it, so divergence fails in the builder — and **pins every rebuilt column against the candidate `.dta` at exactly `0.0`**. Attaches the commuting zone, state and division keys via `14`'s `load_rungs`, and the CZ's state via `27`'s predetermined assignment, repaired onto the frozen support where it points outside the 48 retained states. Caps the shock and the instrument **separately, within start year**, at five interquartile ranges, storing capped and uncapped side by side. Stores **county rows only**: the rungs above need a weight vector and that choice is not frozen | `_data/temp/ladder_first_stage/panel.parquet`, `<era>_h<h>.dta`, `ladder_first_stage_design.json`, `..._share_audit.csv`, `..._panel_audit.csv` |
| `code/stata/52_ladder_first_stages.do` | Estimates the ladder: 3 rungs × 2 eras × 3 horizons × 2 winsorization arms × 2 absorption arms = **72 fits**. Structural clone of the candidate package's `02_run_candidate_first_stage.do` — FWL two-step, `e(sample)` fixed once **per absorption arm** on the **uncapped** shock so the two winsorization arms share rows within an arm, manual `sqrt((N−1)/(N−df_a−2))`. Collapses to the commuting zone and the state as `sum(ω x)/sum(ω)` written out longhand, `ω_ct = income_ct`, so the weighted mean and the plain sums cannot be confused in one `collapse` call. **Caps nothing** — it aggregates `51`'s capped columns and re-derives `51`'s Python cap in Stata to assert they agree. Asserts the aggregation identities and refuses to proceed if `reghdfe` absorbs away a single unit-year | `ladder_first_stage.csv`, `ladder_first_stage_collapse.csv` |
| `code/py/53_verify_ladder_first_stage.py` | 16 checks. Checks 1–8 are the preflight, one per axis the specification moved on, so a script that quietly kept an old setting fails here rather than producing a plausible table. **Check 9 is the pin**: the county rung merged onto the certified 252-row candidate result, integers exact and estimates to `1e-10` relative — and if it fails the aggregate rungs are not evaluated at all. Checks 10–12 are the aggregation identities and the frozen support; check 13 refits all 72 cells in `pyfixest`, both absorption arms. **Checks 14–15 are the absorption arms**: 14 requires the two to agree at exactly 0.0 at the state rung, where the unit is the state and both reduce to unit + year, and 15 pins the 36 pre-existing `sxy` cells to the git-tracked `code/reference/ladder_first_stage_sxy_v1.csv` at 0.0 so an addition cannot become a revision. Whether any *F* clears 9 is **reported and never asserted**, and so is the comparison against step_2's own county *F* | `ladder_first_stage_verification.csv` |
| `code/py/54_export_ladder_first_stage.py` | Renders §1.8.12's and §1.8.13's tables into step_1's `result/` folder as two tabulars, one per absorption arm, and writes the `h = 4` block wide over the winsorization arm as the file later runs read instead of re-estimating. *F* leads and β follows because *F* is the only number comparable across the two arms — capping the instrument removes most of its variance, so β rises by an order of magnitude on the same data | `step_1/result/TableF1_ladder_first_stage.tex`, `_preview.tex`, `ladder_first_stage_headline.csv`, Dropbox mirrors |
| `code/py/55_run_ladder_first_stage.py` | Orchestrator for `51`→`52`→`53`→`54`, reusing `44`'s gate and step runner as `50` does. Two modes, `--certified-cache` (default) and `--from-raw`. Rebuilds the candidate panel first when its gitignored Stata inputs are absent, so a fresh checkout fails at a missing input rather than three steps later inside a pin | `55_run_ladder_first_stage.log`, `ladder_first_stage_run_receipt.json` |
| `code/py/dcads_common.py` | shared by all of them: path resolution, the zoned-decimal table, fiscal-to-calendar time, the layout registry, the chunked reader, the ZALF reader, the ZARK crosswalk, `SUPERSEDED`, `EFSY_RESIDUAL_AS_COUNTY` (the DC recode, which `10` and `11` both need), the `CPI` table, and `build_contract_panel` — the contract collapse, factored out here rather than inlined in `13` so that the disaggregation collapses contracts the same way |  |

---

### 2.1 The symmetric nonoverlapping pipeline — `40`–`50`

**This is the only documentation of scripts `40` through `50`, which build the panel of record.**
It sat unnumbered at the foot of the file until 2026-08-18, after §4 and referenced by no
cross-reference; it is a numbered subsection of §2 now because `41`'s `panel.parquet` is what
everything downstream reads. `§1.8.12`'s own run block is §1.8.12, and `55` reuses `44`'s gate and
step runner rather than restating them.

The entry point is `code/py/44_run_symmetric_nonoverlap_pipeline.py`. It has two modes:

Run the code from a local Git checkout. All inputs and generated datasets are
read from or written to the shared Dropbox project. The default Dropbox project
root is
`~/Library/CloudStorage/Dropbox-Matray/_AdrienTeam/Chenzi_Adrien/Invariance_Agg_Disagg/`.
The path resolver also recognizes a team share mounted directly as
`~/Library/CloudStorage/Dropbox-Matray/Invariance_Agg_Disagg/`. Set
`INVARIANCE_DROPBOX_ROOT` before running only if the shared project is mounted at
another location. Set `INVARIANCE_DROPBOX_HOME` as well when restricted-source
checks use a nonstandard Dropbox account mount.

```bash
# Verify and reuse the certified raw-derived county caches, then rebuild and estimate.
4_Macro_version/.venv/bin/python \
  4_Macro_version/step_1/code/py/44_run_symmetric_nonoverlap_pipeline.py \
  --certified-cache

# Execute the authoritative raw-data producers before rebuilding and estimating.
4_Macro_version/.venv/bin/python \
  4_Macro_version/step_1/code/py/44_run_symmetric_nonoverlap_pipeline.py \
  --from-raw
```

The common design is:

- signed award-year obligations, in real 2006 dollars, with no nominal dollar floor;
- county support fixed by positive employment over FY1975–2006;
- states containing at least two assigned commuting zones;
- fixed 4-, 5-, and 10-year allocation shares ending immediately before each period;
- period-anchored nonoverlapping windows at `h = 2, 3, 4`;
- cumulative contracts from `t+1` through `t+h`, divided by personal income in `t`;
- the same cumulative leave-one-county-out national flow on the instrument;
- LHS-only 1st–99th percentile winsorization within start year through `winsor4`;
- county and state-by-year fixed effects, county clustering, and no weights.

The pre-period shares use FY1975–78, FY1974–78, and FY1969–78. The post-period shares use
FY1994–97, FY1993–97, and FY1988–97. `41_build_symmetric_nonoverlap_panel.py` writes the six
period-horizon inputs, `42_symmetric_nonoverlap_first_stages.do` estimates them through
Stata-MCP, and `43_verify_symmetric_nonoverlap_pipeline.py` performs the independent raw-cell
and pyfixest checks.

| Output | Contents |
|---|---|
| Dropbox project: `_data/temp/symmetric_nonoverlap_first_stage/panel.parquet` | Combined pre/post analysis panel |
| `batch_logs/macro_version/symmetric_nonoverlap_first_stage.csv` | Coefficients, clustered standard errors, and first-stage statistics |
| `batch_logs/macro_version/symmetric_nonoverlap_share_audit.csv` | Share windows, denominators, and adding-up checks |
| `batch_logs/macro_version/symmetric_nonoverlap_panel_audit.csv` | Start years, window counts, sample counts, and input paths |
| `batch_logs/macro_version/symmetric_nonoverlap_verification.csv` | Raw-source, construction, estimator, and post-grid verification gates |
| `batch_logs/macro_version/symmetric_nonoverlap_run_receipt.json` | Run mode, commands, timestamps, and return codes |

## 3. Verification

### 3.1 After downloading — `03_verify_dcads.py`

The NARA catalog's recorded sizes disagree with what the server actually serves: every
zip arrives 160 bytes larger than its catalog figure and `DCADS_TSS.pdf` 2,171 bytes
smaller. Both are valid files, so byte equality against the catalog produces false
failures. The script checks what each file actually **is** instead — a zip's central
directory parses and every member passes CRC, a PDF starts with `%PDF` and contains an
EOF marker, text decodes, a data file is non-empty and its size divides evenly by the
record length where one is declared.

> **211 of 211 objects verified. 0 failed, 0 missing.**
> 155 PDFs with header and EOF marker, 12 zips with every member passing CRC, 3 text
> files decoded, 41 data files non-empty with record arithmetic reported.
> Sizes differing from the catalog but verifying clean: −2,171, −79, +2, +7, +100, +160.

### 3.2 After parsing — `06_verify_dcads_parse.py`

Nine checks, ending on the ones whose failure would mean the numbers are wrong rather
than merely internally inconsistent.

> **42 of 42 fiscal years verified. Exit 0.** About 10 minutes.
>
> - **Row counts** — every year matches an independently sourced count: the declared
>   count in `DCADS_TSS.pdf` where there is one, size ÷ record length otherwise, and a
>   raw line count for the three delimited years.
> - **Counties sum to the state, against the raw records** — 2,290 state-years compared,
>   **exact in all 42 fiscal years**. See below.
> - **The record's own fiscal year** matches the filename on 100.0000% of records, in
>   every year.
> - **Every dollar amount decodes** — zero failures across 13,653,309 rows.

**The county check, and why it is the load-bearing one.** Contract dollars summed across
the counties of a state in a given year have to equal that state's year total **read
straight from the raw records**, to the dollar. The county side comes from the parquet;
the state side comes from an independent re-read of the source bytes. That is what
separates it from every other geography check: both sides of the others are views of the
same parse, so a field read from the wrong offset agrees with itself. Here it cannot.
And wherever the county code carries its own state — FY1984 onward, county `SSCCC` — the
rollup is keyed on the **county** rather than the state field, which puts the county
itself under test: a misread county sends its dollars to a state key that does not exist,
or to the wrong one, and both states break at once.

Two differences are permitted, and both are accounted for by name so the comparison stays
exact rather than tolerant.

**The four domestic rows that name no county.** These are the whole of the county-state
difference across 42 years — 4 rows and **\$1,703,000** out of 13.65M records:

| year | dollars | state | city | contractor | what it was |
|---|---|---|---|---|---|
| FY1984 | \$1,225,000 | Pennsylvania | — (city code `4249608`) | SBA Western International | repair of recreational structures, Corps of Engineers |
| FY1985 | \$120,000 | Maryland | Baltimore Highlands | Commercial Movers Inc | cargo and freight services |
| FY1985 | \$186,000 | Florida | Saint Petersburg Beach | E-Systems Inc | maintenance of aircraft ground equipment |
| FY1989 | \$172,000 | Florida | Jacksonville | PHP Healthcare Corporation | nursing services |

They are **not unidentifiable locations**: the state is known in all four and the city in
three, with a city code in the fourth. Only the 3-digit county field is blank in the
source, so in principle a city-to-county crosswalk would recover every one of them. They
are neither dropped nor reassigned — they sit in the unplaceable bucket, so a county panel
loses them and a state panel keeps them. At 0.0001% to 0.001% of a state-year they are
immaterial, and the reason the check counts them individually rather than tolerating a
threshold is that a county column read from the wrong bytes would come back blank too, and
would show up here as most of the file instead of as four rows.

**One row whose county names a different state.** FY1984 carries a record pairing
California's `06` with Suffolk County New York's `36103`, moving \$78,000 from CA to NY
under a county-keyed rollup. The source's own error, not a parse defect; it is moved back
before comparing, reported, and bounded by the field-domain check.

**Recorded and monitored, not failures.** FY2003 has 24 domestic rows with a 2-character
county code (0.004%, under the 0.01% threshold); FY1995 has one row on the `98` catch-all;
FY1987–93 have 43–109 foreign rows a year breaking the county-begins-with-state invariant,
all Cuba as `CU`/`00000`; FY2006 has 8 rows with a day-first date, re-dated from
`action_date`; FY2004's 290 repaired rows and 1,207 imputed values (§1.5) pass with zero
domain violations. The script also prints a comparison against Pallante's published
state-year series, which is informational here and is left in place unchanged so that
rerunning 06 unmodified still proves the parse did not move. **That comparison is now
superseded by `09_pallante_test.py`** (§3.4), which runs it on all 42 years rather than
only the FIPS-bearing ones, keys it on `state_fips2`, and makes it a pass criterion.

**Two limits to know.**

1. **Only the dollar field was swept for zoned-decimal signs.** Check any other numeric
   field — `n_actions`, `multiyear_value`, set-aside values — before using it in the
   pre-2004 years.
2. **One field in the 272-byte layout is mislabelled for part of the era.** Byte 36,
   `southeast_asia`, is a minority-business code from FY1977. It is unused downstream, and
   every field Part 1 needs is stable across all 19 years. DoD documents separate layouts
   for pre-1979, FY1979 and FY1983; only the pre-1979 one has been read.

### 3.3 After the county FIPS — `08_verify_county_fips.py`

Scoped narrowly on purpose. Row counts, dollar decoding, the record's own fiscal year and
the county-sums-to-state invariant are 06's job and are not repeated; 06 is left untouched
and rerun afterwards, which is what shows the rewrite changed nothing. This asks the one
question 06 cannot: is the geography right.

> **42 of 42 fiscal years verified. Exit 0.** Then 06, unmodified: 42 of 42, 2,290
> state-years exact, 13,653,309 records — unchanged.
>
> - **0 domain violations** in 13.65M rows: `geo_class` only ever takes a declared value,
>   `state_fips2` always equals `county_fips5[:2]` where both exist, `98` never reaches
>   `state_fips2`, and `county` holds exactly the rows in the 50 states and DC with a county.
> - **0 unresolved** — every well-formed county code in the 50 states and DC resolved, in
>   every year. The 24 remaining FY2003 rows whose county field holds the 2-character state
>   code are malformed in the source, cannot be fixed by any crosswalk, and are unplaceable.
> - **0 counties in the wrong state.**

**The name check**

> **2,725 ZALF county records reach a FIPS code that FY1984–96 also names. Four disagree,
> and all four are renames rather than misreads** — accounted for individually, in the same
> style as check 9, so that a fifth fails:
>
> | fips | ZALF | FY1984–96 | why |
> |---|---|---|---|
> | 51650 | `47/615` Fort Monroe | Hampton (city) | an installation inside Hampton; ZALF gives it Hampton's FIPS, as it does for ZARK 650 |
> | 56029 | `51/024` Yellowstone Natl Pk | Park | the Wyoming side of the park lies in Park County |
> | 02070 | `02/070` Bristol Bay | Dillingham | the 1970-census Bristol Bay Division was renamed the Dillingham Census Area, keeping 02070. Not the Bristol Bay Borough, 02060 |
> | 32510 | `29/013` Ormsby | Carson City (city) | Ormsby County became Carson City in 1969, keeping 32510 |

**The FY1983/84 seam** (reported). The last ZARK year and the first FIPS year are built by
completely different code paths, so a broken crosswalk would show as a step. There is none:
**2,072 counties in FY1983, 2,121 in FY1984**, 96.5% of FY1983's codes present again in
FY1984–88. The 72 codes in FY1983 and not FY1984–88 are sparse rural counties, with one
exception worth knowing about — **Caddo Parish, Louisiana (22017) goes from 212 actions in
FY1982 and 93 in FY1983 to zero for FY1984–89, while adjacent Bossier Parish (22015,
Barksdale AFB) goes from 89 to 203.** The crosswalk is right and FY1984 needs no crosswalk
at all; the source reassigned its Shreveport-area work across the parish line. Anything
using Louisiana counties around 1984 has to know that.

**Coverage** (reported). The ZARK era resolves to **2,919** distinct counties, against
2,978 in FY1984–96, 2,901 in FY1997–2003 and 2,939 in FY2004–06. **55 codes appear in the
ZARK era and in no FIPS-era year**, and they split cleanly: 42 are counties that still
exist and simply drew no defence contract after FY1983, mostly one or two rows each. The
other **13 stopped existing before FY1984 and so cannot appear again — these are exactly
what a time-consistent county key has to map forward** (§1.6 does it, and this table is
about a fifth of that work), and 08 prints them with their
dollar weights:

| code | ZARK name | k\$ 1966–83 | successor |
|---|---|---|---|
| 13510 | Columbus | 168,047 | consolidated into Muscogee County GA 13215 in 1971 |
| 02120 | Kenai-Cook Inlet | 148,302 | Kenai Peninsula 02122 |
| 02260 | Valdez-Chitina-Whittier | 115,733 | Valdez-Cordova 02261 |
| 02160 | Kuskokwim | 84,661 | Bethel 02050 |
| 02040 | Barrow/North Slope | 80,188 | North Slope Borough 02185 |
| 02080 | Cordova-McCarthy | 11,439 | Valdez-Cordova 02261 |
| 02210 | Seward | 4,057 | Kenai Peninsula 02122 |
| 51123 | Nansemond | 2,740 | Nansemond County VA, abolished 1974 → Suffolk 51800 |
| 02250 | Upper Yukon | 2,412 | Yukon-Koyukuk 02290 |
| 02230 | Skagway-Yakutat | 771 | Skagway-Yakutat-Angoon 02231 |
| 02200 | Prince of Wales | 435 | Prince of Wales-Outer Ketchikan 02201 |
| 02190 | Outer Ketchikan | 164 | Prince of Wales-Outer Ketchikan 02201 |
| 51695 | Nansemond | 132 | Nansemond city VA, 1972–74 only → Suffolk 51800 |

### 3.4 Against an outside source — `09_pallante_test.py`

Check 2 of `PLAN_macro_fiscal_multiplier.md` §1.5. Every other geography check compares the parse against
itself, so a systematically wrong parse would agree with itself. Pallante builds his
series from his own extract of the same archival records, with his own code, keyed on the
state **name** rather than on any numeric code. Agreement is therefore evidence, and
disagreement is a finding.

**The benchmark is variant A: RAW**, `all_dod_statename_noadj.csv` — the only one of his
eight aggregates built without a geography correction, a reporting threshold or an FMS
exclusion, and so the only one an unadjusted parse can be compared against at all. The
51-code universe is taken from that file rather than from `state_codes.txt`, which carries
56 rows including the territories his series has no rows for. FY1976 is compared with TQ76
folded in, because his FY1976 covers the fifteen months the source reports under that
label.

The script runs four stages: **A** the test against his series exactly as he ships it, **B**
the diagnosis, **C** the same comparison with the negative-zero rows repaired on *his* side,
**D** every surviving deviation named individually so a new one fails.

> **41 of 41 overlapping fiscal years accounted for. Exit 0.** 2,091 state-years compared.
> His series runs to FY2019 and the contract archive ends at FY2006, so 13 of his years
> are out of scope rather than deviations.
>
> - **Stage A, against his series as published: 29 fiscal years exact in all 51 states**
>   — FY1977–2001 and FY2003–06. FY1966–76 deviate, and FY2002 by one record.
> - **Stage C, with the sign byte read correctly on both sides: 37 of 41 exact.** The 8
>   MPCF years that had no other issue close to **zero in every one of the 51 states**.
> - **4 years carry a named deviation**: FY1967, FY1973, FY1976, FY2002 — \$45,230,684 in
>   total, **0.0011%** of his FY1966–2006 total.

**This test found the §1.5 sign-byte error.** What used to stand here was an eleven-year
FY1966–76 band of \$10.317bn attributed to "a record-universe difference between his MPCF
extract and ours", recorded and not closed. It was not that. It was byte 86 of the dollar
field, misread on both sides, and the old diagnosis had explicitly *ruled decoding out* on
the grounds that FY1977–83 use the same layout, the same zoned-decimal table and the same
code path and match to the dollar. That was true and beside the point: the blank sign byte
never occurs after FY1975, so the control years never exercised that branch. **A control
year only controls for a code path it actually runs.**

**Why our side and his differ, in one paragraph.** Byte 86 is a zoned-decimal zone byte —
it carries the amount's last digit and its sign together. In FY1966–75 negative zero is
written there as a blank (§1.5). Read as padding, which is what R's `as.numeric()` does
with a trailing space, the amount comes out a tenth of the truth and positive: that is his
series. Read as positive zero, the amount is right and the sign inverted: that was ours.
The archive says negative. So on those rows his published value is `+magnitude/10` where
the truth is `−magnitude`, and **his state-year total is high by 1.1 × magnitude** — which
is the only thing stage C subtracts. Nothing else about his series is touched.

**Stage A — against his series as published.** Our FY1966–75 totals now run **2.3% to 4.9%
below** his, because his are too high by that 1.1×:

| fy | ours, \$ | his, \$ | gap, \$ | gap % | states off |
|---|---:|---:|---:|---:|---:|
| 1966 | 32,547,622,000 | 33,636,292,000 | −1,088,670,000 | −3.237% | 51 |
| 1967 | 38,160,052,000 | 39,745,371,000 | −1,585,319,000 | −3.989% | 47 |
| 1968 | 38,048,619,000 | 39,314,521,000 | −1,265,902,000 | −3.220% | 50 |
| 1969 | 35,891,840,000 | 37,663,269,000 | −1,771,429,000 | −4.703% | 50 |
| 1970 | 30,347,544,000 | 31,400,838,000 | −1,053,294,000 | −3.354% | 50 |
| 1971 | 29,434,023,000 | 30,501,243,000 | −1,067,220,000 | −3.499% | 49 |
| 1972 | 33,144,720,000 | 33,936,434,000 | −791,714,000 | −2.333% | 48 |
| 1973 | 31,854,304,000 | 33,511,844,000 | −1,657,540,000 | −4.946% | 49 |
| 1974 | 33,866,611,000 | 35,106,817,000 | −1,240,206,000 | −3.533% | 51 |
| 1975 | 38,361,723,000 | 39,369,961,000 | −1,008,238,000 | −2.561% | 49 |
| 1976 | 50,653,174,000 | 50,627,352,000 | +25,822,000 | +0.051% | 34 |
| 1977–2001, 2003–06 | — | — | **0** | 0.000% | **0** |
| 2002 | 158,737,107,072 | 158,737,291,756 | −184,684 | −0.000% | 1 |

**Stage B — the diagnosis, in six tables.** Each is a dollar amount or a count, never a
verdict asserted without one. B1 is the negative-zero census and B2 the `range_code`
adjudication, both set out in §1.5; the `range_code` test is worth one number here — of the
**17,115** ambiguous rows, **17,095 fall inside their own record's declared magnitude band**
read as a zone byte, against **2,869** read as padding, with **14,246** falling below their
own floor. B3 asks whether the ZARK state code agrees with the record's own `state_name`:
**4,236,190 of 4,236,226** ZARK-era rows agree, and the 36 that do not are the source
contradicting itself. B4 re-keys our rows on his key, B5 counts byte-identical repeat
records, B6 measures his aggregation restrictions.

B3 is also **counted in both directions now, and that is what found the largest single
geography difference in the panel.** It used to skip any row whose `state_name` his mapping
does not cover, which silently hid the 14 FY1967 records coded to Montgomery County,
Pennsylvania whose `state_name` reads `BERMUDA`.

**Stage C — both sides corrected.** The eight MPCF years with nothing else wrong go to
**exactly zero in all 51 states**:

| fy | ours, \$ | his corrected, \$ | gap, \$ | states off | removed from his |
|---|---:|---:|---:|---:|---:|
| 1966 | 32,547,622,000 | 32,547,622,000 | **0** | **0** | 1,088,670,000 |
| 1967 | 38,160,052,000 | 38,140,856,000 | +19,196,000 | 8 | 1,604,515,000 |
| 1968 | 38,048,619,000 | 38,048,619,000 | **0** | **0** | 1,265,902,000 |
| 1969 | 35,891,840,000 | 35,891,840,000 | **0** | **0** | 1,771,429,000 |
| 1970 | 30,347,544,000 | 30,347,544,000 | **0** | **0** | 1,053,294,000 |
| 1971 | 29,434,023,000 | 29,434,023,000 | **0** | **0** | 1,067,220,000 |
| 1972 | 33,144,720,000 | 33,144,720,000 | **0** | **0** | 791,714,000 |
| 1973 | 31,854,304,000 | 31,854,276,000 | +28,000 | 1 | 1,657,568,000 |
| 1974 | 33,866,611,000 | 33,866,611,000 | **0** | **0** | 1,240,206,000 |
| 1975 | 38,361,723,000 | 38,361,723,000 | **0** | **0** | 1,008,238,000 |
| 1976 | 50,653,174,000 | 50,627,352,000 | +25,822,000 | 34 | 0 |
| **total removed** | | | | | **12,548,756,000** |

Eight years of two independently built extracts of the same archival records, keyed on two
incompatible geography systems, agreeing **to the dollar in all 51 states**. That is a far
stronger result than the old "bounded and recorded", and it is what the state dimension of
the ZARK crosswalk now rests on.

**Stage D — the four surviving deviations, \$45,230,684 in all.** Nothing below moves a
state-year by more than 1.15%, and only Pennsylvania FY1967 and Illinois FY1976 exceed 0.3%.

| year | states off | net deviation | % of the year | what it is |
|---|---:|---:|---:|---|
| FY1967 | 8 of 51 | +19,196,000 | +0.050% | the coded geography against the record's own `state_name` |
| FY1973 | 1 of 51 | +28,000 | +0.00009% | one record whose `state_name` field is blank |
| FY1976 | 34 of 51 | +25,822,000 | +0.051% | 276 byte-identical repeat records his extract does not hold |
| FY2002 | 1 of 51 | −184,684 | −0.00012% | one record we class foreign and he places in Georgia |

**FY1967, +19,196,000 across 8 states.** Gross +19,674,000 / −478,000. One item plus noise:
every state but Pennsylvania is under 0.13%.

| state | ours | his corrected | deviation | % of state |
|---|---:|---:|---:|---:|
| pennsylvania | 1,686,851,000 | 1,667,522,000 | **+19,329,000** | +1.146% |
| california | 6,741,803,000 | 6,742,110,000 | −307,000 | −0.005% |
| oregon | 143,673,000 | 143,490,000 | +183,000 | +0.127% |
| virginia | 674,140,000 | 674,057,000 | +83,000 | +0.012% |
| florida | 834,339,000 | 834,260,000 | +79,000 | +0.010% |
| new jersey | 1,236,931,000 | 1,236,997,000 | −66,000 | −0.005% |
| new york | 3,270,101,000 | 3,270,164,000 | −63,000 | −0.002% |
| alabama | 315,506,000 | 315,548,000 | −42,000 | −0.013% |

The records behind it, our code against his name:

| `state_name` | our code says | his name says | rows | dollars |
|---|---|---|---:|---:|
| `BERMUDA` | pennsylvania, 39/046 Montgomery | unmapped | 14 | +19,133,000 |
| `CALIFORNIA` | oregon, 38/003 Clackamas | california | 7 | +183,000 |
| `NEW YORK` | pennsylvania, 39/046 Montgomery | new york | 2 | +105,000 |
| `CALIFORNIA` | virginia, 47/007 Arlington | california | 1 | +83,000 |
| `CANADA` | florida, 10/005 Brevard | unmapped | 2 | +63,000 |
| `ALABAMA` | new york, 33/012 Cortland | alabama | 1 | +42,000 |
| `CALIFORNIA` | pennsylvania, 39/046 Montgomery | california | 1 | +25,000 |
| `CALIFORNIA` | florida, 10/005 Brevard | california | 1 | +16,000 |
| `NEW JERSEY` | pennsylvania, 39/046 Montgomery | new jersey | 2 | ±66,000, nets 0 |

**The 14 Bermuda records are ambiguous, and this is the honest statement of it.** All 14 are
General Electric, every one coded `39/046` — Montgomery County, Pennsylvania — with
`county_name` `MONTGOMERY` and a blank city; the largest are \$6,661k and \$6,275k on
contract `F04695400580`. Genuine Bermuda rows elsewhere in the same file carry `state_code`
`BD` and are classed foreign, so the two fields cannot both be right. But **either could be
the true place of performance**: GE's Re-entry Systems plant was in Montgomery County PA,
Bermuda hosted an Air Force downrange tracking station, and these are Air Force Space
Systems contracts (`F0469…`). We keep the coded geography because that is the rule the parse
applies in every other year, **not** because `BERMUDA` is demonstrably wrong. Nor is this a
"three fields to one" case like the FY1984 record of §1.5: state code and county code plus
name are one geography block populated together, so it is two facts against one, in the same
block, from the same reporting office.

**FY1973, +28,000, one record.** Contract `F047013C0207`, Atlantic Science Corp, `state_code`
`10`, `county_code` `005`, `county_name` `BREVARD`, `city_name` `INDIALANTIC` — Brevard
County, Florida, unambiguously — and a `state_name` field of ten blanks, so his name key has
nothing to place it by. FSC `A433`, action 73D.

**FY1976, +25,822,000 across 34 states.** Gross +28,280,000 / −2,458,000. Of the 34, **28
are under \$1m and twelve under \$50,000**; New York is the only negative, a duplicated
deobligation.

| state | ours | his corrected | deviation | % of state |
|---|---:|---:|---:|---:|
| california | 10,994,439,000 | 10,986,177,000 | +8,262,000 | +0.075% |
| illinois | 673,645,000 | 667,670,000 | +5,975,000 | **+0.887%** |
| virginia | 1,927,693,000 | 1,922,833,000 | +4,860,000 | +0.252% |
| **new york** | 4,438,845,000 | 4,441,303,000 | **−2,458,000** | −0.055% |
| hawaii | 431,489,000 | 429,848,000 | +1,641,000 | +0.380% |
| arizona | 760,499,000 | 759,213,000 | +1,286,000 | +0.169% |
| district of columbia | 677,939,000 | 676,989,000 | +950,000 | +0.140% |
| maryland | 1,230,721,000 | 1,229,948,000 | +773,000 | +0.063% |
| pennsylvania | 1,672,397,000 | 1,671,689,000 | +708,000 | +0.042% |
| minnesota | 795,992,000 | 795,523,000 | +469,000 | +0.059% |
| connecticut | 2,366,100,000 | 2,365,674,000 | +426,000 | +0.018% |
| ohio | 1,228,144,000 | 1,227,725,000 | +419,000 | +0.034% |
| massachusetts | 2,525,297,000 | 2,524,884,000 | +413,000 | +0.016% |
| florida | 1,281,981,000 | 1,281,642,000 | +339,000 | +0.026% |
| rhode island | 112,709,000 | 112,413,000 | +296,000 | +0.263% |
| new jersey | 1,340,075,000 | 1,339,829,000 | +246,000 | +0.018% |
| missouri | 2,854,301,000 | 2,854,062,000 | +239,000 | +0.008% |
| texas | 2,865,835,000 | 2,865,610,000 | +225,000 | +0.008% |
| colorado | 435,744,000 | 435,567,000 | +177,000 | +0.041% |
| louisiana | 740,599,000 | 740,457,000 | +142,000 | +0.019% |
| north carolina | 498,706,000 | 498,609,000 | +97,000 | +0.019% |
| mississippi | 1,071,864,000 | 1,071,788,000 | +76,000 | +0.007% |
| washington | 1,682,391,000 | 1,682,342,000 | +49,000 | +0.003% |
| nebraska | 74,074,000 | 74,030,000 | +44,000 | +0.059% |
| georgia | 656,537,000 | 656,506,000 | +31,000 | +0.005% |
| iowa | 338,456,000 | 338,434,000 | +22,000 | +0.007% |
| utah | 176,488,000 | 176,469,000 | +19,000 | +0.011% |
| south carolina | 207,019,000 | 207,002,000 | +17,000 | +0.008% |
| tennessee | 471,323,000 | 471,307,000 | +16,000 | +0.003% |
| wisconsin | 343,258,000 | 343,243,000 | +15,000 | +0.004% |
| kentucky | 314,703,000 | 314,690,000 | +13,000 | +0.004% |
| new mexico | 168,637,000 | 168,624,000 | +13,000 | +0.008% |
| michigan | 1,263,217,000 | 1,263,206,000 | +11,000 | +0.001% |
| oregon | 130,928,000 | 130,917,000 | +11,000 | +0.008% |

**\$25.8m net is the residue of a much larger set that mostly cancels.** The 276 repeats are
**\$262,268,000 gross in absolute value, and 90.2% of it cancels inside the repeat set** —
twelve magnitude groups are exact +/− pairs, because the FY76 file holds an obligation, its
reversal, *and* a duplicate copy of each:

| \$000 | state | contract | contractor |
|---:|---|---|---|
| +41,215 / −41,215 | california | `N000306C0045` | Lockheed Missiles & Space, Sunnyvale |
| +23,432 / −23,432 | california | `N000306C0053` | Lockheed Missiles & Space, Sunnyvale |
| +13,282 / −13,282 | new york | `N000246C4119` | Sperry Rand, Great Neck |
| +7,636 / −7,636 | massachusetts | `N000306C0037` | General Electric, Pittsfield |

So this is not \$25.8m of duplicated spending; it is an unpaired tail. TQ76 contributes only
9 repeats, none above \$61k.

**FY2002, −184,684, one record.** Georgia: ours 5,814,196,067 against his 5,814,380,751.
Contract `DACA8101D0229`, `state_code` `KS` — the FIPS 10-4 country code for the Republic of
Korea — and `state_name` `KOREA, REP`. Both fields agree it is foreign; his series places it
in Georgia.

Two things worth knowing across all four. **His name key resolves typos but not foreign
names** — `NEVEDA` in FY1967 and `WIS.;NSIN` in FY1973 are both flagged by B3 and produce
**no deviation at all**, so he places them exactly as we do; what his key cannot place is a
domestic record whose `state_name` is a foreign country or blank, which is the whole of
FY1967 and FY1973 above. And **FY1976 is not a deduplication rule of his**: FY1978–83 carry
6,300 to 11,661 byte-identical repeats worth \$120m to \$382m in the 51 states and match him
to the dollar anyway, so it is a difference in what his FY1976 extract holds. The declared
record count in `DCADS_TSS.pdf` for FY1976 is 223,130, which is exactly what we read, so
the repeats are in the archival file as distributed and dropping them is not something we
can justify.

FY1984 used to be a fifth entry, a matched ∓\$78,000 pair across California and New York.
That row is now corrected in the data instead (§1.5) and FY1984 is exact in all 51 states.
That his independent extract also reads the record as California is part of why the
correction is defensible.

**What the test does not rest on.** Stages B and C read the raw source bytes alongside the
parse, and the positional join between them is **verified rather than assumed**: all
4,557,244 records of the 272-byte era must decode to the value the parquet holds, or the
script fails before diagnosing anything.

---

## 4. Status and open

**Items keep their numbers.** They are cited from outside this file — `22_winsor_construction_error.py`
names item 9, step_2's README names items 5 and 8 — so the numbers are identifiers, not an ordering.
What follows is grouped by *audited* status rather than by number, because the labels had drifted:
six items marked open concern scripts that are no longer the live pipeline, and one item marked open
at position 17 is the sharpest qualification on the headline result in the file.

Re-audited 2026-08-18 against the settled design.

### Open, and load-bearing on the current result

**5. Shift-share inference is untouched, and it applies to the settled instrument.** `z_ct` is one
common national shifter times a time-invariant county loading, so Adão, Kolesár and Morrison (2019)
applies and clustering on the rung's own unit does not address it. Every interval in §1.8.12 and
§0.3 could be too narrow for that reason. step_2's README §2.10 carries the same caveat, so this is
open in both halves of the project.

> The rest of item 5 as it used to read is retired. Its numbers — β = +0.076 (0.027), *F* = 72.5, the
> county → CZ → state ladder of +0.076 → +0.132 → +0.297 — are the legacy instrument's and are in
> `APPENDIX.md` §A.4. Its `15`/`16`/`17` mtime problem and its `14` demeaner pin are moot: see the
> retired-machinery group below.

**7. The Denver International Airport boundary move** (§1.6.5). Denver annexed 43.31 unpopulated
square miles from Adams County in 1988 and the airport opened in 1995, so Denver's employment steps
up seven years after the boundary did. Neither a level-break sweep nor a change list catches that on
its own. Nothing has touched it.

**8. The FY1984 and FY2000 reporting breaks are inside the identifying variation.** With no floor,
`ΔG^nat` moves when DoD's reporting threshold moves. FY1983–84 sits inside the Reagan buildup, and
the FY1984 robustness split §1.5 flagged is **required rather than optional; it has not been run.**
FY2000 sits inside the flat post-1998 stretch, so there the reporting break and the absence of a
buildup **cannot be separated at all** — a reason the post-1998 half is weak rather than a caveat on
it. Still live in step_2 §2.10.

**17. No IQR rule is mild on this shock.** `shock`'s interquartile range at `h = 2` is 0.0092 and its
maximum is 14.07 — **1,529 IQRs above the median** — so `p50 ± L·IQR` binds hard at any `L`, and
`p50 ± 1000·IQR` still binds. `APPENDIX.md` §A.9 reports the price in every panel rather than tuning
`L` around it, and its check 7 had to be rewritten because the "absurd `L` is a no-op" test it was
meant to be is not implementable on this variable.

**This is no longer a note about an exploratory exercise. The settled design caps at
`method(iq) level(5)` on this variable**, and §1.8.12's own table is what it costs: pre-era county
*F* = 4.90 uncapped against **395.32** capped, a factor of 80, with **none of the 36 uncapped cells
clearing 9 and all 36 capped cells clearing it**. The uncapped arm is reported as co-equal for
exactly this reason. What is owed is a decision about how the paper states a first stage that exists
only under a cap this severe — not a smaller `L`, since there isn't one.

**EFSY's 2017 vintage straddle is unresolved in the post era.** `11_build_employment_cells.py` records
`VINTAGE_BREAKS = (1998, 2017)`; a growth rate straddles the second break whenever
`2017 ≤ y ≤ 2016 + h`, which is `{2017, 2018}` at every horizon the design uses. The design refuses
the straddle at the 1998 break and **does not refuse it at the 2017 one.** `APPENDIX.md` §A.6 reading
7 shows what those two years were worth on one legacy loading: *F* barely moves while four of six β
change sign, and the standard errors *fall* as data is removed. The settled post era ends at 2018 and
inherits the question. Not previously numbered; recorded here.

### Done, and confirmed against the settled design

**1. Part 1.3 — done**, 2026-08-06, extended 2026-08-17. The EFSY employment panel is built into
county-by-industry cells at two levels of detail, rectangular on all 3,101 CCGs in all 44 years, by
`11_build_employment_cells.py`. The allocation ladder and the three properties of EFSY that bite the
results are §1.6.10. §1.6's check 6 establishes that after harmonisation the contract unit set sits
**inside** the employment unit set, year by year and not merely pooled (§1.6.7): 77 CCGs have
employment and never a contract. **The negative-cell deletion of 2026-08-17 is item 19.**

**2. Part 1.4 — done**, §1.7. BEA county personal income, contracts collapsed to `ccg × year`, and
the shock.

> **Restated 2026-08-18.** This item used to say `county_panel.parquet` is "3,101 CCGs × 32 years"
> and that it "is what Part 2 estimates on". Neither is current. The panel `13` writes is **136,136
> rows = 3,094 CCGs × 44 years** (`batch_logs/macro_version/county_panel_report.csv`), and Part 2
> estimates on `41`'s symmetric nonoverlapping panel re-scaled by the candidate builder, not on this
> one directly. `county_panel.parquet` is upstream of `41` and is still read directly by step_2's
> older `22`/`27`/`28` exhibits.

**3. Closed 2026-08-10: `11_build_employment_cells.py` is documented**, §1.6.10. Its write-up had
lived in an earlier version of `../step_2/README.md`, deleted on 2026-08-07 and never committed, so
it was not recoverable from history; it was rewritten from the code and from
`batch_logs/macro_version/11_build_employment_cells.log`. The number that travels downstream is that
**979,542 cells, 5.389% of the genuine cells, are positive only because of allocation**, which moves
them out of the entry term and into the capped term. **`naics4_raw` is not a control and is not
estimated on.**

> The test that this does not drive step 2's result is in `../step_2/README.md` §2.7: the cutoff
> collapse across the NAICS break appears at `naics2`, the rung the ladder never runs on
> (185.5 → 5.5), as well as at `naics4_alloc` (465.8 → 8.4). Since `naics2` needs no imputation, the
> break is EFSY's rather than the ladder's. This item used to cite a `§2.0 Finding 3` that does not
> exist in that file, and used to add that the imputation *attenuates* the tail — **that second
> claim is withdrawn**, because on the current build the imputed rung has the larger break (55×
> against 34×), not the smaller. §1.6.10 has the correction.

**The lesson, which is the reason this cost a rewrite:** `../step_2/README.md` was untracked until
2026-08-10. **Documentation that is not committed is not documentation.**

**4. Part 1.5, the instrument — done**, and superseded twice on the way. §1.8.12 is the answer to
check 5 and §1.8.13 reports it under both admissible absorptions.

> **The whole of this item as it used to read is retired.** It argued for `fixed` over `rolling` on
> validity grounds at `h = 2` — "the exposure is `fixed`, and the choice is not made on the *F*" —
> and the settled instrument is `fixed5` on the candidate panel with no leave-one-out, forward-dated,
> at `h = 4`. `APPENDIX.md` §A.1 and §A.4 reading 6 keep the argument, which was sound about the
> exposure family it was made about.
>
> Its one open sub-item is moot with it: §A.4's Table I Panel C and Table II's 1998–2006 block cannot
> be reproduced from the canonical outputs, because `"1998-2006"` is a key in `14`'s `PERIODS` but
> not in `DEFAULT_PERIODS`. Closing it meant `14 --periods 1998-2006 ...`, which runs into item 9's
> overwrite hazard. `14` is not the live instrument, so this is a gap in a superseded record and is
> labelled as one in §A.4 rather than carried as work.

**6. Decided 2026-08-10 by Adrien: the macro application is stated as a 1975–1998 design, and
post-1998 is reported as a side table.** No second source of post-1998 variation is being sought.
The empirical support at the time: over 1998–2006 no rung identified anything, and extending the
panel to FY2018 — which adds the Iraq/Afghanistan buildup and the 2011–15 drawdown — added only 17%
to the sd of the national buildup and returned a county first stage of *F* = 6.3 with β = −0.168
(0.088), sign flipped (§1.9.5).

> **The basis for this decision has moved, and the decision has not been revisited.** On the settled
> instrument §1.8.12 estimates **both** eras, and post-1998 clears at county *F* = **254.88** under
> the cap and 205.33 at the commuting zone — against 5.95 and 5.01 uncapped. §1.8.12 reading 3 calls
> the conclusion **arm-dependent**: the post era identifies under the cap and not without it. So
> "no rung identifies anything post-1998" is no longer what the data say, though what does identify
> is a capped quantity and item 17 is the reason that is not a clean win. Step 2 nonetheless
> estimates `pre` only, and its five typeset tables are all `_pre_h4_fixed5`.
>
> **This is a decision whose premise changed, not a conclusion to reword. It needs Adrien.** Stated
> here rather than resolved.

**A distinction the "1975–1998" label does not carry on its own, and every write-up has to.** It is
the framing — step 1's window convention and the span of the identifying buildup. It is **not** the
estimation window of anything that uses employment cells: there `t−h` cannot reach before 1975,
EFSY's first year, and 1998 straddles the NAICS break. The two conventions differ on purpose.

**10. Step 2's exhibits — superseded by its own 2026-08-18 run, and this item is stale.**

> When written on 2026-08-14 this item recorded step 2 running at `h = 2` on `fixed_7074`, 1977–1997,
> with post-1998 dropped entirely, and Table D1 as the open half. **That is now the older of two
> vintages in step 2.** Its current exhibits are `Table1`–`Table5` in `4_Macro_version/Results/`,
> all `_pre_h4_fixed5` on the candidate instrument, written 2026-08-18 by scripts `29`–`44`. The
> `22`/`27`/`28` family survives as imported machinery and as the older `TableW*` / `TableD1*`
> exhibits.
>
> Two things in this item are still live and worth keeping. The **cross-check** that step 2's
> `22`/`27`/`28` reproduce §A.5's Table V county cell — +0.1450 (0.0444), *F* = 61.4, N = 65,085
> against `first_stage_table.csv`'s +0.145009 (0.044408), 61.4177, 65,085 — is the only external pin
> those three have, and §A.5 carries a do-not-move banner on that cell for it. And the **`11`
> `--industry-support` change**: the default was `county-own`, which gave a county rows only for the
> industries it had *ever* reported, leaving the four-digit panel 745,143 pairs a year against the
> 3,101 × 303 cross product — short by 194,460, **20.7%** — in violation of plan §2.1. The default
> is now `national`, step 2's check 2e asserts the support is the cross product, and a rebuild under
> the old flag fails rather than passing quietly.
>
> **The live consequence recorded here is now moot.** It warned that `load_cell_counts` weights the
> cell-rung first stage by `n_cells`, which under `national` is uniform, so the `county × NAICS2` and
> `county × NAICS4` rows of `instrument_first_stage.csv` would move. That file is `14`'s and is
> legacy; the affected rows are §A.4's Table IV, which is labelled stale there. Nothing in the
> settled pipeline reads cell counts.

**12, 14, 15, 16, 18 — the five exploratory exercises, all done and all now in `APPENDIX.md`.**

| item | date | what | now |
|---|---|---|---|
| 12 | 2026-08-12 | the county set screened five ways instead of reweighted, `23_sample_screen_grid.py` | §A.7 |
| 14 | 2026-08-12 | the county set screened on the exposure's own right tail, `24_exposure_screen_grid.py` | §A.8 |
| 15 | 2026-08-13 | the rolling exposure held at a fixed distance from the sample, `rolling_t5`, and Tables V–IX re-cut on it | §A.1, §A.5 |
| 16 | 2026-08-13 | winsorizing the outlying counties instead of dropping them, `26_winsor_screen_grid.py` | §A.9 |
| 18 | 2026-08-13 | the raw, dropped and winsorized grids at `h = 3` and `h = 4` — **no new run**, a view of two frozen CSVs | §A.10 |

Their durable findings are promoted to **§1.8.14**; the reversals and dead ends stay in the appendix
entries. Two new inputs came out of item 12 and are still in use:
`code/reference/county_cbsa_2003.csv` from `21_ingest_cbsa.py`, and `population_ccg_year.parquet`
from `12`, which now also reads CAINC1 LineCode 2. Item 16 needed
`25_ingest_bea_employment.py`, because its window rule needs 1969–74 employment and EFSY starts in
1975 (§1.7.1).

**19. NEW — the EFSY negative-cell deletion, 2026-08-17.** Cited four times elsewhere in this file
and never written; written here.

> **Assembled from §1.6.10 item 3, `11_build_employment_cells.py`'s `--negative-cells` flag and the
> run log — not from a witnessed run.** The distinction matters and is why this reads as a pointer
> rather than as a run record. Anyone needing the run's own numbers should rerun `11` and replace
> this paragraph with them.

EFSY's negative employment cells are **deleted at four digits**. The two-digit and county rungs are
summed from the cleaned cells, so no rung carries a negative; `--negative-cells keep` recovers the
previous build, which conserved raw EFSY exactly. §1.6.10 item 3 records what it does and what it
costs: **134.44 employee-years changed by hand**, and 826 county-years over 1981–1997 move, which
shifts every fit with employment growth on the left by 4.1e-05 relative at worst.

**It is a declared departure from A.3.1 in spirit.** No cell leaves the panel and no baseline unit
enters or leaves — the cell returns as a zero when the panel is rectangularised — but values were
changed by hand. step_2's `load_rung` now asserts `emp >= 0` on every file it opens as a **hard
failure**, which is what replaces the old negative-base branch: with the branch deleted, a stale
employment build would silently produce a negative weight and an omega that does not sum to one.

**Two consequences.** Every §1.8 figure in `APPENDIX.md` predates this and was measured on the 3,101
universe (§A.0's banner). And **step_2's `27` and `28` had not been rerun as of that date** — their
`TableD1*` and `TableC1` outputs are the older vintage; step_2's README §4 item 8 is the record.

### Retired machinery — kept as recorded traps, not as work

These three were open against `14`, `19` and `20`. **None touches `41`–`55`**, which pin to the
candidate `.dta` at exactly 0.0 and run `reghdfe` at `tolerance(1e-12)`. They are kept because the
traps recur, not because anything is owed.

**9. `14` overwrites its own outputs on every variant run.** `--no-loo`, `--delta flow`,
`--floor 25000`, `--exposure` and `--periods` subsets all write to the same
`instrument_ccg_year.parquet` and `instrument_first_stage.csv`. §A.3's flow table was produced by
backing both up first and restoring afterwards.

> **Still live for step_2's `22`/`27`/`28`**, which import `14` and rebuild `z_fixed_7074` in memory
> — that is exactly why `22`'s assertion −1 exists and why it records both input parquets' mtime and
> sha256, the git SHA and the library versions. Moot for the settled pipeline: `16` sidesteps the
> hazard by building in memory and writing only a comparison file, `19`, `20` and `23` do the same,
> and `51` writes to its own path and pins to a git-tracked reference.

**11. `14`'s fits are not bit-reproducible.** `bi.fit_rung` takes pyfixest's default demeaner, whose
alternating projections stop on a tolerance rather than at a fixed point, so repeated canonical runs
move at the eighth significant figure — three runs of `19` agreeing with the same frozen anchor to
0.0, 1.4e-14 and 5.4e-08. The fix is one keyword argument at `14`'s two `pf.feols` calls; the reason
it was never applied is that doing so rewrites `instrument_first_stage.csv` and every table §A.4
sources from it. `fixef_rm` must be left alone: pyfixest 0.60.0 already defaults to `"singleton"`.

> **Refined 2026-08-12: the pin bounds this within a session and not across them.** Two consecutive
> runs of `20` that day gave a byte-identical CSV, but against the copy written on 2026-08-11 the
> same file differed by up to 1.2e-12 relative on *F*. Read any "byte-identical" claim in
> `APPENDIX.md` as within-session. Moot for the settled pipeline, which estimates in Stata.

**13. `20`'s check 2 is vacuous.** It calls `sym.only_added`, which closes over `19`'s own copy of
`14` — a different module object from the one `g20.register_variants` mutates, because
`importlib.util.module_from_spec` builds a fresh module on every `_load`. The check compares a
pristine snapshot against a pristine dict and passes whatever happened. `23` restates `only_added`
against the module it actually mutated. `20` was left alone deliberately: editing the script behind
a frozen table to fix a check that has never failed is the wrong trade.

> **Kept because the trap recurs.** The same module-identity mistake is what made `23`'s first run
> raise `KeyError: 'pre-h1'`, which is how it was found. Any script that `_load`s another and then
> mutates its dicts inherits it.

### Closed

~~**The Pallante test.**~~ **Closed** (§3.4). Against his series as published, 29 of the 41
overlapping years are exact in all 51 states. Correcting the sign byte on both sides takes that to
**37 of 41**, with four named deviations totalling \$45,230,684 — 0.0011% of his FY1966–2006 total.
What used to be recorded as an open \$10.317bn "record-universe difference" in his unavailable MPCF
extract was one byte, and the extract is no longer needed to explain it. The test has caught two real
errors: the FY1984 CA/NY record, corrected in the data, and the sign byte, which was ours.

**Worth telling him.** His FY1966–75 state-year totals are high by \$12,548,756,000 — 2.3% to 4.9% a
year — because `as.numeric()` reads the blank sign byte as padding. It is a one-line fix on his side
and it would make eight of those ten years agree with an independent extract to the dollar. Our
contract files are also what his `08_StataRoutineAggregation_NS14.R` needs to run at all, so there is
something to trade.

~~**The time-consistent county set.**~~ **Done** (§1.6). `county_fips5` is contemporaneous, and
`code/reference/county_ccg_crosswalk.csv` maps all 3,103 county codes the 42 years use onto **3,101
fixed units — 3,056 single counties and 45 groups**, time-invariantly, with dollars conserved exactly
in every year and every unit-year carrying an employment denominator bar two declared cells holding
\$0.70m between them (§1.6.7). The 13-row table of §3.3 turned out to be about a fifth of it: the
rest is counties created or abolished inside the panel, Virginia, a source that keeps using retired
codes for up to twenty years, and three codes that are not counties at all.

~~**§1.6.6's three open items.**~~ **Closed** (§1.7.2): the reporting-threshold floor is **none**, as
of 2026-08-10, with `--floor` to change it; the contract panel is rectangularised on the universe
rather than built from observed rows; and the negative county-year cells needed no rule at all once
the shock was scaled by income — 780 of them at no floor, and only 764 of those survive as negative
shocks once `G` is cumulated. The one place the crosswalk knowingly puts money in a unit that did not
hold it is still Broomfield CO, \$0.14m, guarded and switchable.

~~**The parse.**~~ **Verified 2026-08-06**, all 42 years, every check passing, in all five scripts.
The FY1966–75 dollar amounts moved on that date and only they: the negative-zero sign byte was
corrected (§1.5), FY1966–75 were re-parsed, and `06` and `08` were rerun and returned their previous
results unchanged.

---

## 5. Two notes on reading this file

**On numbers that appear in more than one place.** Where §0 quotes a figure it is quoting §1.8.12 or
§1.9, and those are the sources. `APPENDIX.md` quotes nothing from here — its numbers are its own and
predate the settled design.

**On what would invalidate what.** The settled first stage rests on `41`'s panel and the candidate
builder's re-scaling of it. If either is rebuilt, §1.8.12's 72 fits and §0.3 move together, and
`53`'s check 15 — the pin to `code/reference/ladder_first_stage_sxy_v1.csv` — is what will say so
rather than letting them move quietly. §1.6 through §1.9 sit upstream of that and would move it too;
§3's verification and §3.4's Pallante test are independent of every instrument choice and would not.

---

## 6. WS2 (NETS) progress

**Step 0, the protocol gate: done, 2026-09-01.**
`Results/nets_benchmark/ws2_step0_protocol_gate.md` answers the eleven
pre-code questions for the NETS context and records the gate verdict (pass,
conditional on four rules the code must carry: one universe rebuilt at both
rungs; the actual-employment cutoff fixed before the step-2 benchmark numbers;
unmatched county FIPS fail loud; no `ccg x year` FE at the disaggregated rung).

**Producers 63-66 are not to be re-run.** The completed staging is on Dropbox
at `_data/temp/nets_fed2025_staging/`; `estab/estab_year_panel.parquet` is
present and materialized with 765,300,955 rows, matching `manifest_66`. Note
the path: the Dropbox project root, not `4_Macro_version/data/temp/` as the
WS2 Inputs line says. The `long/` and `firm/` tables are online-only
placeholders and hydrate on first read. DuckDB (1.5.5) is installed and is
what both step-1 producers run on.

**Step 1, first half, the NETS-to-CCG bridge: done, 2026-09-01.**
`step_1/code/py/72_nets_ccg_bridge.py`, memo
`Results/nets_benchmark/ws2_step1_bridge_report.md`. Every FIPS in the panel
places into a ccg: no `unmatched_other`, no Connecticut 2022 planning regions
in the data (only the eight legacy counties, 1990-2023), and the Alaska 2019
codes `02063`/`02066` never appear, so that patch is inert. 3,100 ccgs are
observed against the locked 3,058, and no locked ccg is empty of NETS
establishments.

**Step 1, second half, the cell collapse: done, 2026-09-01.**
`step_1/code/py/73_nets_cell_collapse.py`, memo
`Results/nets_benchmark/ws2_step1_cells_report.md`, ~15 minutes end to end.
It imports 72's classifier rather than re-implementing it, and arms the
fail-loud gate: any `unmatched`/`unplaceable` FIPS raises before anything is
written. Two outputs, under the Dropbox project's `_data/temp/nets_cells/`:

| File | Key | Rows |
|---|---|---:|
| `nets_ccg_year.parquet` | `universe, ccg, year` | 210,800 |
| `nets_ccg_industry_year.parquet` | `universe, ccg, year, naics4, sic2` | 44,404,379 |

These two files are the ONLY NETS input steps 2-5 read; the 765M-row
establishment panel is not opened again.

Three construction choices, all following from the step-0 gate:

- the fine file is keyed on `(naics4, sic2)` **jointly**. Each establishment
  carries exactly one NAICS and one SIC, so this is still a partition of the
  county-year and re-collapses exactly to the NAICS4, NAICS2 and SIC2 rungs
  and to the county file — the whole A.1.7 fineness ladder out of one pass;
- `universe` is a column, `all` and `actual` (`any_actual_emp`), both rungs
  grouped from the same retained establishments, which is gate rule 1. The
  cutoff is fixed here, ahead of any step-2 number, which is gate rule 2;
- every stored measure is a sum or a count (employment, establishment and line
  counts, `emp_imputed`, the 4-way EmpC composition from the primary line, and
  the small/education/blacklist/quarantine diagnostics). No shares: a ratio
  would not re-collapse. Nothing is dropped for being small or imputed.

The county file is aggregated INDEPENDENTLY from the panel, not re-collapsed
from the fine file, so the gate's Q8 additivity check compares two independent
aggregations instead of asserting itself. It passes: `max |E_ct - Σ_m E_cmt| =
0` exactly on employment and on all 20 other measures over all 210,800
universe × ccg × year pairs, `emp_imputed` alone at 1.9e-09 (it is the one
fractional sum). Row and employment conservation pass; the EmpC = 0 share is
42.156% against the 42.5% in `NETS_FED2025.md`.

Coverage loss, to be priced against EFSY in step 2, not absorbed: 18,478,569
establishment-years (2.41%) are absent from both files — 17,316,714 with a
blank FIPS (no county at all; the decision was to drop rather than recover
them from the address spells) and 1,161,855 in the territories.

**One finding step 2 must handle.** The `actual` universe's share of placed
employment is itself a coverage series, and it breaks: 58.4% in 2021, 32.2% in
2022 (-26.2 points), against a 1990-2021 range of 58-79%. A window straddling
that break measures who reports an actual figure, not employment. It sits
outside the contract window (FY2018) so the headline is unaffected, but any
NETS exhibit reaching 2022-23 in the `actual` universe has to say so.

**Step 2's BDS benchmark input: ingested, 2026-09-01.**
`step_1/code/py/74_ingest_bds.py`, memo
`Results/nets_benchmark/ws2_bds_ingest_report.md`. Ingest only — it computes
no growth rate and no benchmark, because the pass criterion the gate requires
has not been written yet. Census BDS release **2023**, pinned in
`dc.BDS_RELEASE` (each release re-estimates the whole 1978-2023 series, so
mixing releases mixes two estimates of the same year); the three source CSVs
are hashed in the memo. Five parquets, the CCG-rung pair being
`bds_ccg_year.parquet` (142,278 rows) and `bds_ccg_sector_year.parquet`
(2,703,282). The county geography goes through producer 72's classifier,
imported, so the vintage rules have one owner; the fail-loud rule is armed.

**BDS is local-first, and that is an exception to the project's Dropbox-only
data rule** (`CLAUDE.md`: "There is no repo-local data fallback"). Decided
2026-09-01. The canonical copy — source CSVs under
`4_Macro_version/data/source/BDS/2023/` and parquets under
`4_Macro_version/data/temp/bds/` — is in the checkout, and **every downstream
consumer must read it through `dc.bds_path(name)`**, which returns the local
path and raises if the file is absent. It deliberately does not fall back to
the Dropbox mirror: a run that silently read the mirror would be reading
whatever another machine last wrote. The Dropbox copy at `_data/temp/bds/` is a
mirror written after the local files, for a co-author without the checkout; a
failed mirror is reported and does not fail the run, and `--no-mirror` skips
it. The exception is cheap here — about 300 MB of public, re-downloadable CSV —
and it takes a network and a Dropbox mount out of every step-2 run.
`4_Macro_version/data/**` is gitignored, so none of it is committed.

Four findings step 2 has to carry into its criterion, all in the memo:

- **the industry axis is BDS `sector`, not NAICS2.** The published
  county-industry table is county × sector: 19 codes, three of them ranges of
  NAICS2 (`31-33`, `44-45`, `48-49`). There is no county × NAICS2 BDS table
  and the finer national tables carry no geography, so the NETS side must be
  collapsed to those 19 codes and the criterion written against sectors. This
  is a coarsening of the plan's stated axis;
- **the industry rung is a third invisible.** `estabs_entry` is suppressed in
  1,025,564 of 2,791,556 county-sector-years (36.7%), against 1.6% at the
  county rung. Suppression sits precisely on the small cells the NETS
  comparison is about, so this is not a benchmark on the full universe;
- **BDS has no Connecticut.** The 2023 release retro-codes Connecticut onto
  the nine 2022 planning regions for every year back to 1978, so `09001`-`09015`
  appear nowhere in the file. With 72's refusal to map planning regions, 8 of
  the locked 3,058 ccgs have no BDS row in any year; the intersection is 3,050;
- **only six count columns add up across geography**, established by summing
  BDS's own county file to its own state file rather than assumed: `estabs`,
  `emp`, `estabs_entry`, `estabs_exit`, `firmdeath_estabs`, `firmdeath_emp`,
  exact to the unit in every clean state-year. Firm counts cannot be summed (a
  firm in two counties is one firm in each and one in the state; off by up to
  50,075 in a state-year) and continuer job flows miss the establishment that
  changes county. The four columns the entry benchmark needs are all in the
  exact set. Non-additive columns are carried on the CCG spine only where the
  group is a single county.

Suppression parses to NaN and is never coerced to zero; BDS's statewide-unknown
`XX999` rows (0.17% of establishments, 1.8% of employment) get their own status
and stay off the CCG spine rather than being dropped silently.

Steps 2-5 are unblocked.

**Step 1, third piece, the birth cell collapse: done, 2026-09-01.**
`step_1/code/py/75_nets_births.py`, memo
`Results/nets_benchmark/ws2_births_report.md`. Producer 73 collapses stocks and
carries nothing about flows, because a birth is a statement about an
establishment's whole history rather than a sum over an establishment-year, so
the entry side of the step-2 benchmark needed its own collapse. Written as a
sibling of 73 rather than an edit to it, importing 73's bridge, universe rule
and `UNKNOWN` sentinel so the four cell files key identically; re-running 73
would have restated numbers already cited elsewhere.

An establishment is born in the first year it appears in the panel. **The 1990
cohort is left-censoring, not entry, and is excluded**: 10,106,064 of 95,437,691
establishments (10.6%) first appear in 1990, the panel's first year, and almost
all of them opened earlier. NETS's own `FirstYear` is deliberately not used --
it is D&B's belief about when the business started, a different object from
appearing in the frame, and not what BDS measures. 83,348,596 births 1991-2023
place into a ccg (97.68%); additivity of the county file against the industry
file passes exactly.

**Step 2, the benchmark: done, 2026-09-01.**
`step_1/code/pipeline/10_nets_benchmark.py`, opening the pipeline's 1X band.
Deliverable: `Results/nets_benchmark/ws2_step2_benchmark.pdf`, which carries
every table, both figures and the divergence document in one file; the
component `Table_*.tex`, the two figures, `nets_benchmark_stats.csv`,
`nets_benchmark_divergence.csv` and `ws2_step2_benchmark_report.md` sit beside
it. Runs in about 45 seconds off the four cell parquets; the establishment
panel is not read.

**No pass criterion was pre-registered.** The plan asked for one and the
decision taken on 2026-09-01 was to drop it, so step 2 is descriptive and
**the step-5 gate is left open**. Every statistic a criterion would have been
written against is printed. A criterion written now would be written against a
sample whose numbers are visible, which is not the same object the plan asked
for; if step 5 is to be gated, that has to be faced explicitly.

The support came out at all 3,058 locked ccgs (no locked ccg is missing from
either NETS or EFSY); 3,050 of them have BDS rows, the eight missing being
Connecticut.

What the benchmark found:

- **levels.** The full NETS panel carries 1.50x EFSY employment on average,
  drifting 1.33 -> 1.83 across 1990-2018, and 1.8x -> 4.3x as many
  establishments. The actual-employment subsample is much closer on employment
  (1.05x average), which is the first argument for the gate's choice of it as
  the headline universe. Some of the ratio's movement is EFSY's own: EFSY's
  county total moves more than 5% between consecutive years in 2009 (-5.3%)
  and 2017 (-11.5%), and the report flags those years so no reader charges
  them to NETS;
- **growth.** Midpoint growth at h = 2, actual universe, correlates 0.20
  (Spearman 0.24) across 82,566 county-years, NETS running +0.024 above EFSY at
  the median and agreeing on the sign in 61% of them. At NAICS4 the correlation
  is 0.02 and sign agreement 42% -- worse than a coin toss, because a cell one
  source calls entry the other often calls a continuing establishment. The
  county cut by size moves the correlation only 0.15 -> 0.33 from the smallest
  to the largest employment decile: the disagreement lives in cells, not
  counties;
- **the mechanism is the small-cell mass.** At NAICS4, 52% of NETS pairs and
  63% of EFSY pairs have a zero endpoint. Those are the cells the log rate
  cannot take and the standard rate divides by, which is why their N falls by
  millions between the panels and why the standard rate's weighted mean gap
  runs to double digits while the midpoint rate's stays near 0.02. Midpoint
  growth is bounded in [-2, 2] and cannot do that;
- **imputation is the cleanest single predictor of the gap.** Median gap by
  the cell's imputed employment share, NAICS4, all universe: 0.000 (none),
  0.023 (<25%), 0.028 (25-50%), 0.080 (50-75%), 0.138 (75-100%), 0.287 (fully
  imputed). NETS's excess growth is an imputation artefact almost in
  proportion to how much of the cell is imputed;
- **entry.** The county birth rate correlates 0.14 with BDS entry over 84,573
  county-years, and 0.04 at the sector rung. Aggregate rates are close on the
  full panel (0.081 vs 0.097 at the median) but the year-by-year ratio swings
  from 0.55 to 3.07: NETS births arrive in bursts as D&B discovers
  establishments, and a burst year is not an entry wave. Any regression of a
  birth outcome on a year-varying shock inherits that. This is the single
  strongest argument against step 5 as specified;
- **the two sources' aggregates behave completely differently across samples,
  and that asymmetry is itself the finding.** BDS's aggregate entry rate sits
  at 0.109-0.112 in all eight rung x universe x rate cells. It is a
  mass-weighted ratio, so it is set by the largest cells, and those are in
  every sample: switching the NETS universe at the county rung moves the BDS
  establishment base 176.5M -> 176.8M (0.1%), and the sector rung loses 36.8%
  of its CELLS to suppression but only 5.0% of its establishment base, because
  `estabs_entry` is additive across sectors and what is withheld is small
  cells. NETS's aggregate on the same samples moves 0.026 -> 0.117, a factor of
  four, purely from which establishments are retained. A four-fold swing in an
  outcome's aggregate level driven by a universe choice is a universe artefact,
  and it is the cross-section counterpart of the burst pattern above. The
  report explains this in place rather than leaving the steady column to be
  misread as agreement;
- **the entry rung is BDS `sector`, not NAICS2**, because BDS publishes no
  county x NAICS2 table and its finer industry tables carry no geography; and
  BDS withholds `estabs_entry` in 36.8% of the county-sector-years on this
  support, so the sector statistics describe the unsuppressed large-cell
  remainder;
- **scope, not growth.** 44 NAICS4 codes appear in NETS and never in EFSY
  (411.8M employee-years, mostly NAICS 99 and 92) and 35 the other way (353.0M).
  Coalescing either to a zero on the other side would print midpoint growth of
  +2 in every county that has them, so they are held out of every statistic
  and sized separately rather than deleted.

`05_baseline_tieout.py` PASS afterwards, every vector at 0.0e+00 and 30/30
frozen rows tying: nothing locked moved.

**Step 2b, the three follow-up cuts: done, 2026-09-01.** New producers
`step_1/code/py/77_nets_births_empc.py` (the birth flow re-collapsed by the
primary line's EmpC code, a sibling of 75 that asserts its counts re-collapse to
75's county file exactly) and `step_1/code/pipeline/11_nets_birth_divergence.py`.
Deliverable `Results/nets_benchmark/ws2_step2b_divergence.pdf`. Stage 11 imports
producer 10 for the support, the universes, the entry rate and the growth rate,
so it cannot drift from step 2; **no pass criterion**, for step 2's reason.

Three questions, asked because step 2 established THAT the sources disagree
without saying where the disagreement comes from:

- **the birth gap is a small-CELL gap, and by industry a COMPOSITION gap.** The
  same pairs, partitioned three ways with BDS the reference. At the county x
  sector rung the smallest surviving cells (2-4 BDS establishments at t-1) give
  a NETS birth rate of 0.086 against BDS's 0.039; the largest, 0.115 against
  0.112. Small counties run the opposite way and carry no mass: the smallest
  decile of the BDS establishment base has NETS *below* BDS (-0.025) and 0.4% of
  national entry, the largest is +0.010 with 70.4%. By sector the 19 terms
  nearly cancel to +0.0017, and what remains is composition rather than a rate
  disagreement -- administrative and support services alone contributes +0.0170,
  of which +0.0160 is composition, because NETS puts 21.3% of its births there
  against BDS's 6.3% of entry. Apportioned by a shift-share of the aggregate
  rate gap into a within term and a composition term, additive and asserted
  against the aggregate. A common-denominator difference is deliberately not
  reported: NETS carries 1.8-4.3x BDS's establishment count, so such a column
  ranks groups by size and says nothing;
- **the year-to-year swing is imputation, not entry, and the "cycle" is a
  sawtooth.** Split by EmpC, D&B's own estimate (code 2) carries 51% of the
  variance of the aggregate birth rate and Walls's estimate (code 3) 48%,
  against 1% for establishments that actually reported a figure (code 0). Step
  2's `actual`/`all` universes cannot see this, because `any_actual_emp` is a
  two-way cut that pools D&B with Walls. The aggregate NETS birth rate has
  AC(1) = -0.04 where BDS entry on the same counties and years has +0.84, and
  its SD is 0.051 against BDS's 0.011. Every class is individually persistent;
  the negative total comes from the cross-lag term, Walls at t against D&B at
  t-1, at -0.32 -- the two imputation engines take turns on the same discovery
  flow. That is what a discovery process looks like and not what an entry wave
  looks like, and it sharpens step 2's burst finding into a mechanism;
- **the flat binscatter is a near-zero correlation, not a NETS growth rate that
  barely moves.** At h = 2 the median annual cross-sectional SD of county
  midpoint growth is 0.093 in NETS against 0.099 in EFSY on the headline
  `actual` universe, a ratio of 0.94. On the full panel NETS *is* narrower --
  0.075 against the same 0.099, a ratio of 0.75 -- because pooling in the
  imputed establishments smooths the county aggregate rather than roughening it.
  Neither ratio resembles the collapse the binscatter appears to show. Step 2's
  `Figure_nets_efsy_scatter.pdf` bins on EFSY and plots the conditional mean of
  NETS, which necessarily collapses onto a flat line at the unconditional mean
  when the correlation is near zero: it is a picture of the correlation, not of
  dispersion, and the flat points should not be read as NETS barely moving.
  `Figure_ws2b_growth_lines.pdf` plots each source's own marginal instead, in
  three rows -- aggregate, median with the IQR, and the cross-county SD -- by
  the two horizons. **That figure IS the year-by-year record for the section**:
  the four tables it replaced were unreadable as tables and the digits live in
  `ws2b_growth_by_year.csv`. One reading note carried in the report: the median
  EFSY county falls -0.66 in 2017 and -0.58 in 2018 at h = 2, EFSY's own break
  -- step 2's year cut carries the same numbers -- and its cross-county SD
  spikes to 0.42 in the same years, so the figure's median and SD rows are
  clipped with every off-scale point labelled while the CSV is unclipped.

`05_baseline_tieout.py` PASS afterwards: neither producer writes anything a
locked producer reads.

**Step 3, the locked baseline and the disaggregation test on NETS: done for
Tables 1-5, 2026-09-01.** New producer `step_1/code/py/76_nets_rung_loader.py`
puts producer 73's long cell parquets into `load_rung`'s shape -- one dense
array per year on a support identical in every year, which means materialising
the full ccg x industry cross product with explicit zeros, because the absent
cells are the extensive margin and are the object of study. `mpgr_disagg.py`
gained a `source` parameter (`efsy` / `nets_actual` / `nets_all`) threaded
through 34, 35, 40, 37 and the renderers 43, 44, 45; `efsy` leaves every
pre-existing filename and number untouched, and the cell cache is now keyed and
fingerprinted on the source so an EFSY cache can never be handed to a NETS run.
Report `Results/nets_benchmark/ws2_step3_baseline_report.md`, exhibits in
`Results/nets_baseline/`, preview `preview_tables_nets.pdf`.

The four results: the midpoint invariance survives the outcome swap exactly
(flat to 1e-11 at both rungs, both universes); the log/OLS cross-level gap is
far larger under NETS and changes sign (+0.212 at NAICS4 on a county
coefficient of -0.124, against -0.199 on +0.771 under CBP), and Table 5
attributes essentially all of it to the Jensen term; the county coefficient
itself does NOT survive (+0.935 CBP against -0.122 NETS on the same 76,450
county-years, so this is a source difference and not an era one -- an
era-matched CBP column is produced for exactly this reason, via 34's new
`--year-floor`); and rung 0' moves the level without disturbing any of it
(`46_table_rung0prime.py`). NETS begins in 1990, so h=4 starts at 1994 and the
swap costs 36,696 county-years; no county is lost and no A.3.4 exclusion binds.

Two things to know before continuing. **The COVID rule is vacuous**: the
composition instrument stops at target year 2018 because the contract data end
FY2018, so no window reaches 2020-21 and nothing was rebuilt to make one.
**Table 6 is not delivered and the producer refuses it by default**: Poisson's
invariance needs a free unit effect to absorb log M_c, and the locked
baseline's division-by-year effect is coarser than the county and cannot, so
the ladder is genuinely not flat there (measured gap 2.5e-07 against a 1e-9
tolerance; ccg + year is exact at 4e-19). `37_table_poisson.py --frame
baseline` therefore requires `--unit-fe ccg` or `--unit-fe divisionyear` and
has no default. Tables 1-5 are unaffected -- their invariance is an exact
weighting identity that holds under any effect coarser than the unit. A run at
`--unit-fe ccg` reproduces the invariance at NAICS2, but `validate_poisson`'s
pyfixest benchmark and the NAICS4 pass are slow enough on the baseline frame
that the table has not been written yet.

`05_baseline_tieout.py` PASS afterwards: 19 constants, every vector at
0.0e+00, 30/30 frozen rows tying.
