# The composition-instrument pipeline

The canonical chain for the macro fiscal-multiplier application. Four stages,
one final dataset, one constants file, and banded numbering so nothing ever
needs a global renumber: 00-09 construction and its gate, 1X the NETS work,
econometric analysis from 2X up (2X the shift-share tests, 3X reserved for
the disaggregation tests, 4X reserved for the aggregation tests, 5X reserved
for dynamics). Everything here is proven numerically equal to the frozen
results (see The gate below).

```
raw per-FY parquets (DCADS 1966-2006, USAspending 2007-2018)
        |
   01_contract_panel.py        county x year dollars (net & positive);
        |                      harmonized top-800 county x category x year
   02_county_panel.py          income, population, employment, rung labels
        |
   03_composition_instrument.py  category shocks g_kt; predicted dollars L;
        |                        z = L / P_{t-h}; intensity z_int
   04_analysis_panel.py        analysis_panel.parquet  <- THE final dataset
        |
   05_baseline_tieout.py       standing correctness gate (run any time)
   06_fsc_industry_concordance.py  estimator of the FSC-to-SIC/NAICS
        |                      reference tables; default mode verifies the
        |                      committed CSVs byte-for-byte, --write
        |                      regenerates them
   07_naics_vintage_concordances.py  the Census NAICS-to-NAICS chain
        |                      (1997->2002->2007->2012->2017) and the
        |                      vintage-stable 4-digit axis the EFSY cells sit
        |                      on. Downloads and parses the four Census
        |                      workbooks; same verify/--write discipline as 06
   08_fsc_concordance_on_efsy_axis.py  06's concordance restated on 07's axis.
                               Goes back to 06's own year-level cells (imported,
                               not reimplemented) because the FY2001-06
                               estimation window is TWO vintages
   09_usaspending_recipient.py  WS3 step 0: recipient identity for every
                               USAspending transaction. Producer 31 selects 54
                               of the archive's 297 columns and NONE of them is
                               a recipient identifier, so the match in WS3 step
                               1 has nothing to join on. This re-reads the same
                               archives in the same row order and writes 22
                               identity columns (DUNS, UEI, CAGE, names, parent,
                               address, the two place-of-performance districts)
                               to a SIDECAR keyed to 31's rows, never into the
                               locked store. Asserts row-for-row agreement with
                               usaspending_year/fy<year>.parquet on every run.
                               Writes LOCAL parquets under 4_Macro_version/
                               data/temp/usaspending_recipient/ (see the README
                               there). Creates no estimand; the DELIVERABLE is
                               the identifier coverage report in
                               batch_logs/macro_version/, which is reported and
                               never asserted. --dry-run checks headers only
        |
   10_nets_benchmark.py        WS2 step 2: how problematic is NETS. Opens the
                               1X (NETS) band. Benchmarks the NETS employment
                               STOCK against EFSY -- levels, and midpoint,
                               standard and log growth at the county, NAICS2
                               and NAICS4 rungs, h = 2,3,4, both universes --
                               and the NETS establishment BIRTH flow against
                               BDS entry at the county and industry rungs.
                               NO PASS CRITERION IS PRE-REGISTERED (decided
                               2026-09-01): this is descriptive and gates
                               nothing, and the report says so. The support is
                               the locked x NETS x EFSY intersection, fixed
                               once before any statistic (gate Q7); it comes
                               out at all 3,058 locked ccgs.
                               THE ENTRY RUNG IS BDS `sector`, NOT NAICS2.
                               BDS publishes no county x NAICS2 table and its
                               finer industry tables carry no geography, so
                               the NETS side is collapsed onto the 19 sector
                               codes and the note says so. BDS withholds
                               estabs_entry in 36.8% of the county-sector-years
                               on this support; those cells leave the pair and
                               are never zeroed.
                               CODES ONE SOURCE HAS AND THE OTHER NEVER DOES
                               (44 NAICS4 codes NETS-only, 35 EFSY-only) are
                               held out of every statistic and sized instead:
                               coalescing them to a zero would print midpoint
                               growth of +2 in every county that has them.
                               Reads only the four cell parquets from
                               producers 73 and 75, never the 765M-row panel.
                               Results/nets_benchmark/, and the whole thing in
                               one PDF, ws2_step2_benchmark.pdf
        |
   11_nets_birth_divergence.py WS2 step 2b: the three follow-up questions to
                               stage 10, on stage 10's sample and estimands
                               (both are IMPORTED, not re-implemented, so this
                               stage cannot drift from them).
                               (1) WHERE the NETS/BDS birth gap lives: the same
                               pairs partitioned by BDS sector, by county size
                               decile and by county x sector cell size, with
                               BDS the reference on both axes of the figure.
                               The apportionment is a shift-share of the
                               aggregate rate gap into a WITHIN term (the two
                               sources disagreeing about the rate inside a
                               group) and a COMPOSITION term (NETS's
                               establishment base sitting in different groups),
                               additive and asserted against the aggregate. A
                               common-denominator difference is deliberately
                               NOT printed: NETS carries 1.8-4.3x BDS's
                               establishment count, so such a column ranks
                               groups by size and says nothing.
                               (2) WHETHER the year-to-year swing is entry or
                               imputation: the birth flow split by the primary
                               line's EmpC code (0 actual / 1 range bottom /
                               2 D&B estimate / 3 Walls estimate / missing),
                               which the actual-vs-all universe split cannot
                               do because any_actual_emp pools D&B with Walls.
                               (3) NETS vs EFSY growth as MARGINAL time series,
                               because stage 10's binscatter bins on EFSY and
                               so plots a conditional mean, which collapses to
                               the unconditional mean when the correlation is
                               near zero -- a picture of the correlation, not
                               of dispersion.
                               NO PASS CRITERION, for stage 10's reason.
                               Needs step_1/code/py/77 for the EmpC birth file.
                               Results/nets_benchmark/, whole thing in one PDF,
                               ws2_step2b_divergence.pdf
        |
   20_shift_share_tests.py     the shift-share tests, both tables in one
        |                      runner: Table_many_shifters (GPSS side) and
        |                      Table_bhj_shock_level (BHJ side)
   21_rotemberg.py             the GPSS Rotemberg decomposition of the
        |                      composition-instrument 2SLS on the FSC axis:
        |                      Table_rotemberg_fsc, six panels, plus the
        |                      per-category and per-year record CSVs
   22_bhj_assumption1.py       the BHJ Assumption-1 discharge: the balance
                               battery over shock-level period effects, the
                               shock-control ladder, and the exposure-robust
                               standard error. Carries CONTINGENCY, the
                               pre-committed decision rule, and refuses to run
                               from an uncommitted working copy
   23_industry_axis_cells.py   WS1 step 1: county x FSC dollars relabelled onto
                               the industry axis. Concordance for FY1966-2006,
                               the record's own naics_code for FY2007-18, both
                               allocations (probabilistic and modal), and a
                               dollar-coverage assertion by year. Writes LOCAL
                               parquets under 4_Macro_version/data/temp/
                               industry_axis/ (see the README there), not to
                               the Dropbox tree - they go there when the
                               instrument is built on them. --dry-run reports
                               coverage without writing
   24_efsy_axis_rung.py        WS1 step 2a: restates producer 11's EFSY
                               employment cells onto the same axis, as a NEW
                               rung naics4_axis (naics4_alloc is left alone).
                               11 rectangularises on a 303-code UNION support
                               that carries 4521/4529 AND 4522/4523 in every
                               year, so those cells lose 100% of their
                               employment across the 2016->2017 vintage change;
                               collapsing to 452X/517X removes it. 298 cells =
                               292 axis + 6 EFSY residual roots. --dry-run
                               asserts without writing (~6 min either way)
   25_industry_instrument.py   WS1 step 2b: the composition instrument on the
                               industry axis, at every locked constant except
                               the LEAVE-ONE-OUT national shock WS1 asks for.
                               Emits z_ind (headline), z_ind_noloo (pooled
                               shock, to split the axis change from the LOO
                               change) and z_ind_modal. NCAT is deliberately
                               not re-applied - 23 already runs on the locked
                               top-800 cut. Writes to a SIBLING store,
                               _data/temp/pipeline_industry/, so the locked
                               store is never shadowed. --top-n is the
                               concentration sensitivity
   26_industry_analysis_panel  joins those columns onto the locked
      .py                      analysis_panel and asserts the row count equals
                               locked.N_COUNTY[h]: same sample, new instrument
   27_industry_axis_comparison WS1 step 5: the four instruments side by side,
      .py                      2SLS / first-stage F / reduced form /
                               cross-correlation, in the 20-21-22 estimation
                               cell. Results/industry_axis/
   28_industry_shift_share    WS1 step 6: the pipeline-20 shift-share tests
      _tests.py                and 22's shock-control ladder, run on the
                               industry axis with inference clustered at the
                               FSC level. Three columns everywhere - the locked
                               FSC instrument, the industry axis under
                               probabilistic allocation, the same under modal -
                               and no intensity column, because every column is
                               a composition-type instrument.
                               HOW IT REACHES THE FSC CLUSTER. The industry
                               cells do not partition the FSC categories, so
                               there is no parent to look up. The SHOCK is
                               restated instead (lib/industry.restate_to_fsc):
                               g~_k = sum_j share_kj g_j, which puts the
                               shock-level machinery back on the LOCKED FSC
                               exposure tensor. Sample, share sum, FSC groups
                               and every cluster level are then the same
                               objects 22 uses, and the only thing that moves
                               between 22 and 28 is the shock - which is why 28
                               calls 22's ladder rather than forking it.
                               Exact where producer 23's era rule sources the
                               exposure window from the concordance (worst
                               3.3e-05 of mean |z| on 103,972 of 113,146 cells
                               at h=4), approximate on the three target years
                               whose window straddles the FY2007 native seam.
                               Measured per target year in
                               industry_fsc_restatement.csv, asserted on the
                               exact years, and no sample is restricted: the
                               pooled estimate moves 4.0532 -> 4.0754 against
                               an SE of 0.65.
                               The ladder is NOT this file's: it imports the
                               frozen ws0_contingency.py, which names WS1's
                               industry axis in carry_forward_on_stop, and runs
                               22's precommitment check against it.
                               Results/industry_axis/
   30_entry_margin_bins.py     WS2 step 4: the entry margin of county
                               employment growth split by an ex-ante industry
                               classification (Mian-Sufi non-tradables,
                               Rajan-Zingales external finance dependence).
                               Opens the 3X band. THREE windows are estimated:
                               both sides of the 1998 NAICS break, neither
                               straddling it, and the pooled full locked span,
                               which straddles it on purpose. The .tex prints
                               pre and full as two column groups and the CSV
                               carries all three (--window pre|post|full);
                               estimand approved in
                               Results/entry_margin/entry_formula_memo.md
```

Run order: 01 → 02 → 03 → 04, then 05 to verify, then 20, 21 and 22 for the
diagnostic tables, and 30 for the entry-margin exhibit. 10 is the NETS
benchmark and stands apart: it needs `step_1/code/py/72`, `73`, `74` and `75`
to have run, reads nothing the locked store writes, and writes nothing any
other producer reads, so it can be run or skipped at any point. 30 additionally needs
the three reference classifications built by `step_1/code/py/69`, `70` and `71`.
The industry axis is its own chain, 07 → 08 → 23 → 24 → 25 → 26 → 27, and 07/08
must run before 23 because they build the classification 23 maps onto; 07 needs
`xlrd` and `openpyxl` in the venv, and downloads through curl because this venv
has no certifi bundle. 24 is independent of 25-27 (it restates the OUTCOME side
and the instrument does not read it) and can be run or skipped on its own; 25
needs 23, and 26-27 need 25. 09 is independent of everything above it — it reads
the raw archives and the parsed `usaspending_year/` store and is read by nothing
in this pipeline, so it can be run at any time and skipped entirely; it is in the
construction band because it is ingest, not analysis. Nothing in this chain writes to the locked store or
edits `locked.py`, and both `05_baseline_tieout.py` and `20` are expected to keep
reproducing their frozen numbers exactly afterwards. The active verified-geography
baseline passes 05 with every vector at 0.0e+00 and 30/30 frozen rows tying. The
locked phase of 20 gives 3.431985 in both the county and shock-level formulations
on 107,030 county-years. `lib/store.py` is the one reader for the persisted
stores; 20's locked phase consumes only those stores, and its
characterization phase (`--phase old`) is the single remaining in-memory
rebuild, needed because the old constants (unharmonized, ten-year window,
net rule) are not persisted. Every runner uses the venv at `4_Macro_version/.venv` and
logs to `batch_logs/macro_version/`. Data outputs land in
`<Dropbox project>/_data/temp/pipeline/`; golden fixtures in `_golden/`
under it; committed frozen copies with SHA256s in `golden/` here.

## The final dataset

`analysis_panel.parquet`: LONG, one row per (ccg, year, h), sorted
(ccg, year, h). Columns: `z` (composition instrument), `z_int` (intensity
instrument, same locked window and rule), `x_pos`/`x_net`
(treatment under the positive/net contract rules), `y_mpgr`/`y_std`/`y_log`
(employment growth over [t-h, t]), `E_b`, `E_t`, `P_b`, `popw` (county
population in 1975, the fixed-weight input), `state`, `division`, `czone`,
`msa`, `cea`, `ea`. The row set per h is the locked estimation sample
(113,146 / 110,088 / 107,030 rows at h = 2/3/4 on 3,058 counties).

Exposure shares C̄ are not materialized per (t, h): they are one call to
`lib/instrument.window_sums` on the stage-1 category parquet (materializing
every window would duplicate the tensor ~40x).

## The lock

`locked.py` is the single source of truth for every constant, each with its
provenance. **For future edits, the pipeline is the authoritative copy.**
`lib/oldpath.check_lock` asserts `locked.py` against the old modules at
runtime (in 00 and 05), so drift in either copy fails loud. Precedence when
sources disagree: the values producer 57 actually executes > producer 54's
defaults as executed by 57 > anything in producers 55/56.

## The gate

`05_baseline_tieout.py` is a standing check; rerun it after any change to
`pipeline/` (and after any change to the old producers, which stay live for
the aggregation tests). Three layers, all fail loud:

- lock check (19 constants against the old modules);
- vector layer: `analysis_panel.parquet` against the golden county frames at
  RTOL_VEC = 1e-12 / ATOL_VEC = 0 (exactness by construction; loosening a
  tolerance is a report-to-Adrien event, not an iteration step);
- coefficient layer: all 30 rows of the frozen `aggregation_baseline.csv`
  re-estimated from the panel at RTOL_COEF = 1e-6.

The chain of evidence: the old code reproduces its frozen CSV (gap 0,
verified against the committed copy in `golden/`); the persisted panel
equals the old in-memory frame (gap 0 on every column, horizon, and seam
year); the estimation from the panel lands on the frozen numbers (worst
relative gap 8e-14). Per-module micro tie-outs live under `__main__` in each
`lib/` module and compare against the `_golden/` fixtures produced by
`00_capture_golden.py`.

## Geography baseline and frozen archive

The active baseline is `verified_production_geography_v1`. Its declaration is
`_golden/active_baseline.json`; stage 05 verifies the SHA256 of the contract
caches, stages 3 and 4 outputs, county fixtures, and the 30-cell aggregation
baseline on every run.

The reported-source geography design is stored at
`<MACRO_DATA_ROOT>/temp/pipeline/freezes/reported_source_geography_v1/`. It
contains the exact stage 3 and 4 outputs, the five registered key-table result
files, the aggregation baseline, producer 95, and the two routing tables.
`SHA256SUMS` covers all ten assets. Producer 96 creates this archive once and
otherwise performs an integrity check without overwriting it.

Producer 97 is the deliberate promotion entry point. It requires the archive
integrity check to pass, captures the verified golden vectors, rebuilds stages
3 and 4, records the 30-cell baseline, writes the active-baseline manifest, and
runs stage 05.

## What is in the old folder (`step_1/code/py/`, 67 files)

The old numbered producers are chronological, not pipeline-ordered. They
stay in place and unmodified; the aggregation-test producers still run from
there. Census:

| Bucket | Files | Status |
|---|---|---|
| Ingest and cleaning (raw → per-FY parquets) | 01-08 (DCADS), 29-33 (USAspending), 59-60 (category bridges) | live upstream of this pipeline |
| Geography, panel, outcome | 10, 11, 12, 13 | live upstream (11-13 feed `EMP_OUT_DIR` / `PANEL_OUT_DIR`) |
| Instruments and treatment | 14 (intensity), 51 (treatment + grid), 54 (composition constructor + grid) | construction logic extracted into `lib/`; 54 remains the engine behind 57 |
| Aggregation tests and exhibits | 57, 68, 52, 67, 61, 62 | live, on the old path (see migration trigger below) |
| Shift-share diagnostics | 55, 56 | superseded by pipeline 20 (see below) |
| Frozen specification-search grids | 15, 16, 17, 19, 20, 23, 24, 26, 28, 35, 38, 39, 40, 41, 43, 44, 46, 47, 48, 49 (+ Stata 36, 37, 42, 45) | **frozen, do not extend** |
| External benchmarks and replications | 09 (Pallante), 18 (Muratori), 25, 34, 50, 53 (Nakamura-Steinsson) | frozen references |
| Reference ingests | 21, 22, 27 | one-shot, outputs in `code/reference/` |
| NETS sub-pipeline | 58_ingest, 63, 64, 65, 66 (cleaning, frozen); 72 (FIPS→ccg bridge), 73 (the stock cell collapse), 75 (the birth cell collapse) (+ `nets_common.py`) | separate series, staged locally; 73 and 75 are the ONLY readers of the establishment panel, and pipeline 10 reads only their four cell parquets |
| Concordances | 58_fsc_industry_concordance | superseded by pipeline 06 (byte-identical port); frozen |

**What is 54.** `54_composition_instrument_grid.py` both defines the
composition-instrument construction (category panel, harmonization hooks,
shares, shocks, z) and runs a 163k-cell robustness grid over constructor
choices. Producers 55/56/57 importlib-load it and rebuild the instrument in
memory. The construction logic is extracted into `lib/{contracts,harmonize,
instrument}.py`, proven bit-identical; 54 itself is untouched and still
serves 57/61/62/52.

## Recorded inconsistencies in the old folder

These are facts about the old files, recorded so nobody re-derives them:

- 54's `HEADLINE` cell (54:141-143) uses a ten-year exposure window and the
  net rule; the lock (57's `BASELINE`, 57:539-542) is the five-year window
  and the positive rule. 54's county-grid headline CSV therefore sits on a
  different cell from the locked baseline.
- 55/56 run SPAN = 10 with `harmonize=False` (55:84-140, 56:72-129): their
  tables in `Results/ns14_diagnostics/` are built on the unharmonized axis
  at a non-locked cell, and 55's LaTeX header hardcodes "300 categories"
  against NCAT = 800. The pipeline's tables from 20 in
  `Results/pipeline_diagnostics/` supersede them; the old tables stay for
  provenance.
- 55/56 winsorize the treatment within year (`wins`); 57's locked
  estimation never winsorizes. The ports in 20 preserve the 55/56
  convention (`locked.WINSOR_DIAG_GROUP`); whether the diagnostics should
  instead run on raw x is an open decision for Adrien.

## Migration trigger

57/68 (the aggregation tests) keep running on the old path. They migrate to
`analysis_panel.parquet` **before any new aggregation number is produced for
the paper**; until then, 05's lock check guarantees the two paths cannot
silently diverge. When they migrate, 05's live old-path reference becomes
the archival record of the original construction and its fixtures in
`_golden/` are kept.

## Extending the pipeline

New work takes a number in its band and consumes the persisted parquets,
never the old importlib chain:

    00-09   construction (stages, the golden capture, the gate)
    1X      NETS (bridge, aggregation, benchmarks, contract merge; 10 the
            benchmark, 11 its three follow-up cuts)
    2X      shift-share tests
    3X      disaggregation tests (30 the WS2 step 4 entry-margin bins)
    4X      aggregation tests (reserved; where 57/68 land when they migrate)
    5X      dynamics / local projections (50 the county lead-lag LP,
            51 the same battery on WS1's industry axis)

WS2 step 3 sits OUTSIDE these bands on purpose. It re-estimates the step_2
disaggregation ladder on a NETS outcome, and rather than forking those producers
it adds an employment `source` to `step_2/code/py/mpgr_disagg.py`, fed by
`step_1/code/py/76_nets_rung_loader.py` (the 7X band, next to the NETS producers
72-75 it reads). So there is no 1X or 3X producer for it; look in step_2. The
one thing to know here: `--source efsy` is the default everywhere and leaves
every existing filename, note and coefficient untouched.

21 (the GPSS Rotemberg exhibit) is built. 22 (the BHJ Assumption-1 discharge)
carries its frozen `CONTINGENCY` and is awaiting sign-off; its tests 8-12 are
not yet implemented. 50 (the county lead-lag local projection, WS4) is built and
opens the 5X band; it is the first producer to construct the composition
instrument at h = 1, which the locked chain does not persist, and its estimand
was approved in `Results/lp/lp_formula_memo.md` before the file existed. 51 runs
that same battery with the shock relabelled onto WS1's NAICS-4 industry axis; it
IMPORTS 50 and calls `run_cell`, `horizon_fit`, `joint_test`,
`increment_lead_test` and the rest unchanged, swapping only `S["gk"]` and
`S["Z1_full"]`, which is the relationship 28 has to 22. The industry shock is
restated onto the locked FSC exposure tensor (`lib/industry.restate_to_fsc`) so
the county/shock-level identity is exact and the clusters stay the 800 FSC
categories; its scope was approved in section 9 of the same memo, and its
outputs go to the sibling `Results/lp_industry/` so nothing in `Results/lp/` is
rewritten. Its locked reference column is re-estimated rather than read, and
`anchor_locked` ties it to the frozen `lp_summary.csv` — that check is what
proves the wrapper faithful, and it currently reproduces F = 3.897953 exactly.
30 (WS2
step 4, the EFSY entry-margin bins) is built and opens the 3X band; it is the
first producer to decompose the locked outcome by cell margin, and its estimand
was approved in `Results/entry_margin/entry_formula_memo.md` before the file
existed. The 1X band hosts the NETS producers WS2 and WS3 specify.

30 estimates on the locked composition frame, not on the candidate frame
`step_2/code/py/31_margin_decomposition.py` uses. So its entry coefficient is a
NEW number and is not a tie-out against 31 on either window, even though the two
now overlap in period; what it borrows from 31 is `partition`, imported rather
than restated. Its construction gates and additivity gates are recorded per
WINDOW and horizon in `Results/entry_margin/entry_bins_identities.csv`, and the
producer refuses to report a coefficient if a construction gate fails.

30 estimated the post-1998 window alone until 2026-08-27, when the pre-1998
window was added, the reduced-form panel was dropped and the panels were
reordered to keep each measure's headline and robustness together. Adding the
window is a SAMPLE move on the approved estimand, in the same class as 50's
flags below, so it needed no new protocol pass - but it did need producer 69's
Mian-Sufi EFSY coverage assertion, which ran 1998-2016 only, to be discharged on
1975-97 first. That assertion now lives in 69 itself over the full span and per
window here as gates G3a and G3b. Two named residual bins (`Unresolved NAICS
root`, `Not in the Mian-Sufi 294`) carry the pre-1998 mass A.3.1 forbids
dropping; both are exactly empty post-1998, and every post-1998 coefficient
reproduces the earlier one-window table exactly, which is the check that the
change did not leak. The RZ quartile cuts are employment-weighted per window
(1975 and 1998), so an EFD quartile is not the same industry set in the two.

A THIRD WINDOW, `full`, WAS ADDED 2026-08-27: the pooled whole locked panel, no
year restriction, n = `locked.N_COUNTY`. Another SAMPLE move on the same
approved estimand, so again no new protocol pass. It straddles the break, which
the other two are built to avoid, so three things guard it. Gate G6 asserts its
row count is `locked.N_COUNTY` and that its county `TOTAL y_mpgr` reproduces the
coefficient `window_context` reads straight off the panel - one estimand, one
set of rows, two code paths; realised relative gap 0.00e+00 at every horizon.
`row_accounting` asserts pre + post + straddle + tail = full per horizon.
Diagnostic D1 sizes the h x 3,058 county-years with `t-h < 1998 <= t` that
neither one-sided window has and whose growth rate compares a SIC-bridged base
period to a native-NAICS outcome year: 3.40% / 5.24% / 7.15% of the pooled
window's entry-margin mass at h = 2 / 3 / 4. D1 is REPORTED, NOT GATED - a
straddle is intrinsic to a pooled window, so a pass threshold would license no
action. The pooled window shares pre's RZ cut year (1975) rather than taking one
of its own, so a quartile row is the same industry set in both printed columns.
Every `pre` and `post` coefficient reproduced the previous run byte for byte,
which is the check that adding the window did not leak.

WHAT THE .tex CARRIES IS NOT WHAT THE PRODUCER ESTIMATES. `TABLE_WINDOWS` is
`("pre", "full")`, so `Table_entry_bins_efsy.tex` prints those two as column
groups under `\cmidrule`d titles and leaves the post-1998 window out;
`entry_bins_coeff.csv` carries all three, keyed by the `window` column, and so
does `entry_bins_identities.csv`. Because the page shows two of three windows of
a coefficient that flips sign, the table note is GENERATED rather than typed:
`note()` fills the county totals of all three windows AND the D1 straddle shares
into it from the run, so a stale number there is not possible. The two-group
tabular runs 21pt over `\linewidth`, so the fragment is wrapped in
`\adjustbox{max width=\linewidth}` like every other exhibit here; the consuming
preamble must load `adjustbox`.

50 carries three robustness flags, added 2026-08-27 under
`plans/2026-08-27_ws4_maximal_sample.md`. Each is a SAMPLE or a CONTROL move on
the approved estimand, so none needed a new protocol pass; the defaults are the
headline, and the default invocation's four exhibits are unchanged.

| Flag | Default | What it adds |
|---|---|---|
| `--sample common\|maximal\|both` | `common` | each horizon on its own longest window (leads 1982-2018, post h on 1982-(2018-h)). Never plotted: the post horizons then sit on different calendar periods |
| `--rung 0'\|0\|5'\|both\|all` | `0'` | `0` drops the share-sum level -- NOT admissible under `ws0_contingency.py:34`, a diagnostic, writes `Figure_lp_irf1.{pdf,tex}`. `5'` ADDS 22's five named predetermined covariates, writes `Figure_lp_irf5.{pdf,tex}`. `both` = 0' and 0 (unchanged); `all` = 0', 0 and 5'. Common sample only |
| `--leads-sensitivity` | off | the lead verdict at the maximal, common and pre-seam windows. Writes `lp_leads_sample_sensitivity.csv` |

Rung 1 (share sum x year) is deliberately NOT implemented in 50 and
`control_names` rejects the label rather than approximating it: the first stage
falls 16.6 to 3.1 the moment the year interaction enters. `50_lp_dynamics.py`
reads `CONTINGENCY` but never writes `ws0_contingency.py`, so the rung flag does
not touch the frozen rule or its pre-commitment gate.

Rung 5' (added 2026-08-31) is rung 0' plus one control per predetermined county
covariate in 22's `RUNG2_COVARIATES` -- income per head, the employment rate, log
population, and the professional/technical services and manufacturing shares. It
sits on the PRIMED fork, so it never pays rung 1's first-stage cost. The set is
IMPORTED from 22 rather than restated, so there is one definition of what the
covariate rung controls for; `flagged_names()` only sorts it and asserts that
every name has a constructor here, so adding a covariate upstream fails loudly
instead of silently leaving a control unbuilt. The covariates are built OFF the
finiteness mask, like S0, so the rung cannot move the sample; `main` asserts both
the row count and the control count. This rung carries no verdict.

**Revised the same day it was added.** As first written, this rung selected its
controls from the covariates whose Panel A balance t at rung 0' exceeded 1.96,
read off 22's frozen CSV. That was retired with 22's own t-based cuts: a
specification chosen on the realised balance t's is a function of the noise in
the test it is answering. Rungs 2/2' upstream and rung 5' here now carry the same
named set, and no rung anywhere in the ladder is selected on a t. The rung-5'
numbers in `Results/lp/` moved when it changed (see `lp_formula_memo.md` §8); the
rung-0' and rung-0 exhibits did not. The label 5' is kept even though 22's twin
of this rung is 2', because the label and the exhibit filename are what the
record already names.

## What 21 does, and one thing it deliberately does not

`21_rotemberg.py` decomposes the SAME 2SLS coefficient producer 20 reports
(3.9358 at h = 4), by Goldsmith-Pinkham, Sorkin and Swift's Proposition 3. It
asserts two identities on every run, at every horizon: the weights sum to one
(exact, because the national category rates are not leave-one-out), and
`sum_k alpha_k beta_k` reproduces `t28.fit2sls`'s beta to `RTOL_COEF`. The
second target is the house estimator, not anything 21 computes, so the check is
not a verifier co-authored with the thing it verifies. It also validates its
own vectorized per-category first stage against `fit2sls` on 25 categories
before using it.

**21 does not share the SAMPLE-BUILD block with 20, on purpose.** The two build
the same sample, weights and per-(k,t) aggregates, and lifting that block into
`lib/` was considered. It was not done because 20's `--phase old` asserts its
tables string-equal to the captured reruns of producers 55 and 56, and putting
that byte-exactness at risk buys nothing. That still stands: 22 writes its own
copy of the block, and the extraction is deferred until 22's verdicts land, so
the interface is designed against three known callers rather than two guessed
ones.

**What 22 did lift, and the proof.** Six objects moved out of 21 into `lib/`
when 22 needed them, verbatim and with operation order preserved:

- `lib/clusterstats.py`: `cluster_sums`, `hhi_effective`, `wls_fe`
- `lib/sectors.py`: `SECTORS`, `OMITTED`, `naics2_shares` (+ `NAICS2_DIR`)

`ar_interval` and `fmt_ci` deliberately did NOT move. WS0 runs no
identification-robust set — BHJ's main text reports the exposure-robust standard
error and their null-imposed LM interval is a supplementary-appendix robustness
— so those two remain in 21, which uses them for the GPSS per-category
weak-IV intervals.

Proof of the lift, all four re-run after it: 21's two identities hold at h = 2,
3, 4 (worst adding-up gap 5.65e-15); all eight files in
`Results/pipeline_diagnostics/` byte-identical by SHA256 to their pre-lift
snapshot; `05_baseline_tieout` PASS; `20 --phase old` still string-equal to
`_golden`.

**One epsilon row is expected to be unstable if the sample block is ever
extracted.** `golden/rerun_bhj_shock_level.csv` carries
`Equivalence gap,5.4e-15` — two significant figures of floating-point
accumulation noise over a 25,600-term sum. Any reassociation flips that string
with no bearing on correctness, and 20:560's string-equality assert would fail
for a spurious reason. Compare the golden frames field-by-field there, not as
strings.

**Pipeline 20's "first-stage share" rows are not Rotemberg weights.** 20 reports
the concentration of `|sum_l w_l s_lk x~_l|`, which omits `g_kt` and is never
normalized to sum to one. 21's `alpha_k` include `g_kt` and do sum to one. The
two disagree by construction (0.239 against 0.191 for the largest category);
the table note says so, and neither number is wrong.
