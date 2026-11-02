"""The WS0 step-2 decision rule, frozen before producer 22 estimates anything.

THIS FILE EXISTS TO BE COMMITTED AND THEN LEFT ALONE. It is separate from
22_bhj_assumption1.py for one reason: the pre-commitment is evidenced by the
commit that introduced the rule, and if the rule shared a file with the code
that implements the tests, every edit to the implementation would move that
commit and destroy the provenance. The rule is frozen; the code that runs it is
not. They therefore cannot share a file.

22 checks this file's introducing commit and refuses to estimate from a dirty
copy of it. Revising the rule after the numbers land is possible -- it is a
separate commit with its reason, and both stand in the record -- but it is
visible, which is the whole point.

House precedent for the shape: 46_build_post_anchor_sweep_panel.py:438 and
62_flow_construction_audit.py:210 freeze a decision_rule dict before the
producer that consumes it runs. Every threshold here carries a why_ key: a
number without a reason is a number that gets renegotiated at 2am on the day
the table comes out.
"""

CONTINGENCY = {
    "statement": (
        "Verdict A (balance) and Verdict B (inference) are read off the h=4 "
        "locked cell, clustered by FSC category, under the bands below. The "
        "first matching band wins; the list is ordered and exhaustive."
    ),

    # ---- the ladder ------------------------------------------------------
    # Rungs 0 and 0' are reported for continuity with 20 and are NOT
    # candidates. BHJ Section 4.3 p.16: with incomplete shares "period FEs need
    # to be interacted with the sum of exposure shares". Our shares sum to
    # contract dollars per head, so rung 1 is the floor, not an option.
    "admissible_rungs": ("0'", 1, 2, 3, 4),
    "reference_rung": "0'",

    # ---- REVISION 1, 2026-08-26, after the first run ----------------------
    # Recorded here rather than silently applied. The original rule (commit
    # f0d0466) set reported_baseline = 1 and admissible_rungs = (1,2,3,4); the
    # first run under it returned Verdict A NO_RUNG_CLEARS / Verdict B FAIL.
    # Adrien has ruled that rung 0' is the baseline instead.
    #
    # What this changes: rung 0' enters the admissible set and becomes the
    # reference, so Verdict B is adjudicated on beta = 3.2248 (SE 0.938,
    # first-stage F 16.6) rather than on rung 1's 2.2616 (SE 2.936, F 3.1).
    # Verdict B flips from FAIL to PASS on that arithmetic alone. Verdict A
    # does NOT change: no rung at any horizon reaches joint p > 0.10, so
    # clears() is empty either way and A stays NO_RUNG_CLEARS.
    #
    # The standing objection, recorded once and not re-litigated: BHJ 4.3 p.16
    # says that with incomplete shares "period FEs need to be interacted with
    # the sum of exposure shares". Rung 0' is a single common slope on the
    # share sum -- the CROSS-SECTIONAL control -- and rung 1 is the panel one
    # their text prescribes. Choosing 0' reports a coefficient identified off
    # variation BHJ 4.3 would have period effects absorb. That is a defensible
    # editorial call about which specification to lead with, and it is
    # Adrien's; it is not a claim that rung 1 is wrong. Both rungs are in the
    # table, so a reader can see the whole ladder either way.
    "revisions": [
        {
            "n": 1,
            "date": "2026-08-26",
            "was": "reported_baseline = 1; admissible_rungs = (1,2,3,4)",
            "now": "reported_baseline = 0'; admissible_rungs = (0',1,2,3,4)",
            "reason": "Adrien's ruling after seeing the first run's ladder",
            "effect": ("Verdict B FAIL -> PASS (adjudicated at 0': F 16.6, "
                       "CI excludes zero). Verdict A unchanged at "
                       "NO_RUNG_CLEARS: no rung reaches joint p > 0.10"),
            "objection_on_record": ("BHJ 4.3 prescribes the period-interacted "
                                    "control under incomplete shares; 0' is "
                                    "the cross-sectional one"),
        },
    ],

    # ---- the reported baseline: RULED, 2026-08-26 -------------------------
    # Adrien's call, taken with the measured numbers in front of him rather
    # than after the ladder ran. This was flagged as upstream of the verdict
    # precisely because rung 1 is also the materiality reference.
    "reported_baseline": "0'",
    "reported_baseline_ruling": (
        "SUPERSEDED BY REVISION 1: rung 0' is the reported baseline, at "
        "beta = 3.2248 (exposure-robust SE 0.938, first-stage F 16.6). "
        "The original ruling below stands in the record as what was frozen. "
        "-- ORIGINAL: Rung 1 is the reported baseline, not rung 0. The "
        "headline moves from beta = 3.9358 (t 4.40, county-clustered) to "
        "beta = 2.2616, a 43% drop. Rungs 0 and 0' are demoted to continuity "
        "rows. "
        "This is a CONTROL change, not an estimand change: y_mpgr, x_pos and z "
        "are untouched and beta is the same coefficient of the same regression "
        "-- what narrows is the identifying variation, from all of g_kt to "
        "within-period g_kt. Under homogeneous effects the estimand is "
        "identical; under heterogeneity BHJ fn.16's convex weights change with "
        "the residualized shocks, so the probability limit is a different "
        "average of the same underlying effects. Stated in the table note."
    ),
    "why_rung_1_and_not_rung_0": (
        "BHJ 4.3 p.16 is not optional: 'period FEs only isolate within-period "
        "shock variation when the exposure shares add up to one. With "
        "incomplete shares, in contrast, period FEs need to be interacted with "
        "the sum of exposure shares.' Our shares sum to contract dollars per "
        "head. Rung 0 therefore reports a coefficient that BHJ's own framework "
        "says is contaminated by variation in total exposure, and rung 0' "
        "(a single common slope on the share sum) is the cross-sectional "
        "control, not the panel one"
    ),
    "baseline_carry_forward": (
        "NOT executed by this producer and NOT yet executed anywhere. Moving "
        "the baseline to rung 1 obliges: Tables 1-6 re-estimated at every "
        "ladder rung with the share-sum x year control (-> producers 57/68, "
        "tripping PIPELINE.md's migration trigger); the deck's journey-ladder "
        "row restated at 2.2616; WS1's industry axis and WS2 step 3's NETS "
        "re-estimation inheriting the same control. WS4 inherits it too but "
        "was never gated on this verdict. Each is a separate authorised piece "
        "of work"
    ),
    "rungs": {
        0: "no control, no intercept -- the PREVIOUSLY reported spec, 3.9358",
        "0'": "shock-level constant == county share-sum level control, 3.2248",
        1: "shock-level period effects == share sum x year (BHJ 4.3), 2.2616",
        2: "+ one q_n per FLAGGED covariate only (see flagged_covariate_map)",
        3: "+ FSC group effects == exposure-weighted group indicators",
        4: "+ FSC group x decade of the target year t",
    },
    "why_rung_1_is_the_floor": (
        "BHJ 4.3 makes the period-interacted control mandatory under "
        "incomplete shares, so rung 1 is where the admissible set begins "
        "rather than an option within it. Rungs 0 and 0' are reported for "
        "continuity with producer 20 and are not candidates"
    ),

    # ---- rule (b): what each flagged covariate becomes --------------------
    # Named in advance so rung 2 cannot become a kitchen sink assembled after
    # the numbers. If Panel A flags covariate v, its q_n is the exposure-
    # weighted v-bar_n that the balance test itself regressed g_n on, entering
    # the county regression as sum_k s_lkt v-bar_kt. One control per flagged
    # covariate and NO OTHERS: a covariate that did not flag never enters.
    "flagged_covariate_map": {
        "income per head":              "ybar_n  -> sum_k s_lkt ybar_kt",
        "employment rate":              "ebar_n  -> sum_k s_lkt ebar_kt",
        "log population":               "pbar_n  -> sum_k s_lkt pbar_kt",
        "pre-period growth":            "gpre_n  -> sum_k s_lkt gpre_kt",
        "manufacturing share":          "mbar_n  -> sum_k s_lkt mbar_kt",
        "prof./technical services share": "fbar_n  -> sum_k s_lkt fbar_kt",
        "non-tradable share":           "nbar_n  -> sum_k s_lkt nbar_kt",
    },
    "sector_share_escape": (
        "if 2 or more of the 3 EFSY sector shares flag, skip rung 2 and go "
        "straight to rung 3: FSC group effects are the coarser and more robust "
        "absorber, because same-group categories share a county-industry "
        "footprint and their individual q_n would be near-collinear"
    ),

    # ---- rule (a): rejections are expected, said in advance ---------------
    "expected_rejections": (
        "7 covariates x 4 admissible rungs = 28 individual balance tests at "
        "h=4. Under the null roughly 1 in 20 rejects at 5%, so 1-2 individual "
        "rejections are the EXPECTED outcome and are not evidence against "
        "Assumption 1. This is why the joint F is the operative statistic and "
        "the individual t's are descriptive."
    ),
    "operative_statistic": "joint F over Panel A only; Panel B is excluded",
    "reference_distribution": "cluster-robust Wald/k against F(k, G-1)",
    "ssc": "t28.ssc_k, the 21.wls_fe convention",
    "why_joint_and_not_individual": (
        "with 7 covariates the chance of at least one |t| > 2.5 under the null "
        "is about 8%, and across 4 rungs and 3 horizons it approaches "
        "certainty. A rule keyed to individual t's would reject with "
        "probability one"
    ),

    # ---- clustering -------------------------------------------------------
    "cluster_level": "category k",
    "cluster_robustness": ("FSC group", "year"),
    "fsc_group_rule": (
        "first character for letter-leading PSC codes (R425 -> R), first two "
        "digits for numeric FSC codes (1510 -> 15). 412 of the 800 locked "
        "categories are alphanumeric PSC codes whose hierarchy is one letter "
        "plus three digits, so a flat 'first two characters' would mix two "
        "namespaces. Yields 93 groups"
    ),
    "why_category_and_not_most_conservative": (
        "'headline = the most conservative' selects the test statistic from "
        "the data AFTER this dict is committed, which un-freezes the "
        "pre-commitment. Category k is also what 20's existing balance rows "
        "use, so the port check is against like. Year clustering gives G=37, "
        "badly sized for a joint F over 7 covariates"
    ),

    # ---- the effective-cluster floor, measured at EVERY level -------------
    # Measured BEFORE the run, not after, because the floor is only meaningful
    # at the level that actually binds. Checking it at the category level alone
    # would have made it unfalsifiable: category clears comfortably and group
    # does not, and nothing would have caught that.
    "effective_cluster_floor": 20,
    "why_20": (
        "BHJ 5.3: the asymptotic approximation 'performs well even with a "
        "Herfindahl concentration index of 1/20'. Below that their own "
        "calibration no longer covers the design and the limitation is stated "
        "rather than papered over"
    ),
    "effective_counts_measured": {
        # 1/HHI of cluster-summed |s_kt|, computed before the run.
        "h=4": {"(k,t)": 1355.9, "category": 44.1, "FSC group": 13.8,
                "year": 35.1},
        "h=3": {"(k,t)": 1413.3, "category": 45.1, "FSC group": 14.0,
                "year": 36.1},
        "h=2": {"(k,t)": 1469.8, "category": 45.9, "FSC group": 14.1,
                "year": 37.1},
    },
    "floor_verdict_known_in_advance": (
        "THE FLOOR FIRES AT THE FSC-GROUP LEVEL AND ONLY THERE: 13.8 effective "
        "groups at h=4 against a floor of 20, at all three horizons. Category "
        "(44.1) and year (35.1) clear. Recorded here as a pre-run fact so it "
        "is not reported later as a discovery. "
        "Consequence, fixed now: the group-clustered exposure-robust SE is "
        "reported as a FIRST-CLASS row rather than a footnote, and it carries "
        "the stated limitation that BHJ 5.3's calibration does not cover it -- "
        "so a CRV1 at 13.8 effective clusters is anti-conservative and the "
        "group SE, already the widest of the three, is if anything still "
        "understated. It is NOT promoted to headline: promoting a level whose "
        "own asymptotics are uncovered would trade a stated limitation for an "
        "unstated one"
    ),
    "what_would_settle_the_honest_cluster": (
        "BHJ 6.2.2 decompose the shocks in a hierarchical random-effects model "
        "(their eq. 11) and cluster at the coarsest level whose ICC is "
        "indistinguishable from zero. That is the principled tie-breaker "
        "between category and group and it is cheap on objects 22 already "
        "builds. NOT in this producer's scope as WS0 writes it; flagged as the "
        "first candidate extension if the group/category SE gap matters to a "
        "conclusion"
    ),

    # ---- rule (c): stability across the admissible set --------------------
    # NOT a pairwise distance from a privileged rung. The question BHJ's own
    # practice asks is whether the estimate is stable across the specifications
    # you would be willing to defend, all of them at once.
    "stable": (
        "over the CLEARING rungs in the admissible set: every beta-hat has the "
        "same sign, AND max(beta) - min(beta) <= 1 x SE(rung 1), the "
        "exposure-robust SE at the reference rung"
    ),
    "why_not_a_fixed_percentage": (
        "a bare 15% line is 0.4 SE at rung 1 (SE ~ 0.92 on beta 2.2616) -- "
        "inside sampling noise -- and the plan's own rung 0 -> 0' move of 22% "
        "would have fired it. A criterion that fires on the framework's own "
        "mandatory control is measuring the wrong thing"
    ),
    "clears": (
        "joint F over Panel A has p > 0.10 AND the rung's exposure-robust "
        "first-stage F >= 10"
    ),
    "why_F10_is_the_weak_iv_guard": (
        "BHJ 5.3: 'the conventional rule of thumb for detecting weak "
        "instruments based on the appropriate constructed first-stage "
        "F-statistic applies equally well to SSIV estimators'. No "
        "identification-robust set is run anywhere in WS0; BHJ's main text "
        "reports the exposure-robust SE and nothing else, and their "
        "null-imposed LM interval is a supplementary-appendix robustness"
    ),

    # ---- Verdict A bands, ordered; first match wins ------------------------
    "verdict_a": [
        ("PASS", "joint p > 0.10 at rung 1",
         "rung 1 is the baseline; nothing downstream changes"),
        ("CONTROLS_BIND_STABLE",
         "joint p <= 0.10 at rung 1, at least one rung clears, and STABLE",
         "lowest clearing rung is the recommendation; rung 1 reported beside "
         "it; no 57/68 re-run"),
        ("CONTROLS_BIND_UNSTABLE",
         "as above but not STABLE",
         "print the ladder and the carry-forward list, STOP for Adrien; "
         "unexplained_move fires"),
        ("NO_RUNG_CLEARS",
         "no rung in 1-4 clears, OR Panel E rejects at rung 1 while Panel A's "
         "joint p <= 0.10",
         "the only genuine rejection. STOP, recommend nothing; "
         "unexplained_move fires"),
    ],
    "why_controls_binding_is_not_a_rejection": (
        "under BHJ a shock-level control that moves beta is the specification "
        "you ADOPT, not evidence the design failed. Their own ADH headline "
        "moves -0.489 to -0.267 across exactly this step and the controlled "
        "column is their preferred one (fn.49: the removed variation 'cannot "
        "be viewed as a quasi-experimental source of variation'). A band "
        "labelled REJECT here would have scored BHJ's own application as a "
        "failure"
    ),
    "why_panel_E_routes_to_band_4": (
        "a covariate imbalance says the shocks correlate with where counties "
        "WERE, which a q_n can absorb. A pre-trend says they correlate with "
        "where counties were ALREADY HEADING -- the outcome path itself -- "
        "which no shock-level control fixes, because the confound sits in "
        "epsilon rather than in anything the control can span. Panel E alone, "
        "with Panel A clean, is reported as a warning and escalated without a "
        "band: that pattern is more likely a dynamics question"
    ),
    "lagged_shock_is_not_a_trigger": (
        "BHJ fn.28: 'Arbitrary serial correlation of shocks can be allowed "
        "here via Assumption 5, with n defining a cluster.' Serial correlation "
        "is an inference problem answered by the cluster choice, not an "
        "Assumption-1 violation. g_{k,t-h} is evidence for Verdict B's "
        "clustering and is excluded from the band logic and the joint F"
    ),

    # ---- Verdict B bands --------------------------------------------------
    "verdict_b": [
        ("PASS", "interval excludes zero and first-stage F >= 10",
         "quote the exposure-robust interval alongside the county SE"),
        ("MARGINAL",
         "excludes zero but F < 10, or SE > 2x the county-clustered SE",
         "the exposure-robust SE replaces the county SE in every exhibit "
         "quoting a t, with the F stated"),
        ("FAIL", "interval contains zero",
         "report beta-hat with the exposure-robust interval and no star; "
         "escalate"),
    ],
    "recorded_limitation": (
        "BHJ fn.47 find the null-imposed interval widens by 30-50% in ADH, "
        "'more so on the left end'. Reporting the SE interval alone is "
        "therefore mildly optimistic in the left tail. Accepted because the "
        "measured exposure-robust SEs sit within 1.28x of the county ones and "
        "the first-stage F is reported alongside. Stated in the table note so "
        "it is a choice and not an omission"
    ),

    # ---- rule (d): what an unexplainable move means -----------------------
    # The sentence WS0 step 2 actually asks for. Frozen here so it is not
    # composed after the numbers are known.
    "unexplained_move": (
        "If beta-hat is unstable across the admissible shock-control "
        "specifications and no single flagged covariate accounts for the "
        "movement, then THE LEVEL OF THE COUNTY FISCAL MULTIPLIER IS NOT "
        "IDENTIFIED BY THIS INSTRUMENT. The deck's journey-ladder row is "
        "demoted from a point estimate to the range across the admissible "
        "set, reported with that range and the specifications producing its "
        "endpoints, and the accompanying claim is restated as BOUNDED rather "
        "than ESTIMATED. "
        "The paper's methodological claim is NOT weakened by this. Aggregation "
        "invariance is a statement about how the estimator behaves across "
        "aggregation rungs, not about the level of beta. It survives -- but it "
        "must then be demonstrated at EVERY admissible shock-control "
        "specification rather than at one, so Tables 1-6 gain a specification "
        "dimension rather than losing their content."
    ),
    "carry_forward_on_stop": (
        "Tables 1-6 at every ladder rung (-> producers 57/68 re-run, tripping "
        "PIPELINE.md's migration trigger); the deck's journey-ladder row; "
        "WS1's industry axis (the concordance relabels the same FSC shocks, so "
        "the same controls apply); WS2 step 3's NETS re-estimation. "
        "WS4 IS EXCLUDED and runs regardless: PLAN_ra_next_tasks.md:509-512 "
        "makes its leads diagnostic rather than confirmatory on a rejection, "
        "and halting the lead/lag design would suppress the evidence that "
        "adjudicates the very verdict that halted it"
    ),

    # ---- what is deliberately not here ------------------------------------
    "not_run": (
        "no identification-robust set (BHJ never mention Anderson-Rubin; their "
        "null-imposed LM interval is appendix-only); no leave-out robustness "
        "(not in WS0's scope as the RA plan writes it); no GPSS test is "
        "attributed to this producer -- 21 carries the GPSS side"
    ),
}
