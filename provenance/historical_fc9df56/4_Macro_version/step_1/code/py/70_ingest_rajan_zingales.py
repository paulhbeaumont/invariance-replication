"""Transcribe the Rajan-Zingales external-finance-dependence measure.

WS2 step 4's second bin, alongside the Mian-Sufi non-tradables of
69_ingest_miansufi_classification.py.

Rajan, Raghuram G., and Luigi Zingales (1998), "Financial Dependence and
Growth", American Economic Review 88(3), 559-586.

THERE IS NO REPLICATION PACKAGE AND NO POSTED DATA FILE. The measure exists
only as Table 1 of the 1998 article, and the AER of that vintage published no
supplementary materials. So this producer is a transcription, not a download,
and the numbers are literals below. That makes accuracy the whole problem, and
the canonical column is therefore carried in TWO independent renderings which
the producer asserts agree.

Writes three reference files:

  rz1998_extfin_isic.csv       the canonical measure, 36 ISIC industries,
                               US manufacturing, 1980s.
  cs_dblm_efd_sic2.csv         the Cetorelli-Strahan recomputation, 60 2-digit
                               SIC sectors, ALL sectors. A DIFFERENT MEASURE.
  cs_dblm_sic2_naics3_map.csv  a published SIC2 -> NAICS3 map for CBP/QCEW.

Sources, all public, all pinned by sha256 in SOURCES below:

  A. rz1998_aer.pdf - course copy of the article at
     https://www.isid.ac.in/~tridip/Teaching/DevEco/Readings/07Finance/07Rajan&Zingales-AER1998.pdf
     A SCAN, and its text layer is unreliable: digits split as "3 1 1",
     "--0.23", and three rows yielding nothing at all. Do not build a table
     from it alone.
  B. fisman_love_w9582.pdf - Fisman and Love (2003), "Financial Dependence and
     Growth Revisited", NBER WP 9582, at
     https://www.nber.org/system/files/working_papers/w9582/w9582.pdf
     Born digital. Their Table 2 reproduces the RZ measure as USEXTFIN at
     three decimals for all 36 industries.
  C. dblm2010_bostonfed.pdf - Duygan-Bump, Levkov and Montoriol-Garriga
     (2010), FRB Boston WP QAU10-6, published as 2015 JME 78, 89-105, at
     https://www.bostonfed.org/-/media/Documents/Workingpapers/PDF/qau1006.pdf
     Born digital. Appendix Table 2 is the SIC2 measure, Appendix Table 3 the
     SIC2-to-NAICS3 map.

The PDFs are not kept in the tree; the URLs and digests above are the record.
Re-download and re-check before trusting any number this producer does not
already assert.

WHAT THE CROSS-CHECK ESTABLISHED. extfin_all is Fisman and Love's USEXTFIN.
extfin_all_aer is RZ's own Table 1 column 1 off the scan. All 33 rows the scan
yields agree to two decimals. The three the scan loses - 323 Leather, 3211
Spinning, 3522 Drugs - rest on Fisman and Love alone; Leather at -0.14 is
independently confirmed by Villani (2020), and Drugs being the maximum at
1.492 matches RZ's own text, "Drugs and Pharmaceuticals emerge as the industry
that uses the most external finance". capex_all is Table 1 column 2,
scan-derived and NOT cross-checked. RZ's mature-company and young-company
columns are deliberately not transcribed: the scan garbles them and no
born-digital reproduction was found. If they are ever needed, install poppler
and read the page images rather than trusting the text layer.

THREE THINGS A CONSUMER MUST NOT MISS.

1. The 36 industries are NOT mutually exclusive, and parent rows are net of
   their children (RZ footnote 10): 321 Textile excludes 3211 Spinning; 341
   Paper excludes 3411 Pulp and paper; 352 Other chemicals excludes 3522
   Drugs; 382 Machinery excludes 3825 Office and computing; 383 Electric
   machinery excludes 3832 Radio; 384 Transportation equipment excludes both
   3841 Ship and 3843 Motor vehicle. The net_of_isic and parent_isic columns
   carry this. A concordance treating 384 as all of transportation equipment
   is wrong, and assigning both a parent and its child to one NAICS4 cell
   double counts.

2. Canonical RZ is manufacturing only - they confine themselves to US SIC
   2000-3999. Manufacturing is 15.8% of EFSY county x industry employment in
   1998, falling to 9.1% by 2016. The bin therefore speaks to under a tenth of
   the entry margin in the later years.

3. The SIC2 variant is a DIFFERENT MEASURE, not RZ's published numbers.
   Cetorelli and Strahan recomputed RZ-style dependence from Compustat over
   1980-1996 on mature firms at 2-digit SIC rather than ISIC. The scale
   differs sharply: leather -0.96 against RZ's -0.140, tobacco -0.92 against
   -0.451, forestry -4.63. It reaches 91.5-92.0% of EFSY employment 1998-2016
   against 9-16% for canonical RZ, which is why it is here at all - but six
   NAICS3 codes inherit more than one EFD value and need an explicit pooling
   rule: 333, 334 and 335 all come from SIC 35, 36 and 38 together (0.01,
   0.22, -0.04); 212 and 213 from SIC 10, 12 and 14 (0.96, 0.13, 0.05); 811
   from SIC 75 and 76 (0.43, -0.25). Uncovered NAICS3 by employment: 551, 522,
   813, 523, 712, 533, 521, 525.

   USING BOTH AND CALLING BOTH "RAJAN-ZINGALES" WOULD BE WRONG. Which one WS2
   bins on is a protocol-gate decision, not a coding detail.

Neither file is concorded to NAICS4. The canonical route needs ISIC rev 2 to
SIC to NAICS by hand, respecting the net-of-children semantics above;
reference/sic87_naics97_census.csv is already in the tree for the second leg.
The variant route needs only the NAICS3 truncation plus the pooling rule.

Usage terms: cite Rajan and Zingales (1998) for the canonical measure;
Cetorelli and Strahan (2006) and Duygan-Bump, Levkov and Montoriol-Garriga
(2015) for the variant.
"""

import sys
from pathlib import Path

REF_DIR = Path(__file__).resolve().parent.parent / "reference"

SOURCES = {
    "rz1998_aer.pdf": (
        "Rajan and Zingales (1998), AER 88(3), 559-586. Course copy: "
        "https://www.isid.ac.in/~tridip/Teaching/DevEco/Readings/07Finance/"
        "07Rajan&Zingales-AER1998.pdf",
        "5e10ca8fc1242f0333fb87d09d5f1749cb348fda2b0f48845b2d6d3709ea2d71"),
    "fisman_love_w9582.pdf": (
        "Fisman and Love (2003), NBER WP 9582. "
        "https://www.nber.org/system/files/working_papers/w9582/w9582.pdf",
        "5b833eef1f2f4889e0c88443d4fff7c1d30aee94a31898b91f21928626b2c428"),
    "dblm2010_bostonfed.pdf": (
        "Duygan-Bump, Levkov and Montoriol-Garriga (2010), FRB Boston WP "
        "QAU10-6, published 2015 JME 78, 89-105. "
        "https://www.bostonfed.org/-/media/Documents/Workingpapers/PDF/"
        "qau1006.pdf",
        "a03cad1276a5553b768f6bbca0965cab512a2db1f1a495483247d51ff5eee4e2"),
}

# --- A: RZ 1998 Table 1, columns 1-2, as read from the scan ---------------
# isic -> (external dependence, capital expenditures); None where the scan's
# text layer yields no value.
AER = {
    "314":  (-0.45, 0.23), "361":  (-0.15, 0.20), "323":  (None, None),
    "3211": (None, None),  "324":  (-0.08, 0.25), "372":  (0.01, 0.22),
    "322":  (0.03, 0.31),  "353":  (0.04, 0.22),  "369":  (0.06, 0.21),
    "313":  (0.08, 0.26),  "371":  (0.09, 0.18),  "311":  (0.14, 0.26),
    "3411": (0.15, 0.20),  "3513": (0.16, 0.30),  "341":  (0.18, 0.24),
    "342":  (0.20, 0.39),  "352":  (0.22, 0.31),  "355":  (0.23, 0.28),
    "332":  (0.24, 0.25),  "381":  (0.24, 0.29),  "3511": (0.25, 0.30),
    "331":  (0.28, 0.26),  "384":  (0.31, 0.31),  "354":  (0.33, 0.23),
    "3843": (0.39, 0.32),  "321":  (0.40, 0.25),  "382":  (0.45, 0.29),
    "3841": (0.46, 0.43),  "390":  (0.47, 0.37),  "362":  (0.53, 0.28),
    "383":  (0.77, 0.38),  "385":  (0.96, 0.45),  "3832": (1.04, 0.42),
    "3825": (1.06, 0.60),  "356":  (1.14, 0.44),  "3522": (None, None),
}

# --- B: Fisman and Love (2003) Table 2, USEXTFIN, ascending ---------------
FL = [
    ("314",  "Tobacco",                          -0.451),
    ("361",  "Pottery",                          -0.146),
    ("323",  "Leather",                          -0.140),
    ("3211", "Spinning",                         -0.088),
    ("324",  "Footwear",                         -0.078),
    ("372",  "Non-ferrous metal",                 0.005),
    ("322",  "Apparel",                           0.029),
    ("353",  "Petroleum refineries",              0.042),
    ("369",  "Non metal products",                0.062),
    ("313",  "Beverages",                         0.077),
    ("371",  "Iron and steel",                    0.087),
    ("311",  "Food products",                     0.137),
    ("3411", "Pulp, paper",                       0.151),
    ("3513", "Synthetic resins",                  0.159),
    ("341",  "Paper and products",                0.176),
    ("342",  "Printing and publishing",           0.204),
    ("352",  "Other chemicals",                   0.219),
    ("355",  "Rubber products",                   0.226),
    ("332",  "Furniture",                         0.236),
    ("381",  "Metal products",                    0.237),
    ("3511", "Basic chemicals excl. fertilizers", 0.253),
    ("331",  "Wood products",                     0.284),
    ("384",  "Transportation equipment",          0.307),
    ("354",  "Petroleum and coal products",       0.334),
    ("3843", "Motor vehicle",                     0.389),
    ("321",  "Textile",                           0.400),
    ("382",  "Machinery",                         0.445),
    ("3841", "Ship",                              0.458),
    ("390",  "Other industries",                  0.470),
    ("362",  "Glass",                             0.528),
    ("383",  "Electric machinery",                0.767),
    ("385",  "Professional goods",                0.961),
    ("3832", "Radio",                             1.039),
    ("3825", "Office and computing",              1.060),
    ("356",  "Plastic products",                  1.140),
    ("3522", "Drugs",                             1.492),
]

# RZ net a broader sector of any subsector reported separately (footnote 10).
NET_OF = {
    "321": "3211", "341": "3411", "352": "3522",
    "382": "3825", "383": "3832", "384": "3841;3843",
}
PARENT = {"3211": "321", "3411": "341", "3522": "352", "3825": "382",
          "3832": "383", "3841": "384", "3843": "384",
          "3511": "351", "3513": "351"}

# --- C: DBLM Appendix Table 2, EFD by 2-digit SIC ------------------------
EFD_SIC2 = [
    ("08", "Forestry", -4.63),
    ("63", "Insurance carriers", -3.96),
    ("31", "Leather and leather products", -0.96),
    ("21", "Tobacco products", -0.92),
    ("23", "Apparel and other finished products made from fabrics", -0.61),
    ("82", "Educational services", -0.55),
    ("83", "Social services", -0.43),
    ("76", "Miscellaneous repair services", -0.25),
    ("20", "Food and kindred products", -0.24),
    ("34", "Fabricated metal products, except machinery and transportation equipment", -0.24),
    ("25", "Furniture and fixtures", -0.23),
    ("32", "Stone, clay, glass, and concrete products", -0.20),
    ("39", "Miscellaneous manufacturing industries", -0.20),
    ("56", "Apparel and accessory stores", -0.16),
    ("73", "Business services", -0.16),
    ("41", "Local and suburban transit and interurban highway passenger transportation", -0.12),
    ("72", "Personal services", -0.12),
    ("27", "Printing, publishing, and allied industries", -0.07),
    ("48", "Communications", -0.07),
    ("87", "Engineering, accounting, research, management, and related services", -0.05),
    ("38", "Measuring, analyzing, and controlling instruments; photographic, medical, and optical goods", -0.04),
    ("37", "Transportation equipment", 0.00),
    ("47", "Transportation services", 0.01),
    ("35", "Industrial and commercial machinery and computer equipment", 0.01),
    ("33", "Primary metal industries", 0.03),
    ("01-02-07", "Agriculture", 0.03),
    ("40", "Railroad transportation", 0.04),
    ("24", "Lumber and wood products, except furniture", 0.04),
    ("30", "Rubber and miscellaneous plastics products", 0.04),
    ("14", "Mining and quarrying of nonmetallic minerals, except fuels", 0.05),
    ("26", "Paper and allied products", 0.06),
    ("29", "Petroleum refining and related industries", 0.09),
    ("51", "Wholesale trade: non-durable goods", 0.10),
    ("22", "Textile mill products", 0.10),
    ("42", "Motor freight transportation and warehousing", 0.10),
    ("53", "General merchandise stores", 0.12),
    ("12", "Coal mining", 0.13),
    ("59", "Miscellaneous retail", 0.16),
    ("54", "Food stores", 0.16),
    ("78", "Motion pictures", 0.17),
    ("79", "Amusement and recreation services", 0.21),
    ("36", "Electronic and other electrical equipment and components, except computer equipment", 0.22),
    ("49", "Electric, gas, and sanitary services", 0.24),
    ("58", "Eating and drinking places", 0.25),
    ("28", "Chemicals and allied products", 0.28),
    ("09", "Fishing, hunting, and trapping", 0.31),
    ("50", "Wholesale trade: durable goods", 0.32),
    ("80", "Health services", 0.35),
    ("65", "Real estate", 0.38),
    ("70", "Hotels, rooming houses, camps, and other lodging places", 0.38),
    ("13", "Oil and gas extraction", 0.40),
    ("55", "Automotive dealers and gasoline service stations", 0.41),
    ("75", "Automotive repair, services, and parking", 0.43),
    ("52", "Building materials, hardware, garden supply, and mobile home dealers", 0.47),
    ("45", "Transportation by air", 0.48),
    ("15-16-17", "Construction", 0.57),
    ("44", "Water transportation", 0.67),
    ("57", "Home furniture, furnishings, and equipment stores", 0.69),
    ("10", "Metal mining", 0.96),
    ("46", "Pipelines, except natural gas", 1.00),
]

# --- D: DBLM Appendix Table 3, SIC2 -> NAICS3 ---------------------------
SIC2_NAICS3 = [
    ("01", "111"), ("02", "112"), ("07", "115"), ("08", "113"), ("09", "114"),
    ("10", "212;213"), ("12", "212;213"), ("14", "212;213"),
    ("13", "211"),
    ("15", "236"), ("16", "237"), ("17", "238"),
    ("20", "311"), ("21", "312"), ("22", "313;314"), ("23", "315"),
    ("24", "321"), ("25", "337"), ("26", "322"), ("27", "323;511;516"),
    ("28", "325"), ("29", "324"), ("30", "326"), ("31", "316"),
    ("32", "327"), ("33", "331"), ("34", "332"),
    ("35", "333;334;335"), ("36", "333;334;335"), ("38", "333;334;335"),
    ("37", "336"), ("39", "339"),
    ("40", "482"), ("41", "485"), ("42", "484;492;493"), ("44", "483"),
    ("45", "481"), ("46", "486"), ("47", "488"), ("48", "515;517"),
    ("49", "221;562"), ("50", "423;425"), ("51", "424"), ("52", "444"),
    ("53", "452"), ("54", "445"), ("55", "441;447"), ("56", "448"),
    ("57", "442;443"), ("58", "722"), ("59", "446;451;453;454"),
    ("62", "523"), ("63", "524"), ("64", "524"), ("65", "531"),
    ("67", "533;551"), ("70", "721"), ("72", "812"),
    ("73", "518;519;532;561"), ("75", "811"), ("76", "811"),
    ("78", "512"), ("79", "487;711;713"), ("80", "621;622;623"),
    ("82", "611"), ("83", "624"), ("84", "712"), ("87", "541"),
]


def check():
    assert len(FL) == 36, len(FL)
    assert len(AER) == 36, len(AER)
    assert set(AER) == {c for c, _, _ in FL}, "code sets differ"

    disagree, checked = [], 0
    for code, name, fl in FL:
        aer = AER[code][0]
        if aer is None:
            continue
        checked += 1
        if round(fl, 2) != aer:
            disagree.append((code, name, fl, aer))
    assert checked == 33, f"cross-checked {checked} rows, expected 33"
    assert not disagree, f"renderings disagree: {disagree}"

    assert len(EFD_SIC2) == 60, len(EFD_SIC2)
    assert len({c for c, _, _ in EFD_SIC2}) == 60, "duplicate SIC2 in EFD"
    assert len(SIC2_NAICS3) == 68, len(SIC2_NAICS3)
    assert len({s for s, _ in SIC2_NAICS3}) == 68, "duplicate SIC2 in map"

    mapped = {s for s, _ in SIC2_NAICS3}
    priced = set()
    for c, _, _ in EFD_SIC2:
        priced.update(c.split("-"))
    missing = sorted(priced - mapped)
    assert not missing, f"EFD sectors with no NAICS3 mapping: {missing}"
    unpriced = sorted(mapped - priced)
    assert unpriced == ["62", "64", "67", "84"], unpriced

    return checked, unpriced


def write_canonical():
    out = REF_DIR / "rz1998_extfin_isic.csv"
    with open(out, "w") as f:
        f.write("# Rajan and Zingales (1998) external dependence on finance, "
                "36 ISIC industries, US manufacturing, 1980s.\n")
        f.write("# Built by 70_ingest_rajan_zingales.py, whose docstring "
                "carries the caveats. TRANSCRIBED - there is no RZ "
                "replication package.\n")
        f.write("# extfin_all: Fisman and Love (2003) NBER WP 9582 Table 2 "
                "USEXTFIN, 3 decimals, all 36 rows. This is the reported "
                "value.\n")
        f.write("# extfin_all_aer, capex_all: Rajan and Zingales (1998) "
                "Table 1 columns 1-2, read from the scanned article, "
                "2 decimals.\n")
        f.write("# The two agree on all 33 rows the scan yields. Blank where "
                "the scan yields nothing: 323 Leather, 3211 Spinning, "
                "3522 Drugs.\n")
        f.write("# capex_all is scan-derived only and is NOT cross-checked "
                "against a second source.\n")
        f.write("# Table 1's mature-company and young-company columns are "
                "not transcribed: the scan garbles them.\n")
        f.write("# net_of_isic: RZ net a broader sector of any subsector "
                "reported separately (footnote 10), so these parent rows do "
                "NOT cover their whole group. Assigning a parent and its "
                "child to one cell double counts.\n")
        f.write("# parent_isic: the group this row sits inside, where that "
                "group is finer than the row's own code.\n")
        f.write("# MANUFACTURING ONLY: 15.8% of EFSY employment in 1998, "
                "9.1% by 2016.\n")
        for fn in ("rz1998_aer.pdf", "fisman_love_w9582.pdf"):
            note, digest = SOURCES[fn]
            f.write(f"# Source: {note} sha256 {digest}\n")
        f.write("isic_code,industry_name,extfin_all,extfin_all_aer,capex_all,"
                "net_of_isic,parent_isic\n")
        for code, name, fl in FL:
            aer, capex = AER[code]
            f.write(f"{code},\"{name}\",{fl:.3f},"
                    f"{'' if aer is None else f'{aer:.2f}'},"
                    f"{'' if capex is None else f'{capex:.2f}'},"
                    f"{NET_OF.get(code, '')},{PARENT.get(code, '')}\n")
    return out


def write_variant(unpriced):
    note, digest = SOURCES["dblm2010_bostonfed.pdf"]

    efd_out = REF_DIR / "cs_dblm_efd_sic2.csv"
    with open(efd_out, "w") as f:
        f.write("# External financial dependence by 2-digit SIC, 60 sectors, "
                "ALL sectors not just manufacturing.\n")
        f.write("# Cetorelli-Strahan recomputation of the Rajan-Zingales "
                "measure, mature Compustat firms, 1980-1996.\n")
        f.write("# THIS IS NOT THE RAJAN-ZINGALES PUBLISHED MEASURE. "
                "Different classification, sample window, firm screen and "
                "scale: leather is -0.96 here against RZ's -0.140, tobacco "
                "-0.92 against -0.451. Never pool or compare as levels.\n")
        f.write("# Reaches 91.5-92.0% of EFSY employment 1998-2016, against "
                "9-16% for canonical RZ. Built by "
                "70_ingest_rajan_zingales.py.\n")
        f.write(f"# Source: {note} sha256 {digest}\n")
        f.write("# Appendix Table 2. Codes 01-02-07 (agriculture) and "
                "15-16-17 (construction) are reported as pooled sectors.\n")
        f.write("sic2,industry_name,efd\n")
        for code, name, efd in EFD_SIC2:
            f.write(f"{code},\"{name}\",{efd:.2f}\n")

    map_out = REF_DIR / "cs_dblm_sic2_naics3_map.csv"
    pairs = 0
    with open(map_out, "w") as f:
        f.write("# 2-digit SIC to 3-digit NAICS mapping for CBP and QCEW. "
                "Built by 70_ingest_rajan_zingales.py.\n")
        f.write(f"# Source: {note} sha256 {digest}\n")
        f.write("# Appendix Table 3. Many-to-many: SIC 35, 36 and 38 all map "
                "to NAICS 333;334;335 jointly, and SIC 10, 12, 14 all map to "
                "212;213 jointly, so those NAICS3 codes inherit more than one "
                "EFD value and need an explicit pooling rule.\n")
        f.write(f"# SIC {', '.join(unpriced)} appear in the mapping but carry "
                "no EFD value in Appendix Table 2.\n")
        f.write("sic2,naics3\n")
        for sic2, naics3 in SIC2_NAICS3:
            for n in naics3.split(";"):
                f.write(f"{sic2},{n}\n")
                pairs += 1

    return efd_out, map_out, pairs


def main():
    checked, unpriced = check()
    REF_DIR.mkdir(parents=True, exist_ok=True)
    canonical = write_canonical()
    efd_out, map_out, pairs = write_variant(unpriced)

    print(f"wrote {canonical}: 36 ISIC industries, {checked} cross-checked "
          "against the AER scan, 0 disagreements")
    print(f"wrote {efd_out}: {len(EFD_SIC2)} SIC2 sectors "
          "(a DIFFERENT measure, not RZ's numbers)")
    print(f"wrote {map_out}: {len(SIC2_NAICS3)} SIC2 codes, {pairs} pairs")
    print("All checks passed.")


if __name__ == "__main__":
    sys.exit(main())
