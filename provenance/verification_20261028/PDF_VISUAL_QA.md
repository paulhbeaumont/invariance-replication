# Final PDF visual QA - October 28, 2026

All five pages were rendered with Poppler and visually inspected. Final SHA-256:
`f2e995c9331f8cd70d238a0166fa1ea1731bb9358409e201d04d7844f40999b6`.

- Page 1: disaggregation columns, counts, notes and waterfall fit; beta_b/beta_d
  labels are visible, with the established positive/negative bar colors.
- Pages 2-3: aggregation tables and ordinary first-stage rows are legible.
- Page 4: all control cases and interaction first stages fit; smaller notes
  leave clear separation from the footer.
- Page 5: five 1978 characteristics and 95%/99% intervals render; income is
  labeled income per resident; population is not plotted.

Only display code changed during QA: Unicode beta fixes PDF symbol-font
dropout; heterogeneity-note font/spacing was adjusted. No input or estimate
changed. The sandbox-denied MiKTeX logs are in `render_attempt1/`; the earlier
successful PDF and first QA in `render_before_font_qa/`; final images in
`pdf_qa/recap-1.png` through `pdf_qa/recap-5.png`. The delivery copy and the
certified run PDF have identical checksums. No original report was overwritten.
