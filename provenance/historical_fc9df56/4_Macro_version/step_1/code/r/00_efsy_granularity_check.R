# EFSY long panel: how much employment resolves to four NAICS digits?
#
# The EFSY NAICS long panel assigns employment to the most detailed NAICS code it
# can and parks the unassignable remainder on shorter roots ("11////" carries the
# sector-11 employment that could not be pushed further down). Truncating to four
# digits therefore produces two kinds of cell: genuine four-digit industries, and
# residual cells for employment that never resolved that far.
#
# This script measures the size of the residual, by year, so the granularity of
# the disaggregation exhibit can be described accurately.

suppressMessages(library(arrow))
suppressMessages(library(dplyr))

# The restricted EFSY source lives only in the shared Dropbox data tree.
.this_script <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args, value = TRUE)
  if (length(m)) return(normalizePath(sub("^--file=", "", m[1])))
  if (!is.null(sys.frames())) {
    f <- sapply(sys.frames(), function(s) tryCatch(s$ofile, error = function(e) NULL))
    f <- Filter(Negate(is.null), f)
    if (length(f)) return(normalizePath(f[[length(f)]]))
  }
  file.path(getwd(), "4_Macro_version/step_1/code/r/00_efsy_granularity_check.R")
}
# r -> code -> step_1 -> 4_Macro_version -> repo root
project_root <- dirname(dirname(dirname(dirname(dirname(.this_script())))))
dropbox_home <- Sys.getenv(
  "INVARIANCE_DROPBOX_HOME",
  path.expand("~/Library/CloudStorage/Dropbox-Matray")
)
efsy_path <- file.path(dropbox_home, "_Source/US/CBP_EFSY/efsy_panel_naics.parquet")
if (!file.exists(efsy_path)) stop("Dropbox-only EFSY source missing: ", efsy_path)

# One batch_logs/ in this project, at the root, one subfolder per pipeline.
out_dir <- file.path(project_root, "batch_logs", "macro_version")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

ds <- open_dataset(efsy_path)

# resolved digits = characters before the first "/"
by_year <- ds %>%
  mutate(res = nchar(gsub("/.*$", "", naics12))) %>%
  group_by(year) %>%
  summarise(
    emp_total  = sum(emp, na.rm = TRUE),
    emp_lt4    = sum(ifelse(res < 4, emp, 0), na.rm = TRUE),
    emp_2dig   = sum(ifelse(res == 2, emp, 0), na.rm = TRUE),
    emp_3dig   = sum(ifelse(res == 3, emp, 0), na.rm = TRUE),
    emp_6dig   = sum(ifelse(res == 6, emp, 0), na.rm = TRUE),
    rows       = n()
  ) %>%
  collect() %>%
  arrange(year) %>%
  mutate(
    share_lt4  = 100 * emp_lt4 / emp_total,
    share_6dig = 100 * emp_6dig / emp_total
  )

write.csv(by_year, file.path(out_dir, "efsy_granularity_by_year.csv"), row.names = FALSE)

cat("\nShare of employment NOT resolving to four NAICS digits, by year\n")
print(as.data.frame(by_year[, c("year", "emp_total", "share_lt4", "share_6dig", "rows")]),
      row.names = FALSE, digits = 4)

# distinct four-digit cells, and how many are residual, in a representative year
cells <- ds %>%
  filter(year == 1990) %>%
  mutate(cell4 = substr(naics12, 1, 4)) %>%
  group_by(cell4) %>%
  summarise(emp = sum(emp, na.rm = TRUE)) %>%
  collect() %>%
  mutate(residual = grepl("/", cell4))

cat("\n1990: distinct four-character cells =", nrow(cells),
    "| residual cells =", sum(cells$residual),
    "| employment share in residual cells =",
    round(100 * sum(cells$emp[cells$residual]) / sum(cells$emp), 2), "%\n")
