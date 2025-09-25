

compute_irr_data <- function(
    t_grid,                         # vector of loan terms in years
    rate_annual = 0.04,             # nominal annual fee rate
    x = 1,                          # customer's deposit (scale; IRR is scale-invariant)
    include_deposit_return = TRUE   # if TRUE, -x at the last month
) {
  # ---- Local helpers (kept private) ----
  npv <- function(rate, cf) {
    t <- seq_along(cf) - 1L
    sum(cf / (1 + rate)^t)
  }
  
  irr_uniroot <- function(cf, grid_min = -0.999, grid_max = 100, grid_points = 2000, tol = 1e-10) {
    if (!any(cf > 0) || !any(cf < 0)) return(NA_real_)
    # hybrid grid: dense near 0 (linear), wide positive tail (log)
    g_neg <- seq(grid_min, 0.5, length.out = max(100, grid_points %/% 2))
    g_pos <- exp(seq(log(1e-6), log(1 + grid_max), length.out = max(100, grid_points %/% 2))) - 1
    r_grid <- sort(unique(c(g_neg, g_pos))); r_grid[r_grid <= -0.999999] <- -0.999999
    
    f <- function(r) npv(r, cf)
    npvs <- vapply(r_grid, f, numeric(1))
    s <- sign(npvs); ok <- is.finite(npvs)
    idx <- which(diff(s) != 0 & ok[-length(ok)] & ok[-1])
    
    roots <- numeric(0)
    for (k in idx) {
      a <- r_grid[k]; b <- r_grid[k + 1]
      fa <- npvs[k]; fb <- npvs[k + 1]
      if (is.finite(fa) && abs(fa) < 1e-12) { roots <- c(roots, a); next }
      if (is.finite(fb) && abs(fb) < 1e-12) { roots <- c(roots, b); next }
      rk <- tryCatch(uniroot(f, c(a, b), tol = tol)$root, error = function(e) NA_real_)
      roots <- c(roots, rk)
    }
    roots <- roots[is.finite(roots)]
    if (!length(roots)) return(NA_real_)
    pos <- roots[roots > 0]
    if (length(pos)) min(pos) else roots[which.min(abs(roots))]
  }
  
  irr_yearly_from_cf <- function(cf) {
    r_m <- irr_uniroot(cf)
    if (is.na(r_m)) return(NA_real_)
    (1 + r_m)^12 - 1
  }
  
  # cashflows: linear fee on 2x, and compound fee on outstanding balance (nominal r/12)
  cf_linear <- function(t_years) {
    n <- as.integer(round(12 * t_years))
    t_eff <- n / 12
    CF0 <- -x
    principal_m <- (2 * x) / n
    fee_m <- (rate_annual * 2 * x * t_eff) / n  # = (rate_annual*2x)/12
    cfs <- rep(principal_m + fee_m, n)
    if (include_deposit_return) cfs[n] <- cfs[n] - x
    c(CF0, cfs)
  }
  cf_compound <- function(t_years) {
    n <- as.integer(round(12 * t_years))
    CF0 <- -x
    r_m <- rate_annual / 12
    principal_m <- (2 * x) / n
    cfs <- vapply(1:n, function(m) {
      bal_before <- 2 * x - (m - 1) * principal_m
      principal_m + r_m * bal_before
    }, numeric(1))
    if (include_deposit_return) cfs[n] <- cfs[n] - x
    c(CF0, cfs)
  }
  
  # ---- Build data for both methods over t_grid ----
  lin <- vapply(t_grid, function(tt) irr_yearly_from_cf(cf_linear(tt)),   numeric(1))
  cmp <- vapply(t_grid, function(tt) irr_yearly_from_cf(cf_compound(tt)), numeric(1))
  
  data.frame(
    t_years = rep(t_grid, times = 2),
    IRR_annual = c(lin, cmp),
    method = factor(rep(c(
      paste0("Linear (", round(100 * rate_annual, 2), "%)"),
      paste0("Compound (", round(100 * rate_annual, 2), "%)")
    ), each = length(t_grid)))
  )
}

plot_irr <- function(df,
                     title = "Bank IRR vs Loan Term",
                     xlab = "Term (years)",
                     ylab = "IRR (annualized)") {
  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      !requireNamespace("scales", quietly = TRUE)) {
    stop("Please install 'ggplot2' and 'scales'.")
  }
  
  ggplot2::ggplot(df, ggplot2::aes(x = t_years, y = IRR_annual,
                                   linetype = method,
                                   color = method)) +  # <-- add color here
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point() +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 0.01)) +
    ggplot2::labs(
      x = xlab,
      y = ylab,
      linetype = "Fee method",
      color = "Fee method",   # <-- add legend title for color
      title = title
    ) +
    ggplot2::theme_minimal(base_size = 12)
}


# ---- Example ---------------------------------------------------------------
t <- seq(0.25, 20, by = 0.25)
df <- compute_irr_data(t, rate_annual = 0.04, x = 1, include_deposit_return = TRUE)
plot_irr(df)
