// Equivalent-work PK slice, not the full varying-source PK/PD posterior.
// Transit RHS and source-column indexing follow Bruno 896137dd.
functions {
  vector transit_rhs(real t, vector y, vector ode, vector source) {
    real net = -ode[2] * y[1] + ode[3] * y[2];
    real input = exp(source[1] + (source[2] - 1) * log(t + 1e-16)
                     - source[3] * t);
    return [-ode[1] * y[1] + input + net, -net]';
  }
  vector unit_response(array[] real ts, vector ode, real rate, real mode,
                       data real rtol, data real atol) {
    int n = size(ts);
    real shape = 1 + rate * mode;
    vector[3] source = [shape * log(rate) - lgamma(shape), shape, rate]';
    vector[n] out = rep_vector(0, n);
    // Every schedule has zero as its first lag. BDF only receives t > 0.
    if (n > 1) {
      array[n - 1] vector[2] amounts = ode_bdf_tol(transit_rhs,
        rep_vector(0, 2), 0, ts[2:n], rtol, atol, 10000, ode, source);
      for (i in 2:n)
        out[i] = amounts[i - 1, 1];
    }
    return out;
  }
  real gp_surface(matrix weights, real x, real y) {
    real value = 0;
    for (i in 1:rows(weights))
      for (j in 1:cols(weights))
        value += weights[i, j] * sin((pi() / 3) * (x + 1.5) * i)
                                * sin((pi() / 3) * (y + 1.5) * j);
    return value / 1.5;
  }
  real effective_dose(real dose, real conc, matrix weights,
                      real dose_slope, real conc_slope, real normalizer) {
    real x = 2 * (log(dose) - log(10000.0))
                 / (log(200000.0) - log(10000.0)) - 1;
    real clamped = conc < 0 ? 0 : (conc > 2000 ? 2000 : conc);
    real y = 2 * log1p(clamped) / log1p(2000.0) - 1;
    return dose * exp(dose_slope * x + conc_slope * y
                     + gp_surface(weights, x, y) - normalizer);
  }
}
data {
  int<lower=1> n_subjects;
  int<lower=1> n_reference;
  int<lower=1> n_dose;
  int<lower=1> n_lag;
  int<lower=1> n_concentration;
  int<lower=1> n_obs;
  array[n_subjects] int reference_ends;
  array[n_subjects] int dose_ends;
  array[n_subjects] int lag_ends;
  array[n_subjects] int concentration_ends;
  vector[n_dose] dose_amount;
  array[n_dose] int dose_index;
  array[n_dose] int treatment_map;
  array[n_lag] real unique_dts;
  array[n_concentration] int concentration_idxs;
  array[n_dose] int dosing_time_idxs;
  array[n_obs] int obs_map;
  vector[n_dose] dose_x;
  vector[n_subjects] age_s;
  vector[n_obs] dv;
  // Canonical ports -> RK layout coordinates, checked by the driver.
  array[17] int<lower=1, upper=17> param_index;
  real<lower=0> rtol;
  real<lower=0> atol;
}
parameters {
  // Already unconstrained RK coordinates. Sigma transform/Jacobian is below.
  vector[17] p;
}
model {
  vector[17] q = p[param_index];
  matrix[2, 2] weights = to_matrix(q[14:17], 2, 2);
  vector[n_reference] concentration = rep_vector(0, n_reference);
  vector[3] ode = exp(q[4:6]);
  real normalizer = -q[12] - q[13] + gp_surface(weights, -1, -1);
  real sigma = exp(q[1]);
  for (s in 1:n_subjects) {
    int rstart = s == 1 ? 1 : reference_ends[s - 1] + 1;
    int dstart = s == 1 ? 1 : dose_ends[s - 1] + 1;
    int lstart = s == 1 ? 1 : lag_ends[s - 1] + 1;
    int cstart = s == 1 ? 1 : concentration_ends[s - 1] + 1;
    int nr = reference_ends[s] - rstart + 1;
    int nd = dose_ends[s] - dstart + 1;
    if (nd > 0) {
      int nl = lag_ends[s] - lstart + 1;
      int nt = max(treatment_map[dstart:dose_ends[s]]);
      matrix[nl, nt] units;
      vector[nr] local_conc = rep_vector(0, nr);
      real log_vc = q[2] + q[3] * age_s[s];
      for (j in 1:nt) {
        // Source j uses the j-th dose column, even when its label first
        // appears later. This is the deployed twin's indexing convention.
        real covariate = dose_x[dose_index[dstart + j - 1]];
        units[, j] = unit_response(unique_dts[lstart:lag_ends[s]], ode,
          exp(q[7] + q[9] * covariate), exp(q[8] + q[10] * covariate),
          rtol, atol);
      }
      for (i in 1:nd) {
        int row = dstart + i - 1;
        real effective = effective_dose(dose_amount[row],
          local_conc[dosing_time_idxs[row]], weights, q[12], q[13], normalizer);
        real factor = effective * exp(q[11] * dose_x[dose_index[row]] - log_vc);
        for (k in 1:nr)
          local_conc[k] += factor * units[
            concentration_idxs[cstart + (i - 1) * nr + k - 1], treatment_map[row]];
      }
      concentration[rstart:reference_ends[s]] = local_conc;
    }
  }
  target += normal_lpdf(dv | concentration[obs_map], sigma);
  target += exponential_lpdf(sigma | 1) + q[1];
  target += normal_lpdf(q[2:17] | 0, 1);
}
