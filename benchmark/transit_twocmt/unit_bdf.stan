// Unit-solve gradient benchmark. RHS, parameter grouping and production
// controls follow Bruno stan/varyingsource3.stan (896137dd), lines 2154–2212.
functions {
  real transit_source(real t, vector source_parameters) {
    return exp(source_parameters[1]
               + (source_parameters[2] - 1) * log(t + 1e-16)
               - source_parameters[3] * t);
  }
  vector twocmt_transit_source(real t, vector y, real dose,
                              vector ode_parameters, vector source_parameters) {
    real net = -ode_parameters[2] * y[1] + ode_parameters[3] * y[2];
    return [-ode_parameters[1] * y[1] + dose * transit_source(t, source_parameters) + net,
            -net]';
  }
}
data {
  int<lower=1> n;
  array[n] real<lower=0> ts;
  real<lower=0> rtol;
  real<lower=0> atol;
}
parameters {
  // Identity coordinates: [k10, k12, k21, rate, shape]. No prior or Jacobian.
  vector[5] p;
}
model {
  vector[3] ode_parameters = p[1:3];
  vector[3] source_parameters = [p[5] * log(p[4]) - lgamma(p[5]), p[5], p[4]]';
  array[n] vector[2] amounts;
  if (ts[1] == 0) {
    amounts[1] = rep_vector(0, 2);
    if (n > 1)
      amounts[2:n] = ode_bdf_tol(twocmt_transit_source, rep_vector(0, 2), 0,
                                ts[2:n], rtol, atol, 10000,
                                1.0, ode_parameters, source_parameters);
  } else {
    amounts = ode_bdf_tol(twocmt_transit_source, rep_vector(0, 2), 0,
                         ts, rtol, atol, 10000,
                         1.0, ode_parameters, source_parameters);
  }
  for (i in 1:n)
    target += amounts[i, 1];
}
