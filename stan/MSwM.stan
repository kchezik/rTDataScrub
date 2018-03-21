data {
  int N; // Number of observations in sequence.
	vector[N] y; // Descriptor of temperature data.
}
parameters {
  vector[N] mu; // Estimate regime probability.
	real beta; // AR1 (Information on the previous time steps data.)
  vector[2] rho; // AR1 (Information on the previous time steps data/probability.)
	vector<lower=0>[3] sigma; // Standard deviation estimates for both models and probability.
}
model {
  // priors
  mu[1] ~ normal(0,.1);
  beta ~ normal(0,1);
  rho ~ normal(0,1);
  sigma ~ normal(0,1);

  // likelihood
  for(t in 2:N){
    mu[t] ~ normal(rho[1]*mu[t-1] + beta* y[t-1], sigma[3]);

    target += log_mix(inv_logit(mu[t]),
      normal_lpdf(y[t] | rho[2] * y[t-1], sigma[1]),
      normal_lpdf(y[t] | rho[2] * y[t-1], sigma[2]));
  }
}
