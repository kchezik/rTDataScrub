data {
  int<lower=1> K; // Number of mixture components.
  int<lower=1> N; // Number of observations.
	vector[N] y; // Observations
}
parameters {
  simplex[K] theta; // Mixing proportions.
  ordered[K] mu; // Location of mixture components.
	vector<lower=0>[K] sigma; // Standard deviation estimates for both models and probability.
}
model {
  // priors
  vector[K] log_theta = log(theta); // Cache log calculation.
  sigma ~ lognormal(0,2);
  mu ~ normal(0, 10);

  // likelihood
  for(n in 1:N){
    vector[K] lps = log_theta;
    for(k in 1:K){
      lps[k] += normal_lpdf(y[n] | mu[k], sigma[k]);
    }
    target += log_sum_exp(lps);
  }
}
