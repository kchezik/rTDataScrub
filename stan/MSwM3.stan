data {
  int<lower=1> N;
  vector[N] y;
}
parameters{
  ordered[N] mu;
  real b0;
  real b1;
  vector[N] theta;
  real rho;
  vector<lower = 0>[3] sigma;
}
model{
  // priors
  b0 ~ normal(0,1); // Intercept (probably 0).
  b1 ~ normal(0,1); // AR1 coefficient
  theta[1] ~ beta(0.5, 0.5);
  mu[1] ~ normal(0, 0.5);
  sigma ~ normal(0,1);
  rho ~ normal(0,1);


  // likelihood
  for(n in 2:N){
    // Define the mean captured in µ[i].
    mu[n] = exp(b0 + b1* y[n-1])/1+exp(b0 + b1* y[n-1]);

    // Moment match µ[i] to the beta distribution.
    theta[n] ~ beta((mu[n]^2 + mu[n]^3 + mu[n]*sigma[3]^2)/sigma[3]^2, // alpha
                    (mu[n]-2*mu[n]^2+mu[n]^3=sigma[3]^2+mu[n]*sigma[3]^2)/sigma[3]^2) // beta

    // Estimate the probability of being in a high or low variance state.
    target += log_mix(theta[n],
                     normal_lpdf(y[n] | rho* y[n-1], sigma[1]),
                     normal_lpdf(y[n] | rho* y[n-1], sigma[2]))
  }
}
