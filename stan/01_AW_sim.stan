data {
  int<lower=1> N;   // number of observations (length)
  vector[N] y;      // observations
}
parameters {
  real alpha;
  real<lower = 0> A;
  real<lower=0, upper=2> tau;
  real<lower = 0> sigma;
}
model {
  // Priors
  alpha ~ normal(0,10);
  A ~ normal(0,10);
  sigma ~ student_t(3,1,2);

  // Model
  for (t in 1:N) {
    y[t] ~ normal(alpha + A*cos(2*pi()*t/N + tau*pi()), sigma);
  }
}
