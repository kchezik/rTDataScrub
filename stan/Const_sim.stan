data {
  int<lower=1> N;   // number of observations
  vector[N] y;      // observation at time T
}
parameters {
  real<lower=0> t0; // mean
  real<lower=0> sigma; // error scale
  real<lower=-1, upper=1> theta; // lag coefficient
}
transformed parameters {
  vector[N] accum; // accumulate log probability
  real err; // error term
  err = y[1] - t0;
  for (t in 1:N){
    accum[t] = normal_lpdf(y[t] | t0 + theta * err, sigma);
    err = (y[t] - t0 - theta * err);
  }
}
model {
  // Priors
  t0 ~ cauchy(0,10);
  sigma ~ cauchy(0, 2.5);
  theta ~ cauchy(0, 2.5);

  // Model
  target += sum(accum);
}
