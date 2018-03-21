// Response: Log Difference Temperature
// Poorly identified models. Can't tell differences in variance because they completely overlap.

// Inference for Stan model: RW_Daily.
// 4 chains, each with iter=1500; warmup=1000; thin=1;
// post-warmup draws per chain=500, total post-warmup draws=2000.
//
//          mean se_mean   sd  2.5%  25%  50%  75% 97.5% n_eff Rhat
// p[1]     0.81    0.09 0.14  0.53 0.68 0.87 0.94  0.97     2 2.61
// p[2]     0.80    0.10 0.15  0.53 0.68 0.87 0.94  0.97     2 2.54
// alpha[1] 0.00    0.00 0.00 -0.01 0.00 0.00 0.00  0.00    11 1.10
// alpha[2] 0.00    0.00 0.00 -0.01 0.00 0.00 0.00  0.00    16 1.08
// sigma[1] 0.01    0.01 0.01  0.00 0.00 0.01 0.02  0.03     2 4.27
// sigma[2] 0.01    0.01 0.01  0.00 0.00 0.01 0.02  0.03     2 4.10

data {
  int N; // Number of observations in sequence.
	vector[N] y; // Log differences in average daily temperature data.
}
parameters {
  vector<lower = 0, upper = 1>[2] p; // Probability of switching between models.
  vector[2] alpha;
  vector<lower = 0>[2] sigma; // Residual error.
  real<lower = 0, upper = 1> xi1_init; // Probability of being in state 1 ...
                                       // ... in the initial unseen prior observation.
}
transformed parameters {
  matrix[N, 2] eta; // Likelihood estimation given the data.
  matrix[N, 2] xi; // Probability of being model 1 vs. model 2.
  vector[N] f; // Weighted average of likelihood contributions.

  // fill in etas (likelihood)
  for(t in 1:N) {
      // Both both models are drawn from a normal distribution where alpha is the mean ...
      // ... daily change in temperature. Here we simply predict that the variance is ...
      // ... large for air and small for water.
      eta[t,1] = exp(normal_lpdf(y[t]| alpha[1], sigma[1]));
      eta[t,2] = exp(normal_lpdf(y[t]| alpha[2], sigma[2]));
  }

  // work out likelihood contributions
  for(t in 1:N) {
    // for the first observation
    if(t==1) {
      f[t] = p[1]*xi1_init*eta[t,1] + // stay in state 1
             (1 - p[1])*xi1_init*eta[t,2] + // transition from 1 to 2
             p[2]*(1 - xi1_init)*eta[t,2] + // stay in state 2
             (1 - p[2])*(1 - xi1_init)*eta[t,1];  // transition from 2 to 1

      xi[t,1] = (p[1]*xi1_init*eta[t,1] +(1 - p[2])*(1 - xi1_init)*eta[t,1])/f[t];
      xi[t,2] = 1.0 - xi[t,1];

    } else {
    // and for the rest
      f[t] = p[1]*xi[t-1,1]*eta[t,1] +  // stay in state 1
             (1 - p[1])*xi[t-1,1]*eta[t,2] +  // transition from 1 to 2
             p[2]*xi[t-1,2]*eta[t,2] +  // stay in state 2
             (1 - p[2])*xi[t-1,2]*eta[t,1]; // transition from 2 to 1

      // work out xi
      xi[t,1] = (p[1]*xi[t-1,1]*eta[t,1] +(1 - p[2])*xi[t-1,2]*eta[t,1])/f[t];

      // there are only two states so the probability of the other state is 1 - prob of the first
      xi[t,2] = 1.0 - xi[t,1];
    }
  }
}
model {
  // priors
  p ~ beta(10, 2);  // Probability of state transition is expected to be rather resiliant and ...
                    // ... remain in the same state rather than transition to another state.
  alpha ~ normal(0, .1);
  sigma ~ cauchy(0, 1); // Error drawn from a cauchy distribution. Not sure why yet.
  xi1_init ~ beta(2, 2);  // We have no strong prior about which state we begin in but ...
                          // ... rather assume the state is probably undefined.

  // likelihood
  target += sum(log(f));
}
