// Response: Log Difference Temperature //
// The models don't identify strongly. The probability of being in one state or the other is quite even. It captures the early air temperature readings but not the ones at the end of the dataset. Overall, it does a really poor job. Possibly at the daily level the AR1 process is completely missing in these data.

// Inference for Stan model: AR1_vs_RW_Daily.
// 4 chains, each with iter=1500; warmup=1000; thin=1;
// post-warmup draws per chain=500, total post-warmup draws=2000.
//
//          mean se_mean   sd  2.5%  25%  50%  75% 97.5% n_eff Rhat
// p[1]     0.80    0.10 0.15  0.51 0.66 0.88 0.94  0.97     2 2.55
// p[2]     0.80    0.11 0.17  0.49 0.65 0.88 0.95  0.98     2 2.73
// alpha[1] 0.00    0.00 0.00 -0.01 0.00 0.00 0.00  0.01  2000 1.00
// alpha[2] 0.00    0.00 0.00  0.00 0.00 0.00 0.00  0.01   578 1.03
// rho      0.68    0.20 0.29  0.33 0.40 0.58 0.95  1.11     2 4.06
// sigma[1] 0.01    0.01 0.01  0.00 0.00 0.01 0.02  0.03     2 3.62
// sigma[2] 0.01    0.01 0.01  0.00 0.00 0.01 0.02  0.03     2 4.16


// Response: Log Temperature
// This model distinguishes between air and water quite well. It captures all the absolutely air temperatures at the beginning and the end. Unfortunately it lumps the earth temperatures with water and doesn't question some of the questionable water data by providing weaker probabilties of being water.

// Inference for Stan model: AR1_vs_RW_Daily.
// 4 chains, each with iter=1500; warmup=1000; thin=1;
// post-warmup draws per chain=500, total post-warmup draws=2000.
//
//          mean se_mean   sd  2.5%   25%  50%  75% 97.5% n_eff Rhat
// p[1]     0.79    0.00 0.10  0.57  0.73 0.81 0.87  0.95  2000    1
// p[2]     0.99    0.00 0.01  0.98  0.99 0.99 0.99  1.00  2000    1
// alpha[1] 0.01    0.00 0.10 -0.19 -0.06 0.01 0.08  0.20  2000    1
// alpha[2] 0.12    0.00 0.06  0.01  0.08 0.12 0.16  0.23  1071    1
// rho      0.97    0.00 0.01  0.95  0.97 0.97 0.98  1.00  1071    1
// sigma[1] 5.18    0.07 2.56  2.50  3.62 4.55 5.99 11.08  1394    1
// sigma[2] 0.00    0.00 0.00  0.00  0.00 0.00 0.00  0.01  2000    1

// Response: Temperature

//           mean se_mean   sd  2.5%   25%  50%   75% 97.5% n_eff Rhat
// p[1]      0.75     0.0 0.10  0.52  0.69 0.77  0.83  0.93  2000    1
// p[2]      0.99     0.0 0.01  0.97  0.98 0.99  0.99  1.00  2000    1
// alpha[1]  0.01     0.0 0.10 -0.19 -0.06 0.01  0.07  0.20  2000    1
// alpha[2]  0.12     0.0 0.06  0.00  0.08 0.12  0.16  0.23  2000    1
// rho       0.98     0.0 0.01  0.95  0.97 0.98  0.98  1.00  1985    1
// sigma[1] 10.93     0.1 4.07  6.16  8.20 9.93 12.57 21.85  1584    1
// sigma[2]  0.48     0.0 0.02  0.43  0.46 0.48  0.49  0.52  2000    1

data {
  int N; // Number of observations in sequence.
	vector[N] y; // Temperature data or transformed temperature data.
}
parameters {
  vector<lower = 0, upper = 1>[2] p; // Probability of switching between models.
  vector[2] alpha;
  real<lower = 0> rho; // Effect of T(t-1) on T(t).
  vector<lower = 0>[2] sigma; // Residual error.
  real<lower = 0, upper = 1> xi1_init; // Probability of being in state 1 ...
                                       // ... in the initial unseen prior observation.
  real y_init; // Unseen initial observation drawn from a probability distribution.
}
transformed parameters {
  matrix[N, 2] eta; // Likelihood estimation given the data.
  matrix[N, 2] xi; // Probability of being model 1 vs. model 2.
  vector[N] f; // Weighted average of likelihood contributions.

  // fill in etas (likelihood)
  for(t in 1:N) {
    if(t==1) {
      // Both both models are drawn from a normal distribution where alpha is the mean ...
      // ... daily change in temperature. Model one is only a random walk while model 2 ...
      // ... is an AR1 process using the previous observation to predict the next. We ...
      // ... expect model one to resemble daily differences in air temperature while ...
      // ... model 2 is the daily differences in water and ground.
      eta[t,1] = exp(normal_lpdf(y[t]| alpha[1], sigma[1]));
      eta[t,2] = exp(normal_lpdf(y[t]| alpha[2] + rho * y_init, sigma[2]));
    }
    else {
      eta[t,1] = exp(normal_lpdf(y[t]| alpha[1], sigma[1]));
      eta[t,2] = exp(normal_lpdf(y[t]| alpha[2] + rho * y[t-1], sigma[2]));
    }
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
  rho ~ normal(1, .1);  // We expect stationarity and small shifts between data points ...
                        // ... so we limit rho to be largely between 0 and 1.
  alpha ~ normal(0, .1);
  sigma ~ cauchy(0, 1); // Error drawn from a cauchy distribution. Not sure why yet.
  xi1_init ~ beta(2, 2);  // We have no strong prior about which state we begin in but ...
                          // ... rather assume the state is probably undefined.
  y_init ~ normal(0, .1); // Observations just before recording was probably most like ...
                          // ... the first few observations.

  // likelihood
  target += sum(log(f));
}
