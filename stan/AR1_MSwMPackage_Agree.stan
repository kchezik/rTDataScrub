data {
  int N; // Number of observations in sequence.
	vector[N] y; // Descriptor of temperature data.
}
parameters {
  vector<lower = 0, upper = 1>[2] p; // Probability of switching between models.
  vector[2] alpha;
  vector<lower = 0>[2] rho; // Effect of T(t-1) on T(t).
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
      // Both both models are drawn from a normal distribution where the mean is described by an average bi-hourly ...
      // ... change in temperature and an AR1 process using the previous observation to predict the next. We ...
      // ... expect the coefficients to be different and variances to be different.
      eta[t,1] = exp(normal_lpdf(y[t]| alpha[1] + rho[1] * y_init, sigma[1]));
      eta[t,2] = exp(normal_lpdf(y[t]| alpha[2] + rho[2] * y_init, sigma[2]));
    }
    else {
      eta[t,1] = exp(normal_lpdf(y[t]| alpha[1] + rho[1] * y[t-1], sigma[1]));
      eta[t,2] = exp(normal_lpdf(y[t]| alpha[2] + rho[2] * y[t-1], sigma[2]));
    }
  }

  // work out likelihood contributions
  for(t in 1:N) {
    // for the first observation
    if(t==1) {
      f[t] = p[1]*xi1_init*eta[t,1] + // stay in state 1
             (1 - p[1])*xi1_init*eta[t,2] + // transition from 1 to 2
             p[2]*(1 - xi1_init)*eta[t,2] + // stay in state 2
             (1 - p[2])*(1 - xi1_init)*eta[t,1]; // transition from 2 to 1

      xi[t,1] = (p[1]*xi1_init*eta[t,1] +(1 - p[2])*(1 - xi1_init)*eta[t,1])/f[t];
      xi[t,2] = 1.0 - xi[t,1];

    } else {
    // and for the rest
      f[t] = p[1]*xi[t-1,1]*eta[t,1] + // stay in state 1
             (1 - p[1])*xi[t-1,1]*eta[t,2] + // transition from 1 to 2
             p[2]*xi[t-1,2]*eta[t,2] + // stay in state 2
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
  p ~ beta(10, 2); // Probability of state transition is expected to be rather resiliant and ...
                   // ... remain in the same state rather than transition to another state.
  rho ~ normal(0, .1); // We expect stationarity and small shifts between data points ...
                          // ... so we limit rho to be largely between 0 and 1.
  sigma ~ cauchy(0, 1); // Error drawn from a cauchy distribution. Not sure why yet.
  xi1_init ~ beta(2, 2); // We have no strong prior about which state we begin in but rather assume the state is probably undefined.
  y_init ~ normal(0, .1); // Observations just before recording was probably most like the first few observations.

  // likelihood is really easy here!
  target += sum(log(f));
}
