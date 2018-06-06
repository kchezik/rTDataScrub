data {
  int N; // Number of observations in sequence.
	vector[N] y; // Temperature data.
}
parameters {
  vector<lower = 0, upper = 1>[3] p; // Probability of switching between models.
  real<lower = 0> alpha; //  Normally distributed y-intercept.
  real<lower = 0> rho; // Effect of T(t-1) on T(t).
  real t0; // Earth threshold temperature.
  vector<lower = 0>[3] sigma; // Residual error.
  real<lower = 0, upper = 1> xi_init; // Probability of being in state 1 ...
                                       // ... in the initial unseen prior observation.
  real y_init; // Unseen initial observation drawn from a probability distribution.
}
transformed parameters {
  matrix[N, 3] eta; // Likelihood estimation given the data.
  matrix[N, 3] xi; // Probability of being model 1 vs. model 2.
  vector[N] f; // Weighted average of likelihood contributions.

  // fill in etas (likelihood)
      // All models are drawn from a normal distribution where alpha/b0 is the mean ...
      // ... daily change in temperature. Model one is only a random walk while model 2 ...
      // ... is an AR1 process using the previous observation to predict the next. Model three
      // ... does not allow b0 to drop below zero thereby described by a half normal. We ...
      // ... expect model one to resemble air temperature, model 2 water and model 3 ground.
  for(t in 1:N) {
      eta[t,1] = exp(normal_lpdf(y[t]| alpha, sigma[1]));
    if(t==1) {
      eta[t,2] = exp(normal_lpdf(y[t]| (y_init - t0)^2 + t0, sigma[2]));
      eta[t,3] = exp(normal_lpdf(y[t]| rho * y_init, sigma[3]));
    }
    else {
      eta[t,2] = exp(normal_lpdf(y[t]| (y[t-1] - t0)^2 + t0, sigma[2]));
      eta[t,3] = exp(normal_lpdf(y[t]| rho * y[t-1], sigma[3]));
    }
  }

  // work out likelihood contributions
  for(t in 1:N) {
    // for the first observation
    if(t == 1) {
      f[t] = p[1]*xi_init*eta[t,1] + // stay in state 1
             (1-p[1])*xi_init*eta[t,2] + // transition from 1 to 2
             (1-p[1])*xi_init*eta[t,3] + // transition from 1 to 3

             (1-p[2])*((1-xi_init)/2)*eta[t,1] + // transition from 2 to 1
             p[2]*((1-xi_init)/2)*eta[t,2] + // stay in state 2
             (1-p[2])*((1-xi_init)/2)*eta[t,3] + // transition from 2 to 3

             (1-p[3])*((1-xi_init)/2)*eta[t,1] + // transition from 3 to 1
             (1-p[3])*((1-xi_init)/2)*eta[t,2] + // transition from 3 to 2
             p[3]*((1-xi_init)/2)*eta[t,3]; // stay in state 3

             // work out xi
             xi[t,1] = (p[1]*xi_init*eta[t,1] +
                       (1-p[2])*((1-xi_init)/2)*eta[t,1] +
                       (1-p[3])*((1-xi_init)/2)*eta[t,1])/f[t];

             xi[t,2] = (p[2]*((1-xi_init)/2)*eta[t,2] +
                       (1-p[1])*xi_init*eta[t,2] +
                       (1-p[3])*((1-xi_init)/2)*eta[t,2])/f[t];

             xi[t,3] = 1.0 - (xi[t,1]+xi[t,2]);

    }
    else {
      f[t] = p[1]*xi[t-1,1]*eta[t,1] + // stay in state 1
             (1-p[1])*xi[t-1,1]*eta[t,2] + // transition from 1 to 2
             (1-p[1])*xi[t-1,1]*eta[t,3] + // transition from 1 to 3

             (1-p[2])*xi[t-1,2]*eta[t,1] + // transition from 2 to 1
             p[2]*xi[t-1,2]*eta[t,2] + // stay in state 2
             (1-p[2])*xi[t-1,2]*eta[t,3] + // transition from 2 to 3

             (1-p[3])*xi[t-1,3]*eta[t,1] + // transition from 3 to 1
             (1-p[3])*xi[t-1,3]*eta[t,2] + // transition from 3 to 2
             p[3]*xi[t-1,3]*eta[t,3]; // stay in state 3

             // work out xi
             xi[t,1] = (p[1]*xi[t-1,1]*eta[t,1] +
                       (1-p[2])*xi[t-1,2]*eta[t,1] +
                       (1-p[3])*xi[t-1,3]*eta[t,1])/f[t];
             xi[t,2] = (p[2]*xi[t-1,2]*eta[t,2] +
                       (1-p[1])*xi[t-1,1]*eta[t,2] +
                       (1-p[3])*xi[t-1,3]*eta[t,2])/f[t];
             xi[t,3] = 1.0 - (xi[t,1]+xi[t,2]);

    }
  }
}
model {
  // priors
  p ~ beta(10, 2);  // Probability of state transition is expected to be rather resiliant and ...
                    // ... remain in the same state rather than transition to another state.
  alpha ~ normal(15,5);
  rho ~ normal(1, .05);  // We expect stationarity and small shifts between data points ...
                        // ... so we limit rho to be largely between 0 and 1.
  t0 ~ normal(5,0.5); // Earth temperature.
  sigma ~ cauchy(0, 1); // Error drawn from a cauchy distribution. Not sure why yet.
  xi_init ~ beta(2, 2);  // We have no strong prior about which state we begin in but ...
                          // ... rather assume the state is probably undefined.
  y_init ~ normal(14, 2.4); // Observations just before recording was probably most like ...
                          // ... the first few observations.

  // likelihood
  target += sum(log(f));
}
