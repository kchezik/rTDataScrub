functions {
  vector normalize(vector x) {
    return x / sum(x);
  }
}

data {
  int<lower=1> N;     // number of observations (length)
  int<lower=1> K;     // number of hidden states
  real y[N];          // observations
  real tau;           // define location of annual cycle
  real n;             // define the number of data points in a annual cycle
  real air_A;         // PRISM air mean temperature estimate
  real air_mean;      // PRISM air annual temperature range (i.e., amplitude)
  real water_A;       // Water amplitude approximation given air_A
  real water_mean;    // Water mean approximation given air_mean
}

parameters {
  // Discrete state model
  simplex[K] A_ij[K]; // transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Seasonal Temperature Model for Air and Water
  real alpha_air;
  real alpha_water;
  positive_ordered[2] A;
  ordered[2] tau_est;
  positive_ordered[2] sigma;
}

transformed parameters {
  // Water model temporary estimates of mean and variance
  real mu[2]; // mean temperature
  real season[2]; // seasonal error term

  // log probabilities
  vector[K] unalpha_tk[N];

  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    for (t in 1:N) {
      if(t == 1){
        mu[1]= alpha_air+ A[2]*cos(2*pi()*t/n+ tau_est[1]*pi());
        season[1]= sigma[2];
        unalpha_tk[t,1]= log(0.5)+ student_t_lpdf(y[t]|3, mu[1], season[1]);

        mu[2]= alpha_water+ A[1]*cos(2*pi()*t/n+ tau_est[2]*pi());
        season[2]= sigma[1];
        unalpha_tk[t,2]= log(0.5)+ student_t_lpdf(y[t]|3, mu[2], season[2]);
      }
      else{
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                           // Murphy (2012) Eq. 17.48
                           // belief state + transition prob + local evidence at t

            if(j == 1){
              mu[j]= alpha_air+ A[2]*cos(2*pi()*t/n + tau_est[j]*pi());
              season[j] = sigma[2];
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                student_t_lpdf(y[t]|3, mu[1], season[1]);
            }
            if(j == 2){
              mu[j]= alpha_water+ A[1]*cos(2*pi()*t/n+ tau_est[j]*pi());
              season[j]= sigma[1];
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                student_t_lpdf(y[t]|3, mu[2], season[2]);
            }
          }
          unalpha_tk[t, j] = log_sum_exp(accumulator);
        }
      }
    }
  } // Forward
}

model {
  // Transition State Priors
    A_ij[1,1] ~ beta(10,2);
    A_ij[2,1] ~ beta(2,10);
    A_ij[1,2] ~ beta(2,10);
    A_ij[2,2] ~ beta(10,2);

  // Prior of temperature alpha parameter.
    alpha_water ~ normal(water_mean, 3);
    alpha_air ~ normal(air_mean, 0.5);

  // Model of temperature amplitude parameter.
    A[1] ~ normal(water_A, 3);
    A[2] ~ normal(air_A, 0.5);

  // Model tau
    tau_est[1] ~ normal(tau-0.01,.01);
    tau_est[2] ~ normal(tau+0.01,.01);

  // Remaining Variance Priors
  sigma ~ student_t(3,1,2);

  // Return log probabilities for each model.
  target += log_sum_exp(unalpha_tk[N]); // Note: update based only on last unalpha_tk
}

generated quantities {
  // Water model temporary estimates of mean and variance
  real mu_gen[2]; // mean temperature
  real season_gen[2]; // seasonal error term

  // forward filtered estimates
  vector[K] alpha_tk[N];

  vector[K] logbeta[N];
  vector[K] loggamma[N];

  vector[K] beta[N];
  vector[K] gamma[N];

  { // Forward algorithm
    for (t in 1:N)
      alpha_tk[t] = softmax(unalpha_tk[t]);
  } // Forward

  { // Backward algorithm log p(y_{t+1:T} | z_t = j)
    real accumulator[K];

    for (j in 1:K)
      logbeta[N,j] = 1;

    for (tforward in 0:(N-2)) {
      int t;
      t = N - tforward;

      for (j in 1:K) {    // j = previous (t-1)
        for (i in 1:K) {  // i in next (t)
                          // Murphy (2012) Eq. 17.58
                          // backwards t + transition prob + local evidenceat t
          if(j == 1) {
            mu_gen[j]= alpha_air+ A[2]*cos(2*pi()*t/n+ tau_est[j]*pi());
            season_gen[j]= sigma[2];
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              student_t_lpdf(y[t]|3, mu_gen[j], season_gen[j]);
          }
          if(j == 2){
            mu_gen[j]= alpha_water+ A[1]*cos(2*pi()*t/n+ tau_est[j]*pi());
            season_gen[j]= sigma[1];
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              student_t_lpdf(y[t]|3, mu_gen[j], season_gen[j]);
          }
        }
        logbeta[t-1,j] = log_sum_exp(accumulator);
      }
    }

    for (t in 1:N)
      beta[t] = softmax(logbeta[t]);
  } // Backward

  { // forward-backward algorithm log p(z_t = j | y_{1:T})
    for(t in 1:N) {
      loggamma[t] = alpha_tk[t] .* beta[t];
    }

    for(t in 1:N)
      gamma[t] = normalize(loggamma[t]);
  } //forward-backward
}

