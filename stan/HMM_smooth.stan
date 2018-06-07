functions {
  vector normalize(vector x) {
    return x / sum(x);
  }
}

data {
  int<lower=1> N;   // number of observations (length)
  int<lower=1> K;   // number of hidden states
  real y[N];        // observations
  real tau;         // define location of annual cycle
  real n;           // define the number of data points in a annual cycle
  real air_A;       // PRISM air mean temperature estimate
  real air_mean;    // PRISM air annual temperature range (i.e., amplitude)
  real water_A;     // Water amplitude approximation given air_A
  real water_mean;  // Water mean approximation given air_mean
}

parameters {
  // Discrete state model
  simplex[K] p_1k;    // initial state probabilities
  simplex[K] A_ij[K]; // transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Seasonal Temperature Model for Air and Water
  real alpha_a;
  real alpha_w;
  positive_ordered[2] A;
  real<lower = 0, upper = 1> tau_est[2];
  positive_ordered[2] sigma;

  // Ground Model
  real<lower=0> sigma_g;
}

transformed parameters {
  // Water model temporary estimates of mean and variance
  real mu[2]; // mean temperature
  real season[2]; // seasonal error term

  // log probabilities
  vector[K] unalpha_tk[N];

  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    mu[1]= alpha_a+ A[2]*cos(2*pi()*1/n+ tau_est[2]*pi());
    season[1]= exp(cos(2*pi()*1/n+ tau_est[2]*pi()))*sigma[2];
    unalpha_tk[1,1]= log(p_1k[1])+ normal_lpdf(y[1]| mu[1], season[1]);

    mu[2]= alpha_w+ A[1]*cos(2*pi()*1/n+ tau_est[1]*pi());
    season[2]= exp(cos(2*pi()*1/n+ tau_est[1]*pi()))*sigma[1];
    unalpha_tk[1,2]= log(p_1k[2])+ normal_lpdf(y[1]| mu[2], season[2]);

    unalpha_tk[1,3]= log(p_1k[3])+ normal_lpdf(y[1]| alpha_w, sigma_g);

    for (t in 2:N) {
      for (j in 1:K) {    // j = current (t) or transition state column.
        for (i in 1:K) {  // i = previous (t-1) or transition state row.
                          // Murphy (2012) Eq. 17.48
                          // belief state + transition prob + local evidence at t
            if(j == 1){
              mu[1]= alpha_a+ A[2]*cos(2*pi()*t/n + tau_est[2]*pi());
              season[1]= exp(cos(2*pi()*t/n+ tau_est[2]*pi()))*sigma[2];
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                normal_lpdf(y[t]| mu[1], season[1]);
            }
            if(j == 2){
              mu[2]= alpha_w+ A[1]*cos(2*pi()*t/n+ tau_est[1]*pi());
              season[2]= exp(cos(2*pi()*t/n+ tau_est[1]*pi()))*sigma[1];
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                normal_lpdf(y[t]| mu[2], season[2]);
            }
            if(j == 3){
              accumulator[i] = unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                normal_lpdf(y[t]| alpha_w, sigma_g);
            }
        }
        unalpha_tk[t, j] = log_sum_exp(accumulator);
      }
    }
  } // Forward
}

model {
  // Prior of temperature alpha parameter.
  alpha_w ~ normal(water_mean, 3);
  alpha_a ~ normal(air_mean, 1);

  // Model of temperature amplitude parameter.
  A[2] ~ normal(air_A, 1);
  A[1] ~ normal(water_A, 3);

  // Model tau
  tau_est[1] ~ normal(tau,.1);
  tau_est[2] ~ normal(tau,.1);

  // Remaining Variance Priors
  sigma ~ student_t(3,.75,1);
  // Ground Priors
  sigma_g ~ normal(0,1);

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
            mu_gen[1]= alpha_a+ A[2]*cos(2*pi()*t/n+ tau_est[2]*pi());
            season_gen[1]= exp(cos(2*pi()*t/n+ tau_est[2]*pi()))*sigma[2];
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+ normal_lpdf(y[t]| mu_gen[1], season_gen[1]);
          }
          if(j == 2){
            mu_gen[2]= alpha_w+ A[1]*cos(2*pi()*t/n+ tau_est[1]*pi());
            season_gen[2]= exp(cos(2*pi()*t/n+ tau_est[1]*pi()))*sigma[1];
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+ normal_lpdf(y[t]| mu_gen[2], season_gen[2]);
          }
          if(j == 3){
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+ normal_lpdf(y[t]| alpha_w, sigma_g);
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
