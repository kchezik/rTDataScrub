functions {
  vector normalize(vector x) {
    return x / sum(x);
  }
}

data {
  int<lower=1> N;             // number of observations (length)
  int<lower=1> K;             // number of hidden states
  real y[N];                  // observations
  real<lower=0> d[N];         // data point d in n
  real<lower=1> n[N];         // define the number of data points in a annual cycle
  real<lower=0> air_A;        // PRISM air mean temperature estimate
  real air_mean;              // PRISM air annual temperature range (i.e., amplitude)
}

parameters {
  // Discrete state model
  simplex[K] A_ij[K]; // transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Seasonal Temperature Model for Air and Water
  real<lower=0> alpha_w;
  positive_ordered[2] A;
  positive_ordered[3] tau_est;
  real<lower=0, upper=3> snow;
  positive_ordered[2] sigma;
}

transformed parameters {
  // Water model temporary estimates of mean and variance
  real annual;    // mean temperature
  real season;    // snow effect terms

  // log probabilities
  vector[K] unalpha_tk[N];

  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    for (t in 1:N) {
      // initial estimate
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                            // Murphy (2012) Eq. 17.48
                            // belief state + transition prob + local evidence at t
           if(j == 1){
             //Water annual cycle
             annual= alpha_w+ A[j]*cos(2*pi()*d[t]/n[t]+ tau_est[j]*pi());
             //Water seasonal adjustment
             season = snow*cos(2*pi()*d[t]/(n[t]/2)+tau_est[3]*pi());
             accumulator[i]= log(A_ij[i,j])+
                             normal_lpdf(y[t]|annual+ season, sigma[j]);

             //if data is continuous consider previous data points' likelihood
             if(t!=1) accumulator[i] += unalpha_tk[t-1,i];
           }
           if(j == 2){
             //Air
             annual= air_mean+ air_A*cos(2*pi()*d[t]/n[t]+ tau_est[j]*pi());
             accumulator[i]= log(A_ij[i,j])+ normal_lpdf(y[t]|annual, sigma[j]);

             //if data is continuous consider previous data points' likelihood
             if(t!=1) accumulator[i] += unalpha_tk[t-1,i];
           }
        }
       unalpha_tk[t, j] = log_sum_exp(accumulator);
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
    alpha_w ~ lognormal(2, 1);

  // Model of temperature amplitude parameter.
    A[1] ~ lognormal(2, 1);

  // Model tau
    tau_est[1] ~ normal(0.9,.15);
    tau_est[2] ~ normal(1.1,.15);
    tau_est[3] ~ normal(1.5,.15);

  // Remaining Variance Priors
    sigma ~ student_t(3,1,2);

  // Enforce order on water amplitude.
    A[2] ~ normal(air_A, 0.01);

  // Return log probabilities for each model.
    target += log_sum_exp(unalpha_tk[N]); // Note: update based only on last unalpha_tk
}

generated quantities {
  // Water model temporary estimates of mean and variance
  real annual_gen;     // mean temperature
  real season_gen;    // snow effect adjustment

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
                          // backwards t + transition prob + local evidence at t
          if(i == 1) {
            annual_gen= alpha_w+ A[i]*cos(2*pi()*d[t]/n[t]+ tau_est[i]*pi());
            season_gen = snow*cos(2*pi()*d[t]/(n[t]/2)+tau_est[3]*pi());
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              student_t_lpdf(y[t]|3, annual_gen + season_gen, sigma[i]);
          }
          if(i == 2){
            annual_gen= air_mean+ air_A*cos(2*pi()*d[t]/n[t]+ tau_est[i]*pi());
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              student_t_lpdf(y[t]|3, annual_gen, sigma[i]);
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
