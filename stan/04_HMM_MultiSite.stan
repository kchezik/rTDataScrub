functions {
  vector normalize(vector x) {
    return x / sum(x);
  }
  // Creates a cosine curve upper bound by zero during the spring ...
  // ... to account for hysteresis in the water temperature curve ...
  // ... due to snow melt cooling the stream. The values of this
  // ... cosine curve is modified by a spring snow effect coefficient.
  real stream_snow(real n, real d, real tau) {
    // calulate the first derivative.
    real rate = -sin(2*pi()*d/n + tau*pi());
    // if the derivative is positive calculate the snow adjustment, ...
    // ... otherwise return 0.
    if(rate>0) {
      return rate*-1;
    } else return 0;
  }
}

data {
  int<lower=1> N;                   // Number of observations
  int<lower=1> S;                   // Number of sites
  int<lower=1> site[N];             // Site ID
  int<lower=1> R;                   // Number of unique site*years
  int<lower=1> rowe[N];             // Continuous count of unique site*year values
  int<lower=1> K;                   // Number of hidden states
  real y[N];                        // Observations
  real<lower=0> d[N];               // Data point d in n
  real<lower=0,upper=2> reset[N];   // First observation at a site (no=0, forwardYes=1, backwardYes=2)
  int<lower=0> accum[R];            // Accumulate final row of each site towards target
  real<lower=0> n[N];               // Define the number of data points in a annual cycle
  vector<lower=0>[R] air_A;         // PRISM air annual temperature range (i.e., amplitude)
  vector[R] air_mean;               // PRISM air mean temperature estimate
  int precip;                       // If 1, run the underlying precipitation model
}

parameters {
  // Discrete state model
  simplex[K] A_ij[K]; // Transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Seasonal Temperature Model for Air and Water
  vector<lower=0,upper=20>[R] alpha_w; // Mean annual water temperature
  positive_ordered[2] A[R];            // Annual temperature amplitude
  positive_ordered[2] tau_est[S];      // Annual location parameter
  vector<lower=0,upper=3>[R] snow;     // Seasonal snow/rain effect
  positive_ordered[2] sigma[S];        // Seasonal variance parameter
}

transformed parameters {
  // Water model temporary estimates of mean and variance
  real annual;  // mean annual temperature cycle
  real season;  // seasonal adjustment

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
             annual= alpha_w[rowe[t]]+ A[rowe[t],j]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],j]*pi());
             if(precip == 1){
               //Water seasonal adjustment
               season = snow[rowe[t]]*cos(2*pi()*d[t]/(n[t]/2)+(tau_est[site[t],j]+0.5)*pi());
               //season = stream_snow(n[t],d[t],tau_est[site[t],j])*snow[rowe[t]];
               accumulator[i]= log(A_ij[i,j])+
                               normal_lpdf(y[t]|annual+ season, sigma[site[t],j]);
             } else{
               accumulator[i]= log(A_ij[i,j])+
                               normal_lpdf(y[t]|annual, sigma[site[t],j]);
             }
             //if data is continuous consider previous data points' likelihood
             if(reset[t]==0 || reset[t]==2) accumulator[i] += unalpha_tk[t-1,i];
           }
           if(j == 2){
             //Air
             annual= air_mean[rowe[t]]+ air_A[rowe[t]]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],j]*pi());
             accumulator[i]= log(A_ij[i,j])+ normal_lpdf(y[t]|annual, sigma[site[t],j]);
             //if data is continuous consider previous data points' likelihood
             if(reset[t]==0 || reset[t]==2) accumulator[i] += unalpha_tk[t-1,i];
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

  // Local Priors
    // Prior of temperature alpha parameter.
      alpha_w ~ lognormal(2, 1);

    // Model of temperature amplitude parameter.
      A[,1] ~ lognormal(2, 1);

    // Model tau
      tau_est[,1] ~ normal(0.85,.05);
      tau_est[,2] ~ normal(0.9,.05);

    // Remaining Variance Priors
      sigma[,1] ~ student_t(10,1,2);
      sigma[,2] ~ student_t(10,4,2);

  // Enforce order on water amplitude.
    A[,2] ~ normal(air_A,0.01);

  // Return log probabilities for each model.
    for(t in 1:R){
      // Note: update based only on last unalpha_tk for each year
      target += log_sum_exp(unalpha_tk[accum[t]]);
    }
}

generated quantities {
  // Water model temporary estimates of mean and variance
  real annual_gen;    // mean temperature
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
      // Reset logbeta at each new site or data gap
      if(reset[t]==1){
        logbeta[t-1,j] = 1;
      }
      // Otherwise, provide probability information for next datapoint.
        for (i in 1:K) {  // i in next (t)
                          // Murphy (2012) Eq. 17.58
                          // backwards t + transition prob + local evidence at t
          if(i == 1) {
            annual_gen= alpha_w[rowe[t]]+ A[rowe[t],i]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],i]*pi());
            if(precip == 1){
              season_gen = snow[rowe[t]]*cos(2*pi()*d[t]/(n[t]/2)+(tau_est[site[t],i]+0.5)*pi());
              //season_gen = stream_snow(n[t],d[t],tau_est[site[t],j])*snow[rowe[t]];
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              normal_lpdf(y[t]|annual_gen + season_gen, sigma[site[t],i]);
            } else{
               accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              normal_lpdf(y[t]|annual_gen, sigma[site[t],i]);
            }
          }
          if(i == 2){
            annual_gen= air_mean[rowe[t]]+ air_A[rowe[t]]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],i]*pi());
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              normal_lpdf(y[t]|annual_gen, sigma[site[t],i]);
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
