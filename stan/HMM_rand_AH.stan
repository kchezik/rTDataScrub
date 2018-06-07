functions {
  vector normalize(vector x) {
    return x / sum(x);
  }
}

data {
  int<lower=1> N;                 // Number of observations
  int<lower=1> S;                 // Number of sites
  int<lower=1> site[N];           // Site ID
  int<lower=1> K;                 // Number of hidden states
  real y[N];                      // Observations
  real<lower=0> d[N];             // Data points d in n
  vector<lower=0,upper=1>[S] tau; // Define location of annual cycle
  real<lower=0> n[S];             // Define the number of data points in a annual cycle
  vector[S] air_A;                // PRISM air annual temperature range (i.e., amplitude)
  vector[S] air_mean;             // PRISM air mean temperature estimate
  vector[S] water_A;              // Water amplitude approximation given air_A
  vector[S] water_mean;           // Water mean approximation given air_mean
  real prior;                     // 1 if only priors are to be used
}

parameters {
  // Discrete state model
  simplex[K] p_1k[S]; // Initial state probabilities
  simplex[K] A_ij[K]; // Transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  //Global
  real b;                 // Mean water temperature at mean air of 0ºC
  real<lower=0> m;        // Change in mean water temperature with 1ºC increase in mean air T
  real<lower=0> sigma_j;  // Mean water temperature variance
  real b_A;               // Annual temperature amplitude at 0ºC mean annual water T
  real<lower=0> m_A;      // Change in amplitude with 1ºC change in mean annual water T
  real<lower=0> sigma_A;  // Water amplitude variance

  // Seasonal Temperature Model for Air and Water
  vector[S] alpha_a;                          // Mean annual air temperature
  vector<lower = 0>[S] alpha_w;               // Mean annual water temperature
  positive_ordered[2] A[S];                   // Annual temperature amplitude
  vector<lower = 0, upper = 1>[2] tau_est[S]; // Seasonal location parameter
  positive_ordered[2] sigma[S];               // Seasonal variance parameter

  // Ground model
  vector<lower=0>[S] sigma_g; // Ground temperature variance
}

transformed parameters {
  // Water model temporary estimates of mean and variance
  real mu[2];     // mean temperature
  real season[2]; // seasonal error term

  // log probabilities
  vector[K] unalpha_tk[N];

  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    for (t in 1:N) {
      // initial estimate
      if(d[t] == 1){
        //Air
        mu[1]= alpha_a[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
        season[1]= exp(cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi()))*sigma[site[t],2];
        unalpha_tk[t,1]= log(p_1k[site[t],1])+ normal_lpdf(y[t]| mu[1], season[1]);

        //Water
        mu[2]= alpha_w[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
        season[2]= exp(cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi()))*sigma[site[t],1];
        unalpha_tk[t,2]= log(p_1k[site[t],2])+ normal_lpdf(y[t]| mu[2], season[2]);

        //Ground
        unalpha_tk[t,3] = log(p_1k[site[t],3])+
          normal_lpdf(y[t]| alpha_w[site[t]], sigma_g[site[t]]);
      }
      else {
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                            // Murphy (2012) Eq. 17.48
                            // belief state + transition prob + local evidence at t
              if(j == 1){
                //Air
                mu[1]= alpha_a[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
                season[1]= exp(cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi()))* sigma[site[t],2];
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+ normal_lpdf(y[t]| mu[1], season[1]);
              }
              if(j == 2){
                //Water
                mu[2]= alpha_w[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
                season[2]= exp(cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi()))* sigma[site[t],1];
                accumulator[i]= unalpha_tk[t-1,i] + log(A_ij[i,j]) +
                                 normal_lpdf(y[t]| mu[2], season[2]);
              }
              if(j == 3){
                //Ground
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                  normal_lpdf(y[t]| alpha_w[site[t]], sigma_g[site[t]]);
              }
          }
          unalpha_tk[t, j] = log_sum_exp(accumulator);
        }
      }
    }
  } // Forward
}

model {
  // Global Priors
  m ~ normal(1,2);
  b ~ normal(0,10);
  sigma_j ~ student_t(3,0,1);
  m_A ~ normal(1,1);
  b_A ~ normal(0,2);
  sigma_A ~ student_t(3,0,1);

  // Prior of temperature alpha parameter.
  alpha_w ~ normal(water_mean, 3);
  alpha_a ~ normal(air_mean, 1);

  // Model of temperature amplitude parameter.
  A[,1] ~ normal(water_A, 3);
  A[,2] ~ normal(air_A, 1);

  // Model tau
  tau_est[,1] ~ normal(tau,.1);
  tau_est[,2] ~ normal(tau,.1);

  // Remaining Variance Priors
  sigma[,1] ~ student_t(3,.75,1);
  sigma[,2] ~ student_t(3,.75,1);

  // Ground Priors
  sigma_g ~ normal(0,1);

  // Global mean temperature and amplitude models
  alpha_w ~ lognormal(log(b + m*alpha_a), sigma_j);
  A[,1] ~ normal(b_A + m_A*alpha_w, sigma_A);

  // Return log probabilities for each model.
  if(prior == 1){
    target += log_sum_exp(unalpha_tk[N]); // Note: update based only on last unalpha_tk
  }
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
      // Reset logbeta at each new site
      if((d[t-1]-d[t])>0){
        for (j in 1:K)
          logbeta[t-1,j] = 1;
      }
      // Otherwise, provide probability information to next datapoint.
      else {
        for (j in 1:K) {    // j = previous (t-1)
          for (i in 1:K) {  // i in next (t)
                            // Murphy (2012) Eq. 17.58
                            // backwards t + transition prob + local evidence at t
            if(j == 1) {
              //Air
              mu_gen[1]= alpha_a[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
              season_gen[1]= exp(cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi()))* sigma[site[t],2];
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+ normal_lpdf(y[t]| mu_gen[1], season_gen[1]);
            }
            if(j == 2){
              //Water
              mu_gen[2]= alpha_w[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
              season_gen[2]= exp(cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi()))* sigma[site[t],1];
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+ normal_lpdf(y[t]| mu_gen[2], season_gen[2]);
            }
            if(j == 3){
              //Ground
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                normal_lpdf(y[t]| alpha_w[site[t]], sigma_g[site[t]]);
            }
          }
          logbeta[t-1,j] = log_sum_exp(accumulator);
        }
      }
    }

    for (t in 1:N)
      beta[t] = softmax(logbeta[t]);
  } // Backward

  { // forward-backward algorithm log p(z_t = j | y_{1:T})
    for(t in 1:N) {
      loggamma[t] = alpha_tk[t] .* beta[t];
      if(sum(loggamma[t])==0)
        loggamma[t] = alpha_tk[t];
    }

    for(t in 1:N){
      gamma[t] = normalize(loggamma[t]);
      for(i in 1:K){
        if(is_inf(gamma[t,i]) || is_nan(gamma[t,i])){
          print("t: ",t," log gamma: ",loggamma[t]," beta: ",beta[t]," alpha_tk: ",alpha_tk[t]," logbeta: ",logbeta[t]);
        }
      }
    }
  } //forward-backward
}
