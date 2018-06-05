functions {
  vector normalize(vector x) {
    return x / sum(x);
  }
}

data {
  int<lower=1> N;                 // number of observations
  int<lower=1> S;                 // number of sites
  int<lower=1> site[N];           // site ID
  int<lower=1> K;                 // number of hidden states
  real y[N];             // observations
  real<lower=0> d[N];             // data points d in n
  vector<lower=0,upper=1>[S] tau; // define location of annual cycle
  real<lower=0> n[S];             // define the number of data points in a annual cycle
  vector[S] air_A;                // PRISM air annual temperature range (i.e., amplitude)
  vector[S] air_mean;             // PRISM air mean temperature estimate
}

parameters {
  // Discrete state model
  simplex[K] p_1k[S];    // initial state probabilities
  simplex[K] A_ij[K]; // transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Relationship between air and water across sites
  real<lower=0> alpha_max;  // Maximum mean water temperature
  real<lower=0.09, upper = 0.21> k;          // Rate of change between lower and upper bounds
  real w_low;      // Mean water temperatures lower bound
  real<lower=0> m;          // Inflection point

  // Seasonal Temperature Model for Air and Water
  vector[S] alpha_air;
  positive_ordered[2] A[S];
  vector<lower = 0, upper = 1>[2] tau_est[S];

  // Unaccounted for variance parameter for each model.
  positive_ordered[2] sigma[S];
  vector<lower=0>[S] sigma_g;
}

transformed parameters {
  // Mean temperature parameters for water and air
  real<lower = 0> alpha_water[S];

  // Water model temporary estimates of mean and variance
  real mu[2]; // mean temperature
  real season[2]; // seasonal error term

  // log probabilities
  vector[K] unalpha_tk[N];

  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    for (t in 1:N) {
      //estimate mean water temperature given mean air temperature estimates from PRISM
      alpha_water[site[t]] = (alpha_max/(1+exp(-k*(air_mean[site[t]]-m)))) + w_low;

      // initial estimate
      if(d[t] == 1){
        mu[1]= alpha_air[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
        season[1]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi())+2)*sigma[site[t],2];
        unalpha_tk[t,1]= log(p_1k[site[t],1])+ normal_lpdf(y[t]| mu[1], season[1]);

        mu[2]= alpha_water[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
        season[2]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi())+2)*sigma[site[t],1];
        unalpha_tk[t,2]= log(p_1k[site[t],2])+ normal_lpdf(y[t]| mu[2], season[2]);

        unalpha_tk[t,3] = log(p_1k[site[t],3])+ normal_lpdf(y[t]| alpha_water[site[t]], sigma_g[site[t]]);
      }
      else {
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                            // Murphy (2012) Eq. 17.48
                            // belief state + transition prob + local evidence at t
              if(j == 1){
                mu[1]= alpha_air[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
                season[1]= (cos(2*pi()*d[t]/n[site[t]] + tau_est[site[t],1]*pi())+2)* sigma[site[t],2];
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+ normal_lpdf(y[t]| mu[1], season[1]);
              }
              if(j == 2){
                mu[2]= alpha_water[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
                season[2]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi())+2)* sigma[site[t],1];
                accumulator[i]= unalpha_tk[t-1,i] + log(A_ij[i,j]) +
                                 normal_lpdf(y[t]| mu[2], season[2]);
              }
              if(j == 3){
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                  normal_lpdf(y[t]| alpha_water[site[t]], sigma_g[site[t]]);
              }
          }
          unalpha_tk[t, j] = log_sum_exp(accumulator);
        }
      }
    }
  } // Forward
}

model {
   // Global priors
   alpha_max ~ normal(27,2);
   k ~ lognormal(log(.11),.01);
   w_low ~ normal(0,.1);
   m ~ normal(16,1);

  // Prior of temperature alpha parameter.
  alpha_air ~ normal(air_mean, 1);

  // Model of temperature amplitude parameter.
  A[,2] ~ normal(air_A, 1);
  A[,1] ~ uniform(0,air_A);

  // Model tau
  tau_est[,1] ~ normal(tau,.03);
  tau_est[,2] ~ normal(tau,.03);

  // Remaining Variance Priors
  sigma[,1] ~ student_t(3,.75,1);
  sigma[,2] ~ student_t(3,.75,1);

  // Ground Priors
  sigma_g ~ student_t(3,.75,1);

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
              mu_gen[1]= alpha_air[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
              season_gen[1]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi())+2)* sigma[site[t],2];
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+ normal_lpdf(y[t]| mu_gen[1], season_gen[1]);
            }
            if(j == 2){
              mu_gen[2]= alpha_water[site[t]] + A[site[t],1]*cos(2*pi()*d[t]/n[site[t]] + tau_est[site[t],2]*pi());
              season_gen[2]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi())+2) * sigma[site[t],1];
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+ normal_lpdf(y[t]| mu_gen[2], season_gen[2]);
            }
            if(j == 3){
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                normal_lpdf(y[t]| alpha_water[site[t]], sigma_g[site[t]]);
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
