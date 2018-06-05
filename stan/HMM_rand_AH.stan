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
  real y[N];                      // observations
  real<lower=0> d[N];             // data points d in n
  vector<lower=0,upper=1>[S] tau; // define location of annual cycle
  real<lower=0> n[S];             // define the number of data points in a annual cycle
  vector[S] air_A;                // PRISM air annual temperature range (i.e., amplitude)
  vector[S] air_mean;             // PRISM air mean temperature estimate
  real prior;
}

parameters {
  // Discrete state model
  simplex[K] p_1k[S]; // initial state probabilities
  simplex[K] A_ij[K]; // transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Seasonal Temperature Model for Air and Water
  vector<lower=0>[S] alpha_water;
  vector[S] alpha_air;
  positive_ordered[2] A[S];
  vector<lower = 0, upper = 1>[2] tau_est[S];

  // Ground model
  vector<lower = 0>[S] t0;

  // Unaccounted for variance parameter for each model.
  positive_ordered[2] sigma[S];
  vector<lower=0>[S] sigma_g;
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
      // initial estimate
      if(d[t] == 1){
        //Air
        mu[1]= alpha_air[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
        season[1]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi())+2)*sigma[site[t],2];
        unalpha_tk[t,1]= log(p_1k[site[t],1])+ normal_lpdf(y[t]| mu[1], season[1]);

        //Water
        mu[2]= alpha_water[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
        season[2]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi())+2)*sigma[site[t],1];
        unalpha_tk[t,2]= log(p_1k[site[t],2])+ normal_lpdf(y[t]| mu[2], season[2]);

        //Ground
        unalpha_tk[t,3] = log(p_1k[site[t],3])+
          normal_lpdf(y[t]| t0[site[t]], sigma_g[site[t]]);
      }
      else {
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                            // Murphy (2012) Eq. 17.48
                            // belief state + transition prob + local evidence at t
              if(j == 1){
                mu[1]= alpha_air[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
                season[1]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi())+2)* sigma[site[t],2];
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+ normal_lpdf(y[t]| mu[1], season[1]);
              }
              if(j == 2){
                mu[2]= alpha_water[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
                season[2]= (cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi())+2)* sigma[site[t],1];
                accumulator[i]= unalpha_tk[t-1,i] + log(A_ij[i,j]) +
                                 normal_lpdf(y[t]| mu[2], season[2]);
              }
              if(j == 3){
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                  normal_lpdf(y[t]| t0[site[t]], sigma_g[site[t]]);
              }
          }
          unalpha_tk[t, j] = log_sum_exp(accumulator);
        }
      }

    }
  } // Forward
}

model {
  // Prior of temperature alpha parameter.
  for(s in 1:S){
    if(air_mean[s]>0)
      alpha_water[s] ~ normal(air_mean[s], 3);
    else
      alpha_water[s] ~ normal(0, 3);
  }
  alpha_air ~ normal(air_mean, 1);

  // Model of temperature amplitude parameter.
  A[,2] ~ normal(air_A, 1);

  // Model tau
  tau_est[,1] ~ normal(tau,.03);
  tau_est[,2] ~ normal(tau,.03);

  // Remaining Variance Priors
  sigma[,1] ~ student_t(3,.75,1);
  sigma[,2] ~ student_t(3,.75,1);

  // Ground Priors
  t0 ~ normal(alpha_water, 3);
  sigma_g ~ student_t(3,.75,1);

  for(s in 1:S){
    print(" air: ",alpha_air[s]," water: ", alpha_water[s]);
  }

  // Return log probabilities for each model.
  if(prior == 1){
    target += log_sum_exp(unalpha_tk[N]); // Note: update based only on last unalpha_tk
  }
}

generated quantities {
  // forward filtered estimates
  vector[K] alpha_tk[N];

  { // Forward algorithm
    for (t in 1:N)
      alpha_tk[t] = softmax(unalpha_tk[t]);
  } // Forward
}
