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
  simplex[K] A_ij[K]; // transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Seasonal Temperature Model for Air and Water
  real alpha_air;
  real alpha_water;
  positive_ordered[2] A;
  real tau_est[2];
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

    mu[1]= alpha_air+ A[2]*cos(2*pi()*1/n+ tau_est[2]*pi());
    season[1]= (cos(2*pi()*1/n+ tau_est[2]*pi())+2)*sigma[2];
    unalpha_tk[1,1]= log(.495)+ normal_lpdf(y[1]| mu[1], season[1]);

    mu[2]= alpha_water+ A[1]*cos(2*pi()*1/n+ tau_est[1]*pi());
    season[2]= (cos(2*pi()*1/n+ tau_est[1]*pi())+2)*sigma[1];
    unalpha_tk[1,2]= log(.495)+ normal_lpdf(y[1]| mu[2], season[2]);

    unalpha_tk[1,3]= log(.01)+ normal_lpdf(y[1]| alpha_water, sigma_g);

    for (t in 2:N) {
      for (j in 1:K) {    // j = current (t) or transition state column.
        for (i in 1:K) {  // i = previous (t-1) or transition state row.
                          // Murphy (2012) Eq. 17.48
                          // belief state + transition prob + local evidence at t
            if(j == 1){
              mu[1]= alpha_air+ A[2]*cos(2*pi()*t/n + tau_est[2]*pi());
              season[1]= (cos(2*pi()*t/n+ tau_est[2]*pi())+2)*sigma[2];
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                normal_lpdf(y[t]| mu[1], season[1]);
            }
            if(j == 2){
              mu[2]= alpha_water+ A[1]*cos(2*pi()*t/n+ tau_est[1]*pi());
              season[2]= (cos(2*pi()*t/n+ tau_est[1]*pi())+2)*sigma[1];
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                normal_lpdf(y[t]| mu[2], season[2]);
            }
            if(j == 3){
              accumulator[i] = unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                normal_lpdf(y[t]| alpha_water, sigma_g);
            }
        }
        unalpha_tk[t, j] = log_sum_exp(accumulator);
      }
    }
  } // Forward
}

model {
  // Prior of temperature alpha parameter.
  alpha_water ~ normal(water_mean, 3);
  alpha_air ~ normal(air_mean, 1);

  // Model of temperature amplitude parameter.
  A[1] ~ normal(water_A, 3);
  A[2] ~ normal(air_A, 1);

  // Model tau
  tau_est[1] ~ normal(tau,.01);
  tau_est[2] ~ normal(tau,.01);

  // Remaining Variance Priors
  sigma[1] ~ normal(.75,1);
  sigma[2] ~ normal(1.5,1);
  // Ground Priors
  sigma_g ~ normal(1,.5);

  // Return log probabilities for each model.
  target += log_sum_exp(unalpha_tk[N]); // Note: update based only on last unalpha_tk
}

generated quantities {
  // forward filtered estimates
  vector[K] alpha_tk[N];

  { // Forward algortihm
    for (t in 1:N)
      alpha_tk[t] = softmax(unalpha_tk[t]);
  } // Forward
}
