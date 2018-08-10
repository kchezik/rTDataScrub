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
  int<lower=1> N;                 // Number of observations
  int<lower=1> S;                 // Number of sites
  int<lower=1> site[N];           // Site ID
  int<lower=1> K;                 // Number of hidden states
  real y[N];                      // Observations
  real<lower=0> d[N];             // Data point d in n
  real<lower=0,upper=2> reset[N]; // First observation at a site (no=0, forwardYes=1, backwardYes=2)
  vector<lower=0,upper=2>[S] tau; // Define location of annual cycle
  real<lower=0> n[N];             // Define the number of data points in a annual cycle
  vector<lower=0>[S] air_A;       // PRISM air annual temperature range (i.e., amplitude)
  vector[S] air_mean;             // PRISM air mean temperature estimate
  vector<lower=0>[S] water_A;     // Water amplitude approximation given air_A
  vector<lower=0>[S] water_mean;  // Water mean approximation given air_mean
}

parameters {
  // Discrete state model
  simplex[K] A_ij[K]; // Transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  // Seasonal Temperature Model for Air and Water
  vector<lower=0>[S] alpha_w;             // Mean annual water temperature
  positive_ordered[2] A[S];               // Annual temperature amplitude
  ordered[2] tau_est[S];                  // Seasonal location parameter
  vector<lower=0,upper=3>[S] snow_w;      // Snow effect parameter
  positive_ordered[2] sigma[S];           // Seasonal variance parameter
}

transformed parameters {
  // Water model temporary estimates of mean and variance
  real mu[2];     // mean temperature
  real spring;    // spring effect on water

  // log probabilities
  vector[K] unalpha_tk[N];

  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    for (t in 1:N) {
      // initial estimate
      if(reset[t] == 1){
        //Water
        mu[1]= alpha_w[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],1]*pi());
        spring = stream_snow(n[t],d[t],tau_est[site[t],1]);
        unalpha_tk[t,1]= log(0.5)+ student_t_lpdf(y[t]|3, (mu[1]+ spring* snow_w[site[t]]), sigma[site[t],1]);

        //Air
        mu[2]= air_mean[site[t]]+ air_A[site[t]]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],2]*pi());
        unalpha_tk[t,2]= log(0.5)+ student_t_lpdf(y[t]|3, mu[2], sigma[site[t],2]);
      }
      else {
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                            // Murphy (2012) Eq. 17.48
                            // belief state + transition prob + local evidence at t
              if(j == 1){
                //Water
                mu[j]= alpha_w[site[t]]+ A[site[t],j]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],j]*pi());
                spring = stream_snow(n[t],d[t],tau_est[site[t],j]);
                accumulator[i]= unalpha_tk[t-1,i] + log(A_ij[i,j]) +
                                 student_t_lpdf(y[t]|3, (mu[j]+ spring* snow_w[site[t]]), sigma[site[t],j]);
              }
              if(j == 2){
                //Air
                mu[j]= air_mean[site[t]]+ air_A[site[t]]*cos(2*pi()*d[t]/n[t]+ tau_est[site[t],j]*pi());
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                student_t_lpdf(y[t]|3, mu[j], sigma[site[t],j]);
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

  // Local Priors
    // Prior of temperature alpha parameter.
      alpha_w ~ normal(water_mean, 5);

    // Model of temperature amplitude parameter.
      A[,1] ~ normal(water_A, 5);

    // Model tau
      tau_est[,1] ~ normal(tau-0.02,.03);
      tau_est[,2] ~ normal(tau+0.02,.03);

    // Remaining Variance Priors
      sigma[,1] ~ student_t(3,1.5,2);
      sigma[,2] ~ student_t(3,3,2);

  // Enforce order on water amplitude.
    A[,2] ~ normal(air_A,0.01);

  // Return log probabilities for each model.
    target += log_sum_exp(unalpha_tk[N]); // Note: update based only on last unalpha_tk
}

generated quantities {
  // forward filtered estimates
  vector[K] alpha_tk[N];

  { // Forward algorithm
    for (t in 1:N)
      alpha_tk[t] = softmax(unalpha_tk[t]);
  } // Forward
}
