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
  vector<lower=0>[S] air_A;       // PRISM air annual temperature range (i.e., amplitude)
  vector[S] air_mean;             // PRISM air mean temperature estimate
  vector<lower=0>[S] water_A;     // Water amplitude approximation given air_A
  vector<lower=0>[S] water_mean;  // Water mean approximation given air_mean
  real<lower=0,upper=1> prior;    // 1 if only priors are to be used
}

parameters {
  // Discrete state model
  simplex[K] A_ij[K]; // Transition probabilities
                      // A_ij[i][j] = p(z_t = j | z_{t-1} = i)

  //Global
  real b_alpha_w;                 // Mean water temperature at mean air of 0ºC
  real<lower=0> m_alpha_w;        // Change in mean water temperature with 1ºC increase in mean air T
  real<lower=0> sigma_alpha_w;    // Mean water temperature variance
  real<lower=0> sigma_A;          // Water amplitude variance
  real<lower=0> mu_sigma_Water;   // Mean water temperature variance across sites
  real<lower=0> sigma_Water;      // Variance around the global mean water temperature variance
  real<lower=0> mu_sigma_Air;     // Mean air temperature variance across sites
  real<lower=0> sigma_Air;        // Variance around the global mean air temperature variance

  // Seasonal Temperature Model for Air and Water
  vector[S] alpha_a;                      // Mean annual air temperature
  vector<lower=0>[S] alpha_w;             // Mean annual water temperature
  positive_ordered[2] A[S];               // Annual temperature amplitude
  vector[2] tau_est[S];                  // Seasonal location parameter
  positive_ordered[2] sigma[S];           // Seasonal variance parameter
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
        season[1]= sigma[site[t],2];
        unalpha_tk[t,1]= log(0.5)+ student_t_lpdf(y[t]|3, mu[1], season[1]);

        //Water
        mu[2]= alpha_w[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
        season[2]= sigma[site[t],1];
        unalpha_tk[t,2]= log(0.5)+ student_t_lpdf(y[t]|3, mu[2], season[2]);
      }
      else {
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                            // Murphy (2012) Eq. 17.48
                            // belief state + transition prob + local evidence at t
              if(j == 1){
                //Air
                mu[j]= alpha_a[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
                season[j]= sigma[site[t],2];
                accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                student_t_lpdf(y[t]|3, mu[j], season[j]);
              }
              if(j == 2){
                //Water
                mu[j]= alpha_w[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
                season[j]= sigma[site[t],1];
                accumulator[i]= unalpha_tk[t-1,i] + log(A_ij[i,j]) +
                                 student_t_lpdf(y[t]|3, mu[j], season[j]);
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

  // Global Priors
    //Mean water temperature
      m_alpha_w ~ normal(.1,.5);
      b_alpha_w ~ normal(1,1);
      sigma_alpha_w ~ student_t(3,0,1);
    //Water temperature amplitude
      sigma_A ~ student_t(3,0.5,1);
    //Water amplitude variance
      mu_sigma_Water ~ student_t(3,1,1);
      sigma_Water ~ student_t(3,0,1);
    //Air amplitude variance
      mu_sigma_Air ~ student_t(3,2,1);
      sigma_Air ~ student_t(3,0,1);

  // Local Priors
    // Prior of temperature alpha parameter.
    alpha_w ~ normal(water_mean, 3);
    alpha_a ~ normal(air_mean, 2);

    // Model of temperature amplitude parameter.
    A[,1] ~ normal(water_A, 3);
    A[,2] ~ normal(air_A, 2);

    // Model tau
    tau_est[,1] ~ normal(tau-0.01,.05);
    tau_est[,2] ~ normal(tau+0.01,.05);

    // Remaining Variance Priors
    sigma[,1] ~ student_t(3,1.5,2);
    sigma[,2] ~ student_t(3,3,2);

  // Global mean temperature, amplitude and variance models
  log(alpha_w) ~ normal(b_alpha_w + m_alpha_w*alpha_a, sigma_alpha_w);   //mean
  A[,1] ~ normal(alpha_w, sigma_A);                                      //amplitude
  sigma[,2] ~ normal(mu_sigma_Air, sigma_Air);                           //air
  sigma[,1] ~ normal(mu_sigma_Water, sigma_Water);                       //water

  // Return log probabilities for each model.
  if(prior == 1){
    target += -log(alpha_w); // Add the Jacobian Adjustment
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
              mu_gen[j]= alpha_a[site[t]]+ A[site[t],2]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],2]*pi());
              season_gen[j]= sigma[site[t],2];
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                                 student_t_lpdf(y[t]|3, mu_gen[j], season_gen[j]);
            }
            if(j == 2){
              //Water
              mu_gen[j]= alpha_w[site[t]]+ A[site[t],1]*cos(2*pi()*d[t]/n[site[t]]+ tau_est[site[t],1]*pi());
              season_gen[j]= sigma[site[t],1];
              accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                                student_t_lpdf(y[t]|3, mu_gen[j], season_gen[j]);
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
    }

    for(t in 1:N){
      gamma[t] = normalize(loggamma[t]);
    }
  } //forward-backward
}
