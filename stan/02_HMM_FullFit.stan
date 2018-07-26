functions {
  vector normalize(vector x) {
    return x / sum(x);
  }

// Creates a cosine curve upper bound by zero during the spring ...
// ... to account for hysteresis in the water temperature curve ...
// ... due to snow melt cooling the stream. The values of this
// ... cosine curve is modified by a spring snow effect coefficient.

  real stream_snow(real n, real d, real tau) {
    real d_low = (n/2)-(tau * (n/2));
    real d_high = n-(tau * (n/2));
    if(d>d_low && d<d_high) {
      return (cos(2*pi()*(d-d_low)/(n/2))-1);
    } else return 0;
  }

// Creates a cosine curve during the spring period to ...
// ... linearize the spring increase in temperature.
// The values of the cosine curve would be modified by ...
// ... an air temperature spring coefficient.

//  real air_snow(real n, real d, real tau) {
//    real d_low = (n/2)-(tau * (n/2));
//    real d_high = n-(tau * (n/2));
//    if(d>d_low && d<d_high) {
//      return (cos(2*pi()*(d-d_low)/(n/2)+1.5*pi()));
//    } else return 0;
//  }
}

data {
  int<lower=1> N;     // number of observations (length)
  int<lower=1> K;     // number of hidden states
  real y[N];          // observations
  real n[N];             // define the number of data points in a annual cycle
  real<lower=0> d[N]; // data point d in n
  real tau;           // define location of annual cycle
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
  real A;
  ordered[2] tau_est;
  real<lower=0, upper=1> snow_w;
  positive_ordered[2] sigma;
}

transformed parameters {
  // Water model temporary estimates of mean and variance
  real mu[2];     // mean temperature
  real spring;    // snow effect terms

  // log probabilities
  vector[K] unalpha_tk[N];

  { // Forward algorithm log p(z_t = j | x_{1:t})
    real accumulator[K];

    for (t in 1:N) {
      if(t == 1){
        //Water
        mu[1]= alpha_water+ A[1]*cos(2*pi()*d[t]/n[t]+ tau_est[1]*pi());
        spring = stream_snow(n[t],d[t],tau_est[1]); //Determine the snow effect
        unalpha_tk[t,1]= log(0.5)+ student_t_lpdf(y[t]|3, mu[1] + spring*snow_w, sigma[1]);
        //Air
        mu[2]= alpha_air+ A[2]*cos(2*pi()*d[t]/n[t]+ tau_est[2]*pi());
        unalpha_tk[t,2]= log(0.5)+ student_t_lpdf(y[t]|3, mu[2], sigma[2]);
      }
      else{
        for (j in 1:K) {    // j = current (t) or transition state column.
          for (i in 1:K) {  // i = previous (t-1) or transition state row.
                           // Murphy (2012) Eq. 17.48
                           // belief state + transition prob + local evidence at t

            if(j == 1){
              mu[j]= alpha_water+ A[j]*cos(2*pi()*d[t]/n[t]+ tau_est[j]*pi());
              spring = stream_snow(n[t],d[t],tau_est[j]);
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                student_t_lpdf(y[t]|3, mu[j] + spring*snow_w, sigma[j]);
            }
            if(j == 2){
              mu[j]= alpha_air+ A[j]*cos(2*pi()*d[t]/n[t]+ tau_est[j]*pi());
              accumulator[i]= unalpha_tk[t-1,i]+ log(A_ij[i,j])+
                                student_t_lpdf(y[t]|3, mu[j], sigma[j]);
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
    alpha_air ~ normal(air_mean, 0.2);

  // Model of temperature amplitude parameter.
    A[1] ~ normal(water_A, 3);
    A[2] ~ normal(air_A, 0.2);

  // Model tau
    tau_est[1] ~ normal(tau-0.01,.03);
    tau_est[2] ~ normal(tau+0.01,.03);

  // Spring snow adjustment
    snow_w ~ normal(.5,.5);

  // Remaining Variance Priors
    sigma ~ student_t(3,1,2);

  // Return log probabilities for each model.
    target += log_sum_exp(unalpha_tk[N]); // Note: update based only on last unalpha_tk
}

generated quantities {
  // Water model temporary estimates of mean and variance
  real mu_gen[2];     // mean temperature
  real spring_gen;    // snow effect adjustment

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
            mu_gen[j]= alpha_water+ A[j]*cos(2*pi()*d[t]/n[t]+ tau_est[j]*pi());
            spring_gen = stream_snow(n[t],d[t],tau_est[j]);
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              student_t_lpdf(y[t]|3, mu_gen[j] + spring_gen*snow_w, sigma[j]);
          }
          if(j == 2){
            mu_gen[j]= alpha_air+ A[j]*cos(2*pi()*d[t]/n[t]+ tau_est[j]*pi());
            accumulator[i]= logbeta[t,i]+ log(A_ij[j,i])+
                              student_t_lpdf(y[t]|3, mu_gen[j], sigma[j]);
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
