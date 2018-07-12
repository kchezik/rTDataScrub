data {
  int<lower=1> N;                   // number of observations
  int<lower=1> S;                   // number of sites
  int<lower=1> site[N];             // site x observation
  real y[N];                        // temperature observations
  real<lower=1> d[N];               // data points d in n
  vector<lower=0,upper=1>[S] tau;   // starting annual cycle location estimate
  real<lower=1> n[S];               // define the number of data points in a annual cycle
  vector[S] air_mean;                 // BC Climate/PRISM mean air temperature estimate
  int<lower=0,upper=1> prior_only;  // simulate given the priors (1) or use the data (0)
}

parameters {
  //Global
    real b_alpha_w;                 // Mean water temperature at mean air of 0ºC
    real<lower=0> m_alpha_w;        // Change in mean water temperature with 1ºC increase in mean air T
    real<lower=0> sigma_alpha_w;    // Mean water temperature variance

  // Seasonal water temperature model
    vector<lower=0>[S] alpha_w;
    vector<lower=0>[S] A;         // Amplitude estimate.
    vector[S] tau_est; // Estimate seasonal cycle given tau inits.

  // Variance parameter for each site.
    real<lower=0> sigma[S]; // Variance around the mean.
}

transformed parameters {
  // Water model temporary estimates
  real mu;      // mean temperature
  real season;  // seasonal error term

  // Accumulate log probabilities.
  vector[N] accumulator;

  // Fit coefs to temperature data across multiple sites
  for (i in 1:N){
    //estimate mean water temperature
    mu = alpha_w[site[i]] + A[site[i]] * cos(2*pi()*d[i]/n[site[i]] + tau_est[site[i]]*pi());
    //estimate seasonal variance around mu
    season = sigma[site[i]];
    //estimate probability of coefs given the data
    accumulator[i] = student_t_lpdf(y[i] |3, mu, season);
  }
}

model {
  // Global priors
    m_alpha_w ~ normal(.1,.5);
    b_alpha_w ~ normal(1,1);
    sigma_alpha_w ~ student_t(3,.5,1);

  // Local variance prior
    sigma ~ student_t(3,1,2);

  // Local seasonal cycle location priors
    tau_est ~ normal(tau,.01);

  // Global Model
    log(alpha_w) ~ normal(b_alpha_w + m_alpha_w*air_mean, sigma_alpha_w);

  // Gather model
  if(!prior_only) {
    // Return log probabilities.
    target += -log(alpha_w);
    target += sum(accumulator);
  }
}
