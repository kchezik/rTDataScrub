data {
  int<lower=1> N;                   // number of observations
  int<lower=1> S;                   // number of sites
  int<lower=1> site[N];             // site x observation
  real y[N];                        // temperature observations
  real<lower=1> d[N];               // data points d in n
  vector<lower=0,upper=1>[S] tau;   // starting annual cycle location estimate
  real<lower=1> n[S];               // define the number of data points in a annual cycle
  real air_mean[S];                 // BC Climate/PRISM mean air temperature estimate
  int<lower=0,upper=1> prior_only;  // simulate given the priors (1) or use the data (0)
}

parameters {
  // Relationship between air and water across sites.
  real<lower=0> alpha_max;  // Maximum mean water temperature.
  real<lower=0.05> k;       // Rate of change between lower and upper bounds.
  real w_low;               // Mean water temperatures lower bound.
  real<lower=0> m;          // Inflection point.

  // Seasonal water temperature model
  vector <lower=0, upper=1>[S] tau_est; // Estimate seasonal cycle given tau inits.
  real<lower=0, upper=30> A[S];         // Amplitude estimate.

  // Variance parameter for each site.
  real<lower=0> sigma[S]; // Variance around the mean.
}

transformed parameters {
  // Site mean water temperature estimates
  real<lower=0> alpha_water[S];

  // Water model temporary estimates
  real mu;      // mean temperature
  real season;  // seasonal error term

  // Accumulate log probabilities.
  vector[N] accumulator;

  // Fit coefs to temperature data across multiple sites
  for (i in 1:N){
    //estimate mean water temperature given mean air temperature estimates from PRISM
    alpha_water[site[i]] = (alpha_max/(1+exp(-k*(air_mean[site[i]]-m))))+w_low;
    //estimate mean water temperature
    mu = alpha_water[site[i]] + A[site[i]] * cos(2*pi()*d[i]/n[site[i]] + tau_est[site[i]]*pi());
    //estimate seasonal variance around mu
    season = (cos(2*pi()*d[i]/n[site[i]] + tau_est[site[i]]) + 2) * sigma[site[i]];
    //estimate probability of coefs given the data
    accumulator[i] = normal_lpdf(y[i] |mu, season);
  }
}

model {
  // Global priors
  alpha_max ~ student_t(3,27,2);
  k ~ lognormal(log(.11),.01);
  w_low ~ normal(0,.1);
  m ~ student_t(3,16,1);

  // Local variance prior
  sigma ~ student_t(3,0,1);

  // Local seasonal cycle location priors
  tau_est ~ normal(tau,.01);

  // Gather model
  if(!prior_only) {
    // Return log probabilities.
    target += sum(accumulator);
  }
}
