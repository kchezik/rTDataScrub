data {
  int<lower=0> S;     // Number of sites
  real air_mean[S];  // BC Climate mean air temperature estimate
}
parameters {
  real<lower=0> alpha_max;  // Maximum mean water temperature.
  real<lower=0> k;          // Rate of change between lower and upper bounds.
  real<lower=0> m;          // Mid-point between alpha_max and w_low.
  real w_low;               // Mean water temperature's lower bound.
}
transformed parameters {
  real alpha_water[S];  // Mean water temperature estimate

  for(s in 1:S){
    alpha_water[s] = (alpha_max/(1+exp(-k*(air_mean[s]-m))))+w_low;
  }
}
model {
  // priors
  alpha_max ~ student_t(3,27,2);
  k ~ normal(.1,.02);
  w_low ~ normal(0,.1);
  m ~ student_t(3,16,2);
}
