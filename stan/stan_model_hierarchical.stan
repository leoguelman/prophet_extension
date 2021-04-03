

data { 
  int N;                              // Number of time periods
  int N_changepoints;                 // Number of changepoints
  vector[N_changepoints] s;           // Changepoints
  matrix[N, N_changepoints] A;        // Changepoint Matrix
  vector[N] t;                        // Time
  int n_fourier;                      // Fourier series expansion
  matrix[N, n_fourier * 2] X;         // Seasonality matrix
  vector[N] y;                        // Time series
  int<lower=0> J;                     // Number of groups
  int<lower=1,upper=J> group[N];      // Group variable (integer from 1 to number of groups)
  
} 
 
parameters { 
    
  vector[J] k;                       // Base trend growth rate by group
  real m;                            // Trend offset
  vector[N_changepoints] delta[J];   // Trend rate adjustments by group
  vector[n_fourier * 2] beta[J];     // Seasonality regressor coefficients by group  
  vector[n_fourier * 2] mu_beta;     // beta global mean 
  real<lower=0> sigma_beta;          // Seasonality  noise global mean 
  vector[N_changepoints] mu_delta;   // delta global mean (changepoints)
  real<lower=0> sigma_delta;         // dela global noise (changepoints)
  real mu_k;                         // k global mean 
  real<lower=0> sigma_k;             // k global noise
  real<lower=0> sigma;               // Observation noise
 
} 

transformed parameters { 
  
  vector[N_changepoints] gamma[J];
  vector[N] mu_trend; 
  vector[N] mu_s;
  
  for(n in 1:N){
      gamma[group[n]] = -s .* delta[group[n]];
      mu_trend[n] = k[group[n]] + A[n] * delta[group[n]] .* t[n] + (m + A[n] * gamma[group[n]]);
      mu_s[n] = X[n] * beta[group[n]]; 
  } 
} 
 
model { 

  // Hyper-priors
  m ~ normal(0, 5);
  sigma ~ normal(0, 0.5);
  mu_beta ~ normal(0, 1);
  sigma_beta ~ normal(0, 1);
  mu_delta ~ normal(0, 1);
  sigma_delta ~ normal(0, 1);
  mu_k ~ normal(0, 1);
  sigma_k ~ normal(0,1);

  // Adaptive priors

  for(j in 1:J){
    k[j] ~ normal(mu_k, sigma_k);
    delta[j] ~ double_exponential(mu_delta, sigma_delta);
    beta[j] ~ normal(mu_beta, sigma_beta);
  }

  // Linear likelihood
  y ~ normal(mu_trend + mu_s, sigma);
} 

generated quantities { 
    
    vector[N] y_hat;
    for (n in 1:N)
        y_hat[n] = normal_rng(mu_trend[n] + mu_s[n], sigma);  
}
