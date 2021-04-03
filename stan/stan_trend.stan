
data { 
  int N;                              // Number of time periods
  int N_changepoints;                 // Number of changepoints
  vector[N_changepoints] s;           // Changepoints
  matrix[N, N_changepoints] A;        // Changepoint Matrix
  vector[N] t;                        // Time
  vector[N] y;                        // Time series
  real<lower=0> tau;                  // Scale on changepoints prior
 
} 
 
parameters { 
    
  vector[N_changepoints] delta;      // Trend rate adjustments
  real k;                            // Base trend growth rate
  real m;                            // Trend offset
  real<lower=0> sigma_trend;         // Observation noise

} 
 
transformed parameters { 
  
  vector[N_changepoints] gamma;
  vector[N] mu_trend; 
  
  gamma = -s .* delta;
  mu_trend = k + A * delta .* t + (m + A * gamma);
   
} 
 
model { 

  // Priors
  
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  sigma_trend ~ normal(0, 0.5);
  
  // Linear likelihood
  
  y ~ normal(mu_trend, sigma_trend);

} 

generated quantities { 
    
    vector[N] y_hat;
    for (n in 1:N)
        y_hat[n] = normal_rng(mu_trend[n], sigma_trend);  
}
