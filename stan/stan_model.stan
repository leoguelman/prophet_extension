
data { 
  int N;                              // Number of time periods
  int N_changepoints;                 // Number of changepoints
  vector[N_changepoints] s;           // Changepoints
  matrix[N, N_changepoints] A;        // Changepoint Matrix
  vector[N] t;                        // Time
  int n_fourier;                      // Fourier series expansion
  matrix[N, n_fourier * 2] X;         // Seasonality matrix
  vector[N] y;                        // Time series
  real<lower=0> tau;                  // Scale on changepoints prior
  real<lower=0> sigma_beta;           // Scale on seasonality prior
  
} 
 
parameters { 
    
  vector[N_changepoints] delta;      // Trend rate adjustments
  real k;                            // Base trend growth rate
  real m;                            // Trend offset
  vector[n_fourier * 2] beta;        // Seasonality regressor coefficients
  real<lower=0> sigma;               // Observation noise

} 

transformed parameters { 
  
  vector[N_changepoints] gamma;
  vector[N] mu_trend; 
  vector[N] mu_s;
  
  gamma = -s .* delta;
  mu_trend = k + A * delta .* t + (m + A * gamma);
  mu_s = X * beta;
   
} 
 
model { 

  // Priors
  
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  beta ~ normal(0, sigma_beta);
  sigma ~ normal(0, 0.5);
  
  // Linear likelihood
  
  y ~ normal(mu_trend + mu_s, sigma);

} 

generated quantities { 
    
    vector[N] y_hat;
    for (n in 1:N)
        y_hat[n] = normal_rng(mu_trend[n] + mu_s[n], sigma);  
}
