
data { 
  int N;                              // Number of time periods
  int n_fourier;                      // Fourier series explansion
  matrix[N, n_fourier * 2] X;         // Seasonality matrix
  vector[N] y;                        // Time series
  real<lower=0> sigma_beta;           // Scale on seasonality prior
  
} 
 
parameters { 
    
  vector[n_fourier * 2] beta;        // Seasonality regressor coefficients
  real<lower=0> sigma_s;             // Observation noise

} 

transformed parameters { 
  
  vector[N] mu_s; 
  
  mu_s = X * beta;
   
} 
 
 
model { 

  // Priors
  
  beta ~ normal(0, sigma_beta);
  sigma_s ~ normal(0, 0.5);
  
  // Linear likelihood
  
  y ~ normal(mu_s, sigma_s);

} 

generated quantities { 
    
    vector[N] y_hat;
    for (n in 1:N)
        y_hat[n] = normal_rng(mu_s[n], sigma_s);  
}

  