Ross\_MCMC
================
Zhanhao Zhang
May 14, 2019

References
----------

[Markov Chain Monte Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)

Required Inputs
---------------

data: a csv with one column for time and some other columns for variables.
hyper-parameters:
  step-size for each parameter
  number of particles
  number of iterations

Packages Required
-----------------

``` r
library(reshape2)
library(deSolve)
library(foreach)
library(doParallel)
library(dplyr)
library(numDeriv)
```

Demo (Ross-Macdonald Model)
---------------------------

Registers workers, change the 4 to a larger number if you have a more powerful computer.

``` r
registerDoParallel(cores=4)
```

Read the data

``` r
ross_read <- read.csv("output1.csv")
```

Set the known parameter values

``` r
E = 11
r = 0.005
b = 0.55
a = 0.27
c = 0.15
H = 2000
```

Calculate the mean and standard deviation of each variable from a given start time to a given end time

``` r
start <- 366
end <- 5000
sd_M2 <- sd(ross_read$M[start:end])
sd_Y2 <- sd(ross_read$Y[start:end])
sd_Z2 <- sd(ross_read$Z[start:end])
sd_I2 <- sd(ross_read$I[start:end])

mean_M2 <- mean(ross_read$M[start:end])
mean_Y2 <- mean(ross_read$Y[start:end])
mean_Z2 <- mean(ross_read$Z[start:end])
mean_I2 <- mean(ross_read$I[start:end])
```

Step function

``` r
get_sim_df <- function(p, l, start_t, end_t){
  output_df <- data.frame(time=start_t, M=ross_read$M[start_t + 1], 
                          Y=ross_read$Y[start_t + 1], 
                          Z=ross_read$Z[start_t + 1], 
                          I=ross_read$I[start_t + 1])
  Kappa <- c*output_df[1,]$I/H
  ER <- a*output_df[1,]$Z/H
  ZZ <- rep(0, E + 1)
  for(i in (start_t + 1):end_t){
    prev_row <- output_df[i - start_t, ]
    M <- prev_row$M + l
    M <- p*M
    Y <- p*prev_row$Y
    Y0 <- a*Kappa*(M - Y)
    Y <- Y + Y0
    Z <- p*prev_row$Z
    Z <- Z + ZZ[1]
    for(j in 1:E){
      ZZ[j] = ZZ[j + 1]
    }
    ZZ[E + 1] = (p^E)*Y0
    I <- prev_row$I - r*prev_row$I + b*ER*(H - prev_row$I)
    Kappa <- c*I/H
    ER <- a*Z/H
    output_df <- rbind(output_df, data.frame(time=i, M=M, Y=Y, Z=Z, I=I))
  }
  return(output_df)
}
```

Calculate likelihood functions based on given set of parameter values (two approaches):

The first approach:
Calculate based on the deviation of estimated value around the mean value of original data, from a specified start time to a specified end time.
Usually it is a good approach to consider only the time interval where each variable has (or almost has) reached its equilibrium, so that the fitted curves will also reach nearly the same equilibriums. If we consider the whole time interval, especially when the data are not yet at its equilibrium, then we will end up getting a very biased result.

``` r
get_scaled_likelihood <- function(p, l){
  out_df <- get_sim_df(p, l, 0, 4999)
  out_df <- out_df[out_df$time %in% ross_read$time[start:end],]
  lik <- 0
  lik_M <- mean(log(dnorm(out_df$M - mean_M2, 0, sd_M2)))
  lik_Y <- mean(log(dnorm(out_df$Y - mean_Y2, 0, sd_Y2)))
  lik_Z <- mean(log(dnorm(out_df$Z - mean_Z2, 0, sd_Z2)))
  lik_I <- mean(log(dnorm(out_df$I - mean_I2, 0, sd_I2)))
  lik <- lik + lik_M + lik_I + lik_Z + lik_Y
  return(lik / 4)
}
```

The second approach:
Calculate based on the deviation of estimated value around the observed value of each variable at each time.
The advantage of this approach is that we can utilize the information from the time intervals where the variables have not yet reached their equilibriums.
The drawback, however, is that this approach is not as robust to outliers or randomness of data as the previous one.

``` r
get_scaled_likelihood2 <- function(p, l){
  start_t <- 0
  end_t <- 4999
  
  out_df <- get_sim_df(p, l, start_t, end_t)
  out_df <- out_df[out_df$time %in% 
                     ross_read$time[start_t + 1:end_t + 1],]
  lik <- 0
  lik_M <- mean(log(dnorm(out_df$M - 
                    ross_read$M[start_t + 1:end_t + 1], 0, sd_M)))
  lik_Y <- mean(log(dnorm(out_df$Y - 
                    ross_read$Y[start_t + 1:end_t + 1], 0, sd_Y)))
  lik_Z <- mean(log(dnorm(out_df$Z - 
                    ross_read$Z[start_t + 1:end_t + 1], 0, sd_Z)))
  lik_I <- mean(log(dnorm(out_df$I - 
                    ross_read$I[start_t + 1:end_t + 1], 0, sd_I)))
  lik <- lik + lik_M + lik_I + lik_Z + lik_Y
  return(lik / 4)
}
```

Run a single Monte Carlo Markov Chain

``` r
MCMC_single <- function(step_size, num_steps){
  p <- runif(1, 0, 1)
  l <- runif(1, 0, 100)
  lik <- get_scaled_likelihood(p, l)
  temp <- data.frame(p=p, l=l, lik=lik)
  for(i in 1:num_steps){
    p_new <- max(0, rnorm(1, p, step_size[1]))
    l_new <- max(0, rnorm(1, l, step_size[2]))
    lik_new <- get_scaled_likelihood(p_new, l_new)
    if(!is.na(lik_new)){
      acceptance_prob <- min(1, exp(lik_new)/exp(lik))
      r_u <- runif(1)
      if(!is.na(acceptance_prob) && r_u <= acceptance_prob){
        p <- p_new
        l <- l_new
        lik <- lik_new
        step_size <- schedule(lik)
      }
    }
    temp <- rbind(temp, c(p=p, l=l, lik=lik))
  }
  row <- temp[which.max(temp$lik),]
  return(row)
}
```

As the likelihood gets higher and higher, we need to turn down the step size of each parameter, so that they won't get overshoot.
The following is a tentative schedule of the step size for each parameter at certain thresholds of likelihood.

``` r
schedule <- function(lik){
  if(lik > -4){
    return(c(p=0.001, l=0.02))
  } else if(lik > -20){
    return(c(p=0.005, l=0.1))
  }
  return(c(p=0.05, l=1))
}
```

Run multiple Monte Carlo Markov Chains

``` r
MCMC <- function(step_size, num_particles, num_steps){
  foreach(i = 1:num_particles, 
          #.errorhandling="remove", .inorder=FALSE,
          .combine = rbind) %dopar% {
          list <- MCMC_single(step_size, num_steps)
          data.frame(p = list[1], l = list[2], lik = list[3])
         } -> all_fits 
  return(all_fits)
}
```

Calculate the hessian matrix for the set of parameters, which will be used to calculate the confidence intervals for fitted parameters.

``` r
# A helper function to get the likelihood for a given set
# of parameter values
get_scaled_likelihood2_helper <- function(params){
  return(get_scaled_likelihood2(params[1], params[2]))
}

# Calculate the hessian matrix, which is comprised of the
# second derivatives of likelihood with respect to each parameter
get_hess <- function(p, l, lik, get_scaled_likelihood2){
  dp <- 0.01
  dl <- 0.2
  
  p_h <- p + dp
  p_l <- p - dp
  l_h <- l + dl
  l_l <- l - dl
  
  ph_l <- get_scaled_likelihood2(p_h, l)
  pl_l <- get_scaled_likelihood2(p_l, l)
  p_lh <- get_scaled_likelihood2(p, l_h)
  p_ll <- get_scaled_likelihood2(p, l_l)
  ph_lh <- get_scaled_likelihood2(p_h, l_h)
  pl_lh <- get_scaled_likelihood2(p_l, l_h)
  ph_ll <- get_scaled_likelihood2(p_h, l_l)
  pl_ll <- get_scaled_likelihood2(p_l, l_l)
  
  dpp <- ((ph_l - lik) / dp - (lik - pl_l) / dp) / dp
  dpl <- ((ph_lh - p_lh) / dp - (ph_l - lik) / dp) / dl
  dlp <- ((ph_lh - ph_l) / dl - (p_lh - lik) / dl) / dp
  dll <- ((p_lh - lik) / dl - (lik - p_ll) / dl) / dl
  
  ret_mat <- matrix(c(dpp, dpl), ncol=2)
  ret_mat <- rbind(ret_mat, c(dlp, dll))
  
  return(ret_mat)
}
```

Run the MCMC algorithm

``` r
print("I am ready to work...")
all_fits <- MCMC(c(p=0.05, l=1), 4, 100)
all_fits
```

par stores the fitted values of parameters

``` r
par <- all_fits %>% filter(lik == max(lik))
par
```

Get hessian matrix and calculate the confidence interval

``` r
hess <- get_hess(par$p, par$l, par$lik, get_scaled_likelihood2)
hess

stdv <- sqrt(diag(solve(hess)))
par$p + c(-1, 1)*1.96*stdv[1]
par$l + c(-1, 1)*1.96*stdv[2]
```

Using the fitted values of parameters
Plot the trajectories of fitted curves VS original curves for each variable

``` r
out_df <- get_sim_df(par$p, par$l, 0, 4999)

jpeg("plots/ross_M_MCMC2.jpeg")
plot(ross_read$time, ross_read$M, main="M",
     xlab="time", ylab="nums", type="l", col="green")
lines(out_df$time, out_df$M, col="black")
dev.off()

jpeg("plots/ross_Y_MCMC2.jpeg")
plot(ross_read$time, ross_read$Y, main="Y",
     xlab="time", ylab="nums", type="l", col="green")
lines(out_df$time, out_df$Y, col="black")
dev.off()

jpeg("plots/ross_Z_MCMC2.jpeg")
plot(ross_read$time, ross_read$Z, main="Z",
     xlab="time", ylab="nums", type="l", col="green")
lines(out_df$time, out_df$Z, col="black")
dev.off()

jpeg("plots/ross_I_MCMC2.jpeg")
plot(ross_read$time, ross_read$I, main="I",
     xlab="time", ylab="nums", type="l", col="green")
lines(out_df$time, out_df$I, col="black")
dev.off()
```
