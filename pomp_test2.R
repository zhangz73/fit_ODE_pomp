# Script for fitting ODE parameters using M_I_interaction_data
# It has the following jobs:
# 1. fit the parameters using particle filter in pomp
# 2. plot the simulation plot
# 3. compute the confidence interval of the parameters
# 4. filter out the outliers in the particles

library(pomp2)
library(magrittr)
library(subplex)
library(foreach)
library(doParallel)
library(ggplot2)
library(dplyr)

registerDoParallel(cores=4)
output_read <- read.csv("M_I_interation_data.csv")

# We'd better rename the columns of the file
# because Csnippet doesn't allow the name of columns
# to collide with the statenames
colnames(output_read) <- c("Time", "M_c", "I_c")
statenames <- c("I", "M")
paranames <- c("b", "g", "N", "l", "p")
params <- c(N=500)

# Gives the initial value of each state
Csnippet("I=397.6663779;
         M=0.35717571;") -> init

# The deterministic step function of the model
# I restrict both I and M to be non-negative numbers
# to be consistent with data in the real world
# and to avoid stochastic functions from breaking
Csnippet("
         I += (b*M*(N - I)/N - g*I)*dt;
         M += (l - p*M)*dt;
         I = I > 0 ? I : 0;
         M = M > 0 ? M : 0;
         ") -> step

# The stochastic step function of the model
# using poisson process and binomial processes
Csnippet("
         double t1 = rpois(l*dt);
         double t2 = rbinom((int)M, p*dt);
         double t3 = rbinom((int)(N - I), b*(M)/N*dt);
         double t4 = rbinom((int)I, g*dt);
         I += t3 - t4;
         M += t1 - t2;
         
         I = I > 0 ? I : 0;
         M = M > 0 ? M : 0;
         ") -> step_stochastic

# rmeas is used to calculate Y given X
# but our model is simple, where we don't have a different variable Y
# so it is equivalent to let Y = X
Csnippet("I_c=I; M_c=M;") -> rmeas

# dmeans calculates the likelihood
# The predicted I and M are supposed to be the same as the I and M in csv file
# The farther they deviate from their expected values,
# the smaller the likelihood will be
Csnippet("lik = dnorm(I - I_c, 0, 10, 0) * dnorm(M - M_c, 0, 10, 0);") -> dmeas

# initialize pomp with the Csnippets we defined in the previous part
output_read %>%
  pomp(
    times="Time", t0=0,
    rinit=init, rmeasure=rmeas,
    rprocess=discrete_time(step,delta.t=1),
    statenames=statenames,
    paramnames=paranames
  ) -> output_pomp

# Code the deterministic skeleton, which is identical to our ODE,
# since we don't have random variables directly appears in the ODE
output_pomp %>%
  pomp(
    skeleton=map(
      Csnippet("DI = b*M*(N-I)/N-g*I; DM = l - p*M;"),
      delta.t=1
    ),
    paramnames=paranames,
    statenames=c("I", "M")
  ) -> output_pomp


# Estimate the parameters using the subplex package
# Currently I don't use it anymore, but I leave it here for debugging
# If this function gives an output without error messages, then we
# haven't made any syntactic errors in the previous parts
output_pomp %>%
  traj_objfun(
    est=c("b", "g", "p", "l"),
    params=c(N=500, b=1, g=1, p=1, l=1),
    dmeasure=dmeas, statenames=c("I", "M")
  ) -> ofun

subplex(c(1.1, 1.1, 1.1, 1.1),fn=ofun) -> fit
fit 


# Generate nseq number of random guesses of each paramter
# with their lower bound and upper bound
# For the paramters we passed in as constants (eg. N)
# We can just set both of their lower and upper bounds to their values (eg. N = 500)
sobolDesign(
  lower=c(b=0, g=0, N=500, p=0, l=0),
  upper=c(b=100,g=1, N=500, p=1, l=100),
  nseq=100
) -> guesses

# Initialize the object for particle filter using random guesses we generated
# Np is the number of particles we use
# Nmif is the number of iterations
# rw.sd is the standard deviation of a paramter when sample 
# it in the particle filter algorithm
# rw.sd of a constant should be set to 0, so that its value won't change
output_pomp %>%
  mif2(
    params=guesses[1,],
    Np=100,
    Nmif=100,
    dmeasure=dmeas,
    partrans=parameter_trans(log=c("b", "g", "N", "p", "l")),
    cooling.fraction.50=0.5,
    rw.sd=rw.sd(b=0.01,g=0.01, N=0, p=0.01, l=0.01),
    paramnames=paranames,
    statenames=statenames,
    filter.traj=TRUE,
    pred.mean=TRUE,
    pred.var=TRUE
  ) -> mf1

# Do a global search
foreach(guess=iter(guesses, "row"),
        .combine=c, .packages=c("pomp2"),
        .errorhandling="remove", .inorder=FALSE) %dopar% {
          mf1 %>% mif2(params=guess)
        } ->mifs

# Get the likelihood of these estimates
foreach (mf=mifs,
         .combine=rbind, .packages=c("pomp2"), 
         .errorhandling="remove", .inorder=FALSE) %dopar% {
           
           replicate(5, 
                     mf %>% pfilter() %>% logLik()
           ) %>%
             logmeanexp(se=TRUE) -> ll
           
           data.frame(as.list(coef(mf)),loglik=ll[1],loglik.se=ll[2])
           
         } -> estimates

# More search efforts
estimates %>% select(-loglik, -loglik.se) -> starts
foreach (start=iter(starts,"row"),
         .combine=rbind, .packages=c("pomp2"),
         .errorhandling="remove", .inorder=FALSE) %dopar% {
           
           mf1 %>%
             mif2(
               params=start,
               partrans=parameter_trans(log=c("b", "g", "N", "p", "l")),
               rw.sd=rw.sd(b=0.01,g=0.01, N=0, p=0.01, l=0.01),
               paramnames=paranames
             ) %>%
             mif2() -> mf
           
           replicate(5, 
                     mf %>% pfilter() %>% logLik()
           ) %>%
             logmeanexp(se=TRUE) -> ll
           
           data.frame(as.list(coef(mf)),loglik=ll[1],loglik.se=ll[2])
         } -> r_prof

# Get the MLE and draw the simulation plots
r_prof[which.max(r_prof[,6]),][1:5] -> mle
mlepomp <- as(mifs[[1]], 'pomp')
coef(mlepomp) <- mle

# Simulation plots for I
mlepomp %>%
  simulate(nsim=1,format="data.frame",include.data=TRUE) %>%
  ggplot(mapping=aes(x=Time,y=I_c,group=.id,alpha=(.id=="data")))+
  scale_alpha_manual(values=c(`TRUE`=1,`FALSE`=0.2),
                     labels=c(`FALSE`="simulation",`TRUE`="data"))+
  labs(alpha="")+
  geom_line()+
  theme_bw()

# Simulation plots for M
mlepomp %>%
  simulate(nsim=3,format="data.frame",include.data=TRUE) %>%
  ggplot(mapping=aes(x=Time,y=M_c,group=.id,alpha=(.id=="data")))+
  scale_alpha_manual(values=c(`TRUE`=1,`FALSE`=0.2),
                     labels=c(`FALSE`="simulation",`TRUE`="data"))+
  labs(alpha="")+
  geom_line()+
  theme_bw()
mle

# Construct the confidence interval for each paramter
par <- c(1, 2, 4, 5)
pars <- c("b", "g", "N", "p", "l")
for(i in par){
  se <- sd(r_prof[,i])/sqrt(length(r_prof$loglik))
  print(paste(pars[i], ": ", mle[, i], sep=""))
  print(mle[, i] + c(-1, 1)*1.96*se)
  print("")
}

# filters out the most unlikely particles
# construct the correlation plots for these paramters
indices1 <- which(r_prof$loglik > min(r_prof$loglik) + 3 * sd(r_prof$loglik))
r_prof[indices1, ] %>%
  pairs(~loglik+b+g+p+l,data=.,
        pch=16,
        col=c("#ff0000ff"))%>% 
  invisible()