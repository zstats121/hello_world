


rm(list=ls())
gc()
gc()
"%ni%" <- Negate("%in%")

##for AWS...
#system('sudo apt-get install libcurl4-openssl-dev libssl-dev htop')
##tmux attach
#install.packages('doParallel')
#install.packages('mvtnorm')
#install.packages('foreach')
#install.packages('numDeriv')
#install.packages('devtools')
#install.packages('lfe')
#setwd('/home/ubuntu/pnn')

library(devtools)
install_github('cranedroesch/panelNNET')

library(panelNNET)
library(doParallel)
library(parallel)
library(mvtnorm)
registerDoParallel(detectCores())


N <- 6000
t = 20
pz <- 5
pid <- N/t
id <- (1:N-1) %/% (N/pid) +1
time <- 0:(N-1) %% (N/pid) +1
#set.seed
k=2
set.seed(k)
#Each group has its own covariance matrix
groupcov <- foreach(i = 1:pid) %do% {
  A <- matrix(rnorm(pz^2), pz)
  t(A) %*% A
}
#and its own mean
groupmean <- foreach(i = 1:pid) %do% {
  rnorm(pz, sd  =5)
}
#and it's own effect that is distinct from its covariate distribution
id.eff <- as.numeric(id)

#this is the data generated from those distributions
Z <- foreach(i = 1:N, .combine = rbind) %do% {
  mvrnorm(1, groupmean[[id[i]]], groupcov[[id[i]]])
}
#outcome minus noise
y <- time +log(dmvnorm(Z, rep(0, pz), diag(rep(1, pz)))) + id.eff
#add noise
u <- rnorm(N, sd = 20)
y <- y+u
id <- as.factor(id)
#training and test and validation
v <- time>max(time)*.9
r <- time %in% time[which(v==FALSE & time %%2)]
e <- time %in% time[which(v==FALSE & (time+1) %%2)]
P <- matrix(time)
#put in data frame and estimate fe model
dat <- data.frame(y, Z, time, id)
mfe <- lm(y~.-1, data = dat[r|e,])
pfe <- predict(mfe, newdata = dat[v,])

lam <- .001
g = c(20)
pl <- NULL
#Batch gradeint descent
pnn <- panelNNET(y[r], Z[r,], hidden_units = g
  , fe_var = id[r], maxit = 500, lam = lam
  , time_var = time[r], param = P[r,, drop = FALSE],  verbose = TRUE
  , gravity = 1.01, convtol = 1e-6, activation = 'lrelu'
  , start_LR = .01, parlist = pl, OLStrick = FALSE
  , initialization = 'enforce_normalization'
  , report_interval = 10
)
pnn <- do_inference(pnn, numerical = FALSE, parallel = TRUE, step = 1e-9, J = NULL, verbose = FALSE, OLS_only = FALSE)
summary(pnn)

pnn <- panelNNET(y[r], Z[r,], hidden_units = g
  , fe_var = id[r], maxit = 10, lam = lam
  , time_var = time[r], param = P[r,, drop = FALSE],  verbose = TRUE
  , gravity = 1.01, convtol = 1e-6, activation = 'lrelu', inference = FALSE
  , start_LR = .01, parlist = pnn$parlist, OLStrick = TRUE
)

pnn <- do_inference(pnn, numerical = FALSE, parallel = TRUE, step = 1e-9, J = NULL, verbose = FALSE, OLS_only = FALSE)
summary(pnn)


pr <- predict(pnn, newX = Z[e,], new.param = P[e,, drop = FALSE], fe.newX = id[e])
plot(y[e], pr)

