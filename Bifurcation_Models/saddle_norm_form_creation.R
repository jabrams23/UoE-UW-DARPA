#sample distance from tipping and a starting value of r and then work out what end point needs to be
#create 10000 instances for now

#set timestep to be h=1/12 to be monthly, between 10 and 300 years away
#can have 50 years of data

distances <- runif(100000,10,300)+50 	#add 50 as the distance is away from the initial value (it gets taken away lower down)
r0 <- runif(100000, -5, -1)	#initial starting bifurcation parameter

r1 <- r0-r0*50/distances	#ending bifurcation parameter based on distance from tipping

r <- array(NA, dim=c(100000,50*12))
for (i in 1:100000) {
	r[i,] <- seq(r0[i],r1[i],(r1[i]-r0[i])/(50*12-1))
}

f <- runif(100000,0.01,0.2)
h <- 1/12
x <- array(NA, dim=c(100000,50*12))
i <- 0
while (i < 100000) {
	print(i)
	i <- i + 1
	x[i,1] <- -sqrt(-r0[i])
	for (j in 2:600) {
		x[i,j] <- x[i,j-1] + h*(r[i,j] + x[i,j-1]^2) + sqrt(h)*rnorm(1, sd=f[i])
	}
	if (length(which(x[i,] > 10) > 0)) {		# check for if tipping has occurred and if so then resmaple the parameters for that specific similation
		distances[i] <- runif(1,10,100)+50
		r0[i] <- runif(1, -5, -1)
		r1[i] <- r0[i]-r0[i]*50/distances[i]
		r[i,] <- seq(r0[i],r1[i],(r1[i]-r0[i])/(50*12-1))
		f[i] <- runif(1,0.01,0.2)
		i <- i - 1
	}
}


# detrend based on the equilibrium state of the system and normalise for DL
# then create a training/validation/test set

x_detrend <- x + sqrt(-r)
x_norm <- array(NA, dim=c(100000, 600))

for (i in 1:100000) {
	x_norm[i,] <- (x_detrend[i,]-mean(x_detrend[i,]))/sd(x_detrend[i,])
}

sample_ind <- sample(100000)

train <- x_norm[sample_ind[1:70000],]
train_target <- distances[sample_ind[1:70000]]-50

validation <- x_norm[sample_ind[70001:85000],]
validation_target <- distances[sample_ind[70001:85000]]-50

test <- x_norm[sample_ind[85001:100000],]
test_target <- distances[sample_ind[85001:100000]]-50

# traditional EWS are calculated below

l <- 600
wl <- 300

ar1 <- array(NA, dim=c(100000,l-wl+1))
vari <- array(NA, dim=c(100000,l-wl+1))

for (i in 1:100000) {
	print(i)
	for (z in 1:(l-wl+1)) {
		ar1[i,z] <- ar.ols(x_norm[i,z:(z+wl-1)], aic=FALSE, order.max=1)$ar
		vari[i,z] <- var(x_norm[i,z:(z+wl-1)])
	}
}

library(Kendall)

ar1_k <- rep(NA, 100000)
vari_k <- rep(NA, 100000)

for (i in 1:100000) {
	ar1_k[i] <- Kendall(wl:l, ar1[i,])$tau
	vari_k[i] <- Kendall(wl:l, vari[i,])$tau
}










