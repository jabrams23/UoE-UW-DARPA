# Chris A. Boulton - University of Exeter, UK
# April 11th 2022
# sea water density functions to run the 4-box AMOC model from Gnanandesikan et al. 2018 paper

sw_smow <- function(T) {
	a0 <- 999.842594
	a1 <- 6.793953e-2
	a2 <- -9.095290e-3
	a3 <- 1.001685e-4
	a4 <- -1.120083e-6
	a5 <- 6.536332e-9
	
	T68 <- T*1.00024
	
	return(a0 + a1*T68 + a2*T68^2 + a3*T68^3 + a4*T68^4 + a5*T68^5)
}

sw_dens0 <- function(S, T) {
	b0 <- 8.24493e-1
	b1 <- -4.0899e-3
	b2 <- 7.6438e-5
	b3 <- -8.2467e-7
	b4 <- 5.3875e-9
	
	c0 <- -5.72466e-3
	c1 <- 1.0227e-4
	c2 <- -1.6546e-6
	
	d0 <- 4.8314e-4
	
	T68 <- T*1.00024
	
	B1 <- b0 + b1*T68 + b2*T68^2 + b3*T68^3 + b4*T68^4
	C1 <- c0 + c1*T68 + c2*T68^2
	
	return(sw_smow(T) + B1*S + C1*S^1.5 + d0*S^2)
}

sw_seck <- function(S, T, P) {
	e0 <- 19652.21
	e1 <- 148.4206
	e2 <- -2.327105
	e3 <- 1.360477e-2
	e4 <- -5.155288e-5
	
	f0 <- 54.6746
	f1 <- -0.603459
	f2 <- 1.099870e-2
	f3 <- -6.167e-5
	
	g0 <- 7.944e-2
	g1 <- 1.6483e-2
	g2 <- -5.3009e-4
	
	h0 <- 3.23990
	h1 <- 1.43713e-3
	h2 <- 1.16092e-4
	h3 <- -5.77905e-7
	
	i0 <- 2.28380e-3
	i1 <- -1.09810e-5
	i2 <- -1.60780e-6
	
	j0 <- 1.91075e-4
	
	k0 <- 8.50935e-5
	k1 <- -6.12293e-6
	k2 <- 5.27870e-8
	
	m0 <- -9.9348e-7
	m1 <- 2.0816e-8
	m2 <- 9.1697e-10
	
	T68 <- T*1.00024
	
	Kw <- e0 + e1*T68 + e2*T68^2 + e3*T68^3 + e4*T68^4
	F1 <- f0 + f1*T68 + f2*T68^2 + f3*T68^3
	G1 <- g0 + g1*T68 + g2*T68^2
	
	K0 <- Kw + F1*S + G1*S^1.5
	
	Aw <- h0 + h1*T68 + h2*T68^2 + h3*T68^3
	A1 <- Aw + (i0 + i1*T68 + i2*T68^2)*S + j0*S^1.5
	
	Bw <- k0 + k1*T68 + k2*T68^2
	B2 <- Bw + (m0 + m1*T68 + m2*T68^2)*S
	
	P <- P/10
	
	return(K0 + A1*P + B2*P^2)
	 
}


sw_dens <- function(S, T, P) {
	densP0 <- sw_dens0(S, T)
	K <- sw_seck(S, T, P)
	P <- P/10
	return(densP0/(1-P/K))
}










