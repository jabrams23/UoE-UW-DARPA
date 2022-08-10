# Chris A. Boulton - University of Exeter, UK
# April 11th 2022
# 4-box AMOC model from Gnanandesikan et al. 2018 paper
# samples parameters from the paper within bounds from a uniform distribution
# 4000 steps to equilibrium and then 1500 afterwards with the noise added (and potentially moving Fn_w[i])

# this code creates 10000 time series where the bifurcation parameter (freshwater flux) is linearly increased between two values

# if at any time the 'AMOC off' equations are used, the run is replaced by a new one with a different set of parameters

#sea water desnity calcs are in another .R script

M_n_tip <- array(NA, dim=c(5501,10000))
Fn_w1_tip <- rep(NA, 10000)
Fn_w2_tip <- rep(NA, 10000)
K_v_tip <- rep(NA, 10000)
M_ek_tip <- rep(NA, 10000)
A_GM_tip <- rep(NA, 10000)
e_tip <- rep(NA, 10000)
Aredi_tip <- rep(NA, 10000)
fsig_tip <- rep(NA, 10000)

source('sw_dens_funcs.R')

N1 <- 4000
N2 <- 1500

Area <- 3.6e14
Area_low <- 2e14
Area_S <- 1e14
Area_N <- 0.6e14

D_high <- 100

dt <- 365*86400/4 # 1/4 of a year I think

#A_GM <- 1000
#e <- 1.2e-4
Fs_w <- 1e6
#M_ek <- 15e6
M_SD <- 15e6
#Aredi <- 1000

Ls_x <- 2.5e7
Ls_y <- 1e6

D0 <- 400
T_N0 <- 2
T_S0 <- 4
T_deep0 <- 3
T_low0 <- 17
S_N0 <- 35
S_S0 <- 36
S_deep0 <- 34.5
S_low0 <- 36

T_N <- rep(NA, N1+N2+1)
T_N[1] <- T_N0
T_S <- rep(NA, N1+N2+1)
T_S[1] <- T_S0
T_low <- rep(NA, N1+N2+1)
T_low[1] <- T_low0
T_deep <- rep(NA, N1+N2+1)
T_deep[1] <- T_deep0

S_N <- rep(NA, N1+N2+1)
S_N[1] <- S_N0
S_S <- rep(NA, N1+N2+1)
S_S[1] <- S_S0
S_low <- rep(NA, N1+N2+1)
S_low[1] <- S_low0
S_deep <- rep(NA, N1+N2+1)
S_deep[1] <- S_deep0

M_n <- rep(NA, N1+N2+1)
M_upw <- rep(NA, N1+N2+1)
M_eddy <- rep(NA, N1+N2+1)
D <- rep(NA, N1+N2+1)
D[1] <- D0

n <- 0

crit_val <- 0.8646
tip_time <- rep(NA, 10000)

while (n < 10000) {
	n <- n + 1
	tip_marker <- 0
	break_marker <- 0
	print(n)
	#Fn_w1 <- runif(1,0,1)
	#Fn_w2 <- runif(1,Fn_w1,1)
	#Fn_w2 <- Fn_w1
	Fn_w1 <- 0
	tip_time[n] <- floor(runif(1,1501,10000))
	Fn_w2 <- 1500*crit_val/tip_time[n]
	Fn_w <- rep(NA, N1+N2)
	Fn_w[1:N1] <- Fn_w1*1e6
	Fn_w[(N1+1):(N1+N2)] <- seq(Fn_w1,Fn_w2,(Fn_w2-Fn_w1)/(N2-1))*1e6

	#K_v <- runif(1,0,5)*1e-5
	K_v <- 1e-6
	#M_ek <- runif(1,15,35)*1e6
	M_ek <- 25*1e6
	#A_GM <- runif(1,500,2000)
	A_GM <- 1000
	#e <- runif(1,0.6,2.4)*1e-4
	e <- 1.2*1e-4
	#Aredi <- runif(1,500,2000)
	Aredi <- 1000

	#fsig <- c(rep(0, N1), rep(runif(1,0.01,0.05), N2))
	fsig <- c(rep(0, N1), rep(0.025, N2))

	for (i in 1:(N1+N2)) {
		sig_N <- sw_dens(S_N[i], T_N[i], 0)
		sig_low <- sw_dens(S_low[i], T_low[i], 0)
		M_LS <- Aredi*2.5e7*D[i]/1e6
		M_LN <- Aredi*5e6*D[i]/1e6
	
		if (!is.na(sig_N) && sig_N > sig_low) {
			gp <- 9.8*(sig_N-sig_low)/sig_N
			M_n[i] <- gp*D[i]^2/e
			M_upw[i] <- K_v*Area_low/min(D[i],3700-D[i])
			M_eddy[i] <- A_GM*D[i]*Ls_x/Ls_y
		
			V_deep=3700*Area-Area_N*D_high-Area_S*D_high-Area_low*D[i]
			V_low <- Area_low*D[i]
			dV_low <- (M_ek-M_eddy[i]-M_n[i]+M_upw[i]-Fs_w-Fn_w[i])*dt
			dV_deep <- -dV_low
			D[i+1] <- D[i]+dV_low/Area_low
		
			dS_low <- (M_ek*S_S[i]-M_eddy[i]*S_low[i]-M_n[i]*S_low[i]+M_upw[i]*S_deep[i]+M_LS*(S_S[i]-S_low[i])+M_LN*(S_N[i]-S_low[i]))*dt
			dS_S <- ((M_eddy[i]+M_LS)*(S_low[i]-S_S[i])+(M_ek+M_SD)*(S_deep[i]-S_S[i])-Fs_w*S_S[i])*dt
			dS_deep <- (M_n[i]*S_N[i]-(M_upw[i]+M_ek+M_SD)*S_deep[i]+(M_eddy[i]+M_SD)*S_S[i]+Fs_w*S_S[i]+Fn_w[i]*S_N[i])*dt
			dS_N <- ((M_n[i]+M_LN)*(S_low[i]-S_N[i])-Fn_w[i]*S_N[i])*dt
 
			dT_low=(M_ek*T_S[i]-M_eddy[i]*T_low[i]-M_n[i]*T_low[i]+M_upw[i]*T_deep[i]+M_LS*(T_S[i]-T_low[i])+M_LN*(T_N[i]-T_low[i])+Area_low*100*(T_low0-T_low[i])/365/86400)*dt;
			dT_S=((M_eddy[i]+M_LS)*(T_low[i]-T_S[i])+(M_ek+M_SD)*(T_deep[i]-T_S[i])+Area_S*100*(T_S0-T_S[i])/365/86400)*dt;
			dT_deep=((M_n[i]+Fn_w[i])*T_N[i]-(M_upw[i]+M_ek+M_SD)*T_deep[i]+(M_eddy[i]+M_SD+Fs_w)*T_S[i])*dt;
			dT_N=((M_n[i]+M_LN)*(T_low[i]-T_N[i])+Area_N*100*(T_N0-T_N[i])/365/86400)*dt;
		
			S_N[i+1] <- S_N[i] + dS_N/(D_high*Area_N)
			S_S[i+1] <- S_S[i] + dS_S/(D_high*Area_S)
			S_low[i+1] <- (S_low[i]*V_low + dS_low)/(V_low + dV_low)
			S_deep[i+1] <- (S_deep[i]*V_deep + dS_deep)/(V_deep + dV_deep)

			T_N[i+1] <- T_N[i] <- dT_N/(D_high*Area_N) + rnorm(1, sd=fsig[i])
			T_S[i+1] <- T_S[i] + dT_S/(D_high*Area_S) + rnorm(1, sd=fsig[i])
			T_low[i+1] <- (T_low[i]*V_low + dT_low)/(V_low + dV_low) + rnorm(1, sd=fsig[i])
			T_deep[i+1] <- (T_deep[i]*V_deep + dT_deep)/(V_deep + dV_deep)
		} else if (is.na(sig_N)) {
			break_marker <- 1
		}
	
		if (!is.na(sig_N) && sig_N <= sig_low) {
			tip_marker <- 1
			gp <- 9.8*(sig_N-sig_low)/sig_N
			M_n[i] <- gp*D_high^2/e
			M_upw[i] <- K_v*Area_low/min(D[i],3700-D[i])
			M_eddy[i] <- A_GM*D[i]*Ls_x/Ls_y
		
			V_deep=3700*Area-Area_N*D_high-Area_S*D_high-Area_low*D[i]
			V_low <- Area_low*D[i]
			dV_low <- (M_ek-M_eddy[i]-M_n[i]+M_upw[i]-Fs_w-Fn_w[i])*dt
			dV_deep <- -dV_low
			D[i+1] <- D[i]+dV_low/Area_low
		
			dS_low <- (M_upw[i]*S_deep[i] + (M_ek + M_LS)*S_S[i] + (M_LN - M_n[i])*S_N[i] - (M_eddy[i] + M_LS + M_LN)*S_low[i])*dt
			dS_S <-  ((M_LS + M_eddy[i])*S_low[i] + (M_ek + M_SD)*S_deep[i] - (M_ek + Fs_w + M_SD + M_eddy[i])*S_S[i])*dt
			dS_deep <- ((Fs_w + M_SD + M_eddy[i])*S_S[i] + Fn_w[i]*S_N[i] - (M_ek + M_upw[i] + M_SD - M_n[i])*S_deep[i])*dt
			dS_N <- (M_LN*S_low[i] - M_n[i]*S_deep[i] - (M_LN + M_n[i] + Fn_w[i])*S_N[i])*dt
		
			dT_low <- (M_upw[i]*T_deep[i] + (M_ek + M_LS)*T_S[i] + (M_LN - M_n[i])*T_N[i] - (M_eddy[i] + M_LS + M_LN + Fs_w + Fn_w[i])*T_low[i]+Area_low*100*(T_low0 - T_low[i])/365/86400)*dt
			dT_S <- ((M_LS + M_eddy[i])*T_low[i] + (M_ek + M_SD)*T_deep[i] - (M_ek + M_SD + M_eddy[i])*T_S[i] + Area_S*100*(T_S0 - T_S[i])/365/86400)*dt
			dT_deep <- ((Fs_w + M_SD + M_eddy[i])*T_S[i] + Fn_w[i]*T_N[i] - (M_ek + M_upw[i] + M_SD - M_n[i])*T_deep[i])*dt
			dT_N <- (M_LN*T_low[i] - M_n[i]*T_deep[i] - (M_LN - M_n[i])*T_N[i] + Area_N*100*(T_N0 - T_N[i])/365/86400)*dt
		
			S_N[i+1] <- S_N[i] + dS_N/(D_high*Area_N)
			S_S[i+1] <- S_S[i] + dS_S/(D_high*Area_S)
			S_low[i+1] <- S_low[i] + dS_low/(V_low + dV_low)
			S_deep[i+1] <- S_deep[i] + dS_deep/(V_deep + dV_deep)

			T_N[i+1] <- T_N[i] + dT_N/(D_high*Area_N) + rnorm(1, sd=fsig[i])
			T_S[i+1] <- T_S[i] + dT_S/(D_high*Area_S) + rnorm(1, sd=fsig[i])
			T_low[i+1] <- T_low[i] + dT_low/(V_low + dV_low) + rnorm(1, sd=fsig[i])
			T_deep[i+1] <- T_deep[i] + dT_deep/(V_deep + dV_deep)
		} else if (is.na(sig_N)) {
			break_marker <- 1
		}
	}
	
	M_n_tip[,n] <- M_n
	Fn_w1_tip[n] <- Fn_w1
	Fn_w2_tip[n] <- Fn_w2
	K_v_tip[n] <- K_v
	M_ek_tip[n] <- M_ek
	A_GM_tip[n] <- A_GM
	e_tip[n] <- e
	Aredi_tip[n] <- Aredi
	fsig_tip[n] <- fsig[N1+N2]
	if (tip_marker == 1) {
		print('TIP!')
		n <- n -1
	}
	if (break_marker == 1) {
		print('BREAK!')
		n <- n -1
	}
}

save.image('AMOC_dialled_back_tip_set2.RData')

save(M_n_tip, tip_time, Fn_w2_tip, file='AMOC_dialled_back_tip_set2_short.RData')


