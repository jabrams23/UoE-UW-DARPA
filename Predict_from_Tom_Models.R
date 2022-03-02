# after installing the tensorflow and keras packages to R, you will need to run their installation set up
library('tensorflow')
library('keras')
library('reticulate')

# install_tensorflow() run this the first time
# install_keras() run this the first time too


normalise <- function(x) {
	# a function to noramlise time series to predict from
	norm_x <- rep(NA, length(x))
	for (i in 1:length(x)) {
		norm_x[i] <- (x[i]-mean(x))/sd(x)
	}
	return(norm_x)
}


predict_probs <- function(x, wl=length(x), len=1500) {
	# a function that returns the probabilies from all 20 models
	# can specifiy a window length (wl) to calculate the probabilites on which increases one time series at a time
	# by default will just use the whole time series
	# len refers to which set of models are used
	l <- length(x)
	probs <- array(NA, dim=c(l-wl+1,4,20))
	count <- 0
	for (i in 1:10) {
		for (j in 1:2) {
			count <- count + 1 #there are 20 models in total and numbered i 1 to 10 and j 1 to 2
			model <- load_model_tf(paste('/Tom_Models/best_model_',i,'_',j,'_len',len,'.pkl',sep=''))
			try(predict(model, t(rnorm(1000))), silent=TRUE) #seems to be a bug where it has to go through a failed prediction first
			for (t in 1:(l-wl+1)) {
				predict_probs[t,,count] <- predict(model, t(x[1:(t+wl-1)]))
			}
		}
	}
	return(probs)
	# the values returned are the probability of the time series approaching a [1] fold, [2] hopf, [3] transcritical or [4] no bifurcation
} 
