library(ncdf4)
libraru(zoo)

data_url <- 'data'
nsidc <- nc_open(data_url)


conc <- ncvar_get(nsidc,'cdr_seaice_conc_monthly')
xgrid <- ncvar_get(nsidc,'xgrid')
ygrid <- ncvar_get(nsidc,'ygrid')
conc2 <- conc
conc2[conc2 == 0] <- NA                                    #Set ice free values to NA


for (i in 1:dim(conc2)[1]){
  for (j in 1:dim(conc2)[2]){
    num_na[i,j] <- sum(is.na(conc2[i,j,]))
  }
}

#Find first time a pixel shows an ice free value (NA)

when_na <- array(NA, c(dim(conc2)[1],dim(conc2)[2]))
for (i in 1:dim(conc2)[1]){
  for (j in 1:dim(conc2)[2]){
    when_na[i,j] <- which(is.na(conc2[i,j,]))[1]
  }
}

#Find first time a pixel drops below a certain threshold (here we use 0.5)

when_low <- array(NA, c(dim(conc2)[1],dim(conc2)[2]))
for (i in 1:dim(conc2)[1]){
  for (j in 1:dim(conc2)[2]){
    when_low[i,j] <- which(conc2[i,j,]<0.5 | is.na(conc2[i,j,]))[1]
  }
}

#Find first time a pixel drops below a higher threshold (here we use 0.8). This is to check if a pixel has begun a slow decline in summer ice cover

pre_when_low <- array(NA, c(dim(conc2)[1],dim(conc2)[2]))
for (i in 1:dim(conc2)[1]){
  for (j in 1:dim(conc2)[2]){
    pre_when_low[i,j] <- which(conc2[i,j,]<0.8)[1]
  }
}

#For loop to analyse the time series of each pixel, remove those which either do not dip below the threshold or do so in the first 6 years (to ensure sufficient time points to analyse).
#Then cut each pixel's time series prior to tipping and calculate the AR(1) and its kendall tau value and save this as a matrix.

ar1_tau_map <- array(NA, c(dim(conc)[1],dim(conc)[2]))

for (i in 1:dim(conc2)[1]){
  for (j in 1:dim(conc2)[2]){
    t <- conc2[i,j,]              #Get time series of each pixel in turn
    if(is.na(t[1])) next          #Skip if there is no ice at the start
    t[t >1.5] <- 1                #Two of the values early in the time series show an erronously large value, here this is set to 1
    k <- when_low[i,j]            #First month below low threshold
    m <- pre_when_low[i,j]        #First month below higher threshold
    if(is.na(k)) next
    if(any(k<60)) next            #Skip if shift appears in first 6 yers
    if(any(m+24 <k)) next         #Skip if higher threshold is more than 2 years earlier than lower threshold
    t_cut <- t[1:(k-mod(k,12))]
    t_cut_season <- rowMeans(matrix(t_cut,nrow=12))
    t_cut_season_rep <- rep(t_cut_season,length(t_cut)/12)
    t_cut_deseasonal <- t_cut - t_cut_season_rep              #Remove seasonal cycle
    l <- length(t_cut_deseasonal)
    wl <- length(t_cut_deseasonal)/2
    ar1 <- rep(0,(l-wl))
    for (z in 1:(l-wl)){                                     #Calculate AR(1)
      ar1[z] <- cor.test(t_cut_deseasonal[z:(z+wl)],t_cut_deseasonal[(z+1):(z+1+wl)])$estimate
    }
    time <- 1:length(ar1)
    tau <- cor.test(ar1,time,method='kendall')$estimate       #Calculate AR(1) trend value
    ar1_tau_map[i,j] <- tau
  }
}



#Now apply Bury DL model to each pixel
#This requires downloading the Bury DL model from https://github.com/jabrams23/UoE-UW-DARPA/tree/main/Bury_Models

#Create a function to normalise the time series prior to reading it into the model

normalise <- function(x) {
  # a function to noramlise time series to predict from
  norm_x <- rep(NA, length(x))
  for (i in 1:length(x)) {
    norm_x[i] <- (x[i]-mean(x))/sd(x)
  }
  return(norm_x)
}


#There are twenty models within the Bury model, this script will read one in to analyse each pixel with the model, before reading the next model in
#The loops have the same preprocessing as the AR(1) loop with the addition of a normalisation stage
#Each of the twenty models will give a probability of a fold, hopf, transcritical or no bifurcation

model_location <- 'whichever_folder_you_save_the_model_in'
predictions_ice <- array(NA, dim=c(dim(conc2)[1],dim(conc2)[2],4,20))
count <- 0                                                                #Don't forget to reset the count each time!
len <- 1500
for (s in 1:10) {
  for (t in 1:2) {
    count <- count + 1
    model <- load_model_tf(paste(model location,'best_model_',s,'_',t,'_len',len,'.pkl',sep=''))
    try(predict(model, t(rnorm(1500))), silent=TRUE) #seems to be a bug where it has to go through a failed prediction first
    for (i in 1:dim(conc2)[1]) {
      for (j in 1:dim(conc2)[2]) {
        print(c(i,j,count))
        t <- short_conc[i,j,]
        if(is.na(t[1])) next
        t[t >1.5] <- 1
        k <- long_when_low[i,j]
        #m <- pre_long_when_low[i,j]
        if(is.na(k)) next
        if(any(k<60)) next            #Skip if shift appears in first 6 yers
        #if(any(m+24 <k)) next
        t_cut <- t[1:(k-mod(k,12))]
        t_cut_season <- rowMeans(matrix(t_cut,nrow=12))
        t_cut_season_rep <- rep(t_cut_season,length(t_cut)/12)
        t_cut_deseasonal <- t_cut - t_cut_season_rep
        ts <- normalise(t_cut_deseasonal)
        predictions_ice[i,j,,count] <- predict(model, t(ts))
        saveRDS(predictions_ice,'/Users/joshbuxton/Library/CloudStorage/OneDrive-UniversityofExeter/DARPA Postdoc/BAE Collaboration/average_probs.rds')
      }
    }
  }
}

#Now average each model to get 1 probability for each outcome

average_prob_ice <- array(NA, dim=c(dim(predictions_ice)[1],dim(predictions_ice)[2],4))
for (i in 1:dim(predictions_ice)[1]) {
  for (j in 1:dim(predictions_ice)[2]) {
    for (k in 1:4) {
      average_prob_ice[i,j,k] <- mean(predictions_ice[i,j,k,])
    }
  }
}

#The probabiliy of a given pixel i,j undergoing the following bifurcation is therefore:

fold <- average_prob_ice[i,j,1]
hopf <- average_prob_ice[i,j,2]
trans <- average_prob_ice[i,j,3]
no_bifur <- average_prob_ice[i,j,4]
total_bifur <- 1-no_bifur

