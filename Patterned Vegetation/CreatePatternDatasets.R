#Author: @jbuxt

library(ptw)
library(abind)
library(tensorflow)
library(keras)
use_condaenv("tensorflow-metal")


#Create pattern classes

rainfall= 1

pattern <- generatePattern(configFile=config,startingPatternFilename=NULL,R=rainfall)

#Create rainfall sets for each pattern
spot_rainfall <- rnorm(10000, mean=0.8, sd=0.1)
lab_rainfall <- rnorm(10000, mean=1.6, sd=0.1)
gap_rainfall <- rnorm(10000, mean=2.3, sd=0.1)

#Create empty arrays to populate for each pattern
spot_patterns_150 <- array(c(pattern*0),dim=c(150,150,length(spot_rainfall)))
lab_patterns_150 <- array(c(pattern*0),dim=c(150,150,length(lab_rainfall)))
gap_patterns_150 <- array(c(pattern*0),dim=c(150,150,length(gap_rainfall)))

#Run loops to generate datasets of 10000 of each pattern
for (i in 1:length(spot_rainfall)){
  spot_patterns_150[,,i] <- generatePattern(configFile=configFile,startingPatternFilename = NULL,R=spot_rainfall[i])
  print(i)
  saveRDS(spot_patterns_150, file="/Users/joshbuxton/Library/CloudStorage/OneDrive-UniversityofExeter/DARPA Postdoc/Pattern Veg DL Model/spot_dataset_150.rds")
}

for (i in 1:length(lab_rainfall)){
  lab_patterns_150[,,i] <- generatePattern(configFile=configFile,startingPatternFilename = NULL,R=lab_rainfall[i])
  print(i)
  saveRDS(lab_patterns_150, file="/Users/joshbuxton/Library/CloudStorage/OneDrive-UniversityofExeter/DARPA Postdoc/Pattern Veg DL Model/lab_dataset_150.rds")
}

for (i in 1:length(gap_rainfall)){
  gap_patterns_150[,,i] <- generatePattern(configFile=configFile,startingPatternFilename = NULL,R=gap_rainfall[i])
  print(i)
  saveRDS(gap_patterns_150, file="/Users/joshbuxton/Library/CloudStorage/OneDrive-UniversityofExeter/DARPA Postdoc/Pattern Veg DL Model/gap_dataset_150.rds")
}

# Turn images to binary (1 for veg, 0 for soil)

for (i in 1:10000){
  img <- gap_patterns_150[,,i]
  img[img < 1] = 0
  img[img >0 ] = 1
  gap_150_binary[,,i] <- img
  print(i)
}


#Create 'no pattern' datasets

#1 - Full soil or vegetation

soil <- matrix(0, nrow=150, ncol = 150)

soil_mat <-  array(soil,dim=c(150,150,2000))

full_veg <- matrix(1, nrow=150, ncol = 150)

full_veg_mat <-  array(full_veg,dim=c(150,150,2000))


#2 - Random noise

noise <- matrix(rnorm(2500), nrow=150, ncol=150)
binary_noise <- (sign(noise) +1 )/2

binary_noise_mat <- array(noise*0,dim=c(150,150,2000))

for (i in 1:2000){
  noise <- matrix(rnorm(2500), nrow=150, ncol=150)
  binary_noise <- (sign(noise) +1 )/2
  binary_noise_mat[,,i] <- binary_noise
}



#3 - Triangle matrix

mat <- matrix(1, nrow=150, ncol=150)
mat[lower.tri(mat)] <- 0

lower_tri <- array(mat,dim=c(150,150,1000))

#And
mat2 <- matrix(1, nrow=150, ncol=150)
mat2[upper.tri(mat2)] <- 0

upper_tri <- array(mat2,dim=c(150,150,1000))


#4 - Stripes

vert_stripes <- matrix(NA, nrow=150, ncol=150)
vert_stripes_mat <- array(vert_stripes,dim=c(150,150,1000))

for (i in 1:nrow(vert_stripes)) {
  
  if (i %% 7 == 0) {
    vert_stripes[i, ] <- 1
  } else {
    vert_stripes[i, ] <- 0
  }
}


#5 - Checkerboard 

n=150
check <- (matrix(1:n,n,n,T)+1:n-1)%%2

#6 - Random large blocks of vegetation/soil in middle of image

#Vegetation block

veg_block_mat <- array(soil,dim=c(150,150,1000))

for (i in 1:1000){
  n <- 2*floor(runif(1,min=2,max=76))
  field <- matrix(1, nrow=n, ncol = n)
  pad1 <- padzeros(field,(150-n)/2,side='both')
  veg_block <- padzeros(t(pad1),(150-n)/2,side='both')
  veg_block_mat[,,i] <- veg_block
}


n <- 2*floor(runif(1,min=2,max=76))
field <- matrix(1, nrow=n, ncol = n)
pad1 <- padzeros(field,(150-n)/2,side='both')
veg_block <- padzeros(t(pad1),(150-n)/2,side='both')


#Soil block
soil_block_mat <- array(soil*0,dim=c(150,150,1000))

for (i in 1:1000){
  n <- 2*floor(runif(1,min=2,max=76))
  field <- matrix(1, nrow=n, ncol = n)
  pad1 <- padzeros(field,(150-n)/2,side='both')
  pad2 <- padzeros(t(pad1),(150-n)/2,side='both')
  soil_block <- (pad2 - 1)*(-1)
  soil_block_mat[,,i] <- soil_block
}


#Bind all non-vegetation matrices into 1 set

no_pattern_150 <- abind(soil_mat, full_veg_mat, binary_noise_mat, lower_tri, upper_tri, veg_block_mat, soil_block_mat, along=3)

# Combine all patterns into one dataset

all_patterns_150 <- abind(gap_150_binary, lab_150_binary, spot_150_binary, no_pattern_150, along=3)

#Create pattern labels
gap_labels <- rep(1,10000)
lab_labels <- rep(2,10000)
spot_labels <- rep(3,10000)
no_pattern_labels <- rep(4,10000)

all_labels_150 <- abind(gap_labels, lab_labels, spot_labels, no_pattern_labels, along=1)


