library(abind)
library(keras)
library(tensorflow)
use_condaenv("tensorflow-metal")


#Import data

all_patterns_150 <- readRDS("filepath")
all_labels <- readRDS("filepath")



# Shuffle arrays

a <- all_patterns_150
z.rand <- sample(dim(a)[3])
a[] <- a[,,z.rand]
shuffled_patterns_150 <- a

b <- all_labels
b[] <- b[z.rand] #This will shuffle the labels in the same way that the patterns have been shuffled
shuffled_labels_150 <- b
#saveRDS(shuffled_labels_150, file="/Users/joshbuxton/Library/CloudStorage/OneDrive-UniversityofExeter/DARPA Postdoc/Pattern Veg DL Model/shuffled_labels_150.rds")
#shuffled_labels <- readRDS(file="/Users/joshbuxton/Library/CloudStorage/OneDrive-UniversityofExeter/DARPA Postdoc/Pattern Veg DL Model/shuffled_labels_50.rds")

# Split into training, validation and testing sets

train_set <- shuffled_patterns_150[,,1:32000]
validation_set <- shuffled_patterns_150[,,32001:36000]
test_set <- shuffled_patterns_150[,,36001:40000]

train_labels <- shuffled_labels_150[1:32000]
validation_labels <- shuffled_labels_150[32001:36000]
test_labels <- shuffled_labels_150[36001:40000]

#Convert labels

train_labels <- to_categorical(train_labels)
validation_labels <- to_categorical(validation_labels)
test_labels <- to_categorical(test_labels)

train_labels2 <- train_labels[,2:5]
validation_labels2 <- validation_labels[,2:5]
test_labels2 <- test_labels[,2:5]

#Reshape arrays to fit in model

reshape_train <- array(c(pattern*0),dim=c(150,150,))

train_set_reshaped <- array(NA, dim=c(32000, 150, 150))
validation_set_reshaped <- array(NA, dim=c(4000, 150, 150))
test_set_reshaped <- array(NA, dim=c(4000, 150, 150))

for (i in 1:32000) {
  train_set_reshaped[i,,] <- train_set[,,i]
}

for (i in 1:4000) {
  validation_set_reshaped[i,,] <- validation_set[,,i]
}

for (i in 1:4000) {
  test_set_reshaped[i,,] <- test_set[,,i]
}



#DL Model Stage


model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150,150,1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 4, activation = "softmax")


early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 15)

history <- model %>% fit(train_set_reshaped, train_labels2, steps_per_epoch = 100, epochs=200, 
                         validation_data = list(validation_set_reshaped, validation_labels2), callbacks = list(early_stop))

metrics <- model %>% evaluate(test_set_reshaped,test_labels2)
                                
