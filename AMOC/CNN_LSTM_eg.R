
load('training_set_mix_batch128.RData') #reading in an R datafile that has all 10,000 samples and targets (I have not sent this)
# I have split into randomly sampled 7000 training, 1500 validation and 1500 test (although test is not used here)
# instead I have just sent 500 of the training samples and targets

library('keras')
use_condaenv('tensorflow-metal') #tensorflow-metal is used for Apple silicone but essentially is using tensorflow 2.8

# hyperparameter settings
pool_size_param <- 2
learning_rate_param <- 0.0005     
batch_param <- 128
dropout_percent <- 0.05
filters_param <- 50  
mem_cells <- 50
mem_cells2 <- 10
kernel_size_param <- 12
epoch_param <- 600
initializer_param <- 'lecun_normal'

# when I originally converted this from a Python code it was fairly easy to get into this R format
# I also understood what was happening in the workshop notebooks so I think it should be fairly easy to
# convert back to R
model <- keras_model_sequential()
layer_conv_1d(model, filters=filters_param, kernel_size=kernel_size_param, activation='relu', padding='same', input_shape=c(1500, 1), kernel_initializer=initializer_param)
layer_batch_normalization(model)
layer_dropout(model, dropout_percent)
layer_max_pooling_1d(model, pool_size=pool_size_param)
layer_conv_1d(model, filters=filters_param*2, kernel_size=kernel_size_param, activation='relu', padding='same', kernel_initializer=initializer_param)
layer_dropout(model, dropout_percent)
layer_max_pooling_1d(model, pool_size=pool_size_param)
layer_conv_1d(model, filters=filters_param*2, kernel_size=kernel_size_param, activation='relu', padding='same', kernel_initializer=initializer_param)
layer_dropout(model, dropout_percent)
layer_max_pooling_1d(model, pool_size=pool_size_param)
layer_lstm(model, mem_cells, return_sequences=TRUE, kernel_initializer = initializer_param)
layer_lstm(model, mem_cells, return_sequences=TRUE, kernel_initializer = initializer_param)
layer_dropout(model, dropout_percent)
layer_lstm(model, mem_cells2, kernel_initializer = initializer_param)
layer_dropout(model, dropout_percent)
layer_dense(model, units=1)

model_name <- '3CNN_2LSTM_dropout_per_05'

chk <- callback_model_checkpoint(model_name, monitor='val_loss', save_best_only=TRUE, mode='min', verbose=1) #saving for best validation loss/mae
early <- callback_early_stopping(monitor='val_loss', patience=20)

# I decided to track mae alongside 'loss mae' and they come out the same each epoch for both training and validation
# The problem seems to be from using the model to predict afterwards on these same data
compile(model, loss='mae', optimizer='rmsprop', metrics=c('mae','mse')) 

# for playing around and testing on the smaller dataset I have sent as csv files, we probably don't want 600 epochs
history = fit(model, train, train_target, epochs=epoch_param, batch_size=batch_param, callbacks=list(chk,early), validation_data=list(validation,validation_target))

save_model_tf(model, '3CNN_2LSTM_dropout_per_05_early') #save the final model alongside the best validation one that has been updating
save.image('3CNN_2LSTM_dropout_per_05_early.RData') #this just saves the R workspace to pull out the history and training/validation sets again (although it doesn't save the tf models which have to be reread in)

### Analysis below
####################

# Assume that this script is on a fresh R workspace

load('3CNN_2LSTM_best_val_rmsprop_epoch600.RData')
library('keras')
use_condaenv('tensorflow-metal')

model_best <- load_model_tf('1CNN_1LSTM_rsmprop_best_val') #read in the two models
model_end <- load_model_tf('1CNN_1LSTM_rmsprop_epoch600')

history_train_mae <- history$metrics$loss #pull out the losses (remember mae tracked as a metric has been matching each epoch)
history_val_mae <- history$metrics$val_loss #these make the time series at the top of the figure

train_results_best <- model_best %>% predict(train) #these are the predictions that are used to make the scatter plots
val_results_best <- model_best %>% predict(validation) #the %>% just means predict(model_best, validation) for example (I think it's similar in Python)

train_results_end <- model_end %>% predict(train)
val_results_end <- model_end %>% predict(validation)










