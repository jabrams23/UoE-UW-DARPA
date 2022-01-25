# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:06:29 2020

@author: Daniel Dylewsky


Train CNN-LSTM model to classify simulated Ising model results

order_param variable controls which version to use (varied temp is a 
second-order phase transition, varied external field h is a first-order
phase transition). "ht" option uses a composite set of both



"""
print("Importing dependencies")

import faulthandler; faulthandler.enable()

import os
import pandas as pd
import numpy as np
import sys
# import resource
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score

import tensorflow as tf
# from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# %%
print("Loading data")

retrain_model = True # If False, just redo statistics and plots using exising model

# order_param = 'h'
# order_param = 'h_lin'
# order_param = 'temp'
# order_param = 'temp_lin'
order_param = 'ht_lin'


model_type = 'CNN_LSTM'
# model_type = 'Inception_LSTM'

train_coord_list = ['all','temporal','spatial']

temporal_coords = np.arange(6)
spatial_coords = np.arange(6,12)

if model_type == 'Inception_LSTM':
    import inception_module_functions as incep
    
# smoothing = None
smoothing = 'gaussian'

if smoothing == 'gaussian':
    # smooth_param = [24,0]
    # smooth_param = [48,0]
    smooth_param = [96,0]

# mask_type = None
mask_type = 'ellipse'

if mask_type is None:
    base_dir = os.path.join('Ising_Output','var_'+order_param)
else:
    base_dir = os.path.join('Ising_Output','var_'+order_param+'_'+mask_type)

if smoothing == None:
    data_dir = os.path.join(base_dir,'Processed')
    out_dir = os.path.join(base_dir,'Trained Models')
    # if order_param == 'temp_lin':
    #     data_dir_2 = os.path.join('Ising_Output','var_temp','Processed')
elif smoothing == 'gaussian':
    data_dir = os.path.join(base_dir,'Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
    out_dir = os.path.join(base_dir,'Trained Models','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
    # if order_param == 'temp_lin':
    #     data_dir_2 = os.path.join('Ising_Output','var_temp','Processed','Gaussian_{}_{}'.format(smooth_param[0],smooth_param[1]))
# if train_coords == 'temporal':
#     out_dir = os.path.join(out_dir,'Temporal')
# elif train_coords == 'spatial':
#     out_dir = os.path.join(out_dir,'Spatial')
    
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Set maximum recursion limit to enable loading of large .pkl file

# if sys.getrecursionlimit() < 10000:
#     print("Setting maximum recursion limit")
#     sys.setrecursionlimit(10000)

    
max_rec = 5000

# May segfault without this line. 0x100 is a guess at the size of each stack frame.
# resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
sys.setrecursionlimit(max_rec)


# train_sequences = pd.read_pickle(os.path.join(home_dir,out_dir,'train_sequences.pkl'))

# for kp in range(4):
# for kp in [1,2]:
    
# %%
    
# for train_coords in train_coord_list[:2]:
for train_coords in train_coord_list:
# for train_coords in ['all']:
    # %%
    with np.load(os.path.join(data_dir,'train_data.npz'), allow_pickle=True) as np_load:
        s = np_load['s']
        null = np_load['null']
        train_groups = np_load['train_classes']
        

        
    run_classes = 1-null
    
    
    if train_coords == 'temporal':
        s = s[:,:,temporal_coords]
    elif train_coords == 'spatial':
        s = s[:,:,spatial_coords]
        
    
    # NaN out all masked values
    s_mask = np.array((s != 0),dtype=float)
    s_mask[s_mask==0] = np.nan
    s = np.multiply(s,s_mask) 
    
    # subtract off mean of each feature
    s = s - np.tile(np.expand_dims(np.nanmean(s,axis=(1)),1),(1,s.shape[1],1))
    
    # normalize to unit std dev
    s = np.divide(s,np.tile(np.expand_dims(np.nanstd(s,axis=(1)),1),(1,s.shape[1],1)))
    
    s = np.nan_to_num(s) # convert NaNs back to zeros
    
    saturation_val = 100 # deal with extreme outliers by capping values
    
    s[s>saturation_val] = saturation_val
    s[s<-saturation_val] = -saturation_val
    
    seq_length = s.shape[1]
    training_vars = s.shape[2]
    
    
    # Plot abruptness
    run_class_dict = {0: 'Null', 1: 'Trans.'}
    # run_class_dict = {0: 'Null', 1: 'Fold', 2: 'TC', 3: 'Pitch.', 4: 'Hopf (Sup)', 5: 'Hopf (Sub)', 6: 'Homoc.'}
    
    
    # %% Prepare train classes
    print("Preparing training classes")
    
    train_x_array_nocrop = s[train_groups==0,:,:]
    train_targets = run_classes[train_groups==0]
    
    test_x_array_nocrop = s[train_groups==1,:,:]
    test_targets = run_classes[train_groups==1]
    
    validate_x_array_nocrop = s[train_groups==2,:,:]
    validate_targets = run_classes[train_groups==2]
    
    
    
    
    ## Define different model hyperparameter combinations
    #                       0     1     2     3     4     5     6     7     8     9     10  11  12  13  14     15  16    17
    CL_par_cnn_layer =     [1,    2,    1,    2,    1,    1,    3,    2,    2,    2,    2,  2,  2,  2,  2,     2,  2,     2]
    CL_par_lstm_layer =    [1,    1,    2,    0,    1,    1,    1,    1,    1,    1,    1,  1,  1,  1,  1,     1,  1,     1]
    CL_par_filters =       [20,   20,   20,   20,   40,   20,   20,   20,   20,   20,   20, 20, 20, 20, 20,   20, 20,    20]
    CL_par_kernel_size =   [8,    8,    8,    8,    8,    16,   8,    8,    8,    8,    8,  8,  8,  8,  8,     8,  8,     8]
    CL_par_dropout_pct =   [0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0,    0.2,  0.4,  0.1,0.1,0.1,0.1,0.1, 0.1,0.1,   0.1]
    CL_par_classif_type =  [2,    2,    2,    2,    2,    2,    2,    2,    2,    2,    1,  2,  2,  2,  2,     2,  2,     2] # 1=binary, 2=5-way
    CL_par_rhs_crop =      [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  5, 10, 20,  0,     0,100,   200]
    CL_par_add_noise =     [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  0,  0,  0,  0.1, 0.5,  0,     0]
    # test accuracies = [0.925,0.956,0.937,0.940,0.928,0.923,0.875]
    
    
    IL_par_filters =        [32,    32, 32, 32, 32]
    IL_par_inc_depth =      [1,     2,  4,  6,  2]
    IL_par_kernel_size =    [41,    41, 41, 41, 41]
    IL_par_LSTM_depth =     [0,     0,  0,  0,  1]
    IL_par_pool_size =      [3,     3,  3,  3,  3]
    IL_par_use_resid =      [1,     1,  1,  1,  1]
    
    # %% Tensorflow diagnostics
    if tf.test.gpu_device_name():
       print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
       print("Please install GPU version of TF")
    
    # %% Train
    print("Training models")
    
    
    # for pj in range(len(par_cnn_layer)):
    for pj in [1]:
    # for pj in [0]:
        ## Set up Keras model
        
        if model_type == 'CNN_LSTM':
            hp_dir = os.path.join(out_dir,'CNN_LSTM','HP_'+ str(pj)) + '_' + train_coords
            
            pool_size_param = 2
            learning_rate_param = 0.001     
            batch_param = 32
            dropout_percent = 0.1
            CNN_layers = CL_par_cnn_layer[pj]
            LSTM_layers = CL_par_lstm_layer[pj]
            filters_param = CL_par_filters[pj]
            kernel_size_param = CL_par_kernel_size[pj]
            classif_type = CL_par_classif_type[pj]
            rhs_crop = CL_par_rhs_crop[pj]
            add_noise = CL_par_add_noise[pj]
            mem_cells = 20
            mem_cells2 = 5
            epoch_param = 20
            initializer_param = 'lecun_normal'
            param_dict = dict([(i, eval(i)) for i in ('CNN_layers', 'LSTM_layers', 'pool_size_param', 
                                                      'learning_rate_param', 'dropout_percent',
                                                      'filters_param', 'mem_cells', 'mem_cells2',
                                                      'kernel_size_param','epoch_param','initializer_param',
                                                      'rhs_crop','add_noise')])
            
            if add_noise > 0:
                input_std = np.std(train_x_array_nocrop,axis=(0,1))
                train_noise = np.random.randn(*train_x_array_nocrop.shape)
                test_noise = np.random.randn(*test_x_array_nocrop.shape)
                validate_noise = np.random.randn(*validate_x_array_nocrop.shape)
                for fj in range(train_x_array_nocrop.shape[2]):
                    train_noise[:,:,fj] = input_std[fj]*train_noise[:,:,fj]
                    test_noise[:,:,fj] = input_std[fj]*test_noise[:,:,fj]
                    validate_noise[:,:,fj] = input_std[fj]*validate_noise[:,:,fj]
                train_x_array_noise = train_x_array_nocrop + train_noise
                test_x_array_noise = test_x_array_nocrop + test_noise
                validate_x_array_noise = validate_x_array_nocrop + validate_noise
            else:
                train_x_array_noise = train_x_array_nocrop
                test_x_array_noise = test_x_array_nocrop
                validate_x_array_noise = validate_x_array_nocrop
            
        elif model_type == 'Inception_LSTM':
            hp_dir = os.path.join(out_dir,'INC_LSTM','HP_'+ str(pj)) + '_' + train_coords
            
            batch_param = 64
            nb_filters_param = IL_par_filters[pj]
            use_residual_param = IL_par_use_resid[pj] # include residual skip layers
            use_bottleneck_param = False # used to reduce dimensionality of input (only necessary for data w/ high spatial dimension)
            inc_depth_param = IL_par_inc_depth[pj] # number of inception modules to use
            kernel_size_param = IL_par_kernel_size[pj] # kernel size
            lstm_depth_param = IL_par_LSTM_depth[pj]
            epoch_param = 200
            pool_size_param = IL_par_pool_size[pj]
            classif_type = 2
            
            param_dict = dict([(i, eval(i)) for i in ('batch_param', 'nb_filters_param', 'use_residual_param', 
                                                      'use_bottleneck_param', 'inc_depth_param', 'kernel_size_param',
                                                      'lstm_depth_param','epoch_param','pool_size_param')])
        
        
        if rhs_crop > 0:
            train_x_array = np.zeros_like(train_x_array_noise)
            train_x_array[:,rhs_crop:,:] = train_x_array_noise[:,:-rhs_crop,:]
            
            test_x_array = np.zeros_like(test_x_array_noise)
            test_x_array[:,rhs_crop:,:] = test_x_array_noise[:,:-rhs_crop,:]
            
            validate_x_array = np.zeros_like(validate_x_array_noise)
            validate_x_array[:,rhs_crop:,:] = validate_x_array_noise[:,:-rhs_crop,:]
        else:
            train_x_array = train_x_array_noise
            test_x_array = test_x_array_noise
            validate_x_array = validate_x_array_noise
        
        
        # hp_dir = os.path.join(out_dir,'varied_smoothing','CNN_LSTM_Model_tv_' + str(training_vars),'HP_'+ str(kp))
        if not os.path.exists(hp_dir):
            os.makedirs(hp_dir)
            
        with open(os.path.join(hp_dir,'model_params.txt'), 'w') as f:
            f.write( str(param_dict) )
        
        if classif_type == 1:
            these_train_targets = train_targets.copy()
            these_test_targets = test_targets.copy()
            if validate_targets is not None:
                these_validate_targets = validate_targets.copy()
                these_validate_targets[these_validate_targets != 0] = 1
            else:
                these_validate_targets = None
            
            these_train_targets[these_train_targets != 0] = 1
            these_test_targets[these_test_targets != 0] = 1
        elif classif_type == 2:
            these_train_targets = train_targets
            these_test_targets = test_targets
            if validate_targets is not None:
                these_validate_targets = validate_targets
            else:
                these_validate_targets = None
            
        # One-hot encoded targets
        train_targets_onehot = to_categorical(these_train_targets)
        if these_validate_targets is not None:
            validate_targets_onehot = to_categorical(these_validate_targets)
        else:
            validate_targets_onehot = None
        test_targets_onehot = to_categorical(these_test_targets)
        
        n_classes = train_targets_onehot.shape[1]
    
        
        if retrain_model:
            print('Training Model')
            if model_type == 'CNN_LSTM':
                # print('Training model ' + str(pj) + '/' + str(len(CL_par_cnn_layer)))
                model = Sequential()
                if CNN_layers == 1:
                    model.add(Conv1D(filters=filters_param, kernel_size=kernel_size_param,
                                     activation='relu', padding='same',
                                     batch_input_shape=(None,seq_length, training_vars),
                                     kernel_initializer = initializer_param))
                elif CNN_layers == 2:
                    model.add(Conv1D(filters=filters_param, kernel_size=kernel_size_param,
                                     activation='relu', padding='same',
                                     batch_input_shape=(None,seq_length, training_vars)))
                    model.add(Conv1D(filters=2*filters_param, kernel_size=kernel_size_param,
                                     activation='relu',
                                     padding='same'))
                elif CNN_layers == 3:
                    model.add(Conv1D(filters=filters_param, kernel_size=kernel_size_param,
                                     activation='relu', padding='same',
                                     batch_input_shape=(None,seq_length, training_vars)))
                    model.add(Conv1D(filters=2*filters_param, kernel_size=kernel_size_param,
                                     activation='relu', padding='same'))
                    model.add(Conv1D(filters=2*filters_param, kernel_size=kernel_size_param,
                                     activation='relu', padding='same'))
                
                model.add(Dropout(dropout_percent))
                model.add(MaxPooling1D(pool_size=pool_size_param))
                
                #    model.add(Flatten())
                
                # model.add(LSTM(mem_cells, return_sequences=True,kernel_initializer = initializer_param))
                # model.add(Dropout(dropout_percent))
                
                if LSTM_layers == 1:
                    model.add(LSTM(mem_cells,kernel_initializer = initializer_param))
                    model.add(Dropout(dropout_percent))
                elif LSTM_layers == 2:
                    model.add(LSTM(mem_cells, return_sequences=True))
                    model.add(LSTM(mem_cells2,kernel_initializer = initializer_param))
                    model.add(Dropout(dropout_percent))
                
                
                model.add(Dropout(dropout_percent))
                model.add(Dense(n_classes, activation='softmax',kernel_initializer = initializer_param))
                
                adam = Adam(lr=learning_rate_param)
                chk = ModelCheckpoint(os.path.join(hp_dir,'model_checkpoint.hdf5'), monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
                csvlog = tf.keras.callbacks.CSVLogger(os.path.join(hp_dir,'model_history_log.csv'), append=True)
                model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', 'categorical_accuracy'])
                                
                if validate_x_array is None:
                    history = model.fit(train_x_array, train_targets_onehot,
                                    epochs=epoch_param, batch_size=batch_param,
                                    callbacks=[chk,csvlog], verbose=1)
                else:
                    history = model.fit(train_x_array, train_targets_onehot,
                                    epochs=epoch_param, batch_size=batch_param,
                                    callbacks=[chk,csvlog], verbose=1,
                                    validation_data=(validate_x_array,validate_targets_onehot))
                hist_df = pd.DataFrame(history.history)
                model.save(hp_dir)
                test_pred_probs = model.predict(test_x_array)
                
            elif model_type == 'Inception_LSTM':
                input_shape = train_x_array.shape[1:]
                
                model = incep.Classifier_INCEPTION(hp_dir, input_shape, n_classes, verbose=True,build=True,
                                                  batch_size=batch_param,
                                                  nb_filters=nb_filters_param,
                                                  use_residual=use_residual_param,
                                                  use_bottleneck=use_bottleneck_param,
                                                  inc_depth=inc_depth_param,
                                                  kernel_size=kernel_size_param,
                                                  lstm_depth=lstm_depth_param,
                                                  nb_epochs=epoch_param,
                                                  pool_size=pool_size_param)
                hist_df = model.fit(train_x_array, train_targets_onehot, validate_x_array, 
                                    validate_targets_onehot, validate_targets,plot_test_acc=True)
        
                test_pred_probs = model.predict(test_x_array, test_targets, train_x_array,
                                                train_targets_onehot,test_targets_onehot,
                                                return_df_metrics=False)
                
            # test_preds = model.predict_classes(test_x_array)
            test_preds = np.argmax(test_pred_probs,axis=1)
            test_preds_onehot = to_categorical(test_preds)
    
            
            hist_df.to_pickle(os.path.join(hp_dir,'training_history.pkl'))
            
            
            # print("Precision: ",precision_score(these_test_targets, test_preds, average="macro"))
            # print ("Accuracy sklearn:",accuracy_score(these_test_targets, test_preds))
            keras_accuracy = tf.keras.metrics.CategoricalAccuracy()
            keras_accuracy.update_state(test_targets_onehot, test_preds_onehot)
            # print ("Accuracy keras:",np.array(keras_accuracy.result))
        
            numpy_vars = {'test_preds':test_preds,
                  'test_pred_probs':test_pred_probs}
            np.savez_compressed(os.path.join(hp_dir,'test_preds.npz'),**numpy_vars,allow_pickle=True, fix_imports=True)
        
            # accuracy_score(these_test_targets, test_preds)
        else:
            # print('Loading trained model ' + str(pj) + '/' + str(len(CL_par_cnn_layer)))
            print('Loading trained model ' +str(pj))
            
            if model_type == 'CNN_LSTM':
                hist_df = pd.read_pickle(os.path.join(hp_dir,'training_history.pkl'))
                model = load_model(hp_dir)
            elif model_type == 'Inception_LSTM':
                hist_df = pd.read_csv(os.path.join(hp_dir,'model_history_log.csv'))
                model = load_model(os.path.join(hp_dir,'best_model.hdf5'))
            # try:
            #     with np.load(os.path.join(hp_dir,'test_preds.npz'),allow_pickle=True) as np_load:
            #         test_preds = np_load['test_preds']
            #         test_pred_probs = np_load['test_pred_probs']
            #     test_preds_onehot = to_categorical(test_preds)
            # except FileNotFoundError:
            #     print('Failed to load test predictions. Recomputing.')
            # test_preds = model.predict_classes(test_x_array)
            test_pred_probs = model.predict(test_x_array)
            test_preds = np.argmax(test_pred_probs,axis=1)
            test_preds_onehot = to_categorical(test_preds)
            # print("Precision: ",precision_score(these_test_targets, test_preds, average="macro"))
            # print ("Accuracy sklearn:",accuracy_score(these_test_targets, test_preds))
            keras_accuracy = tf.keras.metrics.CategoricalAccuracy()
            keras_accuracy.update_state(test_targets_onehot, test_preds_onehot)
            # print ("Accuracy keras:",np.array(keras_accuracy.result))
        
            numpy_vars = {'test_preds':test_preds,
                  'test_pred_probs':test_pred_probs}
            np.savez_compressed(os.path.join(hp_dir,'test_preds.npz'),**numpy_vars,allow_pickle=True, fix_imports=True)
        
    
        
        # %% Plot training history

        title_string = order_param + ' | ' + train_coords
        if smoothing is not None:
            title_string = title_string + ' | ' + str(smooth_param)
        
        plot_dir = os.path.join(hp_dir,'Plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        plt.figure()
        plt.plot(hist_df['accuracy'])
        try:
            plt.plot(hist_df['val_accuracy'])
        except KeyError:
            print('val_accuracy not present in hist_df')
            
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(plot_dir,'Train_history.png'))
        plt.close()
        
        # summarize history for loss
        plt.figure()
        plt.plot(hist_df['loss'])
        try:
            plt.plot(hist_df['val_loss'])
        except KeyError:
            print('val_loss not present in hist_df')
        plt.title(title_string)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(plot_dir,'Loss_history.png'))
        plt.close()
        
        # %% Plot confusion matrix
        if classif_type == 1:
            legend_dict = {0: 'Null', 1: 'Bif.'}
        elif classif_type == 2:
            legend_dict = {0: 'Null', 1: 'Trans.'}
            # legend_dict = {0: 'Null', 1: 'Fold', 2: 'TC', 3: 'Pitch.', 4: 'Hopf (Sup)', 5: 'Hopf (Sub)', 6: 'Homoc.'}
        class_names = list(legend_dict.values())
        
        confusion_matrix = tf.math.confusion_matrix(these_test_targets,test_preds).numpy()
        confusion_matrix_normalize = np.zeros(confusion_matrix.shape)
        for cr in range(confusion_matrix.shape[0]):
            confusion_matrix_normalize[cr,:] = confusion_matrix[cr,:]/np.sum(confusion_matrix[cr,:])
        
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix_normalize,display_labels=class_names)
        disp.plot()
        plt.title(title_string)
        plt.savefig(os.path.join(plot_dir,'Confusion_Matrix.png'))
        plt.close()
        
        # %% Plot ROC curves
        plt.figure()
        for target_class in np.unique(test_preds):
            these_targets = these_test_targets.copy()
            # for each ROC curve, lump all other classes together
            these_targets[these_targets != target_class] = -1
            these_targets[these_targets == target_class] = 1
            these_targets[these_targets == -1] = 0
            
            these_preds = test_pred_probs[:,target_class]
            fpr, tpr, thresholds = roc_curve(these_targets, these_preds)
            this_auc = roc_auc_score(these_targets, these_preds)
            
            plt.plot(fpr,tpr,label=class_names[target_class] + ' [AUC {:.2f},n={}]'.format(this_auc,len(these_targets)))
        plt.xlabel('True Pos. Rate')
        plt.ylabel('False Pos. Rate')
        plt.title(title_string)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'--')
        plt.savefig(os.path.join(plot_dir,'ROC_curves.png'))
        plt.close()
        
        
        
        # %% 
        seq_length_unpadded = s.shape[1]
        pred_step = 4
        
        long_run_length = int(250 - rhs_crop/2)
        
        long_runs = np.where(test_x_array[:,long_run_length,0] != 0)[0]
        
        n_test_runs = 16
        
        if len(long_runs) < n_test_runs:
            print('Not enough runs meet length threshold for plotting: Ignoring threshold')
            long_runs = np.arange(test_x_array.shape[0])
        
        
        ROC_averaging_steps = 12 # this many time steps leading up to the transition are used when computing ROC curves
        
        all_pred_val = np.zeros((n_test_runs,ROC_averaging_steps,n_classes))
        all_true_val = np.zeros((n_test_runs,ROC_averaging_steps,n_classes))
        
        # Test every time step in the last ROC_averaging_steps steps
        # query_steps = np.hstack((np.arange(0,seq_length_unpadded-ROC_averaging_steps,pred_step),
        #                         np.arange(seq_length_unpadded-ROC_averaging_steps,seq_length_unpadded)))
        query_steps = np.hstack((np.arange(0,ROC_averaging_steps),
                                np.arange(ROC_averaging_steps,seq_length_unpadded,pred_step)))
        
        
        
        for k,nr in enumerate(np.random.choice(long_runs,size=n_test_runs,replace=False)):
            pred_time = []
            pred_val = []
            
            
            this_test_x = test_x_array[nr,:,:]
            this_test_x_pad = np.vstack((np.zeros((seq_length_unpadded,test_x_array.shape[2])),test_x_array[nr,:,:]))
            this_test_x_pad = np.expand_dims(this_test_x_pad,0)
                    
            for j in query_steps:
                this_test_x = this_test_x_pad[:,-(j+seq_length_unpadded):,:]
                this_test_x = this_test_x[:,:seq_length_unpadded,:]
                
                # this_test_x = np.vstack((np.zeros((seq_length_unpadded,x_test.shape[2])),x_test[nr,:,:]))
                # this_test_x = this_test_x[j:(j+seq_length_unpadded),:]
                # this_test_x = np.expand_dims(this_test_x,0)
                # if np.ndim(this_test_x) == 2:
                #     this_test_x = np.expand_dims(this_test_x,2)
                
                this_pred = model.predict(this_test_x)[0]
                
                pred_time.append(seq_length_unpadded-j-1)
                pred_val.append(this_pred)
                
            
            pred_val_array = np.zeros((len(pred_val),len(pred_val[0])))
            for j in range(len(pred_val)):
                pred_val_array[j,:] = pred_val[j]
            
            all_pred_val[k,:,:] = pred_val_array[:ROC_averaging_steps,:] # pred_val is ordered backwards in time
            all_true_val[k,:,:] = these_test_targets[nr]*np.ones((ROC_averaging_steps,n_classes))
            
            fig, axs = plt.subplots(2,sharex=True)
            axs[0].plot(np.squeeze(test_x_array[nr,:,:]))
            pred_plot = axs[1].plot(pred_time,pred_val_array,'-')
            axs[1].legend(pred_plot, list(legend_dict.values()),loc='upper left')
            axs[1].hlines(0.5,0,690,colors='k',linestyles=':')
            axs[0].set_title('Test Run ' + str(nr) + ' (True = ' + legend_dict[these_test_targets[nr]] + ', Pred.= ' + legend_dict[test_preds[nr]] + ')')
            
            for ax in axs:
                ax.set_xlim([0,test_x_array.shape[1]])
            
            plt.savefig(os.path.join(plot_dir,'Pred_test_run_' + str(nr) + '.png'))
            plt.close()
        # Compute averaged ROC curves
        
        fig, axs = plt.subplots(1)
        
        for target_class in np.unique(test_preds):
            these_targets = all_true_val[:,:,target_class].copy().flatten()
            # for each ROC curve, lump all other classes together
            these_targets[these_targets != target_class] = -1
            these_targets[these_targets == target_class] = 1
            these_targets[these_targets == -1] = 0
            
            these_preds = all_pred_val[:,:,target_class].flatten()
            fpr, tpr, thresholds = roc_curve(these_targets, these_preds)
            this_auc = roc_auc_score(these_targets, these_preds)
            
            plt.plot(fpr,tpr,label=class_names[target_class] + ' [AUC {:.2f},n={}]'.format(this_auc,len(these_targets)))
            
        plt.ylabel('True Pos. Rate')
        plt.xlabel('False Pos. Rate')
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'--')
        plt.title('ROC Avg. over last ' + str(ROC_averaging_steps) + ' time steps\n' + title_string)
        plt.savefig(os.path.join(plot_dir,'ROC_curves_mean.png'))
        plt.close()    
    