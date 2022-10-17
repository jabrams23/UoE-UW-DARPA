import os
import sys
import distutils
import glob

from tensorflow.keras.models import load_model


def load_model_long(model_dir,temp_dir=None):
    if temp_dir is None:
        temp_dir = os.get_cwd()
        
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    distutils.file_util.copy_file(os.path.join(model_dir,'saved_model.pb'), os.path.join(temp_dir,'saved_model.pb'))
    distutils.file_util.copy_file(os.path.join(model_dir,'keras_metadata.pb'), os.path.join(temp_dir,'keras_metadata.pb'))
    distutils.dir_util.copy_tree(os.path.join(model_dir,'variables'), os.path.join(temp_dir,'variables')) 
    model = load_model(temp_dir)
    
    os.remove(os.path.join(temp_dir,'saved_model.pb'))
    os.remove(os.path.join(temp_dir,'keras_metadata.pb'))
    distutils.dir_util.remove_tree(os.path.join(temp_dir,'variables'))

    os.rmdir(temp_dir)
        
    return model
        
def save_model_long(model,model_dir,temp_dir=None):
    if temp_dir is None:
        temp_dir = os.get_cwd()
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    model.save(temp_dir)
    
    for temp_file in glob.glob(os.path.join(temp_dir,'*.*')):
        temp_file_name = os.path.split(temp_file)[-1]
        if os.path.isdir(temp_file):
            distutils.dir_util.copy_tree(temp_file,os.path.join(model_dir,temp_file_name))
        else:
            os.replace(temp_file,os.path.join(model_dir,temp_file_name))
    
    distutils.remove_tree(temp_dir)