import os
import numpy as np
from PIL import Image

def get_filepath(filepath,path_read=None):
    if path_read==None:
        path_read=[]
    temp_list=os.listdir(filepath)
    for temp_list_each in temp_list:
        if os.path.isfile(filepath+'/'+temp_list_each):
            temp_path=filepath+'/'+temp_list_each
            if os.path.splitext(temp_path)[-1]=='.png':
                path_read.append(temp_path)
            else:
                continue
        else:
            path_read=get_filepath(filepath+'/'+temp_list_each,path_read)
    return path_read


