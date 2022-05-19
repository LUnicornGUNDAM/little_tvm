# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:31:57 2019

@author: wuzhe
"""
import cv2
from math import floor
import numpy as np
import dhash
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 2300000000
from os import listdir, walk
from os.path import isfile, join, isdir, getsize
import itertools

from pathlib import Path
import imagededup
##### Pre-set Fileinfo ######
total_file_size = 0 # in Byte
block_range = 0

def get_all_path(open_file_path):
    global total_file_size 
    rootdir = open_file_path
    path_list = []
    list = listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        com_path = join(rootdir, list[i])
        #print(com_path)
        if isfile(com_path):
            if getsize(com_path) > 4096:
                total_file_size += getsize(com_path)
            else:
                total_file_size += 4096
            path_list.append(com_path)
        if isdir(com_path):
            path_list.extend(get_all_path(com_path))
    #print(path_list)
    return path_list

from multiprocessing import Manager

mgr_phash = Manager()
img_phash=mgr_phash.dict();

mgr_ahash = Manager()
img_ahash=mgr_ahash.dict();

mgr_dhash = Manager()
img_dhash=mgr_dhash.dict();

mgr_whash = Manager()
img_whash=mgr_whash.dict();

img_cnn = {}

mgr_result = Manager()
result=[]=mgr_result.list()
      
def clear_result(result):
    return_result = []
    for i,j in result.items():
        if any(j):
            front = i
            back = j[0][0]
            temp_list = [front,back]
            if temp_list not in return_result:
                if [temp_list[1],temp_list[0]] in return_result:
                    continue
                else:
                    return_result.append(temp_list)
    return return_result


def img_similarity_compare():
    
    try:
        phash_result = phash.find_duplicates(encoding_map=img_phash,max_distance_threshold=1,scores=True)
        phash_result = clear_result(phash_result)
        phash_file=open('result_phash.txt',mode='w')
        for i in phash_result:
            phash_file.write("{}-----{}\n".format(i[0],i[1]))
        phash_file.close()
        
        ahash_result= ahash.find_duplicates(encoding_map=img_ahash,max_distance_threshold=1,scores=True)
        ahash_result = clear_result(ahash_result)
        ahash_file=open('result_ahash.txt',mode='w')
        for i in ahash_result:
            ahash_file.write("{}-----{}\n".format(i[0],i[1]))
        ahash_file.close()
        
        dhash_result= dhash.find_duplicates(encoding_map=img_dhash,max_distance_threshold=1,scores=True)
        dhash_result = clear_result(dhash_result)
        dhash_file=open('result_dhash.txt',mode='w')
        for i in ahash_result:
            dhash_file.write("{}-----{}\n".format(i[0],i[1]))
        dhash_file.close()
        
        whash_result= whash.find_duplicates(encoding_map=img_whash,max_distance_threshold=1,scores=True)
        whash_result = clear_result(whash_result)
        whash_file=open('result_whash.txt',mode='w')
        for i in whash_result:
            whash_file.write("{}-----{}\n".format(i[0],i[1]))
        whash_file.close()
        
        '''
        cnn_result= cnn.find_duplicates(encoding_map=img_cnn,min_similarity_threshold=0.85,scores=True)
        cnn_file=open('cnn_result.txt',mode='w')
        for i,j in cnn_result.items():
            if any(j):
                cnn_file.write("{},{}\n".format(i,j))
        cnn_file.close()
        '''
    except:
        print("compare error occurred!")
    
    
from imagededup.methods import PHash, AHash, DHash, WHash, CNN
phash = PHash()
ahash = AHash()
dhash = DHash()
whash = WHash()
#cnn = CNN()
def img_similarity_calculate(img_list):
    img_path=img_list
    try:
        #print('{:s} & {:s}'.format(img1_path,img2_path))
        
        phash_temp = phash.encode_image(img_list)
        if phash_temp is not None:
            img_phash[img_path] = phash_temp
            
        ahash_temp = ahash.encode_image(img_list)
        if ahash_temp is not None:
            img_ahash[img_path] = ahash_temp
            
        dhash_temp = dhash.encode_image(img_list)
        if dhash_temp is not None:
            img_dhash[img_path] = dhash_temp
        
        whash_temp = whash.encode_image(img_list)
        if whash_temp is not None:
            img_whash[img_path] = whash_temp
        
        #print('{:s}'.format(img_path), end="\r", flush=True)
    
    except:
        print("cannot identify image file '{}'".format(img_list))
        #print('check the following images:{:s}\r'.format(img_path))


########### Pre-set Memory ##############
machine_name = 'Cherudim GUNDAM'
memory_speed = 2400 #in MHz
memory_channel = 4 
memory_width = 64 #in Bits
bit_to_byte = 8
theoretic_memory_putthrough = memory_speed * memory_width * memory_channel / bit_to_byte
    
from multiprocessing import Pool
import time
image_dict =[]

#from memory_profiler import profile
#@profile
def main():
    start_time=time.time()
    import os
    from multiprocessing import cpu_count
    file_size = 0
    print("Total CPU:{}".format(cpu_count()))
    for img_path in get_all_path("images"):
        if '.jpg' in img_path.lower():
            image_dict.append(img_path)
            file_size += os.path.getsize(img_path)
        elif '.jpeg' in img_path.lower():
            image_dict.append(img_path)
            file_size += os.path.getsize(img_path)
        elif '.png' in img_path.lower():
            image_dict.append(img_path)
            file_size += os.path.getsize(img_path)
        elif '.tiff' in img_path.lower():
            image_dict.append(img_path)
            file_size += os.path.getsize(img_path)
        elif '.tif' in img_path.lower():
            image_dict.append(img_path)
            file_size += os.path.getsize(img_path)
    print("Total images:{}".format(str(len(image_dict))))
    s_hash_time=time.time()
    
    #Hash
    hash_pool=Pool(cpu_count())
    hash_pool.map(img_similarity_calculate,image_dict)
    hash_pool.close()
    hash_pool.join()
    
    e_hash_time=time.time()
    
    print('############## Hash Performance #################')
    print('Hash time:{:.2f}s\nHash efficiency:{:.1f}pics/s\nTotal_IO:{:.1f}MB'.format(e_hash_time-s_hash_time,
                                                                    len(image_dict)/(e_hash_time-s_hash_time),
                                                                    file_size/1024/1024
                                                                    )
    )
    print('#################################################\n')
    '''
    s_cnn_time=time.time()
    
    #CNN
    for img in image_dict:
        try:
            cnn_2d_array = cnn.encode_image(img)
            if cnn_2d_array is not None:
                cnn_1d_array = cnn_2d_array.flatten()
                img_cnn[img] = cnn_1d_array
        except:
            print("cannot identify image file '{}'".format(img))
    
    e_cnn_time=time.time()
    
    print('############## CNN Performance ##################')
    print('CNN time:{:.2f}s\nCNN efficiency:{:.1f}pics/s\n'.format(e_cnn_time-s_cnn_time,
                                                                    len(image_dict)/(e_cnn_time-s_cnn_time)
                                                                    )
    )
    print('#################################################\n')
    '''
    '''
    print('#################################################\n')
    for keys,values in img_phash.items():
        print (keys,values)
    print('#################################################\n')
    for keys,values in img_ahash.items():
        print (keys,values)
    print('#################################################\n')
    for keys,values in img_dhash.items():
        print (keys,values) 
    print('#################################################\n')
    for keys,values in img_whash.items():
        print (keys,values)        
    print('#################################################\n')
    for keys,values in img_cnn.items():
        print (keys,values)          
    '''
    
    possible_combinations=len(image_dict)*(len(image_dict)-1)/2

    s_comp_time=time.time()
    img_similarity_compare()
    e_comp_time=time.time()
    
    cmp_time=e_comp_time-s_comp_time
    print("Total combinations:{}".format(str(int(possible_combinations))))
    print('############# Compare Performance ###############')
    print('Compare time:{:.2f}s\nCompare efficiency:{:.1f}combs/s'.format(cmp_time,
                                                                    possible_combinations/(cmp_time)
                                                                    )
    )
    print('#################################################\n')

    print("Used time:{:.2f}s".format(time.time() - start_time))
    
    

    

if __name__ == '__main__':
    main()
