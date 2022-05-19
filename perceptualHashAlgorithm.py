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
Image.MAX_IMAGE_PIXELS = 2300000000
from os import listdir, walk
from os.path import isfile, join, isdir, getsize
import itertools

##### Pre-set Fileinfo ######
total_file_size = 0 # in Byte


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

def getAllFilesInDirectory(directoryPath: str):
    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]
"""
author: zhenyu wu
time: 2019/12/04 16:03
function: 均值哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 均值哈希指纹计算结果
"""
def HashValue(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j] = floor(img[i,j]/4)
    avg = np.sum(img)/64*np.ones((8, 8))
    temp = img-avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0
    temp = temp.reshape((1,64))
    return temp


"""
author: zhenyu wu
time: 2019/12/04 16:04
function: 根据均值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def Hash(img1, img2):
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    #if result<=5:
    #    print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/04 16:06
function: 感知哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 感知哈希指纹计算结果
"""
def pHashValue(img):
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img = cv2.dct(img)
    img = img[:8, :8]
    avg = np.sum(img)/64*np.ones((8, 8))
    temp = img-avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0
    temp = temp.reshape((1,64))
    return temp


"""
author: zhenyu wu
time: 2019/12/04 16:06
function: 根据感知哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def pHash(img1, img2):
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    #if result<=5:
    #    print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/09 09:14
function: 差值哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 差值哈希指纹计算结果
"""
def DHashValue(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img2 = []
    for i in range(8):
        img2.append(np.array(img[:,i])-np.array(img[:,i+1]))
    img2 = np.mat(img2).T
    img2[img2 >= 0] = 1
    img2[img2 < 0] = 0
    img2 = img2.reshape((1,64))
    return img2


"""
author: zhenyu wu
time: 2019/12/09 09:13
function: 根据差值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def DHash(img1, img2):
    result = np.nonzero(img1-img2)
    result = np.shape(result[0])[0]
    #if result<=5:
    #    print('Same Picture')
    return result


"""
author: 
time: 
function: 根据包中的差值哈希算法计算
params: 
    img: 输入的图片
return:
    temp: 差值哈希指纹计算结果
"""
def dHashValue_use_package(img_path):
    image = Image.open(img_path)
    row, col = dhash.dhash_row_col(image)
    temp = int(dhash.format_hex(row, col), 16)
    return temp
    
"""
author: zhenyu wu
time: 2019/12/09 09:37
function: 根据包中的差值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""
def dHash_use_package(img1, img2):
    result = dhash.get_num_bits_different(img1, img2)
    #if result<=5:
    #    print('Same Picture')
    return result

from multiprocessing import Manager
mgr_hash = Manager()
img_hash=mgr_hash.dict();

mgr_phash = Manager()
img_phash=mgr_phash.dict();

mgr_dhash = Manager()
img_dhash=mgr_dhash.dict();

#mgr_dhash_pkg = Manager()
#img_dhash_pkg=mgr_dhash_pkg.dict()

mgr_result = Manager()
result=[]=mgr_result.list()
def img_similarity_list(img_path):
    #print (img_path)
    i = image_dict.index(img_path) + 1
    
    for i in range(i, len(image_dict)):
        #print (i)
        img_similarity_compare(img_path, image_dict[i])
    
    
def img_similarity_compare(img_list1,img_list2):
    
    try:
        img1_path=img_list1;
        img2_path=img_list2;
        #print('{:s} & {:s}'.format(img1_path,img2_path), end="\r", flush=True)
        #print(img_hash.get(img_list[0]))
        hash_flag=False
        phash_flag=False
        dhash_flag=False
        #dhashp_flag=False

        hash_result = Hash(img_hash[img1_path], img_hash[img2_path])
        if hash_result<=0:
            hash_flag=True
        #print('Hash Hanming Distance: %d' % (hash_result))
        
        phash_result = pHash(img_phash[img1_path], img_phash[img2_path])
        if phash_result<=0:
            phash_flag=True
        #print('pHash Hanming Distance: %d' % (phash_result))
        
        dhash_result = DHash(img_dhash[img1_path], img_dhash[img2_path])
        if dhash_result<=0:
            dhash_flag=True
        #print('DHash Hanming Package Distance: %d' % (dhash_result_pkg))
        
        #dhash_result_pkg =dHash_use_package(img_dhash_pkg[img_list[0]], img_dhash_pkg[img_list[1]])
        #if dhash_result_pkg<=0:
        #    dhashp_flag=True
        #print('{:s} & {:s}\r'.format(img1_path,img2_path))
        
        #if hash_flag and phash_flag and dhash_flag and dhashp_flag:
        if hash_flag and phash_flag and dhash_flag:
            result.append(str(img1_path+" & "+img2_path)+
                          "\nhash_result:"+str(hash_result)+
                          "\nphash_result:"+str(phash_result)+
                          "\ndhash_result:"+str(dhash_result)+
        #                  "\ndhash_result_pkg:"+str(dhash_result_pkg)+
                          "\n"
                          )
    except:
        print('check the following images:{:s} & {:s}\r'.format(img1_path,img2_path))
    
    
        '''
    if dhash_result_pkg<=5:
    
        print('HH:',hash_flag)
        print('pHH:',phash_flag)
        print('dHH:',dhash_flag)
        print('dHHp:',dhashp_flag)
        #print('\n')
      
    if hash_flag or phash_flag or dhash_flag or dhashp_flag:
        print(img1_path+' & '+img2_path)
        print('HH:',hash_flag)
        print('pHH:',phash_flag)
        print('dHH:',dhash_flag)
        print('dHHp:',dhashp_flag)
        print('\n')
        '''    
    
    

def img_similarity_calculate(img_list):
    img = cv2.imread(img_list)
    if img is None:
        return
    img_path=img_list
    try:
        #print('{:s} & {:s}'.format(img1_path,img2_path))
            
        hash_value = HashValue(img)
        img_hash[img_path] = hash_value
        
        phash_value = pHashValue(img)
        img_phash[img_path] = phash_value
            
        dhash_value = DHashValue(img)
        img_dhash[img_path] = dhash_value
            
        #dhash_value_pkg =dHashValue_use_package(img_path)
        #img_dhash_pkg[img_path] = dhash_value_pkg
        
        #print('{:s}'.format(img_path), end="\r", flush=True)
    
    except:
        if img_hash[img_path] is not None:
            del img_hash[img_path]
        if img_phash[img_path] is not None:
            del img_phash[img_path]
        if img_dhash[img_path] is not None:
            del img_dhash[img_path]
        #if img_dhash_pkg[img_path] is not None:
        #    del img_dhash_pkg[img_path]
        print('check the following images:{:s}\r'.format(img_path))


def get_dict_memory_size():
    from sys import getsizeof
    hash_memory_size = 0
    for i in img_hash.items():
        if getsizeof(i) > 4096:
            x = getsizeof(i)
            hash_memory_size = hash_memory_size - x
        else:
            x = getsizeof(i)
            hash_memory_size = hash_memory_size - x
        for j in img_hash.items():
            if getsizeof(i) > 4096:
                x = x + getsizeof(i)
                hash_memory_size = hash_memory_size + x + getsizeof(i)
            else:
                x = x + getsizeof(i)
                hash_memory_size = hash_memory_size + x + getsizeof(i)
                
    phash_memory_size = 0
    for i in img_phash.items():
        if getsizeof(i) > 4096:
            x = getsizeof(i)
            phash_memory_size = phash_memory_size - x
        else:
            x = getsizeof(i)
            phash_memory_size = phash_memory_size - x
        for j in img_phash.items():
            if getsizeof(i) > 4096:
                x = x + getsizeof(i)
                phash_memory_size = phash_memory_size + x + getsizeof(i)
            else:
                x = x + getsizeof(i)
                phash_memory_size = phash_memory_size + x + getsizeof(i)
        
    dhash_memory_size = 0
    for i in img_dhash.items():
        if getsizeof(i) > 4096:
            x = getsizeof(i)
            dhash_memory_size = dhash_memory_size - x
        else:
            x = getsizeof(i)
            dhash_memory_size = dhash_memory_size - x
        for j in img_dhash.items():
            if getsizeof(i) > 4096:
                x = x + getsizeof(i)
                dhash_memory_size = dhash_memory_size + x + getsizeof(i)
            else:
                x = x + getsizeof(i)
                dhash_memory_size = dhash_memory_size + x + getsizeof(i)
        
    total_memory_size = hash_memory_size + phash_memory_size + dhash_memory_size    
    
    return (total_memory_size)    

########### Pre-set Memory ##############
machine_name = 'Cherudim GUNDAM'
memory_speed = 1866 #in MHz
memory_channel = 4 
memory_width = 64 #in Bits
bit_to_byte = 8
theoretic_memory_putthrough = memory_speed * memory_width * memory_channel / bit_to_byte
    
from multiprocessing import Pool
import time
image_dict =[]
if __name__ == '__main__':
    from multiprocessing import cpu_count
    print("Total CPU:{}".format(cpu_count()))
    for img_path in get_all_path("images"):
        if '.jpg' in img_path.lower():
            image_dict.append(img_path)
        elif '.jpeg' in img_path.lower():
            image_dict.append(img_path)
        elif '.png' in img_path.lower():
            image_dict.append(img_path)
        elif '.tiff' in img_path.lower():
            image_dict.append(img_path)
        elif '.tif' in img_path.lower():
            image_dict.append(img_path)
    print("Total images:{}".format(str(len(image_dict))))
    s_hash_time=time.time()
    hash_pool=Pool(cpu_count())
    hash_pool.map(img_similarity_calculate,image_dict)
    hash_pool.close()
    hash_pool.join()
    e_hash_time=time.time()
    total_iop = len(image_dict)*(1 + 8 * 8*(4*3+1+2+2+1+3))
    total_iop = total_iop + len(image_dict)*(32*32*(4*4*3*3+1+1+15+1+2+3)+1)
    total_iop = total_iop + len(image_dict)*(9*8*(4*4*3*3+1+1)+9*3)
    
    print('############## Hash Performance #################')
    print('Hash time:{:.2f}s\nCore efficiency:{:.1f}pics/s\nTotal_IOP:{:.1f}GOP\nCore_IOPS:{:.1f}KOPS'.format(e_hash_time-s_hash_time,
                                                                    len(image_dict)/(e_hash_time-s_hash_time)/cpu_count(),
                                                                    total_iop/1000/1000/1000,
                                                                    total_iop/1000/(e_hash_time-s_hash_time)/cpu_count()
                                                                    )
    )
    print('#################################################\n')
    
    
    possible_combinations=len(image_dict)*(len(image_dict)-1)/2
    print("Total combinations:{}".format(str(int(possible_combinations))))
    
    #time.sleep(2000)
    


    s_comp_time=time.time()
    pool=Pool(cpu_count())
    pool.map(img_similarity_list, image_dict)
    pool.close()
    pool.join()
    e_comp_time=time.time()
    dict_memory_size = get_dict_memory_size()
    #print(dict_memory_size)
    print('############# Compare Performance ################')
    print('Compare time:{:.2f}s\nCore efficiency:{:.1f}combs/s\nMemory_putthrough:{:.1f}MB/s\nBandwidth_Percentage:{:.1f}%'.format(e_comp_time-s_comp_time,
                                                                    possible_combinations/(e_comp_time-s_comp_time)/cpu_count(),
                                                                    dict_memory_size/1024/1024/(e_comp_time-s_comp_time),
                                                                    dict_memory_size/1024/1024/(e_comp_time-s_comp_time)/theoretic_memory_putthrough*100,
                                                                    )
    )
    print('#################################################\n')
    '''
    for k in range(len(possible_combinations)):
        print(len(possible_combinations))
        #print(possible_combinations[k])
        #print('\x1B[K{:s}\r'.format(str(possible_combinations[k])),end='', flush=True)
        img_similarity(possible_combinations[k][0],possible_combinations[k][1])
    '''
    file_result=open('result.txt',mode='w')
    print("result:")
    for i in result:
        file_result.write(i)
        #print("\r{:s}".format(i),end='\n')
    file_result.close()
    
    
