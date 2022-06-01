# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-scheduling a Neural Network for NVIDIA GPU
===============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole neural
network for NVIDIA GPU with the auto-scheduler.

To auto-tune a neural network, we partition the network into small subgraphs and 
tune them independently. Each subgraph is treated as one search task.
A task scheduler slices the time and dynamically allocates time resources to
these tasks. The task scheduler predicts the impact of each task on the end-to-end
execution time and prioritizes the one that can reduce the execution time the most.

For each subgraph, we use the compute declaration in :code:`tvm/python/topi` to
get the computational DAG in the tensor expression form.
We then use the auto-scheduler to construct a search space of this DAG and search
for good schedules (low-level optimizations).

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any
schedule templates. In other words, the auto-scheduler only uses the compute declarations
in :code:`tvm/python/topi` and does not use existing schedule templates.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""


import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor

import time
import os
import gc

import pynvml
def get_gpu_name():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    gpu_name = gpu_name.replace("NVIDIA GeForce","").replace(" ","")
    return gpu_name


#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# For convolutional neural networks, although auto-scheduler can work correctly
# with any layout, we found the best performance is typically achieved with NHWC layout.
# We also implemented more optimizations for NHWC layout with the auto-scheduler.
# So it is recommended to convert your models to NHWC layout to use the auto-scheduler.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mobilenetv2_1.0', help='a chosen model, like resnet18_v2', required=False)
parser.add_argument("--tune", action='store_true', default=False)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument('--mode', type=int, default=0)
parser.add_argument('--ratio', type=float, default=1.00)

args = parser.parse_args()
print(args)

pre_define_conf=[
    ["mobilenetv2_1.0","cuda","cuda:0",640,3200,3200],
    ["mobilenetv2_1.0","llvm","cpu",320,3200,3200],
    ["resnet152_v2","cuda","cuda:0",320,3200,3200],
    ["resnet152_v2","llvm","cpu",320,3200,3200]
]
    
dm,tp1,tp2,bs,cbs,imgts = pre_define_conf[args.mode]
ratio = args.ratio
start_time = time.time()
####################################################################
# Read Image
if args.tune ==False:
    print("######################################")
    from os import listdir
    from os.path import isfile, join, isdir
    def get_all_path(open_file_path):
        global total_file_size 
        rootdir = open_file_path
        path_list = []
        list = listdir(rootdir)  # 列出文件夹下所有的目录与文件
        for i in range(0, len(list)):
            com_path = join(rootdir, list[i])
            #print(com_path)
            if isfile(com_path):
                path_list.append(com_path)
            if isdir(com_path):
                path_list.extend(get_all_path(com_path))
        #print(path_list)
        return path_list


    from PIL import Image
    import cv2
    image_dict_list =[]
    for img_path in get_all_path("images"):
        if '.jpg' in img_path.lower():
            image_dict_list.append(img_path)
            #file_size += os.path.getsize(img_path)
        elif '.jpeg' in img_path.lower():
            image_dict_list.append(img_path)
            #file_size += os.path.getsize(img_path)
        elif '.png' in img_path.lower():
            image_dict_list.append(img_path)
            #file_size += os.path.getsize(img_path)
        elif '.tiff' in img_path.lower():
            image_dict_list.append(img_path)
            #file_size += os.path.getsize(img_path)
        elif '.tif' in img_path.lower():
            image_dict_list.append(img_path)
            #file_size += os.path.getsize(img_path)
    print("Total images:{}".format(str(len(image_dict_list))))

    print("Read images time:{:.2f}s".format(time.time() - start_time))

####################################################################
# Compile Model

batch_size = bs
layout = "NCHW"
target = tvm.target.Target(tp1)
dtype = "float32"
log_file = "%s-%s-B%d-%s-%s.json" % (dm, layout, batch_size, target.kind.name, get_gpu_name())
lib_file = "./lib_model/%s-%s-B%d-%s-%s.tar" % (dm, layout, batch_size, target.kind.name, get_gpu_name())

print("######################################")
compile_start_time = time.time()
def get_network(name, batch_size, layout="NCHW", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    # an example for mxnet model
    from mxnet.gluon.model_zoo.vision import get_model

    assert layout == "NCHW"

    block = get_model(name, pretrained=True)
    mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
    net = mod["main"]
    net = relay.Function(
        net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
    )
    mod = tvm.IRModule.from_expr(net)

    desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
    # Convert the layout to NHWC
    # RemoveUnunsedFunctions is used to clean up the graph.
    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                    relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod) 

    return mod, params, input_shape, output_shape
if args.tune == True:
    mod, params, input_shape, output_shape = get_network(args.model, batch_size, layout, dtype=dtype)
else:
    mod, params, input_shape, output_shape = get_network(dm, batch_size, layout, dtype=dtype)


if args.tune ==True:

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    def run_tuning():
        print("Begin tuning...")
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

        tuner = auto_scheduler.TaskScheduler(tasks, task_weights,load_log_file=log_file)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=300 * len(tasks),  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)
    run_tuning()
    print("Finish Tuning! Batch = {}, Network = {}".format(batch_size,dm))
    exit()


if not os.path.exists(lib_file):
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, params=params)
            lib.export_library(lib_file)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))
else:
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = tvm.runtime.load_module(lib_file)

    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

print("Compile model time:{:.2f}s".format(time.time() - compile_start_time))


####################################################################
# Batch Image

result_output = None
flag = True
image_list = []
for step in range(0,len(image_dict_list),imgts):
    image_dict=image_dict_list[step:min(len(image_dict_list),step+imgts)]
    from multiprocessing import Pool, cpu_count, Manager
    print("######################################")
    batch_start_time = time.time()
    image_batch = None
    mgr_batch = Manager()
    image_batch_list=mgr_batch.dict()
    def multithread_batch_image(image):
        try:
            x=Image.open(image).resize((224, 224))
        except:
            print("cannot identify image file '{}'".format(image))
            return
        #x = cv2.imread(image).resize((224, 224))
        #print(x)
        x = x.convert("RGB")
        def transform_image(image):
            image = np.array(image) - np.array([123.0, 117.0, 104.0])
            image /= np.array([58.395, 57.12, 57.375])
            image = image.transpose((2, 0, 1))
            #image = image[np.newaxis, :]
            return image
        x = transform_image(x)
        image_batch_list[image] = x
        return
        
    hash_pool=Pool(cpu_count())
    hash_pool.map(multithread_batch_image,image_dict)
    hash_pool.close()
    hash_pool.join()


    image_batch = np.array(list(dict(image_batch_list).values()))
    count_batch = len(image_batch)
    print (image_batch.shape)

    print("Batch images time:{:.2f}s".format(time.time() - batch_start_time))
    image_list.extend(list(image_batch_list.keys()))


####################################################################
# Postprocess Data

    print("######################################")
    postprocess_start_time = time.time()
    from math import ceil
    fill_zero = ceil(len(image_batch)/batch_size)*batch_size -len(image_batch)
    fill_batch = np.zeros((fill_zero,3,224,224))
    image_batch = np.concatenate((image_batch,fill_batch), axis=0)
    print(image_batch.shape)

    print("Postprocess images time:{:.2f}s".format(time.time() - postprocess_start_time))


####################################################################
# Calculate Similarity
    result_list = []
    print("######################################")
    calculate_start_time = time.time()
    for till in range(0,len(image_batch),batch_size):
        module.set_input("data", tvm.nd.array(image_batch[till:till+batch_size].astype(dtype)))
        module.run()
        # get outputs
        tvm_output = module.get_output(0)
        result = tvm_output.numpy()
        result_list.append(result)

    flag_part = True
    result_output_part = None
    for result in result_list:
        if flag_part:
            result_output_part = result
            flag_part = False
        else:
            result_output_part = np.concatenate((result_output_part,result), axis=0)

    result_output_part = result_output_part[:count_batch]
    
    if flag:
        result_output = result_output_part
        flag = False
    else:
        result_output = np.concatenate((result_output,result_output_part), axis=0)
    print(result_output.shape)

    print("Calculate model time:{:.2f}s".format(time.time() - calculate_start_time))


####################################################################
# Compare Similarity

Cbatch_size = cbs
#fill_zero = ceil(len(result_output)/Cbatch_size)*Cbatch_size -len(result_output)
#fill_batch = np.zeros((fill_zero,1000))
#result_output = np.concatenate((result_output,fill_batch), axis=0)
import torch
print("######################################")
compare_start_time = time.time()

class CosineSimilarityTest(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityTest, self).__init__()

    def forward(self, x1, x2):
        x2 = x2.t()
        x = x1.mm(x2)
    
        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)
    
        final = x.mul(1/x_frobenins)
        return final

def cos(x1,x2):
    #  加载数据到设备中CP\GPU
    device = torch.device(tp2 if torch.cuda.is_available() else "cpu")

    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
        
    model = CosineSimilarityTest().to(device)

    # 同时需要多卡计算时需要
    # model = torch.nn.DataParallel(model)

    final_value = model(x1, x2)
    #print(final_value)
    return final_value

final_result = []


for i in range(0,len(result_output),Cbatch_size):
    for j in range(0, i+1, Cbatch_size):
        final_temp=cos(result_output[i:min(i+Cbatch_size,len(result_output))],result_output[j:min(j+Cbatch_size,len(result_output))])
        if i==j:
            final_temp = np.tril(final_temp,-1)
        else:
            final_temp=final_temp.numpy()
        #print(final_temp.shape)
        final_result.append([i,j,final_temp])

#final = np.triu(final,1)
#print(final)s
print("Compare model time:{:.2f}s".format(time.time() - compare_start_time))


####################################################################
# Final process

print("######################################")
fp_start_time = time.time()
mgr_final = Manager()
print_result = mgr_final.list()
def final_process_data(data):
    i, j, final_temp = data
    for index_i in range(0,final_temp.shape[0]):
        for index_j in range(0,final_temp.shape[1]):
            if final_temp[index_i][index_j] > ratio:
                print_result.append("{} and {}. Ratio is {}".format(image_list[i+index_i],image_list[j+index_j],final_temp[index_i][index_j]))

final_pool=Pool(cpu_count())
final_pool.map(final_process_data,final_result)
final_pool.close()
final_pool.join()
print("Final process time:{:.2f}s".format(time.time() - fp_start_time))


####################################################################
# Conclude

fp = open("CNN_hash.txt","w")
for line in print_result:
    fp.writelines(line+"\n")
fp.close()
print("######################################")
print("Used time:{:.2f}s".format(time.time() - start_time))
