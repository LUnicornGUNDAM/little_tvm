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
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to official installation guide.
https://mxnet.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt

block = get_model("resnet18_v1", pretrained=True)
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
img_path = download_testdata(img_url, "cat.png", module="data")
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())
image = Image.open('./86913906_p1.jpg').resize((224, 224))
image = image.convert("RGB")
#plt.imshow(image)
#plt.show()


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


x = transform_image(image)
print("x", x.shape)

image = Image.open('./86913906_p2.jpg').resize((224, 224))
image = image.convert("RGB")

x1 = transform_image(image)
y = np.concatenate((x,x1),axis=0)
print("y", y.shape)
######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
shape_dict = {"data": y.shape}
mod, params = relay.frontend.from_mxnet(block, shape_dict)
## we want a probability so add a softmax operator
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

######################################################################
# now compile the graph
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_executor

dev = tvm.cuda(0)
dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# set inputs



m.set_input("data", tvm.nd.array(y.astype(dtype)))
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)
#print(tvm_output.numpy())
top1 = np.argmax(tvm_output.numpy()[0])
print("TVM prediction top-1:", top1, synset[top1])
top1 = np.argmax(tvm_output.numpy()[1])
print("TVM prediction top-1:", top1, synset[top1])

test_output = tvm_output.numpy()

import torch
import time

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    #for i in range(1):

        #x1 = torch.randn(10000, 1000).to(device)
        #x2 = torch.randn(10000, 1000).to(device)
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
        
    model = CosineSimilarityTest().to(device)

        # 同时需要多卡计算时需要
        # model = torch.nn.DataParallel(model)

    final_value = model(x1, x2)
    print(final_value)

	# 输出排序并输出topk的输出
    #value, indec = torch.topk(final_value, 3, dim=0, largest=True, sorted=True)
    #print(value)

    end_time = time.time()
    print(device)
    print("消耗时间为:{}".format(end_time - start_time))

cos(test_output,test_output)
