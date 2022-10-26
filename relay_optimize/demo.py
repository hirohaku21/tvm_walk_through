import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay

import sys
sys.path.append("..")
from visualize import RelayVisualizer

def auto_optimize(mod,target,params):
  mod,params=relay.optimize(mod, target=target, params=params)
  visualizer=RelayVisualizer()
  visualizer.visualize(mod,path="visualizes/AlexNet_Opt4.prototxt")
  return mod,params

def debug_optimize(mod,target,params):
  mod["main"]=relay.build_module.bind_params_by_name(mod["main"],params)
  #add transform passes
  seq = tvm.transform.Sequential(
    [
      relay.transform.SimplifyInference(),
      #relay.transform.BackwardFoldScaleAxis(),
      #relay.transform.ForwardFoldScaleAxis(),
      #relay.transform.FoldConstant(),
      #relay.transform.AlterOpLayout(),
      #relay.transform.FoldConstant(),
      #relay.transform.FuseOps(),
    ]
  )
  with target:
    mod=seq(mod)

  visualizer=RelayVisualizer()
  visualizer.visualize(mod,path="visualizes/simpleNN.prototxt")
  return mod,params

if __name__=='__main__':
  #prepare model and input
  #model = models.resnet18(pretrained=True)
  model_name = "alexnet"
  model = getattr(models, model_name)(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(model,fake_input)
  #main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  #visualizer=RelayVisualizer()
  #visualizer.visualize(mod,path="visualizes/simpleNN.prototxt")
  #optimize the mod
  #step 1 create target
  target = tvm.target.Target("llvm", host="llvm")
  #step 1 create PassContext
  with tvm.transform.PassContext(opt_level=4):
    #step 3 optimize
    mod,params=auto_optimize(mod,target,params)
    #mod,params=debug_optimize(mod,target,params)
  #print("optimize func "+str(mod["main"]))
