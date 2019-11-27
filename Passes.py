
import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm
from tvm.autotvm import feature
import math

NAIVE = 0
NONNAIVE = 1

PRINT = 0

class ConfigInsertionPass:

    def __init__(self,task):
        self.task = task
        self.space = self.task.config_space
        self.target = self.task.target

    def instantiate_task(self,index):
        config = self.space.get(index)
        with self.target:
            sch,args = self.task.instantiate(config)
            #print(tvm.lower(sch,args,simple_mode=True))
        return sch,args

    def print_info(self,index):
        sch,args = self.instantiate_task(index)
        print(self.space.get(index))
        print(feature.get_itervar_feature(sch,args))
        print(feature.get_itervar_feature_flatten(sch,args))


    def distance(self,indexOne,indexTwo):
        a = feature.get_itervar_feature_flatten(*self.instantiate_task(indexOne))
        b = feature.get_itervar_feature_flatten(*self.instantiate_task(indexTwo))
        return np.linalg.norm(np.array(a)-np.array(b))

def check_factors(size,extent,min_factor = 1,max_factor = 1024):
    # for some reason when tiling occurs all the possible configurations are reported to this function as having -1 
    def factor_inbounds(factor):
        return factor >= min_factor and factor <= max_factor

    dot = 1
    i = 0
    for element in size:
        if(element == -1):
            continue
        if(not factor_inbounds(element)):
            return False
        dot *= element
        i += 1
    actual_tile = []
    if(i != len(size)):
        last_element = extent // dot
        if(not factor_inbounds(last_element)):
            return False
        actual_tile = [last_element]+size[1:]
    return actual_tile == list(reversed(sorted(actual_tile)))

def splitting_outputs(size):
    #This should create enough options for a variety of caches
    if(size >= 16384):
        return 4
    elif(size >= 1024):
        return 3
    else:
        return 2

def tiling_pass(schedule,computes,config):
    #TODO: Use feature information.     
    tiled_axes = dict()
    for compute in computes:
        if PRINT:
            print("Found compute in tiling pass")
        stage = schedule[compute]
        axes = stage.op.axis
        axes = axes
        tiled_inner_axis = []
        tiled_outer_axis = []
        index = 0
        for axis in axes:
            if PRINT:
                print("Found axis in compute in tiling pass:",axis)
            extent = int(axis.dom.extent)
            split_factor = splitting_outputs(int(extent))            
            config.define_split("tile_" + str(index),axis,num_outputs = split_factor,policy='verbose')
            new_axes = config["tile_" + str(index)].apply(schedule,compute,axis)
            midpoint = math.ceil(len(new_axes)/2)
            tiled_inner_axis += new_axes[0:midpoint]
            tiled_outer_axis += new_axes[midpoint:len(new_axes)]
            index += 1
        stage.reorder(*tiled_outer_axis,*stage.op.reduce_axis,*tiled_inner_axis)
        tiled_axes[compute] = [] + tiled_outer_axis + list(stage.op.reduce_axis) + tiled_inner_axis
    return tiled_axes


def annotate_pass(schedule,computes,config,tiled_axes_dict=None):

    for compute in computes:
        if PRINT:
            print("Found compute in annotate pass")
        stage = schedule[compute]
        axes = stage.op.axis if tiled_axes_dict == None else tiled_axes_dict[compute]    
        axes = axes[math.floor((len(axes)-1)/2):len(axes)]
        config.define_annotate("annotate",axes,policy='try_unroll_vec')
        config['annotate'].apply(schedule,compute,axes)

def reordering_pass(schedule,computes,config,tiled_axes_dict = None):

    #TODO: Use other policies and arguments
    for compute in computes:
        if PRINT:   
            print("Found compute in reordering pass")
        stage = schedule[compute]
        axes = [*stage.op.axis, *stage.op.reduce_axis] if tiled_axes_dict == None else tiled_axes_dict[compute]
        if(len(axes) > 5):
            policy = 'interleave' if len(axes) > 5 else 'all'
        reordering = list(axes[math.floor((len(axes)-1)/2):len(axes)])
        config.define_reorder("reorder",reordering,policy = 'all')
        config["reorder"].apply(schedule,compute,reordering)

def enable_autotune(schedule,computes,config,mode=NAIVE):
    if PRINT:
        print("Begin creating space")
    #TODO: try tuning without tiles first then with tiles.
    if(mode == NONNAIVE):
        fused_passes(schedule,computes,config)
    elif(mode == NAIVE):
        tiled_axes_dict = tiling_pass(schedule,computes,config)
        reordering_pass(schedule,computes,config,tiled_axes_dict)
        annotate_pass(schedule,computes,config,tiled_axes_dict)
    if PRINT:
        print("Done creating space")

def fused_passes(schedule,computes,config):
    for compute in computes:
        if PRINT:
            print("Found compute in fused pass")
        stage = schedule[compute]
        axes = list(stage.op.axis)[max(0,len(stage.op.axis)-3):] #+ list(stage.op.reduce_axis[-1:]) #limit the 
        
        tiled_inner_axis = []
        tiled_outer_axis = []
        index = 0
        for axis in axes:
            if PRINT:
                print("Found axis in compute in tiling pass:",axis)
            extent = int(axis.dom.extent)
            split_factor = splitting_outputs(int(extent))
            config.define_split("tile_" + str(index),axis,num_outputs = split_factor,policy='verbose',filter=lambda x: check_factors(x.size,extent))
            new_axes = config["tile_" + str(index)].apply(schedule,compute,axis)
            midpoint = math.ceil(len(new_axes)/2)
            tiled_inner_axis += new_axes[0:midpoint]
            tiled_outer_axis += new_axes[midpoint:len(new_axes)]
            index += 1
        stage.reorder(*tiled_outer_axis,*stage.op.reduce_axis,*tiled_inner_axis)

        config.define_reorder("reorder",tiled_inner_axis,policy='all')
        config["reorder"].apply(schedule,compute,tiled_inner_axis)
        config.define_annotate("annotate",tiled_inner_axis,policy='try_unroll_vec')
        config['annotate'].apply(schedule,compute,tiled_inner_axis)

'''x`
 * ((
 *   ('_itervar_',  var),
 *   ('_attr_',     length, nest_level, topdown, bottomup, one_hot_annotation),
 *   ('_arith_',    add_ct, mul_ct, div_ct),
 *   ('data_vec_0', stride, mod, count, reuse, thread_count, thread_reuse),
 *   ('conv_0',     stride, mod, count, reuse, thread_count, thread_reuse),
 * ),
 * (
 *   ('_itervar_',    var2),
 *   ('_attr_',       length, nest_level, one_hot_annotation),
 *   ('_arith_',      add_ct, mul_ct, div_ct),
 *   ('kernel_vec_0', stride, mod, count, reuse, thread_count, thread_reuse),
 *   ('conv_1',       stride, mod, count, reuse, thread_count, thread_reuse),
 * ))
 *
 '''

if(__name__ == "__main__"):

    @autotvm.template
    def vectoradd_auto(K,dtype):
        A = tvm.placeholder((K,), name='A', dtype=dtype)
        B = tvm.placeholder((K,), name='B', dtype=dtype)
        C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
        s = tvm.create_schedule(C.op)

        #### ADDED AUTO AUTOTUNING ####
        enable_autotune(s,[C],autotvm.get_config(),mode=NAIVE)
        ###############################

        return s, [A, B, C]


    def test_similarity(kernel,args,print_similar=False):

        task = autotvm.task.create(kernel, args=args, target='llvm')
        c = ConfigInsertionPass(task)
        pruned = set()
        print(c.space)
        for i in range(len(task.config_space)):
            if(i in pruned):
                continue
            for j in range(i+1,len(task.config_space)):
                if(j in pruned):
                    continue
                dis = c.distance(i,j)
                if(dis <= 0.0000000001):
                    pruned.add(j)
                    if(print_similar):
                        print(i,j)
                        c.print_info(i)
                        c.print_info(j)
                        print() 
        return len(pruned) / len(c.space)

    task = autotvm.task.create(vectoradd_auto, args=[512, 'float32'], target='llvm')
    c = ConfigInsertionPass(task)
    print(c.space)

    #print(test_similarity(GEMM.matmul_auto,[512,512,512,'float32'],True))
    #print(test_similarity(vectoradd_auto,[512,'float32'],True)) #0.8333333333333334
    #print(test_similarity(vectoradd_auto,[4096,'float32'],False)) #0.9444444444444444

