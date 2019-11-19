
import logging
import sys

import numpy as np
import tvm

# the module is called `autotvm`
from tvm import autotvm


def reordering_pass(schedule,computes,config,tiled_axes_dict = None):
    #TODO: Use other policies and arguments
    for compute in computes:
        stage = schedule[compute]
        axes = [*stage.op.axis, *stage.op.reduce_axis,] if tiled_axes_dict == None else tiled_axes_dict[compute]
        config.define_reorder("reorder",axes,policy = 'all')

def tiling_pass(schedule,computes,config):
    #TODO: Try other policies and filter
    tiled_axes = dict()
    for compute in computes:
        stage = schedule[compute]
        axes = stage.op.axis
        tiled_inner_axis = []
        tiled_outer_axis = []
        index = 0
        for axis in axes:
            config.define_split("tile_" + str(index),axis,num_outputs = 2,policy='verbose', filter = lambda x:x.size[-1] >= 2)
            #TODO: try to figure out how to use these in reorder pass
            outer,inner = config["tile_" + str(index)].apply(schedule,compute,axis)
            tiled_inner_axis.append(inner)
            tiled_outer_axis.append(outer)
            index += 1
        stage.reorder(*tiled_outer_axis,*stage.op.reduce_axis,*tiled_inner_axis)
        tiled_axes[compute] = [] + tiled_outer_axis + list(stage.op.reduce_axis) + tiled_inner_axis
    return tiled_axes


def annotate_pass(schedule,computes,config):
    for compute in computes:
        stage = schedule[compute]
        axes = stage.op.axis
        config.define_annotate("annotate",axes,policy="try_unroll_vec")

def enable_autotune(schedule,computes,config):

    tiled_axes_dict = tiling_pass(schedule,computes,config)
    reordering_pass(schedule,computes,config,tiled_axes_dict)
    annotate_pass(schedule,computes,config)
