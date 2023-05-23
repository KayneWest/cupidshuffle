

import re
import subprocess
import sys
import os
import argparse
import tempfile
import logging
import json
from time import time
from typing import Dict, Tuple, Any, Optional, Union

import torch
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm import autotvm
from tvm.contrib import graph_executor
from tvm.contrib import utils
from tvm.relay.backend.executor_factory import GraphExecutorFactoryModule
from tvm.ir.module import IRModule
from tvm.relay.function import Function
from tvm.autotvm.task.task import Task
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

from model import CupidShuffle

logging.getLogger('autotvm').setLevel(logging.DEBUG)

class TVMCompiler:
    def __init__(
        self,
        batch_size: int = 1,
        channels: int = 3,
        height: int = 224, 
        width: int = 224, 
        input_name: str = "input0",
        dtype: str = "float32",
        target: str = 'llvm', # llvm -device=arm_cpu -mtriple=aarch64-linux-gnu
        save_path: str = 'cupidshufflenet_tvm',
        log_filename: str = 'cupidshufflenet_tvm.log',
        graph_opt_sch_file : str = 'cupidshufflenet_tvm_graph_opt.log',
        tuner: str =  'xgb',
        n_trial: int =  2000,
        early_stopping: int =  600,
        use_transfer_learning: bool =  True, # this failed?
        thresh: float = 0.5,
        gpu: bool = False,
        measure_option: Any = autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(
              number=20, repeat=3, timeout=4, 
              min_repeat_ms=150, enable_cpu_cache_flush=True 
              )
            ),
    ) -> None:
        """
        TODO: write what all these do
        height: int
          height of the image to be baked
        width: int
          width of the image to be baked
        """
        self.height = height
        self.width = width
        self.input_name = input_name
        self.dtype = dtype
        self.target = tvm.target.Target(target, host=target)      
        self.log_filename = log_filename
        self.graph_opt_sch_file = graph_opt_sch_file
        self.tuner = tuner
        self.n_trial = n_trial
        self.early_stopping = early_stopping
        self.use_transfer_learning = use_transfer_learning
        self.thresh = thresh
        self.measure_option = measure_option
        self.save_path = save_path
        self.thresh = thresh
        self.gpu = gpu
        self.batch_size = batch_size
        self.channels = channels
        self.input_shape = [self.batch_size, self.channels, self.height, self.width]
      
        # https://docs.tvm.ai/tutorials/autotvm/tune_nnvm_cuda.html#scale-up-measurement-by-using-multiple-devices
        #self.measure_option = autotvm.measure_option(
        #    builder=autotvm.LocalBuilder(timeout=10),
        #    runner=autotvm.RPCRunner(
        #        args.board,  # change the device key to your key
        #        '0.0.0.0', 9190,
        #        number=20, repeat=3, timeout=4, min_repeat_ms=150)
        # 
        # dev = tvm.cpu(0) 

    def relay(
        self, 
        model: Any, 
        save_mods: bool = True
    ) -> Tuple[IRModule, dict]:
        '''
        function builds IRModule and param dict for tvm
        Args:
        ----
        model: pytorch model
          the CupidShuffle model
        save_mods: bool
          store the mod and params
        Returns:
        --------
        mod: tvm.ir.module.IRModule
          tvm IRModule
        params: dict
          the model's parameters
        '''
        input_data = torch.randn(self.input_shape)
        # trace model   
        scripted_model = torch.jit.trace(model, input_data).eval()
        # create a relay
        shape_list = [(self.input_name, input_data.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        if save_mods:
          self.mod = mod
          self.params = params
        return mod, params

    def export(
        self,
        model: Any,
        save_mods: bool = True
    ) -> None:
        '''
        function to compile a model down to tvm
        Args:
        ----
        model: pytorch model
        save_mods: bool
          store the mod and params
        '''
        mod, params = self.relay(model)
        # compile the model 
        lib = self.tvm_compile(mod, params)
        # export model
        lib.export_library(f"{self.save_path}.so")
        with open(f"{self.save_path}.json", "w") as fo:
            fo.write(lib.get_graph_json())
        with open(f"{self.save_path}.params", "wb") as fo:
            fo.write(relay.save_param_dict(lib.get_params()))

        # add whatever else you want here
        cpp_params = {
          "deploy_lib_path": f"{self.save_path}.so",
          "deploy_graph_path": f"{self.save_path}.json",
          "deploy_param_path": f"{self.save_path}.params",
          "device_id": 0,
          "width": self.width,
          "height": self.height,
          "gpu": self.gpu,
          "tresh": self.thresh
        }
        with open(f"cpp.json", "w") as fo:
          json.dump(cpp_params, fo)

    def tvm_compile(
        self, 
        mod: IRModule, 
        params: dict,
        opt_level: int = 3
    ) -> GraphExecutorFactoryModule:
        '''
        function builds IRModule and param dict for tvm
        Args:
        ----
        mod: tvm.ir.module.IRModule
          tvm IRModule
        params: dict
          the model's parameters
        opt_level: int
          The optimization level of this pass.
        Returns:
        --------
        lib: tvm.relay.backend.executor_factory.GraphExecutorFactoryModule
          the compiled tvm lib
        '''
        # compile the model 
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(
                mod, 
                target=self.target, 
                params=params
            )
        return lib

    #TODO fix this func
    def tune_graph(
        self,
        graph: Function, 
        use_DP=True
    ) -> None:
        """
        Function to tune the graph of the model. 
        Use graph tuner to achieve graph level optimal schedules

        Args:
        -----
        graph: Function
          the mod['main'] object, the function of the model in tvm
        use_DP: bool
          whether to use the DPTuner or not.
        """
        target_op = [relay.op.get("nn.conv2d"),]
        Tuner = DPTuner if use_DP else PBQPTuner
        executor = Tuner(graph, {'data': self.input_shape}, self.log_filename, target_op, self.target)
        executor.benchmark_layout_transform(min_exec_num=self.n_trial)
        executor.run()
        executor.write_opt_sch2record_file(self.graph_opt_sch_file)

    # You can skip the implementation of this function for this tutorial. 
    def tune_tasks(
        self, tasks: Task
    ) -> None:
        """
        this function tunes the tasks set by autotune

        Args:
        ----
        tasks: tvm.autotvm.task.task.Task
          the set of optimization task for the tvm compiler

        """
        tmp_log_file = self.log_filename  + ".tmp"
        tmp_task_log_file = self.log_filename + '.task.tmp'
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        for i, tsk in enumerate(reversed(tasks)):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            # create tuner
            if self.tuner == "xgb":
                tuner_obj = XGBTuner(tsk, loss_type="reg")
            elif self.tuner == "xgb_knob":
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
            elif self.tuner == "xgb_itervar":
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
            elif self.tuner == "xgb_curve":
                tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
            elif self.tuner == "xgb_rank":
                tuner_obj = XGBTuner(tsk, loss_type="rank")
            elif self.tuner == "xgb_rank_knob":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
            elif self.tuner == "xgb_rank_itervar":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
            elif self.tuner == "xgb_rank_curve":
                tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
            elif self.tuner == "xgb_rank_binary":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
            elif self.tuner == "xgb_rank_binary_knob":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
            elif self.tuner == "xgb_rank_binary_itervar":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
            elif self.tuner == "xgb_rank_binary_curve":
                tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
            elif self.tuner == "ga":
                tuner_obj = GATuner(tsk, pop_size=50)
            elif self.tuner == "random":
                tuner_obj = RandomTuner(tsk)
            elif self.tuner == "gridsearch":
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)
            if self.use_transfer_learning and os.path.isfile(tmp_log_file):
                try:
                    # https://github.com/apache/incubator-tvm/blob/master/python/tvm/autotvm/tuner/xgboost_cost_model.py
                    # https://github.com/apache/incubator-tvm/blob/master/python/tvm/autotvm/tuner/xgboost_cost_model.py#L222
                    # when inp.task.name != self.task.name
                    # nothing is appended to 'data' var
                    # this errors out, so we'll just have to avoid it here
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
                except:
                    pass
            

            tsk_trial = min(self.n_trial, len(tsk.config_space))
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=self.early_stopping,
                measure_option=self.measure_option,
                callbacks=[
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(tmp_log_file),
                ],
            )
                
        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, self.log_filename)
        os.remove(tmp_log_file)

    # https://discuss.tvm.ai/t/transfer-learning-doesnt-work-tuner-obj-load-history/5328/3
    # https://discuss.tvm.ai/t/solved-can-we-resume-an-autotuning-session/3329/6
    def tune_and_evaluate(self, model: Any) -> None:
        """
        this function tunes and compiles a model for deployment to a given target
        """
        # extract workloads from relay program
        print("Extracting tasks...")
        mod, params = self.relay(model, save_mods=True)
        tasks = autotvm.task.extract_from_program(
            self.mod["main"], 
            target=self.target,
            params=self.params,
            ops=(relay.op.get("nn.conv2d"),)
        )

        # run tuning tasks
        print("Tuning...")
        self.tune_tasks(tasks)
        # errors now, ridding until i can get it 
        # to work
        # self.tune_graph(self.mod["main"])

        # compile kernels with history best records
        with autotvm.apply_history_best(self.log_filename):
            print("Compiling...")
            self.tvm_compile(mod, params)
            print("Exported!")

if __name__ == '__main__':

    net = CupidShuffle(start_channels=28, token_dim=28, repeats=[1,4,1])
    # load our weights
    net.load_state_dict(torch.load("weights/cupidshuffle.pth"))
    # load our compiler
    compiler = TVMCompiler(
        height = 224, 
        width = 224, 
        input_name = "input0",
        dtype = "float32",
        target = 'llvm', # for arm llvm -device=arm_cpu -mtriple=aarch64-linux-gnu
        save_path = 'cupidshufflenet_tvm',
        log_filename = 'cupidshufflenet_tvm.log',
        graph_opt_sch_file  = 'cupidshufflenet_tvm_graph_opt.log',
        tuner =  'xgb',
        n_trial =  10,
        early_stopping =  None,
        use_transfer_learning =  True,
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(
              number=20, repeat=3, timeout=4, 
              min_repeat_ms=150, enable_cpu_cache_flush=True
              )
            ),
        )
    # export our model
    compiler.export(net, True)
    # compiler.tune_and_evaluate(net)






