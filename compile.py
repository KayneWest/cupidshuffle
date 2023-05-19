

from typing import Dict, Tuple, Any
import torch
import tvm
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib import utils

from models import CupidShuffleNet


class TVMCompiler:
    def __init__(self) -> None
    # define our model

    def set_model_params(self, 
        height: int, 
        width: int, 
        input_name: str = "input0"
        dtype: str = "float32"
        target: str = 'llvm'
        save_path: str = 'cupidshufflenet_tvm'
      ) -> None:
        self.height = height
        self.width = width
        self.input_name = input_name
        self.dtype = dtype
        if self.target == 'llvm':
            dev = tvm.cpu(0)
        else:
            dev = tvm.gpu(0)
        self.target = tvm.target.Target(target, host=target)
        self.dev = dev
      
    def relay(
          self, model
    ) -> Tuple[tvm.ir.module.IRModule, dict]:
        '''
        function builds IRModule and param dict for tvm
        Args:
        ----
        height: int
          height of the image to be baked
        width: int
          width of the image to be baked
        Returns:
        --------
        mod: tvm.ir.module.IRModule
          tvm IRModule
        params: dict
          the model's parameters
        '''
        input_data = torch.randn([1, 3, self.height, se;f/width])
        # trace model   
        scripted_model = torch.jit.trace(model, input_data).eval()
        # create a relay
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        return mod, params

    def tvm_compile(
        self,
        model: Any
    ) -> None
        print('[*] Compile To Target {}'.format(target))
        
        mod, params = self.relay(model)

        # compile the model 
              
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(
                mod, 
                target=self.target, 
                params=params
            )
            save_graph = lib.get_graph_json()
            save_params = lib.get_params()

        lib.export_library(f"{self.save_path}.so")
        print('lib export success')
        with open(f"{self.save_path}.json") as fo:
            fo.write(graph)
        print("graph export success")
        with open(f"{self.save_path}.params") as fo:
            fo.write(relay.save_param_dict(save_params))
        print("params export success")

    def prune_old_tasks(tasks, log_file):
        if os.path.isfile(log_file):
            new_tasks = []
            history = autotvm.record.ApplyHistoryBest(log_file)
            for task in tasks:
                if history._query_inside(task.target, task.workload) is None:
                    new_tasks.append(task)
            return new_tasks
        else:
            return tasks

    # Use graph tuner to achieve graph level optimal schedules
    # Set use_DP=False if it takes too long to finish.
    def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
        target_op = [relay.op.get("nn.conv2d"),]
        Tuner = DPTuner if use_DP else PBQPTuner
        executor = Tuner(graph, {'data': dshape}, records, target_op, target)
        executor.benchmark_layout_transform(min_exec_num=2000)
        executor.run()
        executor.write_opt_sch2record_file(opt_sch_file)

    # You can skip the implementation of this function for this tutorial. 
    def tune_tasks(tasks,
                  measure_option,
                  tuner='xgb',
                  n_trial=200,
                  early_stopping=None,
                  log_filename='tuning.log',
                  use_transfer_learning=True,
                  try_winograd=True,
                  quantization=False):

        if quantization:
            for i in range(len(tasks)):
                output_channel = tsk.workload[1][0]
                input_channel = tsk.workload[1][1]
                if output_channel % 4 == 0 and input_channel % 4 == 0:
                    tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                              tasks[i].target, tasks[i].target_host, 'int8')
                    tasks[i] = tsk

        tmp_log_file = log_filename  + ".tmp"
        tmp_task_log_file = log_filename + '.task.tmp'
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)

        for i, tsk in enumerate(reversed(tasks)):
            #if i == 0:
            #    continue
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            # create tuner
            if tuner == 'xgb' or tuner == 'xgb-rank':
                tuner_obj = XGBTuner(tsk, loss_type='rank')
            elif tuner == 'ga':
                tuner_obj = GATuner(tsk, pop_size=100)
            elif tuner == 'random':
                tuner_obj = RandomTuner(tsk)
            elif tuner == 'gridsearch':
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)
            if use_transfer_learning and os.path.isfile(tmp_log_file):
                try:
                    # https://github.com/apache/incubator-tvm/blob/master/python/tvm/autotvm/tuner/xgboost_cost_model.py
                    # https://github.com/apache/incubator-tvm/blob/master/python/tvm/autotvm/tuner/xgboost_cost_model.py#L222
                    # when inp.task.name != self.task.name
                    # nothing is appended to 'data' var
                    # this errors out, so we'll just have to avoid it here
                    tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
                except:
                    pass
            

            with tempfile.NamedTemporaryFile() as tmp_task_log_file:
                # do tuning
                tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                            early_stopping=early_stopping,
                            measure_option=measure_option,
                            callbacks=[
                                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                                autotvm.callback.log_to_file(tmp_task_log_file.name)])

                with open(tmp_log_file, 'a') as tmp_log_f:
                    tmp_log_f.write(tmp_task_log_file.read().decode('utf8'))
                                
        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)


    # https://discuss.tvm.ai/t/transfer-learning-doesnt-work-tuner-obj-load-history/5328/3
    # https://discuss.tvm.ai/t/solved-can-we-resume-an-autotuning-session/3329/6
    def tune_and_evaluate(tuning_option, target_host, cc=None):
        # extract workloads from relay program
        print("Extract tasks...")

        # extract the model
        mod, params, target = tuning_option['model']['compile'](use_compiler=False)

        # place holder
        target_host = False

        if tuning_option['quantization']:
            with relay.quantize.qconfig(store_lowbit_output=False):
                mod['main'] = relay.quantize.quantize(mod['main'], params=params)

        if target_host:
            tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host,
                                                      params=params,  ops=(relay.op.get("nn.conv2d"),))#ops=(relay.op.nn.conv2d,))
        else:
            tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                                      params=params, ops=(relay.op.get("nn.conv2d"),))#ops=(relay.op.nn.conv2d,))
        if tuning_option['quantization']:
            for i in range(len(tasks)):
                tsk = tasks[i]
                input_channel = tsk.workload[1][1]
                output_channel = tsk.workload[1][0]
                if output_channel % 4 == 0 and input_channel % 4 == 0:
                    tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                              tasks[i].target, tasks[i].target_host, 'int8')
                    tasks[i] = tsk

        # run tuning tasks
        print("Tuning...")
        tune_tasks(tasks,
                  tuning_option['measure_option'],
                  tuner=tuning_option['tuner'],
                  n_trial=tuning_option['n_trial'],
                  early_stopping=tuning_option['early_stopping'],
                  log_filename=tuning_option['log_filename'],
                  use_transfer_learning=tuning_option['use_transfer_learning'],
                  try_winograd=tuning_option['try_winograd'],
                  quantization=tuning_option['quantization'])
        # compile kernels with history best records
        tune_graph(mod["main"], tuning_option['model']['shape'], tuning_option['log_filename'], tuning_option['graph_opt_sch_file'])

        #with autotvm.apply_history_best(tuning_option['log_filename']):
        with autotvm.apply_graph_best(tuning_option['graph_opt_sch_file']):
            print("Compile...")
            # level 3 optimization gave TVMError: Check failed: ir: :
            # VerifyMemory(x, target->device_type): Direct host side access to
            # device memory is detected in fused_nn_contrib_conv2d_winograd_weight_transform_3. Did you forget to bind?
            # So I set opt_level=2 and that worked for compiling
            tvm_compiler(tuning_option['model']['name'], mod, params, target)
            print("exported")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-height', default=224, required=False, help='image height')
    parser.add_argument('-width', default=224, required=False, help='image width')
    parser.add_argument('-weights', default='cupidshufflenet/best.pth', help='the weight path of cupid')
    parser.add_argument('-savepath', default='cupid.tar', help='the save name of the .tar model')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-blocks', action='store_true', default=False, help='block training')
    parser.add_argument('-device', default='cuda:0')
    parser.add_argument('-tensorboard', default=False)
    args = parser.parse_args()

    # load our weights
    net.load_state_dict(torch.load(best_weights_path))

    # declare tvm specific data
    input_data = torch.randn([1, 3, args.height, args.width])
    input_name = "input0"
    dtype = "float32"
    shape_list = [(input_name, input_data.shape)]

    # trace model   
    scripted_model = torch.jit.trace(model, input_data).eval()
    # create a relay
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)



    temp = utils.tempdir()
    path_lib = temp.relpath("deploy_lib.tar")
    lib.export_library(path_lib)
    print(temp.listdir())

# load the module back.
loaded_lib = tvm.runtime.load_module(path_lib)
input_data = tvm.nd.array(data)

module = graph_executor.GraphModule(loaded_lib["default"](dev))
module.run(data=input_data)
out_deploy = module.get_output(0).numpy()

# Print first 10 elements of output
print(out_deploy.flatten()[0:10])



    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
    m.run()
    tvm_output = m.get_output(0)












import re
import subprocess
import sys
import os
import argparse
import tempfile
from time import time

import numpy as np
from tvm import relay
from gluoncv import model_zoo, data, utils
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

import tvm
from tvm.contrib import graph_runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from tvm import autotvm

import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)


def build_save_dict():
    pass

def export(save_dict):
    confg = save_dict['config']
    graph = save_dict['graph']
    lib = save_dict['lib']
    params = save_dict['params']
    opt_path = save_dict['opt_path']
    pose_pipe = save_dict['pose_pipe']
    config_path = os.path.join(opt_path)


def tvm_compiler(name, mod, params, target='llvm', n_dets=None):
    print('[*] Compile To Target {}'.format(target))
    
    # declare tvm specific data
    input_data = torch.randn([1, 3, args.height, args.width])
    input_name = "input0"
    dtype = "float32"
    shape_list = [(input_name, input_data.shape)]

    # trace model   
    scripted_model = torch.jit.trace(model, input_data).eval()
    # create a relay
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # compile the model 
    target = tvm.target.Target(target, host=target)
    if target == 'llvm':
        dev = tvm.cpu(0)
    else:
        dev = tvm.gpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
        graph = lib.get_graph_json()
        params = lib.get_params()

    print(type(graph), type(lib), type(params))
    lib.export_library(
        "{}.so".format(MODEL_CONFIG[name]['output_name']))
    print('lib export success')
    with open("{}.json".format(MODEL_CONFIG[name]['output_name']), "w") as fo:
        fo.write(graph)
    print("graph export success")
    with open("{}.params".format(MODEL_CONFIG[name]['output_name']), "wb") as fo:
        fo.write(relay.save_param_dict(params))
    print("params export success")
    print(MODEL_CONFIG[name]['output_name'])


def evaluate_model(config, loaded_json, loaded_lib, loaded_params, ctx):
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    print('[*] Graph RunTime is Created')
    tvm_input = np.random.randn(*config['shape']).astype(config['dtype'])
    tvm_input = tvm.nd.array(tvm_input, ctx=ctx)
    module.set_input('data', tvm_input)
    print('[*] Run Test')
    avg = []
    for i in range(100):
        tvm_input = np.random.randn(*config['shape']).astype(config['dtype'])
        tvm_input = tvm.nd.array(tvm_input, ctx=ctx)
        module.set_input('data', tvm_input)
        start = time()
        module.run()
        ctx.sync()
        e = time() - start
        print('Time Cost : ', e)
        # print('anchor sum : ', anchor_boxes.sum())
        print('=========================')
        avg.append(e)

    print('[!] Evaluation Done')
    print('[!] First pass time: {}'.format(avg[0]))
    print('[!] Average time: {}'.format(np.mean(avg[1:])))

def load_raw_model(path_base):
    loaded_json = open("{}.json".format(path_base)).read()
    loaded_lib = tvm.runtime.load_module("{}.so".format(path_base))
    loaded_params = bytearray(open("{}.params".format(path_base), "rb").read())
    return loaded_json, loaded_lib, loaded_params

if __name__ == '__main__':
    available_models = {'object_detector', 'simple_pose','mobile_pose', 'face_detector', 'face_embedder', 'hand_detector', 'nonms_hand_detector','nonms_object_detector', 'pose_pipeline_nonms', 'pose_pipeline'}
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--network', type=str, default='nonms_object_detector', help='Network Architecture')
    parser.add_argument('--target', type=str, default='cuda', help='Deploy Target')
    parser.add_argument('--board', type=str, default='titanx', help='board')
    parser.add_argument('--dtype', type=str, default='float32', help='Data Type')
    parser.add_argument('--tune', type=int, default=0, help='whether to tune the models for the current arch')
    parser.add_argument('--ctx', type=int, default=0, help='TVM')
    parser.add_argument('--cc', type=str, default=None, help='if on x86, use "aarch64-linux-gnu-g++" to compile for aarch64 - might not work')
    parser.add_argument('--n_trial', type=int, default=2000, help='TVM')
    parser.add_argument('--quantization', type=bool, default=False, help='TVM')
    parser.add_argument('--custom_savename', type=str, default=None, help='TVM')
    parser.add_argument('--profile_speed', type=int, default=0, help='TVM')
    parser.add_argument('--profile_speed_name', type=str, default=None, help='TVM')
    parser.add_argument('--opt_level', type=int, default=2, help='TVM')
    parser.add_argument('--early_stopping', type=int, default=600, help='TVM') 
    parser.add_argument('--rpc', type=int, default=2, help='TVM')
    parser.add_argument('--override-shape', type=int, default=0, help='TVM')
    parser.add_argument('--override-width', type=int, default=0, help='TVM')
    parser.add_argument('--override-height', type=int, default=0, help='TVM')
    args = parser.parse_args()

    # get available n threads
    num_threads = available_cpu_count()

    if args.network not in available_models:
        raise Exception("{0} not in list of acceptable models: {1}".format(args.network, list(available_models)))
    
    if args.ctx!=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.ctx)

    print(args)
    # maybe input your own board here???
    DEVICE_CUDA_ARCH = {
        "tx2": {
            "arch": "sm_62"
            },
        "xavier": {
            "arch": "sm_72"
            },
        "nano": {
            "arch": "sm_53"
        },
        "titanx": { # maxwell
            "arch": "sm_52"
        },
        "1080ti": { # maxwell
            "arch": "sm_61"
        },
        "turing": {
          "arch" : "sm_75"
        }
    }

    if args.board not in {'tx2', 'xavier', 'nano'} :
        ARCH = 'x86'
    else:
        ARCH = 'aarch64'

    if 'cuda' in args.target:
        TARGET_ARCH = DEVICE_CUDA_ARCH[args.board]['arch']
        set_cuda_target_arch(TARGET_ARCH)
        CUDA = True
        suffix = 'gpu'
    else:
        CUDA = False
        suffix = 'cpu'

    MODEL_CONFIG = {
        'nonms_object_detector':
            {
                'shape': (1, 3, 512, 512),
                'output_name': 'mnet1.0.yolo.nonms.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_nonms_object_detector,
                'name': 'nonms_object_detector',
            },
        'object_detector':
            {
                'shape': (1, 3, 512, 512),
                'output_name': 'mnet1.0.yolo.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_object_detector,
                'name': 'object_detector',
            },
        'simple_pose':
            {
                'shape': (1, 3, 256, 192),
                'output_name': 'pose.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_simple_pose,
                'name': 'simple_pose',
            },
        'mobile_pose':
            {
                'shape': (1, 3, 256, 192),
                'output_name': 'pose.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_mobile_pose,
                'name': 'mobile_pose',
            },
        'face_detector':
            {
                'shape': (1, 3, 480, 640), # yes, this is correct, the height is 640 and width is 480
                'output_name': 'mnet.25.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_face_detector,
                'name': 'face_detector',
            },
        'face_embedder':
            {
                'shape': (1, 3, 112, 112),
                'output_name': 'mnet.facerec.{}.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_face_embedder,
                'name': 'face_embedder',
            },
        'hand_detector' :
            {
                'shape': (1, 3, 320, 320),
                'output_name': 'mnet.1.{}.hands.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_hand_detector,
                'name': 'hand_detector',
            },
        'nonms_hand_detector' :
            {
                'shape': (1, 3, 320, 320),
                'output_name': 'mnet.1.nonms.{}.hands.{}'.format(ARCH, suffix),
                'dtype': 'float32',
                'cuda': CUDA,
                'compile': compile_nonms_hand_detector,
                'name': 'nonms_hand_detector',
            }
    }

    if args.override_shape:
        height = args.override_height
        width = args.override_width
        MODEL_CONFIG[args.network]['shape'] = (1, 3, height, width)
        MODEL_CONFIG[args.network]['output_name'] = "{}.{}.{}".format(height, width, MODEL_CONFIG[args.network]['output_name'])

    MODEL_CONFIG[args.network]['output_name'] = tvm.__version__ + "." + MODEL_CONFIG[args.network]['output_name']


    if args.profile_speed and args.profile_speed_name:
        config = MODEL_CONFIG[args.network]
        loaded_json, loaded_lib, loaded_params = load_raw_model(args.profile_speed_name)

        if CUDA:
            ctx = tvm.gpu(0)
        else:
            ctx = tvm.cpu()
        evaluate_model(config, loaded_json, loaded_lib, loaded_params, ctx)

    # compile model
    #print(args.tune)
    elif not args.tune:
        MODEL_CONFIG[args.network]['compile'](True)
        if CUDA:
            ctx = tvm.gpu(0)
        else:
            ctx = tvm.cpu()
        loaded_json, loaded_lib, loaded_params = load_raw_model(MODEL_CONFIG[args.network]['output_name'])
        evaluate_model(MODEL_CONFIG[args.network], loaded_json, loaded_lib, loaded_params, ctx)

    else:
        TUNING_OPTION = {
            'log_filename': args.network + '.log',
            'graph_opt_sch_file' : args.network + "_graph_opt.log",
            'tuner': 'xgb',
            'n_trial': int(args.n_trial),
            'early_stopping': args.early_stopping,
            'use_transfer_learning': True, # this failed?
            'try_winograd': True,
            'measure_option': autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)
                ),
            'model': MODEL_CONFIG[args.network],
            'quantization': args.quantization
            }
        if args.rpc:
            # https://docs.tvm.ai/tutorials/autotvm/tune_nnvm_cuda.html#scale-up-measurement-by-using-multiple-devices
            TUNING_OPTION['measure_option'] = autotvm.measure_option(
                builder=autotvm.LocalBuilder(timeout=10),
                runner=autotvm.RPCRunner(
                    args.board,  # change the device key to your key
                    '0.0.0.0', 9190,
                    number=20, repeat=3, timeout=4, min_repeat_ms=150)
                )
        tune_and_evaluate(TUNING_OPTION, False)
