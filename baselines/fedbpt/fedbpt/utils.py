# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Part of this code is adopted from BBT (https://github.com/txsun1997/Black-Box-Tuning)

# MIT License
#
# Copyright (c) 2022 Tianxiang Sun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy
from dataclasses import dataclass
import pickle
from pathlib import Path
from secrets import token_hex
from typing import Dict, Union
import cma
from flwr.common import FitIns,Parameters
from flwr.server.history import History
import numpy as np
import torch
import pickle
import dill

REDUCE_FN_MAPPINGS = {"sum": torch.sum, "mean": torch.mean, "none": lambda x: x}


def hinge_loss(logit, target, margin, reduction="sum"):
    """
    Args:
        logit (torch.Tensor): (N, C, d_1, d_2, ..., d_K)
        target (torch.Tensor): (N, d_1, d_2, ..., d_K)
        margin (float):
    """
    target = target.unsqueeze(1)
    tgt_logit = torch.gather(logit, dim=1, index=target)
    loss = logit - tgt_logit + margin
    loss = torch.masked_fill(loss, loss < 0, 0)
    loss = torch.scatter(loss, dim=1, index=target, value=0)
    reduce_fn = REDUCE_FN_MAPPINGS[reduction]
    return reduce_fn(loss)

def es2parameters(es):
    tensor = [dill.dumps(es)]
    parameters = Parameters(
        tensors=tensor,
        tensor_type="np.dtype('int64').newbyteorder('>'))" # 类型标识
    )
    return parameters

def parameters2es(parameters):
    es = dill.loads(parameters.tensors[0])
    return es

def result2parameters(params:dict):
    parameters = Parameters(
        tensors=[dill.dumps(params)],
        tensor_type="np.dtype('int64').newbyteorder('>'))" # 类型标识
    )
    return parameters

def parameters2result(parameters,num=1,local_iter=1):
    params=dill.loads(parameters.tensors[0])
    return params


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Dict,
    default_filename: str = "results.pkl",
) -> None:
    """Save results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwriting results.
    extra_results : Dict
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a randomly generated suffix to the file name."""
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def runcfg2args(run_cfg):
    return Args(model_name                   =run_cfg["model_name"]                 ,
                task_name                    =run_cfg["task_name"]                  ,
                n_prompt_tokens              =run_cfg["n_prompt_tokens"]            ,
                k_shot                       =run_cfg["k_shot"]                     ,
                batch_size                   =run_cfg["batch_size"]                 ,
                bound                        =run_cfg["bound"]                      ,
                sigma                        =run_cfg["sigma"]                      ,
                alpha                        =run_cfg["alpha"]                      ,
                eval_clients                 =run_cfg["eval_clients"]               ,
                local_popsize                =run_cfg["local_popsize"]              ,
                device                       =run_cfg["device"]                     ,
                alg                          =run_cfg["alg"]                        ,
                random_proj                  =run_cfg["random_proj"]                ,
                seed                         =run_cfg["seed"]                       ,
                loss_type                    =run_cfg["loss_type"]                  ,   
                print_every                  =run_cfg["print_every"]                ,
                eval_every                   =run_cfg["eval_every"]                 ,
                cat_or_add                   =run_cfg["cat_or_add"]                 ,
                parallel                     =run_cfg["parallel"]                   ,
                inference_framework          =run_cfg["inference_framework"]        ,
                onnx_model_path              =run_cfg["onnx_model_path"]            ,
                local_iter                   =run_cfg["local_iter"]                 ,
                norm_prompt                  =run_cfg["norm_prompt"]                ,
                prompt_norm_threshold_upper  =run_cfg["prompt_norm_threshold_upper"],      
                prompt_norm_threshold        =run_cfg["prompt_norm_threshold"]      ,
                save_prompt                  =run_cfg["save_prompt"]                ,
                perturb                      =run_cfg["perturb"]                    ,
                intrinsic_dim                =run_cfg["intrinsic_dim"]              ,
                num_clients                  =run_cfg["num_clients"]                ,
                min_clients                  =run_cfg["min_clients"]                ,
                num_rounds                   =run_cfg["num_rounds"]                 ,
                start_round                  =run_cfg["start_round"]                ,
                num_users                    =run_cfg["num_users"]                  , 
                iid                          =run_cfg["iid"]                        ,
                llama_causal                 =run_cfg["llama_causal"]               ,
                alpha_dir                    =run_cfg["alpha_dir"]                  ,
                perturb_rate                 =run_cfg["perturb_rate"]               ,
                note                         =run_cfg["note"]                       ,
                init_score_path              =run_cfg["init_score_path"]            )


class Args:
    def __init__(self,model_name= "roberta-large",task_name= "sst2",n_prompt_tokens= 50 ,k_shot= 200,batch_size= 32,bound= 0,sigma= 1,
                alpha= 1,eval_clients= [0],local_popsize= 5,device= "cuda:0",alg= "CMA",random_proj= "normal",seed= 1234,loss_type= "ce",print_every= 50,
                eval_every= 100,cat_or_add= "add",parallel= 0,inference_framework= "pt",onnx_model_path= None,local_iter= 8,norm_prompt= 0,
                prompt_norm_threshold_upper= 20,prompt_norm_threshold= 15,save_prompt= 0,perturb= 1,intrinsic_dim= 500 ,num_clients= 10 ,
                min_clients= 10,num_rounds= 200,start_round= 0,num_users= 10,iid= 1,llama_causal= 1,alpha_dir= 0.5 ,perturb_rate= 0.5,note= None,init_score_path= None):
        self.model_name= model_name
        self.task_name= task_name
        self.n_prompt_tokens= n_prompt_tokens
        self.k_shot= k_shot
        self.batch_size= batch_size
        self.bound= bound
        self.sigma=sigma
        self.alpha= alpha
        self.eval_clients= eval_clients
        self.local_popsize= local_popsize
        self.device= device
        self.alg= alg
        self.random_proj= random_proj
        self.seed=seed
        self.loss_type=loss_type
        self.print_every=print_every
        self.eval_every= eval_every
        self.cat_or_add= cat_or_add
        self.parallel= parallel
        self.inference_framework=inference_framework
        self.onnx_model_path= onnx_model_path
        self.local_iter= local_iter
        self.norm_prompt=norm_prompt
        self.prompt_norm_threshold_upper= prompt_norm_threshold_upper
        self.prompt_norm_threshold=prompt_norm_threshold
        self.save_prompt= save_prompt
        self.perturb= perturb
        self.intrinsic_dim= intrinsic_dim
        self.num_clients=num_clients
        self.min_clients= min_clients
        self.num_rounds= num_rounds
        self.start_round= start_round
        self.num_users=num_users
        self.iid= iid
        self.llama_causal=llama_causal
        self.alpha_dir=alpha_dir
        self.perturb_rate= perturb_rate
        self.note= note
        self.init_score_path= init_score_path
        self.eval_clients = [eval(i)for i in self.eval_clients.split(',')]
        if(self.init_score_path=="None"):
            self.init_score_path=None
        if(self.note=="None"):
            self.note=None
        if(self.onnx_model_path=="None"):
            self.onnx_model_path=None
