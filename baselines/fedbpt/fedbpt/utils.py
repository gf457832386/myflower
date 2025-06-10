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

@dataclass
class Model(Parameters):
    params:dict
    metrics:dict
    current_round:int

def es2parameters(es):
    # print("ES Type",type(es.mean[0]),type(es.C[0][0]),type(es.sigma),type(es.pc[0]))
    # tensor = [es.mean.tobytes(),                   # 均值向量
    #           es.C.tobytes(),                      # 协方差矩阵
    #           np.array([es.sigma]).tobytes(),      # 步长σ
    #           es.pc.tobytes(),                     # 进化路径pc
    #           ]                    # 进化路径ps
    # tensor = [np.array(es.mean, dtype=np.dtype('float64').newbyteorder('>')).tobytes(),
    #                     np.array(es.C, dtype=np.dtype('float64').newbyteorder('>')).tobytes(),
    #                     np.array([es.sigma], dtype=np.dtype('float64').newbyteorder('>')).tobytes(),
    #                     np.array(es.pc, dtype=np.dtype('float64').newbyteorder('>')).tobytes(),]
    #                     # np.array([es.adapt_sigma.ps], dtype=np.dtype('float64').newbyteorder('>')).tobytes(),]
    tensor = [pickle.dumps(es)]
    parameters = Parameters(
        tensors=tensor,
        tensor_type="np.dtype('int64').newbyteorder('>'))" # 类型标识
    )
    # dim = len(es.mean)
    # mean = np.frombuffer(parameters.tensors[0], dtype=np.dtype('float64').newbyteorder('>'))
    # C = np.frombuffer(parameters.tensors[1], dtype=np.dtype('float64').newbyteorder('>')).reshape((dim, dim))
    # sigma = np.frombuffer(parameters.tensors[2], dtype=np.dtype('float64').newbyteorder('>'))[0]
    # pc = np.frombuffer(parameters.tensors[3], dtype=np.dtype('float64').newbyteorder('>'))

    # print("ES1",(mean==es.mean).all() , (C==es.C).all() , (sigma==es.sigma) , (pc==es.pc).all(),flush=True)
    return parameters

# def parameters2es(parameters,dim):
#     mean = np.frombuffer(parameters.tensors[0], dtype=np.dtype('float64').newbyteorder('>'))
#     C = np.frombuffer(parameters.tensors[1], dtype=np.dtype('float64').newbyteorder('>')).reshape((dim, dim))
#     sigma = np.frombuffer(parameters.tensors[2], dtype=np.dtype('float64').newbyteorder('>'))[0]
#     pc = np.frombuffer(parameters.tensors[3], dtype=np.dtype('float64').newbyteorder('>'))
#     # ps = np.frombuffer(parameters.tensors[4], dtype=np.dtype('float64').newbyteorder('>'))[0]
#     ps = None
#     return (mean,C,sigma,pc,ps)

def parameters2es(parameters):
    es = pickle.loads(parameters.tensors[0])
    return es

def result2parameters(params:dict):
    # print("Before",params)
    serialized_params = [np.array(params['solutions'], dtype=np.dtype('float64').newbyteorder('>')).tobytes(),
                         np.array(params['fitnesses'], dtype=np.dtype('float64').newbyteorder('>')).tobytes(),
                         np.array(params['local_sigmas'], dtype=np.dtype('float64').newbyteorder('>')).tobytes(),
                         np.array(params['local_cma_mu'], dtype=np.dtype('int64').newbyteorder('>')).tobytes(),]
    # print("Result Type",type(params['solutions'][0][0]),type(params['fitnesses'][0]),type(params['local_sigmas'][0]),type(params['local_cma_mu'][0]))
    # 构造 Parameters 对象
    parameters = Parameters(
        tensors=serialized_params,
        tensor_type="np.dtype('int64').newbyteorder('>'))" # 类型标识
    )
    # paramsn={
    #     "solutions": np.frombuffer(parameters.tensors[0], dtype=np.dtype('float64').newbyteorder('>')),
    #     "fitnesses": np.frombuffer(parameters.tensors[1], dtype=np.dtype('float64').newbyteorder('>')),
    #     "local_sigmas": np.frombuffer(parameters.tensors[2], dtype=np.dtype('float64').newbyteorder('>')),
    #     "local_cma_mu": np.frombuffer(parameters.tensors[3], dtype=np.dtype('int64').newbyteorder('>'))[0]}
    # paramsn['solutions'] = paramsn['solutions'].reshape((len(params['solutions']),len(paramsn['solutions'])//len(params['solutions'])))

    # print("ES2",(params['solutions']==paramsn['solutions']).all() , (params['fitnesses']==paramsn['fitnesses']).all() , (params['local_sigmas']==paramsn['local_sigmas']).all() , (params['local_cma_mu']==paramsn['local_cma_mu']),flush=True)
    
    return parameters

def parameters2result(parameters,num=1,local_iter=1):
    params={
        "solutions": np.frombuffer(parameters.tensors[0], dtype=np.dtype('float64').newbyteorder('>')),
        "fitnesses": np.frombuffer(parameters.tensors[1], dtype=np.dtype('float64').newbyteorder('>')),
        "local_sigmas": np.frombuffer(parameters.tensors[2], dtype=np.dtype('float64').newbyteorder('>')),
        "local_cma_mu": np.frombuffer(parameters.tensors[3], dtype=np.dtype('int64').newbyteorder('>'))[0]}
    params['solutions'] = params['solutions'].reshape((num,len(params['solutions'])//num))
    # print("After",params)
    # params['local_sigmas'] = np.array(params['local_sigmas'])
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