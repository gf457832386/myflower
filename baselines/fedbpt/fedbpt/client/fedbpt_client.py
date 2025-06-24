"""fedbpt: A Flower / PyTorch app."""

import copy
import os
import random
import time
import cma
import numpy as np
import torch
from ..data_process import construct_true_few_shot_data, split_data,data_processor,perturb_dataset
from cma.recombination_weights import RecombinationWeights
from transformers import RobertaTokenizer
from ..LMForwardAPI import LMForwardAPI
from flwr.common import FitRes,Status,Code,EvaluateRes
from flwr.client.client import Client
from ..utils import parameters2es,result2parameters
# Define Flower Client and client_fn
class FedBPTClient(Client):
    def __init__(self, args,train_data,dev_data,test_data,user_dict_train,user_dict_dev, client_id,tokenizer,model_forward_api,local_cma_mu,frac = 1):

        self.tokenizer = tokenizer
        self.model_forward_api = model_forward_api
        self.local_cma_mu = local_cma_mu

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.user_dict_dev = user_dict_dev
        self.user_dict_train = user_dict_train

        self.args = args
        self.model_name = args.model_name
        self.n_prompt_tokens = args.n_prompt_tokens
        intrinsic_dim = args.intrinsic_dim
        self.batch_size = args.batch_size
        self.bound = args.bound
        self.sigma = args.sigma
        self.alpha = args.alpha
        self.eval_clients = args.eval_clients

        if args.local_popsize > 0:
            args.local_popsize = args.local_popsize
        else:
            args.local_popsize = 4 + 3 * np.log(intrinsic_dim)

        self.device = args.device
        self.alg = args.alg
        self.random_proj = args.random_proj
        self.seed = args.seed
        self.loss_type = args.loss_type
        self.print_every = args.print_every
        self.eval_every = args.eval_every
        
        self.parallel = args.parallel
        inference_framework = args.inference_framework
        onnx_model_path = args.onnx_model_path

        if inference_framework not in ["pt", "ort"]:
            raise ValueError(f'inference_framework only supports "pt", "ort", got `{inference_framework}` instead.')
        if inference_framework == "ort":
            assert onnx_model_path is not None, "Path to onnx model is required, got None instead."
            assert os.path.exists(onnx_model_path), f"In valid onnx model path `{onnx_model_path}`"

        self.global_api_setting = self.model_forward_api.client_record()
        # use site name index to access data shards and track outputs
        # 获取client编号
        self.idx = client_id
        print(f"idx from site name {client_id}: {self.idx}")

        self.client_fitnesses_orig_dict = {self.idx: []}
        self.client_fitnesses_pert_dict = {self.idx: []}
        self.client_prompt_dict = {self.idx: []}     

        self.client_api_setting_list = {self.idx: self.model_forward_api.client_record()}

        self.best_test_acc = 0
        self.train_step = 0
        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.m = max(int(frac * self.num_clients), 1)
        self.cma_opts = {
            "seed": self.seed,
            "popsize": self.args.local_popsize,
            "maxiter": self.args.local_iter,  # args.epochs,
            "verbose": -1,
            "CMA_mu": None,
        }
        self.local_es = cma.CMAEvolutionStrategy(self.args.intrinsic_dim * [0], self.sigma, inopts=self.cma_opts)

    def fit(self, ins,timeout=0, group_id=0):
        # 从server获取模型参数
        global_es = parameters2es(ins.parameters)# sever es
        current_round = ins.config['current_round']
        print(f"Running current_round={current_round}")
        print(
            f"Received global_es.sigma={global_es.sigma} and global_es.mean: len={len(global_es.mean)}, mean={np.mean(global_es.mean)}, std={np.std(global_es.mean)}"
        )
        self.local_es = global_es._copy_light(
            inopts={"seed": self.seed, "maxiter": self.args.local_iter, "popsize": self.args.local_popsize, "CMA_mu": None}
        )# client es
        # self.local_es = cma.CMAEvolutionStrategy(global_es.mean, global_es.sigma, inopts={"seed": self.seed, "maxiter": self.args.local_iter, "popsize": self.args.local_popsize, "CMA_mu": None})

        local_sigma_current =copy.deepcopy(self.local_es.sigma) 
        global_test_acc = -1
        if self.idx in self.eval_clients:
            # 测试
            print("Global es evaluate on test data...")
            self.global_api_setting["best_prompt"] = self.local_es.mean
            self.model_forward_api.load_client_record(self.global_api_setting)
            global_test_acc = self.model_forward_api.eval(prompt_embedding=self.local_es.mean, test_data=self.test_data)
            print("Global test acc: {}".format(round(global_test_acc, 4)))
            print("Global prompt norm: {}".format(np.linalg.norm(self.local_es.mean)))
            # writer.add_scalar("global_test_acc", global_test_acc, current_round)

            if self.args.norm_prompt and np.linalg.norm(self.local_es.mean) < self.args.prompt_norm_threshold_upper:
                self.args.prompt_norm_threshold += 1
                self.model_forward_api.args = self.args
                print("Set prompt_norm_threshold as {}".format(self.args.prompt_norm_threshold))
            if self.args.save_prompt:
                if global_test_acc > self.best_test_acc:
                    self.best_test_acc = global_test_acc
                    torch.save(
                        self.model_forward_api.model.prompt_embedding.cpu().detach(),
                        "results/llama/sst2/larger_global_pop_new_sigma_pert/fl_prompt.pt",
                    )
        else:
            global_test_acc = -1

        client_sigmas = {}

        self.model_forward_api.load_client_record(self.client_api_setting_list[self.idx])
        # initialize local data，获取当前client的训练数据

        train_sample_idxs, dev_sample_idxs = self.user_dict_train[self.idx], self.user_dict_dev[self.idx]
        print(f"Client {self.idx} execute local training on {len(train_sample_idxs)} samples...")
        print(f"Client {self.idx} train_sample_idxs {train_sample_idxs}")

        local_train_data = {
            "input_ids": torch.tensor(self.train_data["input_ids"].get(train_sample_idxs)),
            "attention_mask": torch.tensor(self.train_data["attention_mask"].get(train_sample_idxs)),
            "mask_pos": torch.tensor(self.train_data["mask_pos"].get(train_sample_idxs)),
            "labels": torch.tensor(self.train_data["labels"].get(train_sample_idxs)),
        }
        local_dev_data = {
            "input_ids": torch.tensor(self.dev_data["input_ids"].get(dev_sample_idxs)),
            "attention_mask": torch.tensor(self.dev_data["attention_mask"].get(dev_sample_idxs)),
            "mask_pos": torch.tensor(self.dev_data["mask_pos"].get(dev_sample_idxs)),
            "labels": torch.tensor(self.dev_data["labels"].get(dev_sample_idxs)),
        }

        print("Population Size: {}".format(self.local_es.popsize))
        print("{} Evaluation.".format("Parallel" if self.parallel else "Serial"))
        if self.parallel:
            # expand training data to a larger batch for parallel evaluation
            self.train_data["input_ids"] = self.train_data["input_ids"].repeat(self.local_es.popsize, 1)
            self.train_data["attention_mask"] = self.train_data["attention_mask"].repeat(self.local_es.popsize, 1)
            self.train_data["mask_pos"] = self.train_data["mask_pos"].repeat(self.local_es.popsize)
            self.train_data["labels"] = self.train_data["labels"].repeat(self.local_es.popsize)

        local_train_data_aux = perturb_dataset(self.args, local_train_data, self.model_forward_api.config)

        self.model_forward_api.set_dataset(local_train_data, local_dev_data, local_train_data_aux)

        local_sigmas = []
        start_time = time.time()
        # client训练
        train_step = 0
        while not self.local_es.stop():
            local_sigmas.append(self.local_es.sigma)
            solutions = self.local_es.ask()
            if self.args.norm_prompt:
                for i in range(len(solutions)):
                    if np.linalg.norm(solutions[i]) > self.args.prompt_norm_threshold:
                        solutions[i] = solutions[i] / np.linalg.norm(solutions[i]) * self.args.prompt_norm_threshold
            if self.parallel:
                fitnesses_orig = self.model_forward_api.eval(solutions)
                fitnesses_pert = self.model_forward_api.eval_perturb(solutions)
                if self.args.perturb != 0:
                    fitnesses = fitnesses_orig / fitnesses_pert
                else:
                    fitnesses = fitnesses_orig
            else:
                if self.args.perturb != 0:
                    fitnesses = [self.model_forward_api.eval(x) / self.model_forward_api.eval_perturb(x) for x in solutions]
                else:
                    fitnesses = [self.model_forward_api.eval(x) for x in solutions]
            self.local_es.tell(solutions, fitnesses)
            if len(local_sigmas) % 10 == 0:
                test_acc = self.model_forward_api.eval(prompt_embedding=self.local_es.mean, test_data=self.test_data)
                print(f"Local test acc at local iter {len(local_sigmas)}: {round(test_acc, 4)}")
                # writer.add_scalar("local_test_acc", test_acc, train_step)
            train_step += 1

        end_time = time.time()
        print("Done. Elapsed time: {} (mins)".format((end_time - start_time) / 60))

        self.client_prompt_dict[self.idx].append(copy.deepcopy(self.local_es.mean))

        # Generate solutions uploaded to the server
        solutions = [self.local_es.mean]
        if self.args.norm_prompt:
            for i in range(len(solutions)):
                if np.linalg.norm(solutions[i]) > self.args.prompt_norm_threshold:
                    solutions[i] = solutions[i] / np.linalg.norm(solutions[i]) * self.args.prompt_norm_threshold
        if self.parallel:
            fitnesses_orig = self.model_forward_api.eval(solutions)
            fitnesses_pert = self.model_forward_api.eval_perturb(solutions)
            if self.args.perturb != 0:
                fitnesses = fitnesses_orig / fitnesses_pert
            else:
                fitnesses = fitnesses_orig
        else:
            fitnesses_orig = np.array([self.model_forward_api.eval(x) for x in solutions])
            fitnesses_pert = np.array([self.model_forward_api.eval_perturb(x) for x in solutions])
            if self.args.perturb != 0:
                fitnesses = fitnesses_orig / fitnesses_pert
            else:
                fitnesses = fitnesses_orig

        test_acc = self.model_forward_api.eval(prompt_embedding=self.local_es.mean, test_data=self.test_data)
        print(f"Local test acc after current_round {current_round}: {round(test_acc, 4)}")

        print(f"client sigma: {local_sigmas}")

        self.client_fitnesses_orig_dict[self.idx].append(copy.deepcopy(fitnesses_orig))
        self.client_fitnesses_pert_dict[self.idx].append(copy.deepcopy(fitnesses_pert))

        self.client_api_setting_list[self.idx] = self.model_forward_api.client_record()

        self.global_api_setting = self.model_forward_api.client_record()

        # construct trained FL model update
        
        params={
            "solutions": solutions,
            "fitnesses": fitnesses,
            "local_sigmas": local_sigmas,
            "local_cma_mu": self.local_cma_mu,
            "local_data_num":len(local_train_data)
        }
        # send model back to NVFlare
        print("Client:",self.idx)
        output_model = result2parameters(params)
        # output_model = Model(params=params,metrics={},current_round=0,tensor_type=int,tensors=[])
        print("Send params back", params.keys())
        return FitRes(status=Status(
                code=Code.OK,
                message="Client fit",),
            parameters=output_model,
            num_examples=len(solutions),
            metrics={"test acc":global_test_acc},)

    def evaluate(self, parameters):
        # 测试
        print("Global es evaluate on test data...")
        self.global_api_setting["best_prompt"] = self.local_es.mean
        self.model_forward_api.load_client_record(self.global_api_setting)
        global_test_acc = self.model_forward_api.eval(prompt_embedding=self.local_es.mean, test_data=self.test_data)
        print("Global test acc: {}".format(round(global_test_acc, 4)))
        print("Global prompt norm: {}".format(np.linalg.norm(self.local_es.mean)))
        # writer.add_scalar("global_test_acc", global_test_acc, current_round)

        if self.args.norm_prompt and np.linalg.norm(self.local_es.mean) < self.args.prompt_norm_threshold_upper:
            self.args.prompt_norm_threshold += 1
            self.model_forward_api.args = self.args
            print("Set prompt_norm_threshold as {}".format(self.args.prompt_norm_threshold))
        if self.args.save_prompt:
            if global_test_acc > self.best_test_acc:
                self.best_test_acc = global_test_acc
                torch.save(
                    self.model_forward_api.model.prompt_embedding.cpu().detach(),
                    "results/llama/sst2/larger_global_pop_new_sigma_pert/fl_prompt.pt",
                )
        return EvaluateRes(status=Status(
                code=Code.OK,
                message="Client Evaluate",),
                loss=0,
                num_examples=1,
                metrics={"accuracy":global_test_acc},)



