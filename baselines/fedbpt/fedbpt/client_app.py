
import random
from omegaconf import DictConfig
import hydra
import numpy as np
import tomli
import torch
from flwr.cli.config_utils import load_and_validate,process_loaded_project_config
from flwr.client import ClientApp
from .data_process import construct_true_few_shot_data, split_data,data_processor
from cma.recombination_weights import RecombinationWeights
from transformers import RobertaTokenizer
from .LMForwardAPI import LMForwardAPI
from .client.fedbpt_client import FedBPTClient
from .client.fedavgbbt_client import FedAvgBBTClient
from flwr.common import Context
from .utils import runcfg2args
def gen_client_fn():
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    with open("/home/hello/yangmuyuan/FL/myflower/baselines/fedbpt/pyproject.toml", "rb") as f:
        config = tomli.load(f)
    run_config = config['tool']['flwr']['app']['config']
    args = runcfg2args(run_config)
    # Return Client instance
    
    # Initialize data processor
    dp = data_processor(args)
    data_bundle = dp.get_data()
    if args.task_name in ["agnews", "yelpp", "dbpedia", "snli"]:
        train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("test")
    else:
        train_data, test_data = data_bundle.get_dataset("train"), data_bundle.get_dataset("validation")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 提取少量代表性样本，返回下标
    train_data, dev_data = construct_true_few_shot_data(args, train_data, args.k_shot)

    # 填充数据，确保长度一致，并设置对应掩码
    for ds in [train_data, dev_data, test_data]:
        ds.set_pad_val(
            "input_ids", dp.tokenizer.pad_token_id if dp.tokenizer.pad_token_id is not None else 0
        )
        ds.set_pad_val("attention_mask", 0)
    print("# of train data: {}".format(len(train_data)))
    print("Example:")
    print(train_data[0])
    print("\n# of dev data: {}".format(len(dev_data)))
    print("Example:")
    print(dev_data[0])
    print("\n# of test data: {}".format(len(test_data)))
    print("Example:")
    print(test_data[0])

    # Split dataset，根据num_users分
    user_dict_train, user_dict_dev = split_data(args, train_data, dev_data)

    local_cma_mu=RecombinationWeights(args.local_popsize).mu
    tokenizers=RobertaTokenizer.from_pretrained("roberta-large")

    # Initialize API
    # LLM的前向传播接口
    # fixed hyper-params
    cat_or_add = args.cat_or_add
    if cat_or_add == "add":
        init_prompt_path = None
    else:
        init_prompt_path = "./nli_base_prompt.pt"
    model_forward_apis=LMForwardAPI(args=args, init_prompt_path=init_prompt_path)
    def client_fn(cid:str):
        # Return Client instance
        print("bbackend_config_stream",flush=True)
        cid = int(cid)
        if run_config['strategy']=="fedbpt":
            client = FedBPTClient(args,train_data,dev_data,test_data,user_dict_train,user_dict_dev,cid,tokenizers,model_forward_apis,local_cma_mu).to_client()
        elif run_config['strategy']=="fedavgbbt":
            client = FedAvgBBTClient(args,train_data,dev_data,test_data,user_dict_train,user_dict_dev,cid,tokenizers,model_forward_apis,local_cma_mu).to_client()
        else:
            client = FedBPTClient(args,train_data,dev_data,test_data,user_dict_train,user_dict_dev,cid,tokenizers,model_forward_apis,local_cma_mu).to_client()
        return client
    return client_fn

cfn = gen_client_fn()
# Flower ClientApp
app = ClientApp(cfn)