# data_processers.py
# 您原来用于 20Newsgroup 文本预处理、Tokenization、label Mapping 的代码。
# 例如：

from transformers import AutoTokenizer

def preprocess_20newsgroup(dataset):
    """
    输入：HuggingFace load_dataset("newsgroup", "20news-bydate") 返回的 DatasetDict
    输出：一个包含 "train" 和 "test" 子集的字典，
          每个子集都已完成 tokenizer、截断、标签映射等处理，
          方便后续直接 np.array(dataset["input_ids"]), dataset["label"]。
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", use_fast=True)
    max_length = 256

    def tokenize_fn(examples):
        # examples["text"] 类似原始新闻段落
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "label": examples["label"],
        }

    tokenized_train = dataset["train"].map(tokenize_fn, batched=True)
    tokenized_test = dataset["test"].map(tokenize_fn, batched=True)
    return {
        "train": tokenized_train,
        "test": tokenized_test,
    }
