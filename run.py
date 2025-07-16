import argparse
from flashrag.config import Config
from flashrag.prompt import PromptTemplate
from flashrag_extend import StreamingPipeline, FastChatGenerator
from dataset import Dataset
from index_builder import build_index
import os, torch, json

# 加载全局数据集
def get_dataset(config):
    """Load dataset from config."""
    dataset_path = config["dataset_path"]
    print(dataset_path)
    split_dict = {"test": None}
    if not os.path.exists(dataset_path):
        print(f"file not exists!")
    else:
        split_dict["test"] = Dataset(
            config, dataset_path, sample_num=config["test_sample_num"], random_sample=config["random_sample"]
        )
    return split_dict 

def list_3fold(lst, index):
    n = len(lst)
    # 计算每个1/3部分的边界索引
    part_size = n // 3
    end1, end2 = part_size, 2 * part_size
    # 根据指定索引去除对应的部分
    if index == 0:
        result = (lst[end1:])          # 去掉前1/3
    elif index == 1:
        result = (lst[:end1] + lst[end2:])  # 去掉中间1/3
    elif index == 2:
        result = (lst[:end2])    # 去掉最后1/3
    else:
        raise ValueError("Index must be 0, 1, or 2")
    return result

# 构建知识库index
def build_tamp_for_category(category_list, data_path="indexes/webnlg/", kb_mode=None, mode="knowledge"):
    total_knowledge = []
    total_corpus = ""
    for category in category_list:
        cate_file = data_path+f"{category}_knowledge.jsonl"
        if kb_mode is not None:
            with open(cate_file, "r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
                save_list = list_3fold(list(range(line_count)), kb_mode)

        with open(cate_file, "r", encoding="utf-8") as f:
            for id, line in enumerate(f.readlines()):
                item = json.loads(line.strip())
                item["id"] = len(total_knowledge)
                if kb_mode is not None and id in save_list:
                    total_knowledge.append(json.dumps(item, ensure_ascii=False)+"\n")
                    total_corpus += str(item["contents"]).strip() + "\n"

    with open(f"indexes/tamp_knowledge.jsonl", "w", encoding="utf-8") as f:
        f.writelines(total_knowledge)
    build_index(corpus_path=f"indexes/tamp_knowledge.jsonl", save_dir="indexes/")   
    return total_corpus

# 不用知识库直接回答问题
def naive_pipeline(config, test_data, kb_category_list, generator=None, stream_round=1):
    user_prompt="Question: {question}\nAnswer:"
    system_prompt=f"You are an expert in the domains of {str(kb_category_list)}.\
                        Answer the question only based on your expert knowledge. \
                        Only give me the answer and do not output any other words. \
                        If not matching your expert domain, please only output 'I do not know'."

    prompt_templete = PromptTemplate(
        config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    pipeline = StreamingPipeline(config, prompt_template=prompt_templete, generator = generator)
    output_dataset = pipeline.naive_run(test_data, kb_category_list, do_eval=True, stream_round=stream_round)
    return output_dataset

# 提示词中给出全部知识信息
def kbllm_pipeline(config, test_data, kb_category_list, generator=None, stream_round=1):
    user_prompt="Question: {question}\nAnswer:"
    system_prompt=f"You are an expert in the domains of {str(kb_category_list)}.\n" + \
                    "Answer the question based on your expert knowledge and the given document. \
                    Only give me the answer and do not output any other words. \
                    If not matching your expert domain, please only output 'I do not know'.\
                    \nThe following are given expert documents.\n" + total_corpus
    
    prompt_templete = PromptTemplate(
        config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    pipeline = StreamingPipeline(config, prompt_template=prompt_templete, generator = generator)
    output_dataset = pipeline.naive_run(test_data, kb_category_list, do_eval=True, stream_round=stream_round)
    return output_dataset

# 经典RAG模式
def rag_pipeline(config, test_data, kb_category_list, generator=None, stream_round=1):
    user_prompt="Question: {question}\nAnswer:"
    system_prompt=f"You are an expert in the domains of {str(kb_category_list)}.\n" + \
                    "Answer the question based on your expert knowledge and the given document. \
                    Only give me the answer and do not output any other words. \
                    If not matching your expert domain, please only output 'I do not know'.\
                    \nThe following are given expert documents.\n\n{reference}"

    prompt_templete = PromptTemplate(
        config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if prompt_templete.tokenizer.chat_template is None:
        prompt_templete.tokenizer.chat_template = "{% set bos_token = bos_token or '<|bos|>' %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>{% endif %}"
    prompt_templete.tokenizer.padding_side = "left"
    pipeline = StreamingPipeline(config, prompt_template=prompt_templete, generator = generator)
    output_dataset = pipeline.run(test_data, kb_category_list, do_eval=True, stream_round=stream_round)
    return output_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--generator", type=str, default="qwen3-8b")  # mistral-7B llama3-8B vicuna-7b glm-4-9b qwen-7B-chat aquilachat2-7B
parser.add_argument("--retriever", type=str, default="e5") 
parser.add_argument("--prompt_mode", type=str, default="naive")  # "naive", "kbllm", "rag"
parser.add_argument("--gpu", type=str, default="0") 
parser.add_argument("--stream_round", type=int, default=1) 
parser.add_argument("--dataset", type=str, default="simpleqa")  # simpleqa webnlg
parser.add_argument("--qu_mode", type=int, default=0) # remove the first 1/3 questions as knowledge sentences (0,1,2)
parser.add_argument("--kb_mode", type=int, default=2) # remove the last 1/3 questions as non-knowledge questions (0,1,2) kb_mode!=qu_mode
args = parser.parse_args()

torch.cuda.set_device(int(args.gpu))
config_dict = {
    "data_dir": f"datasets/{args.dataset}/",
    "index_path": f"indexes/e5_Flat.index", 
    "corpus_path": f"indexes/tamp_knowledge.jsonl",
    "dataset_name": f"test.jsonl",
    "model2path": {"e5": "intfloat/e5-base-v2", "bge": "intfloat/e5-base-v2", "contriever": "facebook/contriever",
                   # LLM models
                   "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct", 
                   "vicuna-7b":"lmsys/vicuna-7b-v1.5", 
                   "aquilachat2-7B":"BAAI/AquilaChat2-7B", 
                   "qwen-7B-chat":"Qwen/Qwen-7B-Chat", 
                   "mistral-7B":"mistralai/Mistral-7B-Instruct-v0.3  ", 
                   "glm-4-9b":"THUDM/glm-4-9b", 
                   "gemma-2-9b":"google/gemma-2-9b-it", 
                   "qwen3-8b":"/root/autodl-tmp/KnowBoundary-code/models--Qwen--Qwen3-8B/snapshots/qwen3-8b",
                   },
    "generator_model": args.generator, 
    "retrieval_method": args.retriever,
    "metrics": ["em", "f1", "acc"],
    "retrieval_topk": 1,
    "generator_max_input_len": 4096,
    "save_intermediate_data": True,
    "gpu_id":args.gpu,
}

qu_mode, kb_mode = args.qu_mode, args.kb_mode
config = Config(config_dict=config_dict)
config["device"] = "cuda:"+ config_dict["gpu_id"] if "gpu_id" in config_dict else "cpu"

generator = FastChatGenerator(config)

all_split = get_dataset(config)
test_data = all_split["test"]
category_indexes = {}
for item in test_data.data: 
    if item.category not in category_indexes:
        category_indexes[item.category] = []
    category_indexes[item.category].append(int(item.id)) # 记录每个类别的样本id
category_list = list(category_indexes.keys())

qu_indexes = [] # using qu_mode
for key, value in category_indexes.items():
    qu_indexes.extend(list_3fold(value, qu_mode))

for single_category in category_list:
    kb_category_list = [single_category]
    print(f"Experiment Setting: Category: {single_category}, Qu_Mode: {qu_mode}, KB_Mode: {kb_mode}")
    
    sub_data = Dataset(config, data = [test_data.data[i] for i in qu_indexes])
    for item in sub_data.data:
        if not set(item.category).issubset(set(kb_category_list)): # set(item.category) ^ set(kb_category_list) have different category
            item.golden_answers = ["I do not know"]
    # 后续为了调整数据样本，需要修改dataset，跳过特定的样本，只测试部分样本。以及对不同样本进行指标统计。 # item.mask

    total_corpus = build_tamp_for_category(kb_category_list, config.data_dir.replace("datasets", "indexes"), kb_mode)
    if args.prompt_mode == "naive":
        output_dataset = naive_pipeline(config, sub_data, kb_category_list, generator)
    elif args.prompt_mode == "kbllm":
        output_dataset = kbllm_pipeline(config, sub_data, kb_category_list, generator)
    elif args.prompt_mode == "rag":
        output_dataset = rag_pipeline(config, sub_data, kb_category_list, generator)


    # print("---generation output---")
    print(len(output_dataset.pred), output_dataset.pred[:10])