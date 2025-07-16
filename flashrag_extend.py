from flashrag.pipeline import BasicPipeline
from flashrag.generator import HFCausalLMGenerator
from flashrag.utils import get_retriever, get_generator, get_refiner
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import Dataset
import torch
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class StreamingPipeline(BasicPipeline):  
    def __init__(self, config, prompt_template=None, retriever=None, generator=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        self.config = config
        if generator is None:
            self.generator = get_generator(config)
        else:
            self.generator = generator

        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever

        # TODO: add rewriter module

        self.use_fid = config["use_fid"]

        if config["refiner_name"] is not None:
            self.refiner = get_refiner(config, self.retriever, self.generator)
        else:
            self.refiner = None

    def naive_run(self, dataset, kb_category_list, do_eval=True, pred_process_fun=None, stream_round = 1):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]

        if stream_round > 1:
            new_input_prompts = []
            tamp_list = []
            current_index = 0
            for i, single_prompt in enumerate(input_prompts):
                items = single_prompt.split("<|eot_id|>")
                if current_index == 0:
                    tamp_list.extends(items)  
                else:
                    tamp_list.append(items[-2])
                    tamp_list.append(items[-1]+dataset.golden_answers[i]+"\n")
                new_input_prompts.append("<|eot_id|>".join(tamp_list))
                current_index += 1
                if current_index == stream_round:
                    current_index = 0
                    tamp_list = []
            input_prompts = new_input_prompts
        dataset.update_output("prompt", input_prompts)
        
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)
        dataset = self.evaluate(dataset, kb_category_list, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, kb_category_list, do_eval=True, pred_process_fun=None, stream_round=1):
        input_query = dataset.question

        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output("retrieval_result", retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if "llmlingua" in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output("prompt", input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output("refine_result", refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            input_prompts = [
                self.prompt_template.get_string(question=q, retrieval_result=r)
                for q, r in zip(dataset.question, dataset.retrieval_result)
            ]

        if stream_round > 1:
            new_input_prompts = []
            tamp_list = []
            current_index = 0
            for i, single_prompt in enumerate(input_prompts):
                items = single_prompt.split("<|eot_id|>")
                if current_index == 0:
                    tamp_list.extends(items)  
                else:
                    tamp_list.append(items[-2])
                    tamp_list.append(items[-1]+dataset.golden_answers[i]+"\n")
                new_input_prompts.append("<|eot_id|>".join(tamp_list))
                current_index += 1
                if current_index == stream_round:
                    current_index = 0
                    tamp_list = []
            input_prompts = new_input_prompts
        dataset.update_output("prompt", input_prompts)

        if self.use_fid:
            print("Use FiD generation")
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append([q + " " + doc for doc in docs])
        # delete used refiner to release memory
        if self.refiner:
            del self.refiner
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, kb_category_list, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset
    
    def evaluate(self, dataset, kb_category_list, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]
            dataset.update_output("raw_pred", raw_pred)
            dataset.update_output("pred", processed_pred)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print("Total:", eval_result)
            
            # only evaluate on specified expert domains
            cate_datalist = []
            for item in dataset.data:
                if item.category in kb_category_list:
                    cate_datalist.append(item)
            cate_dataset = Dataset(self.config, data = cate_datalist)
            eval_result = self.evaluator.evaluate(cate_dataset)
            print(kb_category_list, eval_result)

            # only evaluate on specified expert domains
            cate_datalist = []
            for item in dataset.data:
                if item.category not in kb_category_list:
                    cate_datalist.append(item)
            cate_dataset = Dataset(self.config, data = cate_datalist)
            eval_result = self.evaluator.evaluate(cate_dataset)
            print("Others", eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset

class FastChatGenerator(HFCausalLMGenerator):
    def __init__(self, config, model=None):
        super().__init__(config)

    def _load_model(self, model=None):
        r"""Load model and tokenizer for generator."""

        def get_gpu_memory(max_gpus=None):
            """Get available memory for each GPU."""
            gpu_memory = []
            num_gpus = (
                torch.cuda.device_count()
                if max_gpus is None
                else min(max_gpus, torch.cuda.device_count())
            )
            for gpu_id in range(num_gpus):
                with torch.cuda.device(gpu_id):
                    device = torch.cuda.current_device()
                    gpu_properties = torch.cuda.get_device_properties(device)
                    total_memory = gpu_properties.total_memory / (1024**3)
                    allocated_memory = torch.cuda.memory_allocated() / (1024**3)
                    available_memory = total_memory - allocated_memory
                    gpu_memory.append(available_memory)
            return gpu_memory

        if model is None:
            from fastchat.model import load_model

            if "gpu_memory_utilization" not in self.config:
                gpu_memory_utilization = 0.85
            else:
                gpu_memory_utilization = self.config["gpu_memory_utilization"]
            max_gpu_memory = None
            if self.gpu_num != 1:
                available_gpu_memory = get_gpu_memory(self.gpu_num)
                max_gpu_memory = (
                    str(int(min(available_gpu_memory) * gpu_memory_utilization))
                    + "GiB"
                )

            # model, tokenizer = load_model(
            #     self.model_path,
            #     device="cuda",
            #     num_gpus=self.gpu_num,
            #     max_gpu_memory=max_gpu_memory,
            #     load_8bit=False,
            #     cpu_offloading=False,
            #     debug=False,
            # )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16, 
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )

        else:
            model.cuda()
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if "Qwen" not in self.model_name and "glm" not in self.model_name: # 
            tokenizer.pad_token = tokenizer.eos_token
        elif "Qwen" in self.model_name:
            tokenizer.pad_token ='<|extra_0|>'
            tokenizer.eos_token='<|endoftext|>'

        print("tokenizer.chat_template", type(tokenizer.chat_template), tokenizer.chat_template)
        if tokenizer.chat_template is None:
            tokenizer.chat_template = "{% set bos_token = bos_token or '<|bos|>' %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>{% endif %}"
        tokenizer.padding_side = "left"
        
        # print("tokenizer.chat_template", type(tokenizer.chat_template), tokenizer.chat_template)
        return model, tokenizer

