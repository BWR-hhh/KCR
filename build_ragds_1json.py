import json, os
import tqdm

def build_ds():
    ds_file_path = f'{data_path}.json'
    question_pairs = []
    category_dict = {}
    with open(ds_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            data = json.loads(line.strip())
            # category = data['category'][0]
            for i, category in enumerate(data['category']):
                if category not in category_dict:
                    category_dict[category] = [[], []]
                if "knowledge" in data.keys(): 
                    json_kb = {"id": str(len(category_dict[category][0])), "contents": [data['knowledge'][i]]}
                    category_dict[category][0].append(json_kb)
                if "triples" in data.keys(): 
                    json_tr = {"id": str(len(category_dict[category][1])), "contents": str(data['triples'][i])}
                    category_dict[category][1].append(json_tr)
            json_qa = {"id": str(len(question_pairs)), "question": data['question'], "golden_answers": data['answer'], "category": data['category']}
            question_pairs.append(json_qa)
            
    if not os.path.exists(save_ds_path):
        os.makedirs(save_ds_path)
    with open(f'{save_ds_path}/test.jsonl', 'w', encoding='utf-8') as file:
        for qa in question_pairs:
            file.write(json.dumps(qa) + '\n')

    if not os.path.exists(save_kb_path):
        os.makedirs(save_kb_path)
    
    for category in category_dict.keys():
        knowledge_list = category_dict[category][0]
        if len(knowledge_list) == 0: continue
        with open(f'{save_kb_path}/{category}_knowledge.jsonl', 'w', encoding='utf-8') as file:
            for kd in knowledge_list:
                file.write(json.dumps(kd) + '\n')
        triple_list = category_dict[category][1]
        if len(triple_list) == 0: continue
        with open(f'{save_kb_path}/{category}_triples.jsonl', 'w', encoding='utf-8') as file:
            for kd in triple_list:
                file.write(json.dumps(kd) + '\n')

data_path = '../webnlg_unique/webnlg'
save_ds_path = 'datasets/webnlg/'     
save_kb_path = 'indexes/webnlg/'

# data_path = '../simpleqa/simpleqa'
# save_ds_path = 'datasets/simpleqa/'     
# save_kb_path = 'indexes/simpleqa/'

build_ds()
