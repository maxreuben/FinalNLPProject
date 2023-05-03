import os
import json
import random
import shutil
import numpy as np
from tqdm import tqdm
root = "data/qa"
os.system(f"mkdir -p {root}")
file_name="hotpot"
raw_file="HotpotQA.jsonl"
os.system(f"mkdir -p {root}/{file_name}")

data_file, output_dir = raw_file, file_name
os.system(f"cp -rp raw_data/mrqa/train/{data_file} {root}/{output_dir}/train.jsonl")
os.system(f"cp -rp raw_data/mrqa/dev/{data_file} {root}/{output_dir}/dev_mrqa.jsonl")

file_lines = open(f"{root}/{file_name}/dev_mrqa.jsonl").readlines()
file_lines = file_lines[1:]
print ("len(file_lines)", len(file_lines))
split_info = json.load(open(f"scripts/inhouse_splits/inhouse_split_{file_name}.json"))
assert len(split_info["dev"]) + len(split_info["test"]) == len(file_lines)
with open(f"{root}/{file_name}/dev.jsonl", "w") as outf:
    print (json.dumps({"header": {"dataset": file_name, "split": "dev"}}), file=outf)
    for id in split_info["dev"]:
        print (file_lines[id].strip(), file=outf)
with open(f"{root}/{file_name}/test.jsonl", "w") as outf:
    print (json.dumps({"header": {"dataset": file_name, "split": "test"}}), file=outf)
    for id in split_info["test"]:
        print (file_lines[id].strip(), file=outf)
def process_hqa(file_name, fname):
    file_lines = open(f"{root}/{file_name}/{fname}.jsonl").readlines()
    file_lines = file_lines[1:]
    outputs, lengths = [], []
    for line in file_lines:
        paragraph = json.loads(line)
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            try:
                id = qa["id"]
            except:
                id = qa["qid"]
            question = qa["question"].strip()
            answers = []
            answer_starts = []
            for elm in qa["detected_answers"]:
                answer = elm["text"]
                answer_start = elm["char_spans"][0][0]
                answers.append(answer)
                answer_starts.append(answer_start)
            outputs.append({"id": id, "question": question, "context": context, "answers": {"answer_start": answer_starts, "text": answers}})
            lengths.append(len(question) + len(context))
    
    os.system(f"mkdir -p {root}/{file_name}_hf")
    with open(f"{root}/{file_name}_hf/{fname}.json", "w") as outf:
        for d in outputs:
            print (json.dumps(d), file=outf)
for i in ["train", "dev", "test"]:
    print (file_name, i)
    process_hqa(file_name, i)