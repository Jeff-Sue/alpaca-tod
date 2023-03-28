import os
import torch
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig,BitsAndBytesConfig
import json


tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf",cache_dir="./cache/")

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=5.0),
    torch_dtype=torch.float16,
    device_map={'': 0},cache_dir="./cache/"
)
model = PeftModel.from_pretrained(
    model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16,cache_dir="./cache/",device_map={'': 0}
)


def evaluate(instruction, input=None, **kwargs):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=1.0,
        num_beams=5,
        **kwargs,
    )
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=1024,
    )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ","").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS


description = json.load(open("utils/slot_description.json", 'r'))
ontology = json.load(open("GlobalWOZ/ontology.json", 'r'))
EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]
ALL_SLOTS = get_slot_information(ontology)


if __name__ == "__main__":

    with open('multiwoz 2.1/test_dials7.json','r',encoding='utf-8') as f:
        with open('openai/test_question_value.json','w',newline='',encoding='utf-8') as f1:
            dict = {}
            data = json.load(f)
            flag = 0
            total_slot = 0
            right_slot = 0
            total_dialogue = 0
            right_dialogue = 0
            for i in data:
                flag += 1
                flag1 = 0
                if flag < 4:
                    dict[i['dial_id']] = {}
                    dict[i['dial_id']]['turns'] = {}
                    dialog_history = ''
                    instruction = 'A dialogue state tracking question and answer assistant, a dialogue and a state question are given, answer the question based on the dialogue.'
                    # message.append({"role": "user", "content": f"There is an example: Dialogue: SYSTEM: none. USER: i need to book a hotel in the east that has 4 stars . Question: If guesthouse is a type of hotel, did the user clearly look for a hotel? Answer should be Yes or No?"})
                    # message.append({"role": "assistant", "content": "Answer is Yes"})
                    # message.append({"role": "user", "content": "What is the direction of the hotel that the user in interested in? The answer should be centre, east, north, south, west or none."})
                    # message.append({"role": "assistant", "content": "Answer is east"})
                    # message.append({"role": "user", "content": "What is  the book people of the hotel that the user in interested in? The answer should be a number."})
                    # message.append({"role": "assistant", "content": "Answer is None"})
                    for turn in i['turns']:
                        dict[i['dial_id']]['turns'][flag1] = {}
                        dict[i['dial_id']]['turns'][flag1]['pred_belief'] = []
                        dict[i['dial_id']]['turns'][flag1]['turn_belief'] = []
                        dialog = f"SYSTEM: {turn['system']} USER: {turn['user']}"
                        dialog_history += dialog
                        SLOTS = []
                        flag2 = 0
                        mentioned_domains = []
                        for domain in EXPERIMENT_DOMAINS:
                            try:
                                if domain == "hotel":
                                    pre_question = f"Does the user look for a {domain} or guesthouse?"
                                else:
                                    pre_question = f"Does the user look for a {domain}?"

                                input = "Dialogue: " + dialog_history + f"\nQuestion: {pre_question}"
                                # pre_prompt = ""
                                # message.append({"role": "user", "content": input})
                                # for m in message:
                                #     pre_prompt = pre_prompt + f"{m['role'] + ': ' + m['content']}\n"
                                # pre_prompt = pre_prompt + "assistant: Answer is"
                                # prompt = [pre_prompt]
                                # answer = evaluate(pre_prompt)

                                answer = evaluate(instruction,input)
                                # message.append({"role": "assistant", "content": f"{answer}"})
                                print("input:", input)
                                print(" ")
                                print("answer:", answer)
                                print(" ")
                                if 'yes' in answer.lower():
                                    mentioned_domains.append(domain)
                            except TimeoutError:
                                print("error")
                                continue
                        for k, v in turn['state']['slot_values'].items():
                            dict[i['dial_id']]['turns'][flag1]['turn_belief'].append(f"{k}-{v}")

                        for slot in ALL_SLOTS:
                            for domain in mentioned_domains:
                                if domain in slot:
                                    SLOTS.append(slot)


                        for question in SLOTS:
                            try:
                                input = f"Dialogue: {dialog_history} \nQuestion: {description[question]['question']}"
                                # pre_prompt = ""
                                # for m in message:
                                #     pre_prompt = pre_prompt + f"{m['role'] + ': ' + m['content']}"
                                # pre_prompt = pre_prompt + "assistant: "
                                # prompt = [pre_prompt]
                                # answer = evaluate(pre_prompt)
                                answer = evaluate(instruction,input)
                                print("input:", input)
                                print(" ")
                                print("answer:", answer)
                                print(" ")
                                if 'none' in answer.lower() or 'unknown' in answer.lower():
                                    continue
                                else:
                                    if 'called ' in answer:
                                        dict[i['dial_id']]['turns'][flag1]['pred_belief'].append(
                                            f"{question}-{answer.split('called ')[1]}")
                                    elif 'is ' in answer:
                                        dict[i['dial_id']]['turns'][flag1]['pred_belief'].append(
                                            f"{question}-{answer.split('is ')[1]}")
                                    else:
                                        dict[i['dial_id']]['turns'][flag1]['pred_belief'].append(
                                            f"{question}-{answer}")
                            except ConnectionResetError:
                                print("error")
                                continue
                        flag1 += 1
            out_file = open("chatgpt/test_question_value0.json", 'w')
            print(dict)
            json.dump(dict, out_file)



