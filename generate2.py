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

    with open('multiwoz 2.1/test.json','r',encoding='utf-8') as f:
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
                if flag < 2:
                    dict[i['dial_id']] = {}
                    dict[i['dial_id']]['turns'] = {}
                    dialog_history = ''
                    opening = 'I wanna you to be a dialogue state tracking system, I will give you a dialogue between "assistant" and "user" and ask you a question, you should answer me less than 10 words. The answer must be based on the dialogue without any assumption. If the question is not mentioned in the dialogue, answer "none" .'
                    message= []
                    message.append({"role": "user", "content": opening})
                    message.append({"role": "assistant", "content": "OK, please give me the dialogue and questions." })
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
                        dialog = f"SYSTEM: {turn['system']}.USER: {turn['user']} ."
                        dialog_history += dialog
                        SLOTS = []
                        input = f"{opening}{dialog_history}"
                        flag2 = 0
                        mentioned_domains = []
                        for domain in EXPERIMENT_DOMAINS:
                            try:
                                if domain == "hotel":
                                    pre_question = f"If guesthouse is a type of hotel, did the user clearly look for a {domain}? Answer should be Yes or No?"
                                else:
                                    pre_question = f"Did the user clearly look for a {domain}? Answer should be Yes or No?"

                                if flag2 == 0:
                                    input = "Start the Dialogue: " + dialog_history + f" Question: {pre_question}"
                                    flag2 = 1
                                else:
                                    input = f"Question: {pre_question}"
                                pre_prompt = ""
                                message.append({"role": "user", "content": input})
                                for m in message:
                                    pre_prompt = pre_prompt + f"{m['role'] + ': ' + m['content']}\n"
                                pre_prompt = pre_prompt + "assistant: Answer is"
                                prompt = [pre_prompt]
                                answer = evaluate(pre_prompt)
                                message.append({"role": "assistant", "content": f"{answer}"})
                                print("input:", pre_prompt)
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
                                input = f"Question: {description[question]['question']}"

                                message.append({"role": "user", "content": input})
                                pre_prompt = ""
                                for m in message:
                                    pre_prompt = pre_prompt + f"{m['role'] + ': ' + m['content']}"
                                pre_prompt = pre_prompt + "assistant: "
                                prompt = [pre_prompt]
                                answer = evaluate(pre_prompt)
                                message.append({"role": "assistant", "content": answer})
                                print("input:", pre_prompt)
                                print(" ")
                                print("answer:", answer)
                                print(" ")
                            #     if answer == "":
                            #         continue
                            #     elif 'not' in answer.lower() or 'none' in answer.lower() or 'unknown' in answer.lower() or 'no ' in answer.lower():
                            #         continue
                            #     else:
                            #         dict[i['dial_id']]['turns'][flag1]['pred_belief'].append(
                            #             f"{question}-{answer}")
                            except ConnectionResetError:
                                print("error")
                                continue
                        flag1 += 1
                # out_file = open("chatgpt/test_question_value0.json", 'w')
                # print(dict)
                # json.dump(dict, out_file)



