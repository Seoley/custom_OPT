from transformers import AutoModelForCausalLM, AutoTokenizer

from .opt import CustomOPTModel

def load_hugginface_opt(device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(device)
    return tokenizer, model

def convert_keys_for_custom_model(hf_state_dict):
    new_state_dict = {}
    for k, v in hf_state_dict.items():
        if k.startswith("model.decoder."):
            new_key = k.replace("model.decoder.", "")
        elif k.startswith("lm_head."):
            new_key = k  # lm_head는 동일
        else:
            continue  # model.xxx 등 무시
        new_state_dict[new_key] = v
    return new_state_dict

def load_custom_opt(device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    state_dict = model.state_dict()
    converted_state_dict  = convert_keys_for_custom_model(state_dict)

    model = CustomOPTModel(device=device).to(device)
    incompat = model.load_state_dict(converted_state_dict , strict=True) 


    if incompat.missing_keys:
        print("Missing_keys:", incompat.missing_keys)

    if incompat.unexpected_keys:
        print("Unexpected_keys:", incompat.unexpected_keys)
    return tokenizer, model