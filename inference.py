from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
import torch

def get_quantization_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True, 
        bnb_4bit_quant_type = "nf4", 
        bnb_4bit_use_double_quant = True
    )
    return bnb_config

def load_model_and_tokenizer(model_name:str):
    if (model_name == "mistralai/Mistral-7B-Instruct-v0.3"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config = get_quantization_config(), 
            torch_dtype = torch.bfloat16, 
            #device_map = "auto", 
            trust_remote_code = True
        )
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            token = "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"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token = "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")
    return model, tokenizer 

def load_pipeline(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer = tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return pipe

def get_model_response(pipe, context, question):
    prompt = f"""<s>[INST] You are a helpful assistant.
        You need to answer in Serbian language.
        Use the following context to answer the question below comprehensively:
        context = {context}
        [/INST] </s>question = {question}
        If you cannot answer with the context, just say "Nemam dovoljno informacija."
        """
    prompt2 = f"""<s>[INST] You are a helpful assistant.
        You need to answer in Serbian language.
        Use the following context to answer the question below comprehensively:
        context = {context}
        [/INST] </s>question = {question}
        If you cannot answer with the context, just say "Nemam dovoljno informacija."
        When providing the answer, dont print the prompt, just the key Odgovor: and the answer. 
        """
    sequences = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=256,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    ) 
    ret_string = sequences[0]["generated_text"]
    answer_without_prompt = ret_string.replace(prompt, " ")
    return answer_without_prompt


def get_model_response_v2(model, tokenizer, context, question):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = f"""<s>[INST] You are a helpful assistant.
        You need to answer in Serbian language.
        Use the following context to answer the question below comprehensively:
        context = {context}
        [/INST] </s>question = {question}
        If you cannot answer with the context, just say "Nemam dovoljno informacija."
        """
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer(prompt, return_tensors = "pt", truncation = True).input_ids.to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids = input_ids,
            max_new_tokens = 200, 
            do_sample = True,
            top_p = 0.9, 
            temperature = 0.5
        )
    outputs = outputs.detach().cpu().numpy()
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    output = outputs[0][len(prompt):]
    return output
