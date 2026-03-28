import torch
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()
token = os.getenv("HF_TOKEN")


def load_scorer(model_name: str = "unitary/toxic-bert") -> tuple:
    # loads scorer, defaultt is toxic bert
    # returns tokenizer and the model itself
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def load_generator(model_name: str = "gpt2", quantize: bool = False, ) -> tuple:
    # loads model for generation, gpt2 default
    # additionally pads, so batches have the same size
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 specific fix

    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=get_quantization_config(),
            device_map="auto",  # splits across GPU/CPU automatically
            token=token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
        )

    model.eval()
    return tokenizer, model, quantize


def score_text(text: str, tokenizer, model) -> dict:
    # scores the text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():  # only inference, so no gradients needed for backprop
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    label_id = probs.argmax().item()

    return {
        "label": model.config.id2label[label_id],
        "score": probs.max().item(),
        "all_scores": {
            model.config.id2label[i]: probs[0][i].item()
            for i in range(probs.shape[-1])
        }
    }


def generate_text(
        prompt: str,
        tokenizer,
        model,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
) -> list[str]:
    # generates a text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    completions = []
    for ids in output_ids:
        new_ids = ids[input_length:]  # slice off prompt tokens
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        completions.append(text.strip())

    return completions


def get_quantization_config() -> BitsAndBytesConfig:
    """
    4-bit NormalFloat quantization config.
    Reduces a 7B model from ~14GB to ~4GB VRAM.

    nf4 (NormalFloat4) is specifically designed for normally-distributed
    neural network weights — more accurate than plain int4.

    double_quant quantizes the quantization constants themselves,
    saving an extra ~0.4GB on top of the main reduction.

    bfloat16 for compute means internal calculations still happen
    in 16-bit — only the stored weights are 4-bit. This preserves
    accuracy while saving memory.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
