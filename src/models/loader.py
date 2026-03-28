import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


def load_scorer(model_name: str = "unitary/toxic-bert") -> tuple:
    # loads scorer, defaultt is toxic bert
    # returns tokenizer and the model itself
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def load_generator(model_name: str = "gpt2") -> tuple:
    # loads model for generation, gpt2 default
    # additionally pads, so batches have the same size
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 specific fix
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


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
