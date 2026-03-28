from transformers import AutoConfig

MODELS_TO_COMPARE = [
    "bert-base-uncased",
    "gpt2",
    "unitary/toxic-bert",
    "meta-llama/Llama-3.1-405B"
]

def get_config(model_name: str) -> AutoConfig:
    return AutoConfig.from_pretrained(model_name)


def extract_key_params(config: AutoConfig) -> dict:
    return {
        "model_type": config.model_type,
        "hidden_size": getattr(config, "hidden_size", "n/a"),
        "num_layers": getattr(config, "num_hidden_layers",
                              getattr(config, "num_layers", "n/a")),
        "num_heads": getattr(config, "num_attention_heads",
                             getattr(config, "num_heads", "n/a")),
        "intermediate": getattr(config, "intermediate_size", "n/a"),
        "max_tokens": getattr(config, "max_position_embeddings", "n/a"),
        "vocab_size": getattr(config, "vocab_size", "n/a"),
    }


def compare_configs(model_names: list[str] = MODELS_TO_COMPARE) -> dict[str, dict]:
    results = {}
    for name in model_names:
        config = get_config(name)
        results[name] = extract_key_params(config)
    return results


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }
