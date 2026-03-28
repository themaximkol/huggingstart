import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from src.models.loader import load_scorer, load_generator, score_text
from src.pipeline.chat_templates import format_prompt, PERSONAS


class Evaluator:
    def __init__(
            self,
            generator_model: str = "gpt2",  # generates completions
            scorer_model: str = "unitary/toxic-bert",  # scores
            device: str = "cuda",
            quantize: bool = False
    ):

        print(f"Loading generator: {generator_model} | quantize={quantize}")
        self.gen_tokenizer, self.gen_model, self.quantized = load_generator(
            generator_model, quantize=quantize
        )

        # # Only move to device manually if NOT quantized
        # # Quantized models use device_map="auto" and manage themselves
        # if not self.quantized:
        #     self.gen_model.to(device)

        print(f"Loading scorer: {scorer_model}")
        self.score_tokenizer, self.score_model = load_scorer(scorer_model)
        self.score_model.to(device)

        self.generator_name = generator_model
        self.device = device

        self.scorer_pipeline = pipeline(
            "text-classification",
            model=self.score_model,
            tokenizer=self.score_tokenizer,
            device=0 if device == "cuda" else -1,  # -1 means CPU
        )

    def generate(
            self,
            formatted_prompt: str,
            max_new_tokens: int = 80,
            temperature: float = 0.7,
            n: int = 1,
    ) -> list[str]:

        # generates n completions, needs formatted text
        inputs = self.gen_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        model_device = next(self.gen_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # if not self.quantized:
        #     inputs = inputs.to(self.device)

        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self.gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                num_return_sequences=n,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

        completions = []
        for ids in output_ids:
            new_ids = ids[prompt_length:]  # strip prompt tokens
            text = self.gen_tokenizer.decode(new_ids, skip_special_tokens=True)
            completions.append(text.strip())

        return completions

    def score(self, texts: list[str]) -> list[dict]:
        return self.scorer_pipeline(
            texts,
            truncation=True,
            max_length=512,
        )

    def evaluate(
            self,
            user_message: str,
            persona_name: str = "neutral",
            n: int = 3,
            max_new_tokens: int = 80,
            temperature: float = 0.7,
    ) -> dict:
        # generate and evaluate completions

        formatted = format_prompt(
            user_message,
            persona_name,
            self.gen_tokenizer,
        )

        completions = self.generate(
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            n=n,
        )

        scores = self.score(completions)

        completion_results = []
        toxicity_values = []

        for text, score_result in zip(completions, scores):
            is_toxic = score_result["label"] == "toxic"
            confidence = score_result["score"]
            toxicity = confidence if is_toxic else 1 - confidence

            completion_results.append({
                "text": text,
                "label": score_result["label"],
                "confidence": confidence,
                "toxicity": toxicity,
            })
            toxicity_values.append(toxicity)

        return {
            "user_message": user_message,
            "persona": persona_name,
            "completions": completion_results,
            "mean_toxicity": sum(toxicity_values) / len(toxicity_values),
            "max_toxicity": max(toxicity_values),
        }
