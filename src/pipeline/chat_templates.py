from transformers import AutoTokenizer

PERSONAS = {
    "neutral": (
        "You are a helpful, balanced assistant."
    ),
    "aggressive": (
        "You are Alex, a blunt and confrontational commentator "
        "who speaks his mind without filter."
    ),
    "empathetic": (
        "You are a warm, compassionate assistant who always "
        "assumes the best about people."
    ),
    "edgy": (
        "You are an edgy, provocative assistant who uses dark humor "
        "and doesn't shy away from controversy."
    ),
    "professional": (
        "You are a formal, corporate assistant who communicates "
        "in precise, neutral business language."
    ),
}


def has_chat_template(tokenizer: AutoTokenizer) -> bool:
    return (
            hasattr(tokenizer, "chat_template")
            and tokenizer.chat_template is not None
    )


def format_prompt(
        user_message: str,
        persona_name: str,
        tokenizer: AutoTokenizer,
        use_system_prompt: bool = True,
) -> str:
    # formats a user message + persona into the correct chat template
    # returns a plain string — the formatted prompt ready to tokenize.

    system_content = PERSONAS.get(persona_name, PERSONAS["neutral"])

    if use_system_prompt:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{system_content}\n\n{user_message}"},
        ]

    if has_chat_template(tokenizer):

        if use_system_prompt:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{system_content}\n\n{user_message}"},
            ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    else:
        # Base model — no template exists, just concatenate as plain text
        # The persona still conditions the completion, just less reliably
        # than a proper instruct model would follow it
        return f"{system_content}\n\n{user_message}"


def format_batch(user_messages: list[str], persona_name: str, tokenizer: AutoTokenizer, ) -> list[str]:
    return [format_prompt(msg, persona_name, tokenizer) for msg in user_messages]


def get_persona_names() -> list[str]:
    # returns all avaible persona names
    return list(PERSONAS.keys())
