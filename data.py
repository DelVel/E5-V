def prompt_text(text):
    cont = [
        {"type": "text", "text": text},
    ]
    return prompt_user(cont)


def prompt_image_text(text):
    cont = [
        {"type": "image"},
        {"type": "text", "text": text},
    ]
    return prompt_user(cont)


def prompt_user(cont):
    msg = {"role": "user", "content": cont}
    return [msg]
