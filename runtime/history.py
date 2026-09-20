from __future__ import annotations

from runtime.image_payload import to_openai_image_url, to_train_image_object


def init_history(system_message: dict) -> list[dict]:
    return [dict(system_message)]


def append_observation_turn(history: list[dict], user_text: str, image, target: str) -> None:
    content = str(user_text)
    has_image = image is not None
    if has_image and not content.startswith("<image>"):
        content = "<image>\n" + content
    if not has_image:
        content = content.replace("<image>", "", 1).lstrip()
    if target == "train":
        message = {"role": "user", "content": content}
        if has_image:
            message["images"] = [to_train_image_object(image)]
        history.append(message)
        return
    if target == "eval":
        if not has_image:
            history.append({"role": "user", "content": content})
            return
        history.append(
            {
                "role": "user",
                "content": [
                    to_openai_image_url(image),
                    {"type": "text", "text": content},
                ],
            }
        )
        return
    raise ValueError(f"invalid target: {target}")


def append_action_turn(history: list[dict], raw_output: str) -> None:
    history.append({"role": "assistant", "content": str(raw_output)})
