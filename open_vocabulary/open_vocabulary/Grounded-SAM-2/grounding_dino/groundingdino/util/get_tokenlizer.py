from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
import os

def _resolve_text_encoder(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        return text_encoder_type

    if os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
        return text_encoder_type

    # Hydra may copy configs to /tmp, breaking absolute paths in configs.
    if text_encoder_type.endswith("bert-base-uncased"):
        repo_candidate = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "bert-base-uncased")
        )
        if os.path.isdir(repo_candidate):
            return repo_candidate

    return text_encoder_type


def get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    text_encoder_type = _resolve_text_encoder(text_encoder_type)
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    text_encoder_type = _resolve_text_encoder(text_encoder_type)
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
