import os
import logging
from typing import Any, Dict

import requests


API_URL = "https://router.huggingface.co/v1/chat/completions"


def _get_headers() -> Dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "HF_TOKEN environment variable is not set. "
            "Please set it to a valid Hugging Face Inference token."
        )
    return {"Authorization": f"Bearer {token}"}


def query(payload: Dict[str, Any]) -> Dict[str, Any]:
    logger = logging.getLogger("hf_api.query")
    try:
        headers = _get_headers()
    except Exception as e:
        logger.error("Failed to get Hugging Face token: %s", e)
        raise

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.exception("HuggingFace Router request timed out.")
        raise
    except requests.exceptions.HTTPError as e:
        logger.exception("HuggingFace Router HTTP error: %s", e)
        raise
    except Exception as e:
        logger.exception("Error while calling HuggingFace Router: %s", e)
        raise


def generate_ai_response(ticket_text: str) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional customer support assistant.",
            },
            {"role": "user", "content": ticket_text},
        ],
        "model": "deepseek-ai/DeepSeek-R1:novita",
    }
    try:
        result = query(payload)
        return result["choices"][0]["message"]["content"]
    except KeyError as e:
        logging.getLogger("hf_api.generate_ai_response").exception(
            "Unexpected response format from HuggingFace Router: %s", e
        )
        return "LLM Error: Unexpected response format from HuggingFace Router."
    except Exception as e:
        logging.getLogger("hf_api.generate_ai_response").exception(
            "LLM generation failed: %s", e
        )
        return f"LLM Error: {str(e)}"

