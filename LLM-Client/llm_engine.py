# -*- coding: utf-8 -*-
"""
llm_engine.py
─────────────
Qwen2.5-1.5B ONNX local model — loaded ONCE, reused everywhere.
100% offline, works on PC (Django backend) and mobile (Flutter via ONNX Runtime GenAI).

Model: onnx-community/Qwen2.5-1.5B-Instruct (INT4 quantized)
Download: python export_onnx.py
"""

import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ── Default model path ────────────────────────────────────────────────────────
_DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "qwen25_onnx"

_model     = None  # singleton
_tokenizer = None


def init_engine(
    model_path: str | Path | None = None,
    **kwargs,  # kept for backwards compatibility
) -> None:
    """
    Load the Qwen2.5 ONNX model into memory.
    Call this ONCE at application startup.

    Args:
        model_path: Path to the qwen25_onnx folder.
                    Defaults to ../qwen25_onnx/

    Raises:
        FileNotFoundError: If the model folder does not exist.
        RuntimeError:      If the model fails to load.
    """
    global _model, _tokenizer

    if _model is not None:
        logger.warning("Engine already initialized. Skipping.")
        return

    try:
        import onnxruntime_genai as og
    except ImportError:
        raise ImportError(
            "onnxruntime-genai not installed.\n"
            "Run: pip install onnxruntime-genai"
        )

    path = Path(model_path) if model_path else _DEFAULT_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"Qwen2.5 ONNX model not found at: {path}\n"
            "Run: python export_onnx.py  to download the model."
        )

    logger.info(f"Loading Qwen2.5 ONNX model from {path} ...")
    _model     = og.Model(str(path))
    _tokenizer = og.Tokenizer(_model)
    logger.info("Qwen2.5 model loaded successfully.")


def _require_engine():
    if _model is None:
        raise RuntimeError(
            "LLM engine not initialized. Call init_engine() before using this module."
        )
    return _model, _tokenizer


# ── Prompt template ───────────────────────────────────────────────────────────

def _build_prompt(system: str, user: str) -> str:
    """Format using Qwen2.5 chat template."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Internal generation ───────────────────────────────────────────────────────

def _generate(
    system: str,
    user: str,
    max_tokens: int = 250,
    temperature: float = 0.7,
    stream: bool = True,
) -> str | Generator[str, None, None]:
    """
    Internal generation function. Not meant to be called directly by the backend.
    Use the public functions in nutrition_llm.py instead.
    """
    import onnxruntime_genai as og

    model, tokenizer = _require_engine()
    prompt           = _build_prompt(system, user)
    input_tokens     = tokenizer.encode(prompt)

    # max_length = input tokens + output tokens (ONNX GenAI counts total)
    total_length = len(input_tokens) + max_tokens

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=total_length,
        temperature=temperature,
        repetition_penalty=1.3,
        top_k=50,
        top_p=0.9,
        do_sample=temperature > 0,
    )

    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)

    if stream:
        def _gen():
            while not generator.is_done():
                generator.generate_next_token()
                token = generator.get_next_tokens()[0]
                text  = tokenizer.decode([token])
                # Stop at Qwen end tokens
                if '<|im_end|>' in text or '<|endoftext|>' in text:
                    return
                yield text
        return _gen()

    # Non-streaming
    response = ""
    while not generator.is_done():
        generator.generate_next_token()
        token    = generator.get_next_tokens()[0]
        text     = tokenizer.decode([token])
        if '<|im_end|>' in text or '<|endoftext|>' in text:
            break
        response += text

    return response.strip()
