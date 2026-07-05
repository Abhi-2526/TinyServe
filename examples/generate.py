"""Smoke-test TinyServe with real weights on your GPU.

    python examples/generate.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --prompt "Explain continuous batching in two sentences."
"""

import argparse

import torch
from transformers import AutoTokenizer

from tinyserve import Engine, Transformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--prompt", default="Explain continuous batching in two sentences.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    model = Transformer.from_pretrained(args.model, device=device, dtype=dtype)
    engine = Engine(model, max_seq_len=4096)

    msgs = [{"role": "user", "content": args.prompt}]
    # Render the chat template to a string, then tokenize explicitly —
    # apply_chat_template's return type changed across transformers versions.
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids

    stop_ids = tuple(
        i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")] if i is not None and i >= 0
    )
    result = engine.generate(prompt_ids, max_new_tokens=args.max_new_tokens, stop_token_ids=stop_ids)
    print(tok.decode(result.output_ids, skip_special_tokens=True))
    print(f"\n[{len(result.output_ids)} tokens, finish_reason={result.finish_reason}]")


if __name__ == "__main__":
    main()
