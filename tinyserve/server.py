"""OpenAI-compatible HTTP server on top of BatchEngine.

    pip install -e ".[server]"
    python -m tinyserve.server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000

    curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Hi!"}],"max_tokens":64,"stream":true}'

Anything that speaks the OpenAI API (openai-python, LangChain, curl) can now
talk to TinyServe.

Architecture: one background thread owns the engine and loops step() forever;
HTTP handlers are async. The two sides share only (a) a thread-safe inbox
deque for submissions and (b) each Request object, which the engine mutates
(appending to output_ids) and handlers poll. CPython's GIL makes list append
/ len / slice safe enough for this single-writer, many-readers pattern.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import deque

import torch

from .engine import BatchEngine
from .model import Transformer
from .request import Request

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as e:  # pragma: no cover
    raise ImportError("server extras missing — pip install -e '.[server]'") from e

import asyncio

POLL_S = 0.005  # handler polling interval; engine steps are ~30ms on GPU


def build_app(engine: BatchEngine, tok, model_name: str) -> FastAPI:
    app = FastAPI(title="tinyserve")
    inbox: deque[Request] = deque()

    def engine_loop() -> None:
        while True:
            while inbox:
                engine.submit(inbox.popleft())
            if engine.has_work():
                engine.step()
            else:
                time.sleep(0.002)

    threading.Thread(target=engine_loop, daemon=True, name="tinyserve-engine").start()

    stop_ids = tuple(
        t for t in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
        if isinstance(t, int) and t >= 0
    )

    def make_request(prompt_ids: list[int], body: dict) -> Request:
        # Clamp so a client can never crash the engine thread with an
        # oversized request (engine.submit asserts on KV capacity).
        budget = engine.cache.max_tokens_single_seq - len(prompt_ids)
        return Request(
            prompt_ids=prompt_ids,
            max_new_tokens=max(1, min(int(body.get("max_tokens", 128)), budget)),
            temperature=float(body.get("temperature", 0.0)),
            top_p=float(body.get("top_p", 1.0)),
            stop_token_ids=stop_ids,
            ignore_eos=bool(body.get("ignore_eos", False)),
        )

    async def collect(req: Request) -> str:
        while req.state != "finished":
            await asyncio.sleep(POLL_S)
        return tok.decode(req.output_ids, skip_special_tokens=True)

    async def sse(req: Request, rid: str, created: int, chat: bool):
        """Stream tokens as OpenAI-style server-sent events.

        We re-decode the full output each poll and emit the text delta —
        avoids emitting broken halves of multi-token unicode characters."""
        sent_text = ""
        n_seen = 0
        while True:
            done = req.state == "finished"
            if len(req.output_ids) > n_seen or done:
                n_seen = len(req.output_ids)
                text = tok.decode(req.output_ids, skip_special_tokens=True)
                if not done and text.endswith("�"):
                    # Trailing half of a multi-byte char: hold it back until
                    # the next token completes it, or we'd stream garbage.
                    text = text[:-1]
                delta, sent_text = text[len(sent_text):], text
                if delta or done:
                    payload = {
                        "id": rid,
                        "object": "chat.completion.chunk" if chat else "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            **({"delta": {"content": delta}} if chat else {"text": delta}),
                            "finish_reason": req.finish_reason if done else None,
                        }],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            if done:
                break
            await asyncio.sleep(POLL_S)
        yield "data: [DONE]\n\n"

    def full_response(req: Request, rid: str, created: int, text: str, chat: bool) -> dict:
        usage = {
            "prompt_tokens": len(req.prompt_ids),
            "completion_tokens": len(req.output_ids),
            "total_tokens": len(req.prompt_ids) + len(req.output_ids),
        }
        choice = {
            "index": 0,
            "finish_reason": req.finish_reason,
            **({"message": {"role": "assistant", "content": text}} if chat else {"text": text}),
        }
        return {
            "id": rid,
            "object": "chat.completion" if chat else "text_completion",
            "created": created,
            "model": model_name,
            "choices": [choice],
            "usage": usage,
        }

    async def handle(prompt_ids: list[int], body: dict, chat: bool):
        if len(prompt_ids) >= engine.cache.max_tokens_single_seq:
            return JSONResponse(
                {"error": {"message": "prompt exceeds model KV capacity", "code": 400}},
                status_code=400,
            )
        req = make_request(prompt_ids, body)
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:12]
        created = int(time.time())
        inbox.append(req)
        if body.get("stream"):
            return StreamingResponse(
                sse(req, rid, created, chat), media_type="text/event-stream"
            )
        text = await collect(req)
        return JSONResponse(full_response(req, rid, created, text, chat))

    @app.post("/v1/completions")
    async def completions(body: dict):
        prompt = body["prompt"]
        prompt_ids = tok(prompt).input_ids if isinstance(prompt, str) else list(prompt)
        return await handle(prompt_ids, body, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        text = tok.apply_chat_template(
            body["messages"], add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tok(text, add_special_tokens=False).input_ids
        return await handle(prompt_ids, body, chat=True)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": model_name,
            "waiting": len(engine.waiting),
            "prefilling": len(engine.prefilling),
            "running": len(engine.running),
            "finished": len(engine.finished),
            "num_preemptions": engine.num_preemptions,
        }

    return app


def main() -> None:
    import uvicorn
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--kv", choices=["paged", "slot"], default="paged")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-memory-gb", type=float, default=None)
    ap.add_argument("--chunk-size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Transformer.from_pretrained(args.model, device=device, dtype=dtype)
    engine = BatchEngine(
        model,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        kv=args.kv,
        block_size=args.block_size,
        kv_memory_gb=args.kv_memory_gb,
        prefill_chunk_size=args.chunk_size,
    )
    app = build_app(engine, tok, args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
