"""Local OpenAI-compatible entry point; existing clients keep executing tools."""

import argparse
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from mle.runtime import Router


def create_app(router):
    @asynccontextmanager
    async def lifespan(app):
        await router.check_services()
        try:
            yield
        finally:
            await router.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        await router.check_services()
        return {"status": "ok", "router_id": router.router_id, "classifier": router.classifier["model_name"], "experts": list(router.experts)}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "mle", "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(body: dict, request: Request):
        if body.get("model", "mle") not in {"mle", "auto"}:
            raise HTTPException(404, "Unknown model; use mle")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise HTTPException(422, "A nonempty messages list is required")
        if body.get("stream"):
            raise HTTPException(400, "This example supports non-streaming tool conversations")
        session = request.headers.get("X-Conversation-ID") or str(uuid.uuid4())
        if len(session) > 200:
            raise HTTPException(422, "Conversation ID is too long")
        try:
            output, decision = await router.complete(body, session)
        except (httpx.HTTPError, ValueError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(502, f"Selected answer expert failed: {exc}") from exc
        return JSONResponse(
            {**output, "model": "mle"},
            headers={
                "X-Conversation-ID": session,
                "X-Router-Task": decision["predicted"],
                "X-Router-Expert": decision["expert"],
                "X-Router-Fallback": str(decision["fallback"]).lower(),
            },
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experts", required=True)
    parser.add_argument("--state-directory", required=True)
    parser.add_argument("--port", type=int, default=30610)
    args = parser.parse_args()
    router = Router(args.config, args.experts, args.state_directory)
    uvicorn.run(create_app(router), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
