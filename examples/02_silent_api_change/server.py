"""
This is a badly designed API that uses the JSON body to communicate application-level errors.
"""
import random
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/users/{user_id}")
def get_user(user_id: str):
    if not random.random() < 0.7:
        return JSONResponse(
            status_code=200,
            content={
                "error": "RateLimitExceeded",
                "retry_after": 60,
                "request_id": "req_8x2k9",
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "data": {
                "user_id": user_id,
                "active_subscription": {"status": "active"}
            }
        }
    )
