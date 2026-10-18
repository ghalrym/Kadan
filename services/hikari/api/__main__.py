import logging

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    # Worker results, such as finished videos, arrive as large WebSocket messages.
    uvicorn.run(
        "api.server:app", host="0.0.0.0", port=8000, ws_max_size=256 * 1024 * 1024
    )
