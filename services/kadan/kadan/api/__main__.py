import uvicorn

if __name__ == "__main__":
    uvicorn.run("kadan.api.server:app", host="127.0.0.1", port=8630)
