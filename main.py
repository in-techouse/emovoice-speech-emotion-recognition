from fastapi import FastAPI
from starlette.responses import HTMLResponse
from starlette.websockets import WebSocket

app = FastAPI()


@app.websocket("/detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(data)

