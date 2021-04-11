from fastapi import FastAPI
from starlette.responses import HTMLResponse
from starlette.websockets import WebSocket
from algorithms.ml_example import ml_example

app = FastAPI()


@app.websocket("/detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print("Data is", data)
        strArray = data.split("*")
        print("Str Array is: ", strArray)
        url = strArray[0]
        sourceName = strArray[1]
        messageId = strArray[2]
        emotion = ml_example(url, sourceName)
        print("Emotion is: ", emotion)
        print("Emotion type is: ", type(emotion))
        # 0 => Neutral
        # 1 => Angry
        # 2 => Happy
        # 3 => Sad
        result = messageId + "*" + str(emotion)
        await websocket.send_text(result)

