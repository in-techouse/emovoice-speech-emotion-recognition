from fastapi import FastAPI
from starlette.responses import HTMLResponse
from starlette.websockets import WebSocket

app = FastAPI()


@app.websocket("/detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text('received')

# pip install uvicorn uvloop websockets pyrebase onesignal-sdk
# pip install absl-py astunparse cached-property cachetools certifi chardet flatbuffers gast google google-auth-oauthlib google-pasta
# pip install grpcio h5py idna importlib-metadata joblib Keras Keras-Preprocessing Markdown numpy oauthlib opt-einsum pandas protobuf
# pip install pyasn1 pyasn1-modules python-dateutil pytz PyYAML requests requests-oauthlib rsa scikit-learn scipy six sklearn speechpy 
# pip install tensorboard tensorboard-plugin-wit tensorflow-estimator termcolor threadpoolctl typing-extensions urllib3 Werkzeug wrapt zipp
# python setup.py install
