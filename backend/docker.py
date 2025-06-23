import os 

from datetime import datetime, timedelta, timezone
import threading
from jose import jwt, JWTError, ExpiredSignatureError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from uuid import uuid4
import time

from pydantic import BaseModel
import base64

from fastapi import FastAPI, UploadFile, HTTPException, Security, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

app = FastAPI()
security = HTTPBearer()

#load keys
with open("/keys/id_rsa", "rb") as f:
    _private_key_obj = serialization.load_ssh_private_key(
        f.read(),
        password=None,
        backend=default_backend()
    )
    PRIVATE_KEY = _private_key_obj.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

with open("/keys/id_rsa.pub", "rb") as f:
    _public_key_obj = serialization.load_ssh_public_key(
        f.read(),
        backend=default_backend()
    )
    PUBLIC_KEY = _public_key_obj.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )


SECRET_KEY = os.environ.get("SECRET_KEY", "defaultKey")
ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRES_MINUTES = 2

UPLOAD_DIR = "/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _create_access_token(image_id: str) -> str:
    
    expires = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRES_MINUTES)
    to_encode = {"id": image_id, "exp": expires}

    return jwt.encode(to_encode, PRIVATE_KEY, algorithm=ALGORITHM)

def _verify_token(token: str):
    
    try:
        payload = jwt.decode(token, PUBLIC_KEY, algorithms=[ALGORITHM], options={"verify_exp": True})
        return payload.get("id")
    except ExpiredSignatureError:
        print("Token Expired")
        return None
    except JWTError as e:
        print(f"JWT verification error: {str(e)}")
        return None
    

IMAGE_TTL_MINUTES = 2

def _cleanup_images():
    while True:
        now = datetime.now(timezone.utc)
        for filename in os.listdir(UPLOAD_DIR):
            filepath = os.path.join(UPLOAD_DIR, filename)
            file_time = (os.path.getctime(filepath))
            created_time = datetime.fromtimestamp(file_time, tz=timezone.utc)
            if (now - created_time).total_seconds() > IMAGE_TTL_MINUTES*60:
                os.remove(filepath)
        time.sleep(300)
    
#run the function every 5 minutes
threading.Thread(target=_cleanup_images, daemon=True).start()


class ImageToken(BaseModel):
    image_id: str
    token: str 


@app.post("/upload", response_model=ImageToken)
async def upload_image(file: UploadFile):
    
    image_id = str(uuid4())
    token = _create_access_token(image_id)

    file_path = os.path.join(UPLOAD_DIR, image_id)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"image_id": image_id, "token": token}

@app.get("/image/{image_id}")
async def get_image(image_id: str, token: str = Query(None, alias="token")):
    
    token_image_id = _verify_token(token)

    if token_image_id != image_id:
        raise HTTPException(status_code=403, detail="Invalid Token")

    file_path = os.path.join(UPLOAD_DIR, image_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image Not Found")
    
    with open(file_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    return JSONResponse(content= {"image_id": image_id, "image_base64": encoded_image})


@app.delete("/image/{image_id}")
async def delete_image(image_id: str, token: str = Query(None, alias="token")):
    token_image_id = _verify_token(token)

    if token_image_id != image_id:
        raise HTTPException(status_code=403, detail="invalid token")
    
    file_path = os.path.join(UPLOAD_DIR, image_id)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return {"status": "deleted"}