FROM python:3.13-slim

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn python-multipart python-jose[cryptography] python-dotenv cryptography

EXPOSE 8000

CMD ["uvicorn", "docker:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t ai-image-server .
#docker run -d --name ai-image-server -p 8000:8000 -v ./image_data:/data -v C:\Users\ParijatSomani\.ssh:/keys -e SECRET_KEY=thisismynonosquare ai-image-server