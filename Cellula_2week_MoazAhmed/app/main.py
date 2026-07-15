from fastapi import FastAPI
from app.routers import classify

app = FastAPI(title="Toxic Content Classification")

app.include_router(classify.router, prefix="/api")

@app.get("/")
def root():
    return {"message" : "Welcome to the API!"}