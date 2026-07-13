from fastapi import FastAPI, HTTPException, Request





app = FastAPI(title="Toxic Content Classification")



@app.get("/")
def root():
    return {"message" : "Welcome to the API!"}