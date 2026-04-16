from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI(title="Embedded Camera Image Classification API")
app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Image Classification API is running. Use /docs for Swagger UI."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)