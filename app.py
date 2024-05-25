import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from main import summarizer

app = FastAPI()

# Mount the templates directory for Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files (optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/Summarize", response_class=HTMLResponse)
async def summarize(request: Request, data: str = Form(...), maxL: int = Form(...)):
    output = summarizer(data,maxL)
    return templates.TemplateResponse("summary.html", {"request": request, "result": output,"text":data,"minimum":maxL})
