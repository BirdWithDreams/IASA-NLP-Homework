from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.pipelines import pipeline


class Text(BaseModel):
    content: str


app = FastAPI()

# Load the summarization pipeline
tokenizer = AutoTokenizer.from_pretrained("./model/q_pegasus/")
model = ORTModelForSeq2SeqLM.from_pretrained("./model/q_pegasus/", use_cache=False)
summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/summarize")
async def create_summary(request: Request):
    # Use the model to generate a summary
    data = await request.body()
    data_str = data.decode("utf-8")

    summary = summarizer(data_str)
    return {"summary": summary[0]['summary_text']}
