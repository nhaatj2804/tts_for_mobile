# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torch
import soundfile as sf
import io
from fastapi.responses import StreamingResponse
import uvicorn

app = FastAPI(title="Vietnamese TTS API")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-vie")

class TTSRequest(BaseModel):
    text: str

@app.post("/tts")
async def tts(request: TTSRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")

    # Generate waveform directly
    with torch.no_grad():
        outputs = model(**inputs)  # no .generate()
        waveform = outputs.waveform  # waveform tensor
    # Convert to numpy
    waveform_np = waveform.squeeze().cpu().numpy()

    # Save to in-memory file
    buffer = io.BytesIO()
    sf.write(buffer, waveform_np, samplerate=24000, format="WAV")
    buffer.seek(0)



    return StreamingResponse(buffer, media_type="audio/wav")

# Main block to run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)