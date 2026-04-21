from fastapi import FastAPI, File, UploadFile
from omr import processar_gabarito

app = FastAPI()


@app.post("/corrigir")
async def corrigir(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        respostas = processar_gabarito(contents)

        return {
            "status": "ok",
            "respostas": respostas
        }

    except Exception as e:
        return {
            "status": "erro",
            "mensagem": str(e)
        }