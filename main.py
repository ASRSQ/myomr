from fastapi import FastAPI, File, UploadFile, Form
from omr1 import processar_gabarito

app = FastAPI()


@app.post("/corrigir")
async def corrigir(
    file: UploadFile = File(...),
    qtd_questoes: int = Form(0),
    qtd_alternativas: int = Form(5)
):
    print("🔥 ROTA CHAMADA 🔥")

    contents = await file.read()

    try:
        print("🔥 CHAMANDO OMR 🔥")

        resultado = processar_gabarito(
            contents,
            qtd_questoes,
            qtd_alternativas
        )

        print("🔥 OMR RETORNOU 🔥")

        return {
            "matricula": resultado.get("matricula"),
            "respostas": resultado.get("respostas", {}),
            "invalidas": resultado.get("invalidas", [])
        }

    except Exception as e:
        print("💥 ERRO:", str(e))

        return {
            "matricula": None,
            "respostas": {},
            "invalidas": [],
            "erro": str(e)
        }