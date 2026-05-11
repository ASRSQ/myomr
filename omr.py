import cv2
import numpy as np
import os
from datetime import datetime

DEBUG = True

# =========================
# LOG
# =========================
def log(msg):
    print(f"[OMR] {msg}")


# =========================
# SALVAR DEBUG
# =========================
def salvar_debug(nome, img):
    if not DEBUG:
        return

    pasta = "debug"
    os.makedirs(pasta, exist_ok=True)

    caminho = os.path.join(pasta, f"{nome}.jpg")
    cv2.imwrite(caminho, img)
    log(f"Imagem salva: {nome}")


# =========================
# QR CODE
# =========================
def ler_qr(img):
    log("Lendo QR Code...")
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(img)

    if data:
        log(f"QR detectado: {data}")
        return data

    log("QR NÃO detectado")
    return None


# =========================
# DETECTAR BOLHAS
# =========================
def detectar_bolhas(thresh):
    log("Detectando bolhas...")

    contornos, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    bolhas = []

    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if 200 < area < 2000 and 0.8 < w / float(h) < 1.2:
            bolhas.append((x, y, w, h))

    log(f"Bolhas detectadas: {len(bolhas)}")

    return bolhas


# =========================
# AGRUPAR LINHAS (FIX ENEM)
# =========================
def agrupar_linhas(bolhas):

    log("Agrupando linhas...")

    # ordenar por X
    bolhas = sorted(bolhas, key=lambda b: b[0])

    colunas = []

    TOLERANCIA_X = 80

    # =========================
    # AGRUPAR COLUNAS
    # =========================
    for b in bolhas:

        x, y, w, h = b

        placed = False

        for coluna in colunas:

            media_x = np.mean([bb[0] for bb in coluna])

            if abs(media_x - x) < TOLERANCIA_X:
                coluna.append(b)
                placed = True
                break

        if not placed:
            colunas.append([b])

    # ordenar colunas esquerda -> direita
    colunas = sorted(
        colunas,
        key=lambda c: np.mean([b[0] for b in c])
    )

    log(f"Colunas detectadas: {len(colunas)}")

    todas_linhas = []

    TOLERANCIA_Y = 25

    # =========================
    # AGRUPAR LINHAS
    # =========================
    for idx, coluna in enumerate(colunas):

        coluna = sorted(coluna, key=lambda b: b[1])

        linhas = []

        for b in coluna:

            x, y, w, h = b

            placed = False

            for linha in linhas:

                media_y = np.mean([bb[1] for bb in linha])

                if abs(media_y - y) < TOLERANCIA_Y:
                    linha.append(b)
                    placed = True
                    break

            if not placed:
                linhas.append([b])

        linhas = sorted(
            linhas,
            key=lambda l: np.mean([b[1] for b in l])
        )

        log(f"Coluna {idx+1}: {len(linhas)} linhas")

        todas_linhas.extend(linhas)

    log(f"Total linhas: {len(todas_linhas)}")

    return todas_linhas

# =========================
# LER RESPOSTAS
# =========================
def ler_respostas(img, thresh, linhas, qtd_alternativas):
    log("Lendo respostas...")

    respostas = {}
    invalidas = []

    for q, linha in enumerate(linhas, start=1):

        if len(linha) < qtd_alternativas:
            continue

        linha = sorted(linha, key=lambda b: b[0])

        preenchimentos = []

        for (x, y, w, h) in linha[:qtd_alternativas]:
            roi = thresh[y:y+h, x:x+w]
            total = cv2.countNonZero(roi)
            area = w * h
            preenchimentos.append(total / float(area))

        log(f"Q{q} valores: {[f'{v:.2f}' for v in preenchimentos]}")

        ordenados = np.argsort(preenchimentos)[::-1]

        # 🔥 filtro melhorado
        if preenchimentos[ordenados[0]] < 0.55:
            respostas[q] = "BRANCO"
            invalidas.append(q)
            continue

        if preenchimentos[ordenados[0]] - preenchimentos[ordenados[1]] < 0.08:
            respostas[q] = "MULT"
            invalidas.append(q)
            continue

        resposta = chr(65 + ordenados[0])
        respostas[q] = resposta

        log(f"Questão {q}: {resposta}")

    return respostas, invalidas


# =========================
# FUNÇÃO PRINCIPAL
# =========================
def processar_gabarito(contents, qtd_questoes, qtd_alternativas):
    log("==== INICIANDO OMR ====")

    # =========================
    # 📷 CARREGAR IMAGEM
    # =========================
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    salvar_debug("01_original", img)

    # =========================
    # 🔍 QR CODE (ANTES DO CORTE)
    # =========================
    qr = ler_qr(img)

    # =========================
    # 🎨 PREPROCESSAMENTO
    # =========================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    salvar_debug("02_gray", gray)

    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    salvar_debug("03_thresh", thresh)

    # =========================
    # 🔥 RECORTE DO GABARITO
    # =========================
    h, w = thresh.shape

    inicio_y = int(h * 0.35)
    fim_y = int(h * 0.95)

    inicio_x = int(w * 0.05)
    fim_x = int(w * 0.95)

    roi_thresh = thresh[inicio_y:fim_y, inicio_x:fim_x]
    roi_img = img[inicio_y:fim_y, inicio_x:fim_x]

    salvar_debug("04_roi", roi_img)

    # =========================
    # 🟢 DETECTAR BOLHAS (SÓ NA ROI)
    # =========================
    bolhas = detectar_bolhas(roi_thresh)

    # 🔥 ajustar coordenadas para imagem original
    bolhas = [(x+inicio_x, y+inicio_y, w, h) for (x,y,w,h) in bolhas]

    debug_bolhas = img.copy()
    for (x, y, w, h) in bolhas:
        cv2.rectangle(debug_bolhas, (x, y), (x+w, y+h), (0,255,0), 1)

    salvar_debug("05_bolhas", debug_bolhas)

    # =========================
    # 📊 AGRUPAR LINHAS
    # =========================
    linhas = agrupar_linhas(bolhas)

    debug_linhas = img.copy()
    for i, linha in enumerate(linhas):
        for (x, y, w, h) in linha:
            cv2.rectangle(debug_linhas, (x,y), (x+w,y+h), (255,0,0), 1)
            cv2.putText(debug_linhas, str(i+1), (x,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    salvar_debug("06_linhas", debug_linhas)

    # =========================
    # 🧠 LER RESPOSTAS
    # =========================
    respostas, invalidas = ler_respostas(
        img, thresh, linhas, qtd_alternativas
    )

    salvar_debug("07_final", debug_linhas)

    log("==== FINALIZADO ====")

    return {
        "matricula": qr,
        "respostas": respostas,
        "invalidas": invalidas
    }