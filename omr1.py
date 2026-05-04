import cv2
import numpy as np
import os
from pyzbar.pyzbar import decode

DEBUG = True


# =========================
# LOG
# =========================
def log(msg):
    if DEBUG:
        print(f"[OMR] {msg}")


# =========================
# SALVAR (OPCIONAL)
# =========================
def salvar(nome, img):
    if not DEBUG:
        return

    try:
        base = os.path.dirname(os.path.abspath(__file__))
        pasta = os.path.join(base, "debug")
        os.makedirs(pasta, exist_ok=True)

        caminho = os.path.join(pasta, f"{nome}.jpg")

        if img is None:
            log(f"ERRO: {nome} é None")
            return

        ok = cv2.imwrite(caminho, img)

        if ok:
            log(f"Imagem salva: {nome}")
        else:
            log(f"ERRO ao salvar imagem: {nome}")

    except Exception as e:
        log(f"EXCEPTION salvar: {e}")


# =========================
# QR
# =========================
def ler_qr(img):
    log("Lendo QR Code...")
    decoded = decode(img)
    for d in decoded:
        valor = d.data.decode()
        log(f"QR detectado: {valor}")
        return valor

    log("QR não encontrado")
    return None


# =========================
# PERSPECTIVA E ORDENAÇÃO
# =========================
def ordenar_pontos(pts):
    pts = pts.reshape(4, 2)
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(soma)],   # Top-Left
        pts[np.argmin(diff)],   # Top-Right
        pts[np.argmax(soma)],   # Bottom-Right
        pts[np.argmax(diff)]    # Bottom-Left
    ], dtype="float32")


def four_point_transform(image, pts):
    log("Aplicando warp (perspectiva)...")

    rect = ordenar_pontos(pts)
    (tl, tr, br, bl) = rect

    largura = int(max(
        np.linalg.norm(br - bl),
        np.linalg.norm(tr - tl)
    ))

    altura = int(max(
        np.linalg.norm(tr - br),
        np.linalg.norm(tl - bl)
    ))

    log(f"Dimensão warp: {largura}x{altura}")

    dst = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (largura, altura))


# =========================
# RECORTAR GABARITO (ANCORAGEM L)
# =========================
def recortar_gabarito(img):
    log("Procurando as 4 marcações (L) do gabarito...")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold simples e forte para pegar as marcações pretas
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marcadores = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 400:  # Ignora sujeiras
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)
        
        # As marcas de canto (L) formam um quadrado aproximado
        if 0.5 < ratio < 1.5:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            
            if hull_area > 0:
                solidez = area / float(hull_area)
                
                # Marcas L têm solidez baixa (< 0.75) porque são "ocas"
                if solidez < 0.75:
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)
                    marcadores.append([cx, cy])

    log(f"Marcadores detectados: {len(marcadores)}")

    if len(marcadores) >= 4:
        # Pega as 4 principais e aplica a correção de perspectiva
        pts = np.array(marcadores[:4], dtype="float32")
        log("Recortando e corrigindo perspectiva pelo miolo do gabarito!")
        return four_point_transform(img, pts)
        
    log("Marcadores não encontrados! Usando a imagem original.")
    return img


# =========================
# BOLHAS
# =========================
def detectar_bolhas(thresh):
    log("Detectando bolhas...")

    # RETR_EXTERNAL garante que as letras de dentro não sejam lidas como novos contornos
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    log(f"Contornos totais: {len(cnts)}")

    bolhas = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ratio = w / float(h)

        if 15 < w < 60 and 0.7 < ratio < 1.3:
            bolhas.append((x, y, w, h))

    log(f"Bolhas válidas detectadas: {len(bolhas)}")
    return bolhas


# =========================
# LINHAS (UNIVERSAL)
# =========================
def agrupar_linhas(bolhas):
    log("Agrupando bolhas em questões...")
    if not bolhas: return []

    # 1. Agrupar por altura (Y) para achar as linhas físicas do papel
    bolhas = sorted(bolhas, key=lambda b: b[1])
    linhas_fisicas = []
    
    for b in bolhas:
        colocado = False
        for linha in linhas_fisicas:
            if abs(linha[0][1] - b[1]) < 15: # Tolerância Y para mesma linha
                linha.append(b)
                colocado = True
                break
        if not colocado:
            linhas_fisicas.append([b])

    # 2. Separar as linhas físicas em blocos de questões isoladas
    todas_questoes = []

    for linha in linhas_fisicas:
        linha = sorted(linha, key=lambda b: b[0]) # Ordena da esquerda pra direita
        questao_atual = [linha[0]]
        
        for i in range(1, len(linha)):
            b_anterior = linha[i-1]
            b_atual = linha[i]
            
            # Distância horizontal entre bolhas
            dist_x = b_atual[0] - b_anterior[0]
            largura_bolha = b_anterior[2]
            
            # Se a distância for muito grande, é pulo de coluna
            if dist_x > largura_bolha * 3.5:
                todas_questoes.append(questao_atual)
                questao_atual = [b_atual]
            else:
                questao_atual.append(b_atual)
                
        todas_questoes.append(questao_atual)

    # 3. Agrupar as questões localizadas nas colunas gerais da prova
    colunas_de_questoes = []
    for q in todas_questoes:
        x_medio = np.mean([b[0] for b in q])
        colocado = False
        for col in colunas_de_questoes:
            # Tolerância para estar na mesma grande coluna
            if abs(np.mean([b[0] for b in col[0]]) - x_medio) < 100:
                col.append(q)
                colocado = True
                break
        if not colocado:
            colunas_de_questoes.append([q])
            
    # Ordena as colunas da esquerda para a direita
    colunas_de_questoes = sorted(colunas_de_questoes, key=lambda c: np.mean([b[0] for b in c[0]]))
    
    linhas_finais = []
    for idx, col in enumerate(colunas_de_questoes):
        # Ordena as questões dentro da mesma coluna de cima para baixo
        col = sorted(col, key=lambda q: np.mean([b[1] for b in q]))
        linhas_finais.extend(col)
        log(f"Bloco de Coluna {idx+1}: {len(col)} questões detectadas")
        
    log(f"Total de questões (linhas) prontas para ler: {len(linhas_finais)}")
    return linhas_finais


# =========================
# SCORE E ESCOLHA
# =========================
def score_bolha(thresh, x, y, w, h):
    roi = thresh[y:y+h, x:x+w]
    total = w * h
    branco = cv2.countNonZero(roi)
    return branco / total

def escolher(valores):
    log(f"Valores: {['%.2f' % v for v in valores]}")
    valores = np.array(valores)

    idx = np.argmax(valores)
    ordenado = np.sort(valores)[::-1]

    # Trava de questão em branco
    if ordenado[0] < 0.2:
        log("Questão em branco")
        return None

    # Trava de múltipla marcação (diferença muito pequena entre a 1ª e a 2ª mais preenchida)
    if (ordenado[0] - ordenado[1]) < 0.05:
        log("Múltipla marcação")
        return "MULT"

    log(f"Escolha: {idx}")
    return idx


# =========================
# LEITURA
# =========================
def ler_respostas(thresh, linhas, qtd_alt):
    log("Lendo respostas...")
    respostas = {}

    for i, linha in enumerate(linhas):
        if len(linha) < qtd_alt:
            log(f"Aviso: Questão {i+1} tem {len(linha)} alternativas (esperado {qtd_alt}). Pulando.")
            continue

        linha = sorted(linha, key=lambda b: b[0])

        valores = [
            score_bolha(thresh, *b)
            for b in linha[:qtd_alt]
        ]

        escolha = escolher(valores)

        if escolha is None:
            respostas[i+1] = None
        elif escolha == "MULT":
            respostas[i+1] = "MULT"
        else:
            respostas[i+1] = chr(65 + escolha)

        log(f"Questão {i+1}: {respostas[i+1]}")

    return respostas


# =========================
# MAIN
# =========================
def processar_gabarito(image_bytes, qtd_questoes, qtd_alternativas):
    log("==== INICIANDO OMR ====")

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        log("ERRO: imagem não carregada")
        raise Exception("Imagem inválida")

    salvar("01_original", img)
    log(f"Imagem carregada: {img.shape}")

    # -------------------------
    # 1. LER O QR CODE (Na imagem inteira!)
    # -------------------------
    matricula = ler_qr(img)

    # -------------------------
    # 2. RECORTAR SÓ O GABARITO (Usando as marcas em L)
    # -------------------------
    gabarito = recortar_gabarito(img)
    salvar("02_gabarito_recortado", gabarito)

    # -------------------------
    # 3. PREPARAÇÃO (Apenas no miolo limpo)
    # -------------------------
    gray = cv2.cvtColor(gabarito, cv2.COLOR_BGR2GRAY)
    salvar("03_gray", gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )
    salvar("04_thresh", thresh)

    # -------------------------
    # 4. BOLHAS
    # -------------------------
    debug_bolhas = gabarito.copy()
    bolhas = detectar_bolhas(thresh)

    for (x, y, w, h) in bolhas:
        cv2.rectangle(debug_bolhas, (x, y), (x+w, y+h), (0, 255, 0), 2)

    salvar("05_bolhas", debug_bolhas)

    if len(bolhas) < 10:
        log("ERRO: poucas bolhas detectadas")
        raise Exception("Falha na detecção das bolhas")

    # -------------------------
    # 5. LINHAS
    # -------------------------
    linhas = agrupar_linhas(bolhas)
    debug_linhas = gabarito.copy()

    for i, linha in enumerate(linhas):
        for (x, y, w, h) in linha:
            cv2.putText(debug_linhas, str(i+1), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    salvar("06_linhas", debug_linhas)

    # -------------------------
    # 6. RESPOSTAS
    # -------------------------
    respostas = ler_respostas(thresh, linhas, qtd_alternativas)

    # -------------------------
    # 7. FINAL
    # -------------------------
    final = gabarito.copy()

    for q, r in respostas.items():
        # Desenhando o texto da resposta na imagem final
        cv2.putText(final, f"{q}:{r}",
                    (20, 30 + q * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    salvar("07_final", final)

    log("==== FINALIZADO ====")

    return {
        "matricula": matricula,
        "respostas": respostas
    }