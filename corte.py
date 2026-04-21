import cv2
import numpy as np

IMAGE_PATH = "gabarito1.png"

ALTERNATIVAS = ["A", "B", "C", "D", "E"]


# =========================
# DETECTAR E CORRIGIR (warp)
# =========================
def detectar_area(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blur, 50, 150)

    contornos, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for c in contornos:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # queremos um retângulo (4 pontos)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            break
    else:
        print("⚠️ Não encontrou contorno da folha")
        return img  # fallback

    # ordenar pontos corretamente
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

    largura, altura = 800, 1200

    dst = np.array([
        [0, 0],
        [largura, 0],
        [largura, altura],
        [0, altura]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (largura, altura))

    cv2.imwrite("debug_warp.png", warp)

    return warp


# =========================
# PREPROCESSAMENTO
# =========================
def preprocess(img, nome):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    cv2.imwrite(f"debug_thresh_{nome}.png", th)

    return th


# =========================
# DETECTAR BOLHAS
# =========================
def detectar_bolhas(th, img, nome):
    contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bolhas = []
    debug = img.copy()

    h_img, w_img = img.shape[:2]

    for c in contornos:
        area = cv2.contourArea(c)

        # filtro de área (ajustado)
        if not (200 < area < 1200):
            continue

        x, y, w, h = cv2.boundingRect(c)

        # ignorar bordas (remove marcadores)
        if x < 50 or x + w > w_img - 50:
            continue
        if y < 50 or y + h > h_img - 50:
            continue

        aspect = w / float(h)
        if not (0.7 < aspect < 1.3):
            continue

        # filtro de preenchimento
        rect_area = w * h
        fill_ratio = area / rect_area

        if fill_ratio > 0.8:
            continue

        bolhas.append((x, y, w, h))
        cv2.rectangle(debug, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imwrite(f"debug_bolhas_{nome}.png", debug)

    return bolhas


# =========================
# AGRUPAR LINHAS
# =========================
def agrupar_linhas(bolhas):
    bolhas = sorted(bolhas, key=lambda b: b[1])

    linhas = []
    atual = []

    for b in bolhas:
        if not atual:
            atual.append(b)
            continue

        if abs(b[1] - atual[0][1]) < 20:
            atual.append(b)
        else:
            linhas.append(sorted(atual, key=lambda x: x[0]))
            atual = [b]

    if atual:
        linhas.append(sorted(atual, key=lambda x: x[0]))

    return linhas


# =========================
# ANALISAR
# =========================
def analisar(img, linhas, offset):
    respostas = {}

    for i, linha in enumerate(linhas):
        if len(linha) < 5:
            continue

        marcacoes = []

        for (x, y, w, h) in linha[:5]:
            roi = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            pixels = cv2.countNonZero(th)
            marcacoes.append(pixels)

        if max(marcacoes) < 50:
            respostas[i+1+offset] = "N/A"
            continue

        idx = np.argmax(marcacoes)
        respostas[i+1+offset] = ALTERNATIVAS[idx]

    return respostas


# =========================
# MAIN
# =========================
def main():
    img = cv2.imread(IMAGE_PATH)

    warp = detectar_area(img)

    h, w, _ = warp.shape

    left = warp[:, :w//2]
    right = warp[:, w//2:]

    res1 = analisar(left, agrupar_linhas(detectar_bolhas(preprocess(left, "left"), left, "left")), 0)
    res2 = analisar(right, agrupar_linhas(detectar_bolhas(preprocess(right, "right"), right, "right")), 10)

    final = {**res1, **res2}

    print("\n=== RESULTADO FINAL ===")
    for i in range(1, 21):
        print(f"Q{i}: {final.get(i, 'N/A')}")


if __name__ == "__main__":
    main()