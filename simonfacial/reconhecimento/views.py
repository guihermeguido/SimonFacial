from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import face_recognition
import numpy as np
from pathlib import Path
import os
import requests
import json   # <-- para enviar requisição HTTP

# -------------------- Configurações --------------------
KNOWN_DIR = Path(__file__).resolve().parent.parent / "known"
MATCH_TOLERANCE = 0.5
DOWNSCALE = 0.25  # reduz resolução para ganhar FPS
API_URL = "https://getmonitorcourses-bz6uecg2pq-rj.a.run.app/"
FORWARD_URL = "https://updatestatus-bz6uecg2pq-rj.a.run.app" 
# -------------------------------------------------------

# Carrega rostos conhecidos
def load_known_faces(known_dir):
    encodings = []
    labels = []
    image_exts = {".jpg", ".jpeg", ".png"}

    if not known_dir.exists():
        print(f"[WARN] Pasta '{known_dir}' não encontrada")
        return encodings, labels

    for person_dir, _, files in os.walk(known_dir):
        person_path = Path(person_dir)
        if person_path == known_dir:
            continue
        person_name = person_path.name
        for f in files:
            if Path(f).suffix.lower() not in image_exts:
                continue
            img_path = person_path / f
            image = face_recognition.load_image_file(str(img_path))
            locations = face_recognition.face_locations(image, model="hog")
            if not locations:
                continue
            encoding = face_recognition.face_encodings(image, known_face_locations=locations)[0]
            encodings.append(encoding)
            labels.append(person_name)
            print(f"[OK] {person_name} <- {img_path}")
    return encodings, labels

# Indexa os rostos conhecidos
known_encodings, known_labels = load_known_faces(KNOWN_DIR)

# Último rosto enviado (para não floodar a API)
last_sent_name = None
last_disciplines = []

# Função auxiliar: notifica backend
def notify_backend(uid):
    global last_sent_name, last_disciplines
    if uid == "Desconhecido":
        return
    if uid == last_sent_name:  # evita enviar repetido
        return
    try:
        payload = {"uid": uid}
        resp = requests.post(API_URL, json=payload, timeout=5)
        data = resp.json()

        # garante que payload seja convertido corretamente
        payload_data = data.get("payload")
        if isinstance(payload_data, str):
            payload_data = json.loads(payload_data)

        disciplinas = []
        if isinstance(payload_data, list):
            for d in payload_data:
                disciplina = d.get("disciplina")
                disciplina_id = d.get("disciplinaId")
                disciplinas.append({
                    "disciplina": disciplina,
                    "disciplinaId": disciplina_id
                })
        last_sent_name = uid
        last_disciplines = disciplinas

        print(f"[API] Usuário {uid}, disciplinas: {disciplinas}")
        
        
    except Exception as e:
        print(f"[API] Erro ao enviar {uid}: {e}")


# Função de streaming de frames
def gen_frames():
    cap = cv2.VideoCapture(0)
    process_toggle = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if process_toggle:
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            encodings = face_recognition.face_encodings(rgb_small, face_locations)
            names = []

            for enc in encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                name = "Desconhecido"
                if len(distances) > 0:
                    best_idx = np.argmin(distances)
                    if distances[best_idx] <= MATCH_TOLERANCE:
                        name = known_labels[best_idx]
                        # 🔹 notifica quando encontrar alguém conhecido
                        notify_backend(name)
                names.append(name)
        process_toggle = not process_toggle

        # Redimensiona coordenadas para frame original
        scale = int(1 / DOWNSCALE)
        for (top, right, bottom, left), name in zip(face_locations, names):
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# View principal
def index(request):
    return render(request, "reconhecimento/index.html")

# Rota do vídeo
def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

def get_disciplines(request):
    return JsonResponse({"uid": last_sent_name, "disciplinas": last_disciplines})

@csrf_exempt
def confirm_discipline(request):
    """
    Recebe POST do frontend com {"uid":..., "disciplinaId":...},
    encaminha para FORWARD_URL e retorna a resposta.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Método não permitido"}, status=405)
    try:
        body = json.loads(request.body.decode("utf-8"))
        uid = body.get("uid")
        disciplina_id = body.get("disciplinaId")
        if not uid or not disciplina_id:
            return JsonResponse({"error": "uid e disciplinaId são obrigatórios"}, status=400)

        payload = {"uid": uid, "disciplinaId": disciplina_id}
        # encaminha para o backend real (porta 3000)
        forward_resp = requests.post(FORWARD_URL, json=payload, timeout=5)

        # tenta parsear JSON de resposta, senão retorna texto
        try:
            forward_json = forward_resp.json()
        except Exception:
            forward_json = forward_resp.text

        return JsonResponse({"status": "ok", "enviado": payload, "resposta": forward_json})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
