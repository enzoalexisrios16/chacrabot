import math
import time
import cv2
import numpy as np
import threading
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI(title="Sistema de Monitoreo de Ganado")

# Cargar modelo YOLOv8 Nano (se descarga automáticamente en el primer uso)
model = YOLO("yolov8n.pt")

# Estado compartido en memoria
system_state = {
    "cow_count": 0,          # vacas detectadas en el fotograma actual
    "total_ingresos": 0,     # acumulador: total histórico de vacas que cruzaron la cerca hacia adentro
    "total_egresos": 0,      # acumulador: total histórico de vacas que cruzaron la cerca hacia afuera
    "vacas_adentro": 0,      # cantidad de vacas actualmente adentro del corral (ingresos - egresos)
    "fps": 0.0,              # cuadros por segundo reales que está entregando el sistema
    "status": "Iniciando..."
}
lock = threading.Lock()

# 0 para cámara web integrada/USB.
# Si usas una cámara IP, reemplaza 0 por la URL RTSP (ej: "rtsp://admin:pass@192.168.1.50:554/stream")
CAMERA_SOURCE = 0

# Las cámaras integradas de notebook suelen entregar el video "en espejo"
# (como te ves vos, invertido izquierda-derecha respecto a la escena real).
# Con esto en True se corrige. Si tu cámara no viene espejada, poné False.
FLIP_CAMERA_HORIZONTAL = True

# --- Calidad de imagen y rendimiento ---
# Resolución que se le pide a la cámara (si la cámara no la soporta, usa la más cercana).
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Tamaño interno que usa YOLO para inferir (no afecta la nitidez de lo que se ve:
# los cuadros se dibujan siempre sobre el fotograma a resolución completa). Un
# número más chico = más FPS pero puede perder vacas muy lejanas/pequeñas.
# Valores típicos: 320 (rápido), 480 (equilibrado), 640 (más preciso, más lento).
YOLO_IMGSZ = 480

# Calidad del JPEG que se transmite por el streaming (0-100). Más alto = mejor
# imagen pero cuadros más pesados.
JPEG_QUALITY = 90

# --- Configuración de la cerca (línea virtual que separa "afuera" de "adentro") ---
# Se definen como fracciones del ancho/alto del fotograma para que funcione con
# cualquier resolución de cámara. Por defecto es una línea horizontal al 60% de la altura.
# Cambiá estos puntos si tu cerca está en otra posición o es diagonal.
LINE_P1_FRAC = (0.0, 0.6)
LINE_P2_FRAC = (1.0, 0.6)

# Si el conteo de "entradas" queda invertido (suma cuando la vaca sale, no cuando entra),
# cambiá este valor a -1.
INSIDE_SIGN = 1

# Banda muerta (en píxeles) a cada lado de la cerca. Una vaca sólo se confirma
# "adentro" o "afuera" cuando su centro se aleja más de este margen de la línea;
# mientras esté justo sobre la cerca no se toca su estado. Esto evita contar
# entradas/salidas de más por el "temblor" normal de la detección cuadro a cuadro.
# Si notás cruces falsos, subilo (ej: 25); si tarda en detectar cruces reales, bajalo.
CROSS_MARGIN_PX = 15

# Cuántos cuadros consecutivos tiene que sostenerse el nuevo lado antes de darlo
# por confirmado y sumar/restar. Más alto = más robusto contra falsos positivos,
# pero un poco más lento para registrar el cruce real.
CONFIRM_FRAMES = 3

# --- Imagen del cerco que se dibuja sobre la línea (reemplaza la raya amarilla) ---
FENCE_IMAGE_PATH = "fence.png"   # tile PNG con canal alfa (transparente entre tablas/postes)
FENCE_HEIGHT_PX = 70             # alto en pixeles al que se escala el cerco sobre el video

# Estado de tracking: por cada ID de vaca...
track_side = {}      # lado CONFIRMADO (True = adentro, False = afuera)
pending_side = {}     # lado que se está evaluando todavía (aún no confirmado)
pending_count = {}    # cuántos cuadros seguidos lleva ese lado pendiente
inside_ids = set()    # IDs actualmente confirmadas como "adentro"


def signed_distance_to_line(px, py, x1, y1, x2, y2):
    """Distancia con signo (en píxeles) del punto a la línea: positivo de un lado,
    negativo del otro. El signo (no la magnitud) indica de qué lado está el punto."""
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return cross / length


def load_fence_tile():
    """Carga el tile del cerco y lo espeja en el eje horizontal (arriba-abajo)."""
    tile = cv2.imread(FENCE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if tile is None:
        return None
    if tile.shape[2] == 3:  # por si el PNG no trae canal alfa
        alpha = np.full(tile.shape[:2], 255, dtype=tile.dtype)
        tile = np.dstack([tile, alpha])
    # Espejado en el eje horizontal (flipCode=0 = espejo arriba/abajo).
    # Si en tu caso lo necesitás espejado izquierda/derecha, usá flipCode=1.
    tile = cv2.flip(tile, 0)
    return tile


def build_fence_strip(tile, length_px, height_px):
    """Escala el tile a la altura deseada y lo repite (tiling) hasta cubrir 'length_px'."""
    th, tw = tile.shape[:2]
    scale = height_px / th
    new_w = max(1, int(round(tw * scale)))
    tile_resized = cv2.resize(tile, (new_w, height_px), interpolation=cv2.INTER_AREA)
    reps = int(math.ceil(length_px / new_w)) + 1
    tiled = np.tile(tile_resized, (1, reps, 1))
    return tiled[:, :max(1, int(length_px)), :]


def rotate_rgba(img, angle_deg):
    """Rota una imagen RGBA manteniendo la transparencia fuera del área rotada."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )


def overlay_rgba(background, overlay, x, y):
    """Pega 'overlay' (RGBA) sobre 'background' (BGR) en la posición (x, y), con alpha-blend."""
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1e, y1e = min(x + ow, bw), min(y + oh, bh)
    if x0 >= x1e or y0 >= y1e:
        return

    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1e - x0), oy0 + (y1e - y0)

    overlay_crop = overlay[oy0:oy1, ox0:ox1]
    alpha = (overlay_crop[:, :, 3:4].astype(np.float32)) / 255.0
    bg_region = background[y0:y1e, x0:x1e].astype(np.float32)
    blended = overlay_crop[:, :, :3].astype(np.float32) * alpha + bg_region * (1 - alpha)
    background[y0:y1e, x0:x1e] = blended.astype(np.uint8)


class CameraStream:
    """Lee la cámara en un hilo aparte y siempre deja disponible el último
    fotograma. Así el bucle principal (inferencia + dibujo) nunca se queda
    esperando a que la cámara entregue el siguiente cuadro: eso es lo que más
    FPS le come a este tipo de pipelines."""

    def __init__(self, source, width, height, fps):
        self.cap = cv2.VideoCapture(source)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                # Buffer chico = menos latencia (algunos backends no lo soportan).
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None

    def is_opened(self):
        return self.cap.isOpened()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.cap.release()


def get_video_stream():
    """Captura video, ejecuta inferencia de YOLOv8, trackea vacas cruzando la cerca
    y genera el streaming MJPEG."""
    camera = CameraStream(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)

    if not camera.is_opened():
        with lock:
            system_state["status"] = "Error al abrir la cámara"
        return
    camera.start()

    with lock:
        system_state["status"] = "Monitoreando en vivo"

    line_pixels = None       # se calcula una vez que conocemos el tamaño real del fotograma
    fence_overlay = None     # imagen del cerco ya escalada/rotada, lista para pegar cada frame
    fence_pos = (0, 0)       # esquina superior izquierda donde se pega el cerco
    fence_tile = load_fence_tile()
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    fps_smoothed = 0.0
    last_tick = time.time()
    frames_without_camera = 0

    while True:
        frame = camera.read()
        if frame is None:
            frames_without_camera += 1
            if frames_without_camera > 150:  # ~varios segundos sin poder leer la cámara
                with lock:
                    system_state["status"] = "Señal perdida"
                break
            time.sleep(0.01)
            continue
        frames_without_camera = 0

        if FLIP_CAMERA_HORIZONTAL:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        if line_pixels is None:
            x1 = int(LINE_P1_FRAC[0] * w)
            y1 = int(LINE_P1_FRAC[1] * h)
            x2 = int(LINE_P2_FRAC[0] * w)
            y2 = int(LINE_P2_FRAC[1] * h)
            line_pixels = (x1, y1, x2, y2)

            if fence_tile is not None:
                length = math.hypot(x2 - x1, y2 - y1)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                strip = build_fence_strip(fence_tile, length, FENCE_HEIGHT_PX)
                fence_overlay = rotate_rgba(strip, angle)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                fence_pos = (
                    int(mx - fence_overlay.shape[1] / 2),
                    int(my - fence_overlay.shape[0] / 2),
                )

        lx1, ly1, lx2, ly2 = line_pixels

        # Inferencia con YOLOv8 + tracking:
        # classes=[19] filtra únicamente la clase "cow" del dataset COCO
        # imgsz achica lo que YOLO procesa internamente (más FPS); las cajas que
        # devuelve igual quedan en las coordenadas del fotograma a resolución completa.
        results = model.track(frame, classes=[19], conf=0.45, persist=True, verbose=False, imgsz=YOLO_IMGSZ)

        boxes = results[0].boxes
        current_cows = len(boxes)

        # --- Detección de cruce de cerca por cada vaca trackeada ---
        seen_ids = set()
        if boxes.id is not None:
            ids = boxes.id.int().tolist()
            xyxy = boxes.xyxy.tolist()
            seen_ids = set(ids)

            for track_id, (bx1, by1, bx2, by2) in zip(ids, xyxy):
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2

                dist = signed_distance_to_line(cx, cy, lx1, ly1, lx2, ly2) * INSIDE_SIGN

                if dist > CROSS_MARGIN_PX:
                    observed_side = True   # claramente adentro
                elif dist < -CROSS_MARGIN_PX:
                    observed_side = False  # claramente afuera
                else:
                    observed_side = None   # zona ambigua, justo sobre la cerca: no decide nada

                if observed_side is None:
                    continue

                confirmed_side = track_side.get(track_id)

                if confirmed_side is None:
                    # Primera vez que vemos esta vaca de forma clara: registramos su
                    # lado sin contar cruce (para no inflar el acumulador con vacas
                    # que ya estaban ahí desde el arranque).
                    track_side[track_id] = observed_side
                    pending_side[track_id] = None
                    pending_count[track_id] = 0
                    if observed_side:
                        inside_ids.add(track_id)
                    continue

                if observed_side == confirmed_side:
                    # Sigue del mismo lado: se cancela cualquier cruce que estuviera
                    # a mitad de confirmar (por ejemplo, se acercó a la cerca y volvió).
                    pending_side[track_id] = None
                    pending_count[track_id] = 0
                    continue

                # El lado observado difiere del confirmado: no se suma/resta todavía,
                # primero tiene que sostenerse CONFIRM_FRAMES cuadros seguidos.
                if pending_side.get(track_id) == observed_side:
                    pending_count[track_id] = pending_count.get(track_id, 0) + 1
                else:
                    pending_side[track_id] = observed_side
                    pending_count[track_id] = 1

                if pending_count[track_id] < CONFIRM_FRAMES:
                    continue

                # Cruce confirmado: se actualiza el estado y se suma/resta una sola vez.
                track_side[track_id] = observed_side
                pending_side[track_id] = None
                pending_count[track_id] = 0

                if observed_side:
                    inside_ids.add(track_id)
                    with lock:
                        system_state["total_ingresos"] += 1
                    print(f"🐄 Vaca #{track_id} cruzó la cerca hacia ADENTRO. "
                          f"Total ingresos: {system_state['total_ingresos']}")
                else:
                    inside_ids.discard(track_id)
                    with lock:
                        system_state["total_egresos"] += 1
                    print(f"🐄 Vaca #{track_id} cruzó la cerca hacia AFUERA. "
                          f"Total egresos: {system_state['total_egresos']}")

        # Vacas que el tracker dejó de ver (salieron de cámara, oclusión larga, etc.):
        # se limpia su estado pendiente para que no arrastren un cruce a medio confirmar.
        for stale_id in list(pending_side.keys() - seen_ids):
            pending_side.pop(stale_id, None)
            pending_count.pop(stale_id, None)

        # FPS reales (promedio suavizado para que no salte de más)
        now = time.time()
        delta = now - last_tick
        last_tick = now
        if delta > 0:
            instant_fps = 1.0 / delta
            fps_smoothed = instant_fps if fps_smoothed == 0 else (0.9 * fps_smoothed + 0.1 * instant_fps)

        with lock:
            system_state["cow_count"] = current_cows
            system_state["vacas_adentro"] = len(inside_ids)
            system_state["fps"] = round(fps_smoothed, 1)

        # Dibujar detecciones y etiquetas en el fotograma
        annotated_frame = results[0].plot()

        # Dibujar el cerco (imagen) sobre la cerca; si no se pudo cargar la imagen,
        # se usa una línea amarilla como respaldo.
        if fence_overlay is not None:
            overlay_rgba(annotated_frame, fence_overlay, *fence_pos)
        else:
            cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)

        cv2.putText(annotated_frame,
                    f"Adentro: {len(inside_ids)}  Ingresos: {system_state['total_ingresos']}  "
                    f"Egresos: {system_state['total_egresos']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Codificar el cuadro a JPEG para transmisión web
        ret, buffer = cv2.imencode('.jpg', annotated_frame, jpeg_params)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.stop()

@app.get("/video_feed")
def video_feed():
    """Endpoint de transmisión continua de video."""
    return StreamingResponse(
        get_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/stats")
def get_stats():
    """API para consultar conteo y estado actual."""
    with lock:
        return system_state

@app.get("/", response_class=HTMLResponse)
def index():
    """Interfaz web de monitoreo."""
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Panel de Monitoreo - Detección de Ganado</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }
            body { background: #0f172a; color: #f8fafc; padding: 24px; }
            .container { max-width: 1100px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
            header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; }
            h1 { font-size: 1.5rem; font-weight: 600; color: #38bdf8; }
            .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
            .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; overflow: hidden; padding: 16px; }
            .video-container { display: flex; justify-content: center; align-items: center; background: #000; border-radius: 8px; overflow: hidden; }
            img { width: 100%; height: auto; display: block; object-fit: cover; }
            .metric-box { display: flex; flex-direction: column; gap: 16px; }
            .metric { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 20px; text-align: center; }
            .metric h3 { font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
            .metric p { font-size: 3rem; font-weight: 700; color: #22c55e; margin-top: 8px; }
            .metric.accum p { color: #f59e0b; }
            .metric.egresos p { color: #ef4444; }
            .metric.inside p { color: #38bdf8; }
            .status-badge { display: inline-block; padding: 6px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; background: #0284c7; color: #fff; }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Detector de Vacas en Tiempo Real</h1>
                <span id="status-badge" class="status-badge">Conectando...</span>
            </header>
            <div class="grid">
                <div class="card">
                    <div class="video-container">
                        <img src="/video_feed" alt="Video Feed" />
                    </div>
                </div>
                <div class="card metric-box">
                    <div class="metric">
                        <h3>Vacas en Pantalla</h3>
                        <p id="counter">0</p>
                    </div>
                    <div class="metric inside">
                        <h3>Vacas Adentro del Corral</h3>
                        <p id="inside-counter">0</p>
                    </div>
                    <div class="metric accum">
                        <h3>Total Ingresos Registrados</h3>
                        <p id="accum-counter">0</p>
                    </div>
                    <div class="metric egresos">
                        <h3>Total Egresos Registrados</h3>
                        <p id="egresos-counter">0</p>
                    </div>
                    <div class="metric">
                        <h3>FPS</h3>
                        <p id="fps-counter" style="font-size: 2rem;">0</p>
                    </div>
                    <div class="metric">
                        <h3>Estado del Sistema</h3>
                        <p id="state-text" style="font-size: 1.2rem; color: #cbd5e1; margin-top: 12px;">Activo</p>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function updateStats() {
                try {
                    const res = await fetch('/stats');
                    const data = await res.json();
                    document.getElementById('counter').innerText = data.cow_count;
                    document.getElementById('inside-counter').innerText = data.vacas_adentro;
                    document.getElementById('accum-counter').innerText = data.total_ingresos;
                    document.getElementById('egresos-counter').innerText = data.total_egresos;
                    document.getElementById('fps-counter').innerText = data.fps;
                    document.getElementById('state-text').innerText = data.status;
                    document.getElementById('status-badge').innerText = data.status;
                } catch (err) {
                    document.getElementById('status-badge').innerText = 'Sin conexión';
                }
            }
            setInterval(updateStats, 1000);
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    # host="127.0.0.1": el panel sólo se ve desde esta misma computadora.
    # Si más adelante querés entrar desde otro equipo de la red local, volvé a "0.0.0.0".
    uvicorn.run(app, host="127.0.0.1", port=8000)
