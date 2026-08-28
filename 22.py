from ultralytics import YOLO
import cv2
import time
import subprocess
from llama_cpp import Llama


model = YOLO("src/models/YOLO/yolo11n_int8.engine")

pipeline = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

displayed_fps = 0.0

while True:
    start_time = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    results = model.predict(
        source=frame,   # source image
        conf=0.25,      # Confidence Threshold
        iou=0.5,        # IoU Threshold
        verbose=False,  # no output prints
        classes=[0],   # selected class
    )

    output_frame = results[0].plot()

    elapsed_time = time.perf_counter() - start_time
    current_fps = 1.0 / elapsed_time

    if displayed_fps == 0:
        displayed_fps = current_fps
    else:
        displayed_fps = 0.9 * displayed_fps + 0.1 * current_fps

    cv2.putText(output_frame, f"FPS: {displayed_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("YOLO Object Detection with FPS", output_frame)

cap.release()
cv2.destroyAllWindows()



MIC_DEVICE = "plughw:3,0"
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
RECORD_SECONDS = 5



subprocess.run(
    [
        "pasuspender", "--",
        "arecord",
        "-D", MIC_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "1",
        "-d", str(RECORD_SECONDS),
        AUDIO_FILE,
    ],
    check=True,
)

result = subprocess.run(
    [
        WHISPER_PATH,
        "-m", WHISPER_MODEL,
        "-f", AUDIO_FILE,
        "-l", "ko",
        "--no-timestamps",
    ],
    text=True,
    capture_output=True,
    check=True,
)


text = result.stdout.strip()


MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"


llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,   # GPU 가속 사용
    n_ctx=2048,        # Context Window 크기
    n_batch=32,        # 한 번의 decode 요청에서 처리할 수 있는 최대 token 수
    n_ubatch=32,       # GPU/CPU가 한 번에 계산하는 token 묶음의 최대 크기
    verbose=False,
)