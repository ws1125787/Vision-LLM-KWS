import os
import re
import time
import base64
import subprocess

import cv2
import numpy as np

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler



TOKEN = "TOKEN"

REPO_ID = "bartowski/google_gemma-4-E2B-it-GGUF"

GEMMA_MODEL_FILE = (
    "google_gemma-4-E2B-it-Q4_K_M.gguf"
)

MMPROJ_FILE = (
    "mmproj-google_gemma-4-E2B-it-f16.gguf"
)

GEMMA_MODEL_DIR = (
    "src/models/Gemma4"
)

GEMMA_MODEL_PATH = os.path.join(
    GEMMA_MODEL_DIR,
    GEMMA_MODEL_FILE
)

MMPROJ_PATH = os.path.join(
    GEMMA_MODEL_DIR,
    MMPROJ_FILE
)




YOLO_MODEL_PATH = (
    "src/models/YOLO/yolo11n_int8.engine"
)



WHISPER_PATH = (
    "whisper.cpp/build/bin/whisper-cli"
)

WHISPER_MODEL = (
    "whisper.cpp/models/ggml-small.bin"
)

AUDIO_FILE = (
    "src/audio/input.wav"
)

MIC_DEVICE = (
    "plughw:2,0"
)

RECORD_SECONDS = 5


PIPER_PYTHON = (
    ".piper_venv/bin/python"
)

PIPER_MODEL = (
    "src/models/Piper/"
    "ko_KR-kss-medium.onnx"
)

OUTPUT_FILE = (
    "src/audio/response.wav"
)

SPEAKER_DEVICE = (
    "plughw:2,0"
)


CONTEXT_WINDOW = 2048

COMMAND_MAX_TOKENS = 30

VISION_MAX_TOKENS = 10


# YOLO
YOLO_CONFIDENCE = 0.50

YOLO_IOU = 0.50


# HSV 판정 기준
CONFIDENT_THRESHOLD = 0.30

AMBIGUOUS_THRESHOLD = 0.20


# Gemma Vision은 3프레임마다 후보 검사
GEMMA_INTERVAL = 3


# Gemma NO 판정은 2초 후 다시 검사
FALSE_CACHE_TTL = 2.0


# 같은 검출 결과를 계속 말하지 않게 함
VOICE_COOLDOWN = 5.0


# ============================================================
# 6. 폴더 생성
# ============================================================

os.makedirs(
    GEMMA_MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    "src/audio",
    exist_ok=True
)


if not os.path.exists(
    GEMMA_MODEL_PATH
):

    print(
        "[SYSTEM] Gemma 모델 다운로드"
    )

    hf_hub_download(
        repo_id=REPO_ID,
        filename=GEMMA_MODEL_FILE,
        token=TOKEN,
        local_dir=GEMMA_MODEL_DIR
    )


if not os.path.exists(
    MMPROJ_PATH
):

    print(
        "[SYSTEM] mmproj 다운로드"
    )

    hf_hub_download(
        repo_id=REPO_ID,
        filename=MMPROJ_FILE,
        token=TOKEN,
        local_dir=GEMMA_MODEL_DIR
    )


# ============================================================
# 8. Gemma Vision Handler
# ============================================================

print(
    "[SYSTEM] Gemma Vision 로딩..."
)


chat_handler = Gemma4ChatHandler(
    clip_model_path=MMPROJ_PATH
)


llm = Llama(

    model_path=GEMMA_MODEL_PATH,

    chat_handler=chat_handler,

    n_gpu_layers=-1,

    n_ctx=CONTEXT_WINDOW,

    n_batch=32,

    n_ubatch=32,

    verbose=False
)


print(
    "[SYSTEM] Gemma 준비 완료"
)


# ============================================================
# 9. YOLO
# ============================================================

print(
    "[SYSTEM] YOLO 로딩..."
)


yolo = YOLO(
    YOLO_MODEL_PATH
)


print(
    "[SYSTEM] YOLO 준비 완료"
)


COLOR_MAP = {

    # BLUE
    "파란색": "blue",
    "파랑": "blue",
    "청색": "blue",

    "하늘색": "blue",
    "연파랑": "blue",
    "남색": "blue",


    # RED
    "빨간색": "red",
    "빨강": "red",
    "적색": "red",
    "진홍색": "red",


    # GREEN
    "초록색": "green",
    "초록": "green",
    "녹색": "green",
    "연두색": "green",


    # YELLOW
    "노란색": "yellow",
    "노랑": "yellow",


    # ORANGE
    "주황색": "orange",
    "주황": "orange",


    # PURPLE
    "보라색": "purple",
    "보라": "purple",


    # BLACK
    "검은색": "black",
    "검정색": "black",
    "검정": "black",


    # WHITE
    "흰색": "white",
    "하얀색": "white",
    "하양": "white"
}


COLOR_RANGES = {

    "red": [

        (
            np.array(
                [0, 50, 50]
            ),

            np.array(
                [15, 255, 255]
            )
        ),

        (
            np.array(
                [165, 50, 50]
            ),

            np.array(
                [180, 255, 255]
            )
        )
    ],


    "blue": [

        (
            np.array(
                [85, 40, 40]
            ),

            np.array(
                [140, 255, 255]
            )
        )
    ],


    "green": [

        (
            np.array(
                [35, 40, 40]
            ),

            np.array(
                [85, 255, 255]
            )
        )
    ],


    "yellow": [

        (
            np.array(
                [18, 60, 60]
            ),

            np.array(
                [38, 255, 255]
            )
        )
    ],


    "orange": [

        (
            np.array(
                [5, 80, 80]
            ),

            np.array(
                [20, 255, 255]
            )
        )
    ],


    "purple": [

        (
            np.array(
                [135, 40, 40]
            ),

            np.array(
                [165, 255, 255]
            )
        )
    ],


    "black": [

        (
            np.array(
                [0, 0, 0]
            ),

            np.array(
                [180, 255, 60]
            )
        )
    ],


    "white": [

        (
            np.array(
                [0, 0, 180]
            ),

            np.array(
                [180, 60, 255]
            )
        )
    ]
}



def speech_to_text():

    print()
    print(
        "[VOICE] 말씀하세요..."
    )


    try:


        subprocess.run(

            [
                "pasuspender",
                "--",

                "arecord",

                "-D",
                MIC_DEVICE,

                "-f",
                "S16_LE",

                "-r",
                "16000",

                "-c",
                "1",

                "-d",
                str(RECORD_SECONDS),

                AUDIO_FILE,
            ],

            check=True
        )


  
        result = subprocess.run(

            [
                WHISPER_PATH,

                "-m",
                WHISPER_MODEL,

                "-f",
                AUDIO_FILE,

                "-l",
                "ko",

                "--no-timestamps"
            ],

            text=True,

            capture_output=True,

            check=True
        )


        text = (
            result.stdout
            .strip()
        )


        print(
            "[USER]",
            text
        )


        return text


    except subprocess.CalledProcessError as e:

        print(
            "[STT ERROR]",
            e
        )

        return None



def text_to_speech(text):

    print(
        "[AI]",
        text
    )


    try:

        subprocess.run(

            [
                PIPER_PYTHON,

                "-m",
                "piper",

                "-m",
                PIPER_MODEL,

                "-f",
                OUTPUT_FILE,

                "--",
                text
            ],

            check=True
        )


        subprocess.run(

            [
                "aplay",

                "-D",
                SPEAKER_DEVICE,

                OUTPUT_FILE
            ],

            check=True
        )


    except subprocess.CalledProcessError as e:

        print(
            "[TTS ERROR]",
            e
        )



def understand_command(
    user_input
):

    messages = [

        {
            "role": "system",

            "content": """
Instruction:
사용자의 한국어 명령에서
찾고 싶은 사람의 옷 색상을 추출하시오.

Constraint:
반드시 아래 형식으로만 출력하시오.

COLOR=색상

예시:

파란색 옷 입은 사람 찾아줘
COLOR=파란색

저기 하늘색 티 입은 사람 어디 있어?
COLOR=하늘색

남색 옷을 입은 사람 찾아줘
COLOR=남색

연두색 옷 입은 사람 찾아줘
COLOR=연두색

다른 설명이나 문장을 추가하지 마시오.

Output Format:
COLOR=색상
"""
        },

        {
            "role": "user",

            "content":
            user_input
        }
    ]


    try:

        response = (
            llm.create_chat_completion(

                messages=messages,

                max_tokens=(
                    COMMAND_MAX_TOKENS
                ),

                temperature=0.1
            )
        )


        answer = (

            response["choices"][0]
            ["message"]["content"]
            .strip()

        )


        print(
            "[GEMMA COMMAND]",
            answer
        )


        match = re.search(

            r"COLOR\s*=\s*([^\n]+)",

            answer
        )


        if match:

            return (
                match
                .group(1)
                .strip()
            )


    except Exception as e:

        print(
            "[GEMMA COMMAND ERROR]",
            e
        )


    return None



def get_voice_color():

    user_input = (
        speech_to_text()
    )


    if not user_input:

        return None, None


    if any(

        word in user_input

        for word in [

            "종료",
            "끝내",
            "그만"

        ]
    ):

        return (
            "EXIT",
            "EXIT"
        )



    spoken_color = (
        understand_command(
            user_input
        )
    )


    if spoken_color is None:

        text_to_speech(
            "색상을 이해하지 못했습니다."
        )

        return None, None


    target_color = (
        COLOR_MAP.get(
            spoken_color
        )
    )


    if target_color is None:

        text_to_speech(

            f"{spoken_color}은 "
            f"아직 지원하지 않는 "
            f"색상입니다."

        )

        return None, None


    print()
    print(
        "================================"
    )

    print(
        "사용자 색상:",
        spoken_color
    )

    print(
        "OpenCV 계열:",
        target_color
    )

    print(
        "================================"
    )


    text_to_speech(

        f"{spoken_color} 옷을 "
        f"입은 사람을 찾아볼게요."

    )


    return (
        spoken_color,
        target_color
    )


def create_color_mask(
    hsv_image,
    color_name
):

    mask = np.zeros(

        hsv_image.shape[:2],

        dtype=np.uint8
    )


    for (
        lower,
        upper
    ) in COLOR_RANGES[color_name]:


        temp_mask = cv2.inRange(

            hsv_image,

            lower,

            upper
        )


        mask = cv2.bitwise_or(

            mask,

            temp_mask
        )


    return mask



def calculate_color_ratio(
    roi,
    color_name
):

    if (
        roi is None
        or
        roi.size == 0
    ):

        return 0.0


 
    blurred = cv2.GaussianBlur(

        roi,

        (5, 5),

        0
    )



    hsv = cv2.cvtColor(

        blurred,

        cv2.COLOR_BGR2HSV
    )



    mask = create_color_mask(

        hsv,

        color_name
    )


    kernel = np.ones(

        (5, 5),

        dtype=np.uint8
    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_OPEN,

        kernel
    )


    mask = cv2.morphologyEx(

        mask,

        cv2.MORPH_CLOSE,

        kernel
    )


    total_pixels = (
        mask.size
    )


    if total_pixels == 0:

        return 0.0


    color_pixels = (
        cv2.countNonZero(
            mask
        )
    )


    ratio = (

        color_pixels
        /
        total_pixels

    )


    return ratio


FINE_GRAINED_COLORS = {

    "하늘색",
    "연파랑",
    "남색",
    "진홍색",
    "연두색"

}


# ============================================================
# 19. OpenCV Image → Base64
# ============================================================

def image_to_base64(
    image
):

    success, buffer = (
        cv2.imencode(
            ".jpg",
            image
        )
    )


    if not success:

        return None


    image_base64 = (

        base64
        .b64encode(buffer)
        .decode("utf-8")

    )


    return (

        "data:image/jpeg;base64,"
        + image_base64

    )



def ask_gemma_vision(
    upper_body,
    spoken_color
):

    print()
    print(
        "[GEMMA VISION]"
    )

    print(
        "판정 요청:",
        spoken_color
    )


    image_data = (
        image_to_base64(
            upper_body
        )
    )


    if image_data is None:

        print(
            "[ERROR] 이미지 인코딩 실패"
        )

        return False


    messages = [

        {
            "role": "system",

            "content": f"""
Instruction:
주어진 이미지에서 사람이 입고 있는
상의의 주된 색상을 판단하시오.

사용자가 찾고 있는 색상은
'{spoken_color}'이다.

Constraint:
이미지에서 실제로 확인되는 상의의 색상이
'{spoken_color}'과 일치하는지 판단하시오.

조명이나 배경색이 아니라
사람이 입고 있는 상의의 색상만 판단하시오.

반드시 아래 두 단어 중 하나만 출력하시오.

YES
NO

다른 설명이나 문장을 추가하지 마시오.

Output Format:
한 단어.
"""
        },


        {
            "role": "user",

            "content": [

                {
                    "type": "text",

                    "text":
                    f"이 사람이 입고 있는 "
                    f"상의의 주된 색상이 "
                    f"{spoken_color}입니까?"
                },


                {
                    "type": "image_url",

                    "image_url": {

                        "url":
                        image_data

                    }
                }
            ]
        }
    ]


    try:

        response = (
            llm.create_chat_completion(

                messages=messages,

                max_tokens=(
                    VISION_MAX_TOKENS
                ),

                temperature=0.0
            )
        )


        answer = (

            response["choices"][0]
            ["message"]["content"]
            .strip()
            .upper()

        )


        print(
            "[GEMMA VISION RESULT]",
            answer
        )


        if answer.startswith(
            "YES"
        ):

            return True


        return False


    except Exception as e:

        print(
            "[GEMMA VISION ERROR]",
            e
        )

        return False



gemma_cache = {}


def save_gemma_result(
    track_id,
    result
):

    gemma_cache[
        track_id
    ] = {

        "result":
        result,

        "time":
        time.time()

    }


def get_cached_gemma_result(
    track_id
):

    if (
        track_id
        not in gemma_cache
    ):

        return None


    cache = (
        gemma_cache[
            track_id
        ]
    )


    result = (
        cache["result"]
    )


    cache_time = (
        cache["time"]
    )



    if result is True:

        return True



    elapsed = (

        time.time()
        -
        cache_time

    )


    if (
        elapsed
        <
        FALSE_CACHE_TTL
    ):

        return False

    del gemma_cache[
        track_id
    ]


    return None


print()
print(
    "================================"
)

print(
    "AI Clothing Color Finder"
)

print(
    "================================"
)


text_to_speech(

    "찾고 싶은 옷 색깔을 "
    "말씀해주세요."

)


spoken_color = None

target_color = None


while target_color is None:


    (
        spoken_color,
        target_color

    ) = get_voice_color()


    if (
        target_color
        ==
        "EXIT"
    ):

        text_to_speech(
            "프로그램을 종료합니다."
        )

        raise SystemExit



pipeline = (

    "nvarguscamerasrc sensor-id=0 ! "

    "video/x-raw(memory:NVMM), "
    "width=1280, "
    "height=720, "
    "framerate=30/1 ! "

    "nvvidconv ! "

    "video/x-raw, "
    "format=BGRx ! "

    "videoconvert ! "

    "video/x-raw, "
    "format=BGR ! "

    "queue "
    "leaky=downstream "
    "max-size-buffers=1 ! "

    "appsink "
    "drop=true "
    "max-buffers=1 "
    "sync=false"

)


cap = cv2.VideoCapture(

    pipeline,

    cv2.CAP_GSTREAMER
)


if not cap.isOpened():

    raise RuntimeError(
        "카메라를 열 수 없습니다."
    )



frame_count = 0

last_detection_time = 0.0

displayed_fps = 0.0


print()
print(
    "[SYSTEM] 카메라 시작"
)

print(
    "V : 새로운 음성 명령"
)

print(
    "Q : 종료"
)



while True:

    frame_start = (
        time.perf_counter()
    )


    # ========================================================
    # Camera
    # ========================================================

    ret, frame = (
        cap.read()
    )


    if not ret:

        print(
            "[ERROR] Frame 읽기 실패"
        )

        break


    frame_count += 1


    frame_height, frame_width = (
        frame.shape[:2]
    )


  
    results = yolo.track(

        source=frame,

        persist=True,

        conf=YOLO_CONFIDENCE,

        iou=YOLO_IOU,

        classes=[0],

        verbose=False

    )


    detected_ids = set()


 
    for result in results:


        if (
            result.boxes is None
        ):

            continue


        for box in result.boxes:


           
            if box.id is None:

                continue


            track_id = int(
                box.id[0].item()
            )


          
            bbox = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )


            x1, y1, x2, y2 = (
                map(
                    int,
                    bbox
                )
            )


            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame_width,
                x2
            )

            y2 = min(
                frame_height,
                y2
            )


            if (
                x2 <= x1
                or
                y2 <= y1
            ):

                continue


           
            person_roi = frame[
                y1:y2,
                x1:x2
            ]


            if (
                person_roi.size
                ==
                0
            ):

                continue


            person_h, person_w = (
                person_roi.shape[:2]
            )


            

            upper_x1 = int(
                person_w
                *
                0.15
            )

            upper_x2 = int(
                person_w
                *
                0.85
            )


            upper_y1 = int(
                person_h
                *
                0.20
            )

            upper_y2 = int(
                person_h
                *
                0.60
            )


            upper_body = (
                person_roi[
                    upper_y1:upper_y2,
                    upper_x1:upper_x2
                ]
            )


            if (
                upper_body.size
                ==
                0
            ):

                continue


        

            ratio = (
                calculate_color_ratio(

                    upper_body,

                    target_color
                )
            )


            detected = False

            method = (
                "OpenCV"
            )


       
            fine_grained = (

                spoken_color
                in
                FINE_GRAINED_COLORS

            )


        

            if (

                ratio
                >=
                CONFIDENT_THRESHOLD

                and

                not fine_grained

            ):

                detected = True

                method = (
                    "OpenCV"
                )


    
            elif (

                fine_grained

                and

                ratio
                >=
                AMBIGUOUS_THRESHOLD

            ):


                cached = (
                    get_cached_gemma_result(
                        track_id
                    )
                )


                if cached is not None:

                    detected = cached

                    method = (
                        "Gemma Cache"
                    )


                else:

                    method = (
                        "Gemma Waiting"
                    )


                    if (

                        frame_count
                        %
                        GEMMA_INTERVAL

                        ==
                        0

                    ):


                        gemma_result = (
                            ask_gemma_vision(

                                upper_body,

                                spoken_color

                            )
                        )


                        save_gemma_result(

                            track_id,

                            gemma_result

                        )


                        detected = (
                            gemma_result
                        )


                        method = (
                            "Gemma"
                        )


         
            elif (

                ratio
                >=
                AMBIGUOUS_THRESHOLD

            ):


                cached = (
                    get_cached_gemma_result(
                        track_id
                    )
                )


                if cached is not None:

                    detected = (
                        cached
                    )

                    method = (
                        "Gemma Cache"
                    )


                else:

                    method = (
                        "Gemma Waiting"
                    )


                    if (

                        frame_count
                        %
                        GEMMA_INTERVAL

                        ==
                        0

                    ):


                        gemma_result = (
                            ask_gemma_vision(

                                upper_body,

                                spoken_color

                            )
                        )


                        save_gemma_result(

                            track_id,

                            gemma_result

                        )


                        detected = (
                            gemma_result
                        )


                        method = (
                            "Gemma"
                        )


         
            else:

                detected = False

                method = (
                    "Rejected"
                )


           
            abs_ux1 = (
                x1
                +
                upper_x1
            )

            abs_uy1 = (
                y1
                +
                upper_y1
            )

            abs_ux2 = (
                x1
                +
                upper_x2
            )

            abs_uy2 = (
                y1
                +
                upper_y2
            )


            cv2.rectangle(

                frame,

                (
                    abs_ux1,
                    abs_uy1
                ),

                (
                    abs_ux2,
                    abs_uy2
                ),

                (
                    255,
                    255,
                    0
                ),

                1
            )


           
            if detected:


                detected_ids.add(
                    track_id
                )


                cv2.rectangle(

                    frame,

                    (
                        x1,
                        y1
                    ),

                    (
                        x2,
                        y2
                    ),

                    (
                        0,
                        255,
                        0
                    ),

                    3
                )


                label = (

                    f"ID:{track_id} "

                    f"{ratio * 100:.1f}% "

                    f"[{method}]"

                )


                cv2.putText(

                    frame,

                    label,

                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.60,

                    (
                        0,
                        255,
                        0
                    ),

                    2
                )


           
            elif (

                ratio
                >=
                AMBIGUOUS_THRESHOLD

            ):


                cv2.rectangle(

                    frame,

                    (
                        x1,
                        y1
                    ),

                    (
                        x2,
                        y2
                    ),

                    (
                        0,
                        255,
                        255
                    ),

                    2
                )


                label = (

                    f"ID:{track_id} "

                    f"{ratio * 100:.1f}% "

                    f"[{method}]"

                )


                cv2.putText(

                    frame,

                    label,

                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (
                        0,
                        255,
                        255
                    ),

                    2
                )


 
    current_time = (
        time.time()
    )


    if (

        len(detected_ids)
        >
        0

        and

        current_time
        -
        last_detection_time

        >=
        VOICE_COOLDOWN

    ):


        count = (
            len(
                detected_ids
            )
        )


        if count == 1:

            text_to_speech(

                f"{spoken_color} 옷을 "
                f"입은 사람을 "
                f"찾았습니다."

            )


        else:

            text_to_speech(

                f"{spoken_color} 옷을 "
                f"입은 사람을 "
                f"{count}명 "
                f"찾았습니다."

            )


        last_detection_time = (
            current_time
        )


 
    elapsed = (

        time.perf_counter()
        -
        frame_start

    )


    if elapsed > 0:

        current_fps = (
            1.0
            /
            elapsed
        )


        if (
            displayed_fps
            ==
            0
        ):

            displayed_fps = (
                current_fps
            )


        else:

            displayed_fps = (

                0.9
                *
                displayed_fps

                +

                0.1
                *
                current_fps

            )



    cv2.putText(

        frame,

        f"Target: {target_color}",

        (
            20,
            40
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        (
            0,
            255,
            255
        ),

        2
    )


    cv2.putText(

        frame,

        f"FPS: {displayed_fps:.1f}",

        (
            20,
            75
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (
            0,
            255,
            0
        ),

        2
    )


    cv2.putText(

        frame,

        "V: Voice / Q: Quit",

        (
            20,
            110
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (
            255,
            255,
            255
        ),

        2
    )


    cv2.imshow(

        "AI Clothing Color Finder",

        frame
    )


    # ========================================================
    # 29. Keyboard
    # ========================================================

    key = (

        cv2.waitKey(1)
        &
        0xFF

    )


    # ========================================================
    # Q
    # ========================================================

    if key == ord("q"):


        text_to_speech(
            "프로그램을 종료합니다."
        )

        break



    elif key == ord("v"):


        text_to_speech(

            "찾고 싶은 색깔을 "
            "말씀해주세요."

        )


        (
            new_spoken_color,
            new_target_color

        ) = get_voice_color()


      
        if (

            new_target_color
            ==
            "EXIT"

        ):

            text_to_speech(
                "프로그램을 종료합니다."
            )

            break


   
        if (
            new_target_color
            is not None
        ):


            spoken_color = (
                new_spoken_color
            )


            target_color = (
                new_target_color
            )


        
            gemma_cache.clear()


          
            last_detection_time = 0.0


            print()
            print(
                "================================"
            )

            print(
                "새 Target:",
                spoken_color
            )

            print(
                "OpenCV 계열:",
                target_color
            )

            print(
                "Gemma Cache 초기화"
            )

            print(
                "================================"
            )


# ============================================================
# 30. 종료
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "[SYSTEM] 프로그램 종료"
)
