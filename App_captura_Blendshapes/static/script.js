import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
const videoBlendShapes = document.getElementById("video-blend-shapes");
const emotionButtons = document.querySelectorAll(".emotionButton");
let faceLandmarker; 
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
let detectingEmotion = false;
const videoWidth = 640;
let countdownInterval; 

// detector faceLandmarker de MediaPipe
async function createFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
    });
}
createFaceLandmarker();

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const timerElement = document.getElementById("timer"); 


function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
// añadimos event listener al botón de activar cámara
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}
// función para habilitar o deshabilitar la cámara
function enableCam(event) {
    if (webcamRunning) {
        webcamRunning = false; // desactiva cámra
        enableWebcamButton.innerText = "Activar Cámara";
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
        }
        liveView.style.display = "none";
        clearInterval(countdownInterval);
        timerElement.textContent = ""; 
    } else {
        webcamRunning = true; // activa cámara
        enableWebcamButton.innerText = "Desactivar Cámara";
        liveView.style.display = "block";

        const constraints = {
            video: true
        };

        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                // la webcam está activa, pero no comenzamos la detección aún.
            });
        }).catch((error) => {
            console.error("Error accessing the webcam:", error);
        });
    }
}
// añadimos event listeners a los botones de emoción
emotionButtons.forEach(button => {
    button.addEventListener("click", () => {
        const emotion = button.getAttribute("data-emotion");
        startDetection(emotion); //comenzamos la detección cuando se pulsa un boton de emocion
    });
});

const detectionTime = 5; // tiempo de detección en segundos
// función para comenzar con la detección facial
function startDetection(emotion) {
    if (!faceLandmarker) {
        console.log("Wait! faceLandmarker not loaded yet.");
        return;
    }
    if (!webcamRunning) {
        console.log("Webcam is not running.");
        return;
    }
    detectingEmotion = true;
    clearInterval(countdownInterval); // reiniciamos temporizador (cuenta regresiva)

    let timeRemaining = detectionTime; 
    timerElement.textContent = `Tiempo restante: ${timeRemaining}s`;

    countdownInterval = setInterval(() => {
        timeRemaining -= 1;
        timerElement.textContent = `Tiempo restante: ${timeRemaining}s`;
        if (timeRemaining <= 0) {
            clearInterval(countdownInterval);
            stopDetection();
        }
    }, 1000);

    predictWebcam(emotion); // pasamos la emoción a la función predictWebcam
}


// función para detener la detección de emociones
function stopDetection() {
    detectingEmotion = false
    clearCanvas();
    clearInterval(countdownInterval); 
    
}

// función para limpiar el canvas de la imagen 
function clearCanvas() {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}

let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);

async function predictWebcam(emotion) {
    if (!detectingEmotion) {
        return;
    }
    
    const radio = video.videoHeight / video.videoWidth;
    video.style.width = videoWidth + "px";
    video.style.height = videoWidth * radio + "px";
    canvasElement.style.width = videoWidth + "px";
    canvasElement.style.height = videoWidth * radio + "px";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceLandmarker.setOptions({ runningMode: runningMode });
    }

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = faceLandmarker.detectForVideo(video, startTimeMs);
    }

    if (results.faceLandmarks) {
        for (const landmarks of results.faceLandmarks) {
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: "#30FF30" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#E0E0E0" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: "#FF3030" });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: "#30FF30" });
        }
        // enviamos los valores blendshape a app.py
        if (results.faceBlendshapes) {
            sendBlendShapesData(results.faceBlendshapes, emotion);
        }
    }
    drawBlendShapes(videoBlendShapes, results.faceBlendshapes);


    if (webcamRunning && detectingEmotion) {
        window.requestAnimationFrame(() => predictWebcam(emotion)); 
    }
}


function drawBlendShapes(el, blendShapes) {
    if (!blendShapes.length) {
        return;
    }
    console.log(blendShapes[0]);
    let htmlMaker = "";
    blendShapes[0].categories.map((shape) => {
        htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
        <span class="blend-shapes-value" style="width: calc(${+shape.score * 100}% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
    });
    el.innerHTML = htmlMaker;
}

// pasamos la información que introduce el usuario 
function sendBlendShapesData(blendshapes, emotion) {
    const userInfoForm = document.getElementById("userInfoForm");
    const formData = new FormData(userInfoForm);

    const blendshapeData = blendshapes[0].categories.map(shape => ({
        name: shape.categoryName,
        //score: shape.score.toFixed(4) //Para redondear a 4 (igual que lo visualizamos en el gráfico)
        score: shape.score
    }));

    const data = {
        Identificador: formData.get("Identificador"),
        Edad: formData.get("Edad"),
        Sexo: formData.get("Sexo"),
        Patologia: formData.get("Patologia"),
        Emocion: emotion,
        blendshapes: blendshapeData

    };
    // obtenemos el ID del dataset por separado, ya que no queremos incluirlo dentro del dataset
    const DatasetID = formData.get("DatasetID");

    fetch('/save_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({data, DatasetID}) // enviamos data y datasetid por separado
    })
    .then(response => response.json())
    .then(data => console.log('Success:', data))
    .catch((error) => {
        console.error('Error:', error);
    });
}
