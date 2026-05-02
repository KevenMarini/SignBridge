const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
const predictionText = document.getElementById('prediction_text');
const connectionStatus = document.getElementById('connection_status');
const historyList = document.getElementById('history_list');
const startBtn = document.getElementById('start_btn');
const stopBtn = document.getElementById('stop_btn');

// Socket.IO Setup
const socket = io();

socket.on('connect', () => {
    connectionStatus.innerText = 'Connected';
    connectionStatus.classList.remove('offline');
    connectionStatus.classList.add('online');
});

socket.on('disconnect', () => {
    connectionStatus.innerText = 'Disconnected';
    connectionStatus.classList.remove('online');
    connectionStatus.classList.add('offline');
});

socket.on('response', (data) => {
    if (data.prediction) {
        updatePrediction(data.prediction);
    } else if (data.error) {
        console.error('Server error:', data.error);
    }
});

let lastPrediction = '';
let predictionTimeout = null;

function updatePrediction(prediction) {
    predictionText.innerText = prediction;
    
    // Add to history if it's a new sign and stable
    if (prediction !== lastPrediction && prediction !== "Waiting...") {
        lastPrediction = prediction;
        addToHistory(prediction);
    }
}

function addToHistory(sign) {
    const item = document.createElement('div');
    item.className = 'history-item';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    item.innerHTML = `<span class="word">${sign}</span> <span class="time">${time}</span>`;
    historyList.prepend(item);
    
    // Keep only last 10
    if (historyList.children.length > 10) {
        historyList.removeChild(historyList.lastChild);
    }
}

// MediaPipe Setup
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

function onResults(results) {
    // Resize canvas to match video
    if (canvasElement.width !== videoElement.videoWidth) {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // We draw the video ourselves if we want mirror effect
    canvasCtx.translate(canvasElement.width, 0);
    canvasCtx.scale(-1, 1);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#6366f1', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: '#ffffff', lineWidth: 2, radius: 4});
            
            // Extract landmarks for inference
            const landmarkData = [];
            landmarks.forEach(lm => {
                landmarkData.push(lm.x, lm.y, lm.z);
            });
            
            // Send to server
            socket.emit('inference', { landmarks: landmarkData });
        }
    } else {
        predictionText.innerText = 'No Hand Detected';
    }
    canvasCtx.restore();
}

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 640,
    height: 480
});

startBtn.addEventListener('click', () => {
    camera.start();
    predictionText.innerText = 'Model Initializing...';
});

stopBtn.addEventListener('click', () => {
    camera.stop();
    predictionText.innerText = 'Camera Stopped';
});
