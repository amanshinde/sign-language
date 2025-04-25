document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('videoInput');
    const canvasElement = document.getElementById('canvasOutput');
    const startBtn = document.getElementById('startBtn');
    const speakBtn = document.getElementById('speakBtn');
    const statusMessage = document.getElementById('statusMessage');
    const gestureOutput = document.getElementById('gestureOutput');
    const translationOutput = document.getElementById('translationOutput');
    const languageSelect = document.getElementById('languageSelect');
    
    // Add progress bar for frame collection
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress-container';
    progressContainer.style.width = '100%';
    progressContainer.style.height = '10px';
    progressContainer.style.backgroundColor = '#f0f0f0';
    progressContainer.style.borderRadius = '5px';
    progressContainer.style.margin = '10px 0';
    progressContainer.style.display = 'none';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'progress-bar';
    progressBar.style.width = '0%';
    progressBar.style.height = '100%';
    progressBar.style.backgroundColor = '#4CAF50';
    progressBar.style.borderRadius = '5px';
    progressBar.style.transition = 'width 0.3s ease';
    
    progressContainer.appendChild(progressBar);
    document.querySelector('.recognition-panel').prepend(progressContainer);

    let isStreaming = false;
    let isRecognizing = false;
    let recognitionInterval;
    let videoRenderInterval;
    const ctx = canvasElement.getContext('2d', { alpha: false }); // Disable alpha for better performance
    
    // For aborting fetch requests
    let abortController = null;
    let processingFrame = false;
    
    // Generate a unique client ID for this session
    const clientId = 'client_' + Date.now();
    console.log("App initialized with client ID:", clientId);
    
    // Frame processing settings
    const frameInterval = 25; // Send a frame every 100ms (10 frames per second) - reduced for less lag
    const videoRenderRate = 30; // Render video at 30fps for smooth display

    // Dictionary for translations - updated to match the actual model actions
    const translations = {
        'Alright': {
            'en': 'Alright',
            'hi': 'ठीक है',
            'kn': 'ಸರಿ',
            'mr': 'ठीक आहे'
        },
        'Good Afternoon': {
            'en': 'Good Afternoon',
            'hi': 'शुभ दोपहर',
            'kn': 'ಶುಭ ಮಧ್ಯಾಹ್ನ',
            'mr': 'शुभ दुपार'
        },
        'Good Evening': {
            'en': 'Good Evening',
            'hi': 'शुभ संध्या',
            'kn': 'ಶುಭ ಸಂಜೆ',
            'mr': 'शुभ संध्याकाळ'
        },
        'Good Morning': {
            'en': 'Good Morning',
            'hi': 'सुप्रभात',
            'kn': 'ಶುಭೋದಯ',
            'mr': 'शुभ सकाळ'
        },
        'Good Night': {
            'en': 'Good Night',
            'hi': 'शुभ रात्रि',
            'kn': 'ಶುಭ ರಾತ್ರಿ',
            'mr': 'शुभ रात्री'
        },
        'Hello': {
            'en': 'Hello',
            'hi': 'नमस्ते',
            'kn': 'ನಮಸ್ಕಾರ',
            'mr': 'नमस्कार'
        },
        'How Are You': {
            'en': 'How Are You',
            'hi': 'आप कैसे हैं',
            'kn': 'ನೀವು ಹೇಗಿದ್ದೀರಿ',
            'mr': 'तू कसा आहेस'
        },
        'Pleased': {
            'en': 'Pleased to meet you',
            'hi': 'आपसे मिलकर खुशी हुई',
            'kn': 'ನಿಮ್ಮನ್ನು ಭೇಟಿಯಾಗಲು ಸಂತೋಷವಾಗಿದೆ',
            'mr': 'तुला भेटून आनंद झाला'
        },
        'Thank You': {
            'en': 'Thank You',
            'hi': 'धन्यवाद',
            'kn': 'ಧನ್ಯವಾದಗಳು',
            'mr': 'धन्यवाद'
        }
    };

    // Start/Stop camera stream
    startBtn.addEventListener('click', async () => {
        if (!isStreaming) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: 640,
                        height: 480,
                        facingMode: 'user'
                    } 
                });
                videoElement.srcObject = stream;
                isStreaming = true;
                startBtn.textContent = 'Stop Camera';
                statusMessage.textContent = 'Camera active - Collecting frames for gesture recognition...';
                progressContainer.style.display = 'block';
                console.log("Camera started");
                
                // Set canvas dimensions to match video
                canvasElement.width = 640;
                canvasElement.height = 480;
                
                // Create new abort controller
                abortController = new AbortController();
                
                // Start video rendering (separate from frame processing)
                startVideoRendering();
                
                // Start recognition
                startRecognition();
            } catch (err) {
                console.error('Error accessing camera:', err);
                statusMessage.textContent = 'Error: Could not access camera';
            }
        } else {
            // First stop recognition to prevent further frame processing
            stopRecognition();
            
            // Stop video rendering
            stopVideoRendering();
            
            // Then stop the camera
            const stream = videoElement.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            videoElement.srcObject = null;
            
            // Clear the canvas
            ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
            
            // Reset UI
            isStreaming = false;
            startBtn.textContent = 'Start Camera';
            statusMessage.textContent = 'Camera stopped';
            progressContainer.style.display = 'none';
            
            // Notify server that this client has stopped
            try {
                fetch('http://localhost:8000/stop', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        clientId: clientId
                    })
                }).catch(error => {
                    // Ignore errors when stopping
                    console.log("Error notifying server of stop (can be ignored):", error);
                });
            } catch (error) {
                console.error('Error notifying server of stop:', error);
            }
            
            console.log("Camera stopped");
        }
    });

    // Language selection
    languageSelect.addEventListener('change', () => {
        const selectedLanguage = languageSelect.value;
        statusMessage.textContent = `Language changed to: ${languageSelect.options[languageSelect.selectedIndex].text}`;
        console.log("Language changed to:", selectedLanguage);
        
        // Update translation if a gesture is already detected
        if (gestureOutput.textContent) {
            updateTranslation(gestureOutput.textContent);
        }
    });

    // Text-to-Speech functionality
    speakBtn.addEventListener('click', async () => {
        const text = translationOutput.textContent;
        const language = languageSelect.value;
        
        if (text && text.trim() !== '') {
            try {
                // Show loading state
                const originalBtnText = speakBtn.innerHTML;
                speakBtn.innerHTML = '<span class="icon">🔄</span> Loading...';
                speakBtn.disabled = true;
                
                // Call the Google TTS API endpoint
                const response = await fetch('http://localhost:8000/tts', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: text,
                        language: language
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Create audio element and play the speech
                    const audio = new Audio(`data:audio/mp3;base64,${result.audio}`);
                    audio.play();
                    
                    // Reset button after audio starts playing
                    audio.onplay = () => {
                        speakBtn.innerHTML = originalBtnText;
                        speakBtn.disabled = false;
                        
                        // Show fallback notice if applicable
                        if (result.fallback) {
                            const fallbackMsg = document.createElement('div');
                            fallbackMsg.className = 'fallback-notice';
                            fallbackMsg.textContent = result.fallback;
                            fallbackMsg.style.fontSize = '12px';
                            fallbackMsg.style.color = '#666';
                            fallbackMsg.style.marginTop = '5px';
                            
                            // Remove any existing fallback notice
                            const existingNotice = document.querySelector('.fallback-notice');
                            if (existingNotice) {
                                existingNotice.remove();
                            }
                            
                            // Add the new notice
                            document.querySelector('.translation-panel').appendChild(fallbackMsg);
                            
                            // Remove the notice after 5 seconds
                            setTimeout(() => {
                                fallbackMsg.remove();
                            }, 5000);
                        }
                    };
                    
                    // Also reset if there's an error
                    audio.onerror = () => {
                        console.error('Error playing audio');
                        speakBtn.innerHTML = originalBtnText;
                        speakBtn.disabled = false;
                    };
                    
                    console.log(`Speaking with Google TTS: "${text}" in ${result.language}`);
                    if (result.fallback) {
                        console.log(result.fallback);
                    }
                } else {
                    // If Google TTS fails, fallback to browser TTS
                    console.warn('Google TTS failed, falling back to browser TTS:', result.error);
                    speakBtn.innerHTML = originalBtnText;
                    speakBtn.disabled = false;
                    
                    // Fallback to browser's built-in speech synthesis
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = language;
                    speechSynthesis.speak(utterance);
                }
            } catch (error) {
                console.error('Error with TTS service:', error);
                speakBtn.innerHTML = '<span class="icon">🔊</span> Speak';
                speakBtn.disabled = false;
                
                // Fallback to browser's built-in speech synthesis
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = language;
                speechSynthesis.speak(utterance);
            }
        }
    });
    
    // Start video rendering (separate from frame processing for better performance)
    function startVideoRendering() {
        if (videoRenderInterval) {
            clearInterval(videoRenderInterval);
        }
        
        videoRenderInterval = setInterval(() => {
            if (isStreaming && videoElement.readyState === 4) {
                // Draw video frame to canvas
                ctx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            }
        }, 1000 / videoRenderRate); // e.g., 1000/30 = ~33ms for 30fps
    }
    
    // Stop video rendering
    function stopVideoRendering() {
        if (videoRenderInterval) {
            clearInterval(videoRenderInterval);
            videoRenderInterval = null;
        }
    }

    // Start recognition process
    function startRecognition() {
        if (isRecognizing) return;
        
        console.log("Starting recognition process");
        isRecognizing = true;
        
        // Create new abort controller if needed
        if (!abortController) {
            abortController = new AbortController();
        }
        
        recognitionInterval = setInterval(() => {
            if (isStreaming && videoElement.readyState === 4 && isRecognizing && !processingFrame) {
                // Get canvas data as base64 image - use a smaller size for processing
                const processingCanvas = document.createElement('canvas');
                processingCanvas.width = 320; // Smaller size for processing
                processingCanvas.height = 240;
                const processingCtx = processingCanvas.getContext('2d');
                processingCtx.drawImage(videoElement, 0, 0, 320, 240);
                
                const imageData = processingCanvas.toDataURL('image/jpeg', 0.6); // Lower quality for faster transmission
                
                // Send to server for processing
                processFrame(imageData);
            }
        }, frameInterval); // Process frames at specified interval
    }

    // Stop recognition process
    function stopRecognition() {
        console.log("Stopping recognition process...");
        
        // Set flag to prevent new processing
        isRecognizing = false;
        
        // Clear interval
        if (recognitionInterval) {
            clearInterval(recognitionInterval);
            recognitionInterval = null;
        }
        
        // Abort any in-progress fetch requests
        if (abortController) {
            console.log("Aborting in-progress fetch requests");
            abortController.abort();
            abortController = null;
        }
        
        console.log("Recognition process stopped completely");
    }

    // Process frame with the server
    async function processFrame(imageData) {
        // Skip if we're already processing a frame or recognition is stopped
        if (processingFrame || !isRecognizing) {
            return;
        }
        
        processingFrame = true;
        
        try {
            // Skip if recognition has been stopped
            if (!isRecognizing || !abortController) {
                processingFrame = false;
                return;
            }
            
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    clientId: clientId
                }),
                signal: abortController.signal
            });
            
            // Skip processing response if recognition has been stopped
            if (!isRecognizing) {
                processingFrame = false;
                return;
            }
            
            const result = await response.json();
            
            // Skip if recognition has been stopped
            if (!isRecognizing) {
                processingFrame = false;
                return;
            }
            
            // Skip detailed logging for skipped frames
            if (!result.skipped) {
                console.log("Server response:", result);
            }
            
            if (result.success) {
                updateGestureOutput(result.action, result.confidence);
                
                // If still collecting frames, update status and progress bar
                if (result.collecting) {
                    const total = result.frames_collected + result.frames_needed;
                    const percent = (result.frames_collected / total) * 100;
                    progressBar.style.width = `${percent}%`;
                    statusMessage.textContent = `Collecting frames: ${result.frames_collected}/${total}`;
                } else {
                    // If we have a prediction, show full progress bar
                    progressBar.style.width = '100%';
                }
            } else if (result.collecting) {
                // Still collecting frames but no prediction yet
                const total = result.frames_collected + result.frames_needed;
                const percent = (result.frames_collected / total) * 100;
                progressBar.style.width = `${percent}%`;
                statusMessage.textContent = `Collecting frames: ${result.frames_collected}/${total}`;
            } else if (result.error) {
                console.error('Recognition error:', result.error);
                statusMessage.textContent = 'Recognition error: ' + result.error;
            }
        } catch (error) {
            // Don't log abort errors as they're expected when stopping
            if (error.name !== 'AbortError') {
                console.error('Error processing frame:', error);
                statusMessage.textContent = 'Connection error: ' + error.message;
            }
        } finally {
            processingFrame = false;
        }
    }

    // Update gesture output with recognition results
    function updateGestureOutput(gesture, confidence) {
        console.log(`Detected gesture: ${gesture} with confidence: ${confidence}`);
        // Only update if confidence is above threshold
        if (confidence > 0.3) { // Lowered threshold for testing
            gestureOutput.textContent = gesture;
            updateTranslation(gesture);
            statusMessage.textContent = `Detected: ${gesture} (${Math.round(confidence * 100)}% confidence)`;
        }
    }
    
    // Update translation based on detected gesture and selected language
    function updateTranslation(gesture) {
        const language = languageSelect.value;
        
        if (translations[gesture] && translations[gesture][language]) {
            translationOutput.textContent = translations[gesture][language];
        } else {
            translationOutput.textContent = gesture; // Fallback to gesture name
        }
        console.log("Translation updated:", translationOutput.textContent);
    }
    
    // Add instructions for users
    
});
