#!/usr/bin/env python3
"""
Praat-based formant extraction server for spectrogram analysis
Requires: praat-parselmouth, flask, flask-cors
"""

import os
import tempfile
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import parselmouth
import numpy as np
import base64
import io
import wave
import logging
import asyncio
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Constants
SAMPLE_RATE = 44100
MAX_FORMANT = 5000.0
MIN_AMPLITUDE = 0.005  # より敏感に設定
MIN_AMP_FILE = 0.005   # より敏感に設定
TIME_STEP = 0.03      # Analysis time step (seconds)
Z_SCORE = 0.674        # Z-score threshold for outlier removal

# English vowel targets for classification
ENGLISH_VOWELS = {
    'æ': {'name': 'cat', 'f1': 700, 'f2': 1700},
    'ɪ': {'name': 'sit', 'f1': 400, 'f2': 2000},
    'ʊ': {'name': 'put', 'f1': 400, 'f2': 1000},
    'ɛ': {'name': 'get', 'f1': 550, 'f2': 1800},
    'ɔ': {'name': 'caught', 'f1': 550, 'f2': 900},
    'ʌ': {'name': 'but', 'f1': 600, 'f2': 1200},
    'ɑ': {'name': 'father', 'f1': 750, 'f2': 1100},
    'ə': {'name': 'about', 'f1': 500, 'f2': 1500}
}

def classify_vowel(f1, f2):
    """
    Classify vowel based on F1 and F2 values

    Args:
        f1: First formant frequency
        f2: Second formant frequency

    Returns:
        str: Detected vowel symbol or None
    """
    if f1 <= 0 or f2 <= 0:
        return None

    best_match = None
    min_distance = float('inf')
    threshold = 250  # Hz threshold for classification (より寛容に設定)

    for vowel, data in ENGLISH_VOWELS.items():
        # Calculate Euclidean distance in formant space
        distance = np.sqrt((f1 - data['f1'])**2 + (f2 - data['f2'])**2)

        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = vowel

    return best_match

def extract_formants_praat(audio_data, sample_rate=22050):
    """
    Extract formants using Praat (parselmouth) with improved algorithm

    Args:
        audio_data: numpy array of audio samples
        sample_rate: sampling rate

    Returns:
        dict: F1, F2, F3 values and confidence scores
    """
    try:
        # Import librosa for STFT analysis
        import librosa

        # Create Praat Sound object
        sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)

        # Constants from the provided code
        MAX_FORMANT = 5000.0
        Z_SCORE = 0.674
        TIME_STEP = 0.03
        MIN_AMP_REALTIME = 0.2

        # Extract formants using improved method
        formant = sound.to_formant_burg(time_step=TIME_STEP, maximum_formant=MAX_FORMANT)
        duration = sound.duration
        max_flat_samples = sound.values.flatten()
        max_amplitude = np.max(np.abs(max_flat_samples))
        times = np.arange(0, duration, TIME_STEP)

        f1_all, f2_all = [], []
        f1_amp_all, f2_amp_all = [], []

        # STFT for amplitude analysis
        y = sound.values.flatten()
        sr = sound.sampling_frequency
        hop_length = int(sr * TIME_STEP)
        n_fft = min(1024, len(y))

        # Amplitude spectrum in dB
        D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        D = librosa.amplitude_to_db(D, ref=np.max)
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        for i, t in enumerate(times):
            try:
                # Extract segment and check amplitude
                segment = sound.extract_part(from_time=t, to_time=min(t + TIME_STEP, duration), preserve_times=False)
                flat_samples = segment.values.flatten()
                amplitude = np.max(np.abs(flat_samples))
                if amplitude != 0:
                    normalized_amplitude = amplitude / max_amplitude
                else:
                    normalized_amplitude = 0

                if normalized_amplitude > MIN_AMP_REALTIME:
                    # Extract formants
                    f1 = formant.get_value_at_time(1, t)
                    f2 = formant.get_value_at_time(2, t)
                    if f1 and f2 and f1 > 0 and f2 > 0:
                        f1_all.append(f1)
                        f2_all.append(f2)

                        # Get corresponding STFT amplitudes
                        f1_idx = np.argmin(np.abs(frequencies - f1))
                        f2_idx = np.argmin(np.abs(frequencies - f2))
                        frame_idx = min(i, D.shape[1]-1)
                        f1_amp_all.append(D[f1_idx, frame_idx])
                        f2_amp_all.append(D[f2_idx, frame_idx])

            except Exception:
                continue

        # Convert to arrays
        f1_vals = np.array(f1_all)
        f2_vals = np.array(f2_all)
        f1_amp_vals = np.array(f1_amp_all)
        f2_amp_vals = np.array(f2_amp_all)

        # Z-score outlier removal
        if len(f1_vals) > 0 and len(f2_vals) > 0:
            f1_mean, f1_std = f1_vals.mean(), f1_vals.std()
            f2_mean, f2_std = f2_vals.mean(), f2_vals.std()

            if f1_std > 0 and f2_std > 0:
                valid_idx = (np.abs((f1_vals - f1_mean) / f1_std) <= Z_SCORE) & \
                           (np.abs((f2_vals - f2_mean) / f2_std) <= Z_SCORE)

                f1_vals = f1_vals[valid_idx]
                f2_vals = f2_vals[valid_idx]
                f1_amp_vals = f1_amp_vals[valid_idx]
                f2_amp_vals = f2_amp_vals[valid_idx]

        # Calculate final values
        if len(f1_vals) > 0 and len(f2_vals) > 0:
            f1_median = np.median(f1_vals)
            f2_median = np.median(f2_vals)
            f1_amp_median = np.median(f1_amp_vals) if len(f1_amp_vals) > 0 else 0
            f2_amp_median = np.median(f2_amp_vals) if len(f2_amp_vals) > 0 else 0

            # Calculate confidence based on amplitude and consistency
            f1_confidence = max(0, min(100, 50 + f1_amp_median * 2))  # Amplitude-based confidence
            f2_confidence = max(0, min(100, 50 + f2_amp_median * 2))

            # Adjust confidence based on data consistency
            if len(f1_vals) > 1:
                f1_std_norm = np.std(f1_vals) / f1_median
                f1_confidence *= max(0.5, 1 - f1_std_norm)
            if len(f2_vals) > 1:
                f2_std_norm = np.std(f2_vals) / f2_median
                f2_confidence *= max(0.5, 1 - f2_std_norm)

            return {
                'f1': round(f1_median) if f1_median > 0 else 0,
                'f2': round(f2_median) if f2_median > 0 else 0,
                'f3': 0,  # Not calculated in this version
                'confidence_f1': round(f1_confidence),
                'confidence_f2': round(f2_confidence),
                'success': True,
                'method': 'praat-parselmouth-improved',
                'samples_used': len(f1_vals),
                'total_samples': len(f1_all)
            }
        else:
            return {
                'f1': 0,
                'f2': 0,
                'f3': 0,
                'confidence_f1': 0,
                'confidence_f2': 0,
                'success': False,
                'error': 'No valid formants detected',
                'method': 'praat-parselmouth-improved'
            }

    except Exception as e:
        return {
            'f1': 0,
            'f2': 0,
            'f3': 0,
            'confidence_f1': 0,
            'confidence_f2': 0,
            'success': False,
            'error': str(e),
            'method': 'praat-parselmouth-improved'
        }

@app.route('/extract_formants', methods=['POST'])
def extract_formants():
    """
    API endpoint to extract formants from audio data
    Expects JSON with base64-encoded WAV data
    """
    try:
        data = request.get_json()

        if not data or 'audio' not in data:
            return jsonify({
                'error': 'Missing audio data',
                'success': False
            }), 400

        # Decode base64 audio data
        audio_base64 = data['audio']
        if audio_base64.startswith('data:'):
            # Remove data URL prefix if present
            audio_base64 = audio_base64.split(',')[1]
        
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as decode_error:
            logger.error(f"Base64 decode error: {decode_error}")
            return jsonify({
                'error': 'Invalid base64 audio data',
                'success': False
            }), 400
            
        if len(audio_bytes) == 0:
            return jsonify({
                'error': 'Empty audio data',
                'success': False
            }), 400

        # Parse audio data (handle both WAV and WebM formats)
        try:
            with io.BytesIO(audio_bytes) as audio_io:
                # Try WAV first
                try:
                    with wave.open(audio_io, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        frames = wav_file.readframes(wav_file.getnframes())

                        # Convert to numpy array
                        if wav_file.getsampwidth() == 2:  # 16-bit
                            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif wav_file.getsampwidth() == 4:  # 32-bit
                            audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:  # 8-bit or other
                            audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                except wave.Error:
                    # If WAV parsing fails, try to use librosa for WebM/other formats
                    logger.info("WAV parsing failed, trying librosa for audio format")
                    import librosa
                    audio_io.seek(0)
                    audio_data, sample_rate = librosa.load(audio_io, sr=None)
                    
        except Exception as audio_error:
            logger.error(f"Audio parsing error: {audio_error}")
            return jsonify({
                'error': f'Audio format not supported: {str(audio_error)}',
                'success': False
            }), 400

        # Extract formants using Praat
        result = extract_formants_praat(audio_data, sample_rate)

        # Add vowel classification
        if result['success'] and result['f1'] > 0 and result['f2'] > 0:
            result['detected_vowel'] = classify_vowel(result['f1'], result['f2'])
        else:
            result['detected_vowel'] = None

        return jsonify(result)

    except Exception as e:
        logger.error(f"Extract formants error: {e}", exc_info=True)
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'praat_available': True,
        'version': '1.0'
    })

@app.route('/', methods=['GET'])
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'spectrogram_test.html')

@app.route('/api', methods=['GET'])
def api_info():
    """API info endpoint"""
    return jsonify({
        'service': 'Praat Formant Extraction Server',
        'endpoints': {
            '/extract_formants': 'POST - Extract formants from audio',
            '/health': 'GET - Health check'
        },
        'requirements': [
            'praat-parselmouth',
            'flask',
            'flask-cors',
            'numpy',
            'librosa'
        ]
    })

# === WebSocket Server for Audio Analysis === #

def analyze_audio_blob(audio_blob):
    """Analyze audio blob using Praat for formant extraction"""
    temp_path = None
    
    try:
        # Validate audio blob
        if not audio_blob or len(audio_blob) == 0:
            return {'error': 'Empty audio data received'}
        
        # Save base64 audio data as a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav.write(audio_blob)
            temp_path = temp_wav.name

        logger.info(f"Analyzing audio file: {temp_path} ({len(audio_blob)} bytes)")

        # Analyze audio with Praat
        sound = parselmouth.Sound(temp_path)
        
        # Validate sound object
        if sound.duration < 0.01:  # Minimum 10ms
            return {'error': 'Audio too short for analysis'}
            
        formant = sound.to_formant_burg(time_step=TIME_STEP, maximum_formant=MAX_FORMANT)
        duration = sound.duration
        max_flat_samples = sound.values.flatten()
        max_amplitude = np.max(np.abs(max_flat_samples))
        
        if max_amplitude == 0:
            return {'error': 'No audio signal detected'}
            
        times = np.arange(0, duration, TIME_STEP)
        f1_all, f2_all = [], []

        for t in times:
            try:
                segment = sound.extract_part(from_time=t, to_time=min(t + TIME_STEP, duration), preserve_times=False)
                flat_samples = segment.values.flatten()
                amplitude = np.max(np.abs(flat_samples))

                if amplitude != 0:
                    normalized_amplitude = flat_samples / max_amplitude
                else:
                    normalized_amplitude = flat_samples

                normalized_amplitude = np.max(np.abs(normalized_amplitude))

                if normalized_amplitude > MIN_AMP_FILE:
                    f1 = formant.get_value_at_time(1, t)
                    f2 = formant.get_value_at_time(2, t)
                    if f1 and f2 and f1 > 0 and f2 > 0:
                        f1_all.append(float(f1))
                        f2_all.append(float(f2))
            except Exception as e:
                logger.warning(f"Error analyzing segment at time {t}: {e}")
                continue

        if f1_all and f2_all:
            f1_vals = np.array(f1_all)
            f2_vals = np.array(f2_all)

            # Z-score outlier removal
            f1_mean, f1_std = np.mean(f1_vals), np.std(f1_vals)
            f2_mean, f2_std = np.mean(f2_vals), np.std(f2_vals)
            
            if f1_std > 0 and f2_std > 0:
                valid_idx = (np.abs((f1_vals - f1_mean)/f1_std) <= Z_SCORE) & \
                            (np.abs((f2_vals - f2_mean)/f2_std) <= Z_SCORE)
                f1_vals = f1_vals[valid_idx]
                f2_vals = f2_vals[valid_idx]

            if len(f1_vals) > 0 and len(f2_vals) > 0:
                result = {
                    'f1': float(np.median(f1_vals)),
                    'f2': float(np.median(f2_vals)),
                    'f1_all': f1_all,
                    'f2_all': f2_all,
                    'samples_analyzed': len(f1_vals),
                    'duration': duration
                }
                logger.info(f"Analysis successful: F1={result['f1']}, F2={result['f2']}")
                return result
            else:
                return {'error': 'No valid formants after filtering'}

        else:
            return {'error': 'No formants detected in audio'}

    except parselmouth.PraatError as e:
        logger.error(f"Praat analysis error: {e}")
        return {'error': f'Praat analysis failed: {str(e)}'}
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return {'error': f'Audio analysis failed: {str(e)}'}
    finally:
        # Delete temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")

async def handle_websocket(websocket, path):
    """WebSocket connection handler for audio analysis"""
    logger.info(f"New WebSocket connection from {websocket.remote_address}")
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if 'audio' in data:
                    # Convert base64 string to binary data
                    audio_base64 = data['audio']
                    if audio_base64.startswith('data:'):
                        audio_base64 = audio_base64.split(',')[1]
                    audio_blob = base64.b64decode(audio_base64)
                    result = analyze_audio_blob(audio_blob)
                    await websocket.send(json.dumps(result))
                else:
                    await websocket.send(json.dumps({'error': 'No audio data received'}))
            except json.JSONDecodeError as e:
                await websocket.send(json.dumps({'error': f'Invalid JSON: {str(e)}'}))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send(json.dumps({'error': str(e)}))
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"WebSocket connection closed: {websocket.remote_address}")
    except websockets.exceptions.InvalidMessage as e:
        logger.warning(f"Invalid WebSocket message: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"WebSocket connection ended: {websocket.remote_address}")

async def websocket_main():
    host = '0.0.0.0'
    # RenderではPORTを同じにして、パスで区別する
    port = int(os.environ.get('WS_PORT', 8765))

    # Handle HTTP requests (health checks) on WebSocket port
    async def process_request(path, request_headers):
        """Process incoming HTTP requests and handle health checks"""
        import http
        
        # Check if this is a proper WebSocket upgrade request
        has_upgrade = False
        has_connection = False
        method = None

        for name, value in request_headers:
            name_lower = name.lower()
            value_lower = value.lower()

            if name_lower == 'upgrade' and 'websocket' in value_lower:
                has_upgrade = True
            elif name_lower == 'connection' and 'upgrade' in value_lower:
                has_connection = True

        # Extract HTTP method from headers (Render sends this in custom header)
        for name, value in request_headers:
            if name.lower() == ':method':
                method = value.upper()
                break

        # If it's a proper WebSocket request, continue with WebSocket handshake
        if has_upgrade and has_connection:
            logger.info(f"WebSocket upgrade request from {path}")
            return None

        # Handle health check requests (GET, HEAD, or any non-WebSocket request)
        logger.info(f"Health check request: {method or 'Unknown'} {path}")
        
        # Return appropriate HTTP response for health checks
        response_body = b"WebSocket server is running\n"
        content_length = str(len(response_body))
        
        return (
            http.HTTPStatus.OK,
            [
                ("Content-Type", "text/plain"),
                ("Content-Length", content_length),
                ("Cache-Control", "no-cache")
            ],
            response_body
        )

    try:
        import http
        server = await websockets.serve(
            handle_websocket,
            host,
            port,
            process_request=process_request,
            # Add ping/pong to keep connections alive
            ping_interval=20,
            ping_timeout=10,
            # Additional settings for better compatibility
            max_size=10*1024*1024,  # 10MB max message size
            max_queue=32
        )
        logger.info(f"Formant analysis WebSocket server running on ws://{host}:{port}")
        logger.info("Server is ready to accept connections")
        await server.wait_closed()
    except Exception as e:
        logger.error(f"WebSocket server error: {e}")
        raise

if __name__ == '__main__':
    # Check if required packages are available
    try:
        import parselmouth
        print("✓ Praat-parselmouth is available")
    except ImportError:
        print("✗ Error: praat-parselmouth not found")
        print("Install with: pip install praat-parselmouth")
        exit(1)

    try:
        import flask_cors
        print("✓ Flask-CORS is available")
    except ImportError:
        print("✗ Error: flask-cors not found")
        print("Install with: pip install flask-cors")
        exit(1)

    try:
        import librosa
        print("✓ Librosa is available")
    except ImportError:
        print("✗ Error: librosa not found")
        print("Install with: pip install librosa")
        exit(1)

    print("Starting Praat Formant Extraction Server...")
    print("Server will be available at: http://localhost:5001")
    print("\nEndpoints:")
    print("  GET  /health - Health check")
    print("  POST /extract_formants - Extract formants from audio")

    # Start Flask app
    # app.run(debug=True, host='0.0.0.0', port=5001)

    # Start WebSocket server
    asyncio.run(websocket_main())