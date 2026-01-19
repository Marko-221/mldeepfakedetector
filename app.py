import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import numpy as np
import io
from PIL import Image
import tempfile
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Inisialisasi Aplikasi Flask
# -------------------------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------------------------
# Konfigurasi & Pemuatan Model
# -------------------------------------------------------------------
MODEL_PATH = 'C:/Users/ASUS/Downloads/deepfake_detectorv2.h5'
AUDIO_MODEL_PATH = 'C:/Users/ASUS/Downloads/fake_vs_real_speech_classifier.h5'  # Model untuk deteksi audio

try:
    model = load_model(MODEL_PATH)
    print(f"--- Model gambar/video berhasil dimuat dari {MODEL_PATH} ---")
except Exception as e:
    print(f"WARNING: Gagal memuat model gambar/video. Pastikan file '{MODEL_PATH}' ada.")
    print(f"Error detail: {e}")
    model = None

try:
    audio_model = load_model(AUDIO_MODEL_PATH)
    print(f"--- Model audio berhasil dimuat dari {AUDIO_MODEL_PATH} ---")
except Exception as e:
    print(f"WARNING: Gagal memuat model audio. Fitur audio akan dinonaktifkan.")
    print(f"Error detail: {e}")
    audio_model = None

LABEL_MAP = {1: 'Palsu (Fake)', 0: 'Asli (Real)'}
IMG_SIZE = (224, 224)

# -------------------------------------------------------------------
# Konfigurasi Audio
# -------------------------------------------------------------------
AUDIO_IMG_SIZE = (128, 128)  # Ukuran spectrogram untuk model audio
SUPPORTED_AUDIO_FORMATS = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']

# -------------------------------------------------------------------
# Fungsi Preprocessing Gambar
# -------------------------------------------------------------------
def preprocess_image(image_file_stream):
    try:
        img = Image.open(image_file_stream)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded)
    except Exception as e:
        print(f"Error saat preprocessing gambar: {e}")
        return None

# -------------------------------------------------------------------
# Fungsi Preprocessing Video
# -------------------------------------------------------------------
def process_video(video_path, max_frames=10):
    """
    Ekstrak frame dari video dan lakukan prediksi pada setiap frame
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    predictions = []
    
    frame_count = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Ambil setiap 5 frame untuk efisiensi
        if frame_count % 5 == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
            # Preprocess
            frame_processed = preprocess_input(np.expand_dims(frame_resized, axis=0))
            frames.append(frame_processed)
            
            # Prediksi
            if model:
                prediction = model.predict(frame_processed)
                score = float(prediction[0][0])
                predictions.append(score)
        
        frame_count += 1
    
    cap.release()
    return predictions

# -------------------------------------------------------------------
# Fungsi Preprocessing Audio
# -------------------------------------------------------------------
def create_spectrogram(audio_path, save_path=None, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    Membuat mel-spectrogram dari file audio
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Save spectrogram as image jika path disediakan
        if save_path:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                log_mel_spec, 
                sr=sr, 
                hop_length=hop_length, 
                x_axis='time', 
                y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        return log_mel_spec, y, sr
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None, None

def preprocess_audio_for_model(audio_path):
    """
    Preprocess audio untuk model
    """
    try:
        # Buat spectrogram temporary
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Buat spectrogram
        spectrogram, audio_data, sr = create_spectrogram(audio_path, temp_path)
        if spectrogram is None:
            return None, None
        
        # Load dan preprocess gambar spectrogram
        img = Image.open(temp_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(AUDIO_IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Hapus file temporary
        os.unlink(temp_path)
        
        # Analisis fitur audio tambahan
        features = analyze_audio_features(audio_data, sr)
        
        return img_array, features
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None

def analyze_audio_features(audio_data, sr):
    """
    Analisis fitur audio tambahan
    """
    try:
        # Hitung berbagai fitur audio
        duration = librosa.get_duration(y=audio_data, sr=sr)
        energy = np.sum(audio_data**2) / len(audio_data)
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
        
        features = {
            'duration': duration,
            'energy': energy,
            'zero_crossing_rate': zero_crossing,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'mfcc_mean': mfccs_mean.tolist()[:5]  # Ambil 5 pertama saja
        }
        
        return features
    except Exception as e:
        print(f"Error analyzing audio features: {e}")
        return None

# -------------------------------------------------------------------
# Rute API untuk Prediksi
# -------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    # Cek tipe file yang dikirim
    file_type = request.form.get('type', '')
    
    # Handle Image Upload
    if 'image' in request.files and request.files['image'].filename != '':
        return predict_image(request.files['image'])
    
    # Handle Video Upload
    elif 'video' in request.files and request.files['video'].filename != '':
        return predict_video(request.files['video'])
    
    # Handle Audio Upload
    elif 'audio' in request.files and request.files['audio'].filename != '':
        return predict_audio(request.files['audio'])
    
    # Handle berdasarkan type parameter
    elif file_type and 'file' in request.files:
        file = request.files['file']
        if file_type == 'image':
            return predict_image(file)
        elif file_type == 'video':
            return predict_video(file)
        elif file_type == 'audio':
            return predict_audio(file)
    
    return jsonify({'success': False, 'error': 'Tidak ada file yang valid terkirim'})

def predict_image(file):
    """
    Handle prediksi gambar
    """
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model gambar tidak tersedia'})
        
        processed_image = preprocess_image(file.stream)
        if processed_image is None:
            return jsonify({'success': False, 'error': 'Gagal memproses gambar'})

        prediction = model.predict(processed_image)
        score = float(prediction[0][0])
        
        if score > 0.5:
            class_id = 1
            confidence = score * 100
        else:
            class_id = 0
            confidence = (1 - score) * 100
            
        predicted_label = LABEL_MAP[class_id]

        return jsonify({
            'success': True,
            'type': 'image',
            'prediction': predicted_label,
            'confidence': f'{confidence:.2f}%',
            'raw_score': score
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})

def predict_video(video_file):
    """
    Handle prediksi video
    """
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model video tidak tersedia'})
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            temp_path = temp_video.name
        
        # Process video
        predictions = process_video(temp_path)
        
        if not predictions:
            return jsonify({'success': False, 'error': 'Tidak bisa mengekstrak frame dari video'})
        
        # Calculate average prediction
        avg_score = np.mean(predictions)
        real_frames = sum(1 for score in predictions if score > 0.5)
        total_frames = len(predictions)
        
        if avg_score > 0.5:
            overall_prediction = "Fake"
            confidence = avg_score * 100
        else:
            overall_prediction = "Real" 
            confidence = (1 - avg_score) * 100
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'type': 'video',
            'prediction': overall_prediction,
            'confidence': f'{confidence:.2f}%',
            'frame_analysis': f'{real_frames}/{total_frames} frame terdeteksi asli',
            'raw_score': float(avg_score)
        })
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing video: {str(e)}'})

def predict_audio(audio_file):
    """
    Handle prediksi audio
    """
    try:
        if audio_model is None:
            return jsonify({
                'success': False, 
                'error': 'Model audio tidak tersedia. Pastikan file model audio sudah diunduh.'
            })
        
        # Cek format file
        filename = audio_file.filename
        if not any(filename.lower().endswith(ext) for ext in SUPPORTED_AUDIO_FORMATS):
            return jsonify({
                'success': False,
                'error': f'Format audio tidak didukung. Gunakan: {", ".join(SUPPORTED_AUDIO_FORMATS)}'
            })
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name
        
        # Preprocess audio
        processed_audio, audio_features = preprocess_audio_for_model(temp_path)
        
        if processed_audio is None:
            os.unlink(temp_path)
            return jsonify({'success': False, 'error': 'Gagal memproses file audio'})
        
        # Predict
        prediction = audio_model.predict(processed_audio, verbose=0)
        score = float(prediction[0][0])
        
        # Interpret hasil
        if score > 0.5:
            result = "Suara Manusia Asli (Real Human Speech)"
            confidence = score * 100
        else:
            result = "Suara AI/Buatan (AI-Generated Speech)"
            confidence = (1 - score) * 100
        
        # Format audio features untuk ditampilkan
        features_text = ""
        if audio_features:
            features_text = f"""
            <strong>Analisis Audio:</strong><br>
            • Durasi: {audio_features['duration']:.2f} detik<br>
            • Energi: {audio_features['energy']:.6f}<br>
            • Zero-Crossing Rate: {audio_features['zero_crossing_rate']:.4f}<br>
            • Spectral Centroid: {audio_features['spectral_centroid']:.2f}<br>
            • Spectral Bandwidth: {audio_features['spectral_bandwidth']:.2f}
            """
        
        # Clean up
        os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'type': 'audio',
            'prediction': result,
            'confidence': f'{confidence:.2f}%',
            'raw_score': score,
            'details': {
                'analysis': f"Berdasarkan analisis spectrogram dan fitur audio",
                'audio_features': features_text
            }
        })
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error processing audio: {str(e)}'})

# -------------------------------------------------------------------
# Rute untuk Health Check
# -------------------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint untuk mengecek status aplikasi dan model
    """
    status = {
        'status': 'healthy',
        'image_video_model_loaded': model is not None,
        'audio_model_loaded': audio_model is not None,
        'supported_audio_formats': SUPPORTED_AUDIO_FORMATS
    }
    return jsonify(status)

# -------------------------------------------------------------------
# Rute untuk Halaman Utama
# -------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# -------------------------------------------------------------------
# Konfigurasi Server
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Buat folder untuk temporary files jika belum ada
    temp_dir = 'temp_files'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Konfigurasi Flask
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    app.config['UPLOAD_FOLDER'] = temp_dir
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION API SERVER")
    print("="*60)
    print(f"1. Model Gambar/Video: {'LOADED' if model else 'NOT LOADED'}")
    print(f"2. Model Audio: {'LOADED' if audio_model else 'NOT LOADED'}")
    print(f"3. Endpoint: http://localhost:5000")
    print(f"4. Health Check: http://localhost:5000/health")
    print("="*60)
    print("\nServer starting...")
    
    # Jalankan server
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)