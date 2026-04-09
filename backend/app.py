from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.filterwarnings('ignore')

try:
    import audio_processor as ap
except:
    from backend import audio_processor as ap



app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RF_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '../models/copd_rf_optimized_bundle.joblib'))
XGB_MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '../models/respiratory_xgb_optimized.joblib'))

print("RF PATH:", RF_MODEL_PATH)
print("XGB PATH:", XGB_MODEL_PATH)

# Global variables
rf_model = None
xg_model = None
rf_scaler = None
xg_scaler = None

# Stage 1 mapping (RF model: COPD vs Non-COPD)
STAGE1_MAPPING = {
    0: "Non-COPD",
    1: "COPD"
}

# Stage 2 mapping (XGB model: Asthma vs Normal vs Pneumonia)
STAGE2_MAPPING = {
    0: "Asthma",
    1: "Normal",
    2: "Pneumonia"
}

def load_models():
    print("RF exists:", os.path.exists(RF_MODEL_PATH))
    print("XGB exists:", os.path.exists(XGB_MODEL_PATH))
    """Load both stage models"""
    global rf_model, xg_model, rf_scaler, xg_scaler
    
    print("=" * 60)
    print("Loading Auscura ML Models...")
    print("=" * 60)
    
    # Load Stage 1: Random Forest (COPD vs Non-COPD)
    try:
        rf_bundle = joblib.load(RF_MODEL_PATH)
        if isinstance(rf_bundle, dict):
            if 'model' in rf_bundle:
                rf_model = rf_bundle['model']
                print("✅ Stage 1 Model (COPD vs Non-COPD) loaded")
            if 'scaler' in rf_bundle:
                rf_scaler = rf_bundle['scaler']
                print("   ✓ Scaler found")
        else:
            rf_model = rf_bundle
            print("✅ Stage 1 Model (COPD vs Non-COPD) loaded")
    except Exception as e:
        print(f"⚠️ Stage 1 model error: {e}")
    
    # Load Stage 2: XGBoost (Asthma vs Normal vs Pneumonia)
    try:
        xgb_bundle = joblib.load(XGB_MODEL_PATH)
        if isinstance(xgb_bundle, dict):
            if 'model' in xgb_bundle:
                xg_model = xgb_bundle['model']
                print("✅ Stage 2 Model (Asthma/Normal/Pneumonia) loaded")
            if 'scaler' in xgb_bundle:
                xg_scaler = xgb_bundle['scaler']
                print("   ✓ Scaler found")
        else:
            xg_model = xgb_bundle
            print("✅ Stage 2 Model (Asthma/Normal/Pneumonia) loaded")
    except Exception as e:
        print(f"⚠️ Stage 2 model error: {e}")
    
    print("=" * 60)

app = Flask(__name__)
CORS(app)

load_models()

def generate_waveform_plot(audio, sr):
    """Generate waveform plot as base64 image"""
    plt.figure(figsize=(10, 4))
    time = np.linspace(0, len(audio)/sr, len(audio))
    plt.plot(time, audio, color='#00d4ff', linewidth=1.5)
    plt.title('Waveform', color='white', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', color='white', fontsize=10)
    plt.ylabel('Amplitude', color='white', fontsize=10)
    plt.gca().set_facecolor('#0a0e17')
    plt.gcf().set_facecolor('#0a0e17')
    plt.tick_params(colors='white')
    plt.grid(alpha=0.2, color='white')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0e17', edgecolor='none')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_spectrogram(audio, sr):
    """Generate spectrogram as base64 image"""
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='plasma')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Spectrogram', color='white', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', color='white', fontsize=10)
    plt.ylabel('Frequency (Hz)', color='white', fontsize=10)
    plt.gca().set_facecolor('#0a0e17')
    plt.gcf().set_facecolor('#0a0e17')
    plt.tick_params(colors='white')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0e17', edgecolor='none')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_mfcc_plot(audio, sr):
    """Generate MFCC plot as base64 image"""
    plt.figure(figsize=(10, 4))
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')
    plt.colorbar(img)
    plt.title('MFCC Features', color='white', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)', color='white', fontsize=10)
    plt.ylabel('MFCC Coefficients', color='white', fontsize=10)
    plt.gca().set_facecolor('#0a0e17')
    plt.gcf().set_facecolor('#0a0e17')
    plt.tick_params(colors='white')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0e17', edgecolor='none')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "models": {
            "stage1_copd_vs_non_copd": rf_model is not None,
            "stage2_asthma_normal_pneumonia": xg_model is not None
        }
    })

@app.route('/api/predict-multiple', methods=['POST', 'OPTIONS'])
def predict_multiple():
    if request.method == 'OPTIONS':
        return jsonify({"success": True}), 200
    
    try:
        # Check for user_type in both headers AND form data
        user_type = request.headers.get('X-User-Type')
        if not user_type:
            user_type = request.form.get('user_type', 'patient')
        
        print(f"\n👤 User type: {user_type}")
        print(f"   Headers: X-User-Type={request.headers.get('X-User-Type')}")
        print(f"   Form data: user_type={request.form.get('user_type')}")
        
        # Collect files
        all_files = []
        for key in request.files:
            if key.startswith('point_') or key == 'audio':
                all_files.append(request.files[key])
        
        if len(all_files) == 0:
            return jsonify({"error": "No audio files provided"}), 400
        
        print(f"📁 Processing {len(all_files)} file(s)...")
        print(f"👨‍⚕️ Doctor mode: {user_type == 'doctor'} (will generate visualizations)")
        
        predictions = []
        successful = 0
        
        if rf_model is None:
            return jsonify({"error": "Models not loaded properly"}), 500
        
        for idx, file in enumerate(all_files):
            if file.filename == '':
                continue
            
            temp_path = f"temp_audio_{idx}.wav"
            file.save(temp_path)
            
            try:
                print(f"\n  📄 File {idx+1}: {file.filename}")
                
                # Load audio for visualizations
                audio, sr = librosa.load(temp_path, sr=16000, duration=5.0)
                
                # Extract features
                features = ap.extract_features_from_file(temp_path)
                if features is None:
                    raise ValueError("Feature extraction failed")
                
                # STAGE 1: Check if COPD or Non-COPD
                if rf_scaler is not None:
                    features_scaled = rf_scaler.transform(features)
                else:
                    features_scaled = features
                
                stage1_pred = int(rf_model.predict(features_scaled)[0])
                stage1_probs = rf_model.predict_proba(features_scaled)[0]
                stage1_conf = float(np.max(stage1_probs)) * 100
                stage1_result = STAGE1_MAPPING.get(stage1_pred, "Unknown")
                
                print(f"    Stage 1 (COPD vs Non-COPD): {stage1_result} ({stage1_conf:.1f}%)")
                
                # Initialize variables
                final_diagnosis = None
                final_confidence = 0
                stage2_result = None
                stage2_conf = 0
                probabilities = None
                
                # If COPD detected, final output is COPD
                if stage1_result == "COPD":
                    final_diagnosis = "COPD"
                    final_confidence = stage1_conf
                    print(f"    ✅ COPD detected directly - skipping Stage 2")
                
                # If Non-COPD, go to Stage 2 for detailed classification
                elif stage1_result == "Non-COPD" and xg_model is not None:
                    if xg_scaler is not None:
                        features_scaled = xg_scaler.transform(features)
                    else:
                        features_scaled = features
                    
                    stage2_pred = int(xg_model.predict(features_scaled)[0])
                    probabilities = xg_model.predict_proba(features_scaled)[0]
                    stage2_conf = float(np.max(probabilities)) * 100
                    stage2_result = STAGE2_MAPPING.get(stage2_pred, "Unknown")
                    
                    print(f"    Stage 2 (Asthma/Normal/Pneumonia): {stage2_result} ({stage2_conf:.1f}%)")
                    
                    # Final diagnosis from Stage 2
                    final_diagnosis = stage2_result
                    final_confidence = stage2_conf
                
                prediction_data = {
                    "point": idx + 1,
                    "filename": file.filename,
                    "prediction": final_diagnosis,
                    "confidence": round(final_confidence, 1),
                    "stage1": {
                        "result": stage1_result,
                        "confidence": round(stage1_conf, 1)
                    },
                    "status": "success"
                }
                
                if stage2_result:
                    prediction_data["stage2"] = {
                        "result": stage2_result,
                        "confidence": round(stage2_conf, 1)
                    }
                    prediction_data["probabilities"] = {
                        "Asthma": round(float(probabilities[0]) * 100, 1),
                        "Normal": round(float(probabilities[1]) * 100, 1),
                        "Pneumonia": round(float(probabilities[2]) * 100, 1)
                    }
                
                # For doctors, add visualizations - FIXED: check user_type correctly
                if user_type == 'doctor':
                    print(f"    📊 Generating visualizations for doctor mode...")
                    try:
                        prediction_data["visualizations"] = {
                            "waveform": generate_waveform_plot(audio, sr),
                            "spectrogram": generate_spectrogram(audio, sr),
                            "mfcc": generate_mfcc_plot(audio, sr)
                        }
                        prediction_data["audio_info"] = {
                            "duration": round(len(audio)/sr, 2),
                            "sample_rate": sr,
                            "file_size": os.path.getsize(temp_path)
                        }
                        print(f"    ✅ Visualizations generated successfully")
                    except Exception as viz_err:
                        print(f"    ⚠️ Visualization error: {viz_err}")
                        prediction_data["visualizations"] = None
                        prediction_data["visualization_error"] = str(viz_err)
                else:
                    print(f"    📱 Patient mode - skipping visualizations")
                
                predictions.append(prediction_data)
                successful += 1
                print(f"    ✅ Final Diagnosis: {final_diagnosis} ({final_confidence:.1f}%)")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                predictions.append({
                    "point": idx + 1,
                    "filename": file.filename,
                    "error": str(e),
                    "prediction": "Error",
                    "status": "failed"
                })
            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        if successful == 0:
            return jsonify({"error": "No valid predictions"}), 400
        
        # Get most common final diagnosis
        disease_counts = {}
        for pred in predictions:
            if pred.get('status') == 'success':
                disease = pred['prediction']
                disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        final_diagnosis = max(disease_counts, key=disease_counts.get)
        avg_confidence = float(np.mean([p['confidence'] for p in predictions if p.get('status') == 'success']))
        
        # Get color, icon, and recommendation based on diagnosis
        diagnosis_info = {
            "Normal": {
                "color": "#00ff88", 
                "icon": "fa-heartbeat", 
                "recommendation": "Your respiratory sounds appear normal. Continue with regular checkups."
            },
            "COPD": {
                "color": "#ff9800", 
                "icon": "fa-lungs", 
                "recommendation": "COPD indicators detected. Please consult a pulmonologist for further evaluation."
            },
            "Asthma": {
                "color": "#4caf50", 
                "icon": "fa-wind", 
                "recommendation": "Asthma pattern detected. Consider consulting a respiratory specialist."
            },
            "Pneumonia": {
                "color": "#f44336", 
                "icon": "fa-virus", 
                "recommendation": "Pneumonia suggested. Immediate medical consultation recommended."
            }
        }
        
        info = diagnosis_info.get(final_diagnosis, {
            "color": "#00d4ff", 
            "icon": "fa-stethoscope", 
            "recommendation": "Please consult a healthcare provider for proper diagnosis."
        })
        
        response = {
            "success": True,
            "total_files": len(all_files),
            "successful": successful,
            "predictions": predictions,
            "final_diagnosis": final_diagnosis,
            "confidence": round(avg_confidence, 1),
            "points_analyzed": successful,
            "color": info["color"],
            "icon": info["icon"],
            "recommendation": info["recommendation"]
        }
        
        print(f"\n📊 Final Summary: {successful}/{len(all_files)} successful")
        print(f"🎯 Overall Diagnosis: {final_diagnosis} ({round(avg_confidence, 1)}%)")
        print(f"👨‍⚕️ Doctor visualizations included: {user_type == 'doctor'}")
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_models()
    
    print("\n" + "=" * 60)
    print("🚀 Starting Auscura Backend with 2-Stage Classification")
    print("=" * 60)
    print("📍 Health check: http://localhost:5000/api/health")
    print("📍 Predict endpoint: http://localhost:5000/api/predict-multiple")
    print("\n📊 Classification Pipeline:")
    print("   Stage 1 (RF): COPD vs Non-COPD")
    print("   If COPD → Final: COPD")
    print("   If Non-COPD → Stage 2 (XGB): Asthma vs Normal vs Pneumonia")
    print("\n👨‍⚕️ Doctor Features:")
    print("   - Waveform visualization")
    print("   - Spectrogram visualization")
    print("   - MFCC features visualization")
    print("   - Audio information (duration, sample rate, file size)")
    print("   - Probability breakdown for Stage 2")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)