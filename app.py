import io
import base64
import gc
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
from engine.biometric import BiometricAnalyzer
from engine.pgd import AegisPGDEngine

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

biometric_scanner = BiometricAnalyzer()
pgd_engine = AegisPGDEngine()

VOLATILE_EXPORT_STORE = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/auth", methods=["POST"])
def authenticate():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    if username and password:
        return jsonify({"status": "success", "message": "Authenticated to Local Edge Environment"})
    return jsonify({"status": "error", "message": "Invalid authentication credentials"}), 400

@app.route("/api/scan_vulnerability", methods=["POST"])
def scan_vulnerability():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No media asset uploaded"}), 400
    
    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
        analysis = biometric_scanner.analyze_and_generate_heatmap(image)
        
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        orig_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "status": "success",
            "landmarks_count": analysis["landmarks_count"],
            "heatmap_preview": analysis["heatmap_preview"],
            "original_preview": f"data:image/jpeg;base64,{orig_b64}"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/shield", methods=["POST"])
def execute_shield():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No media asset uploaded"}), 400
    
    file = request.files["file"]
    epsilon = float(request.form.get("epsilon", 0.005))
    iterations = int(request.form.get("iterations", 10))

    try:
        image = Image.open(file.stream).convert("RGB")
        analysis = biometric_scanner.analyze_and_generate_heatmap(image)
        
        results = pgd_engine.optimize_shield(
            image_pil=image,
            spatial_mask=analysis["spatial_mask"],
            epsilon=epsilon,
            iterations=iterations
        )

        buf = io.BytesIO()
        results["shielded_image"].save(buf, format="JPEG", quality=95)
        shielded_bytes = buf.getvalue()
        shielded_b64 = base64.b64encode(shielded_bytes).decode("utf-8")

        session_id = "local_active_tensor_session"
        VOLATILE_EXPORT_STORE[session_id] = shielded_bytes

        return jsonify({
            "status": "success",
            "shielded_preview": f"data:image/jpeg;base64,{shielded_b64}",
            "baseline_confidence": results["baseline_confidence"],
            "shielded_confidence": results["shielded_confidence"],
            "protection_score": results["protection_score"],
            "ssim": results["ssim"]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/export", methods=["GET"])
def export_and_purge():
    session_id = "local_active_tensor_session"
    purge_requested = request.args.get("purge", "true").lower() == "true"

    if session_id in VOLATILE_EXPORT_STORE:
        if purge_requested:
            file_data = VOLATILE_EXPORT_STORE.pop(session_id)
            gc.collect()
        else:
            file_data = VOLATILE_EXPORT_STORE[session_id]

        return send_file(
            io.BytesIO(file_data),
            mimetype="image/jpeg",
            as_attachment=True,
            download_name="aegis_shielded_protected.jpg"
        )
    return jsonify({"status": "error", "message": "No protected asset in volatile memory"}), 404

@app.route("/api/purge", methods=["POST"])
def purge_memory():
    VOLATILE_EXPORT_STORE.clear()
    gc.collect()
    return jsonify({"status": "success", "message": "Memory purge complete"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
