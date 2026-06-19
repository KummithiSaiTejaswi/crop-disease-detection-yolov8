"""
app.py
======
Crop Disease Classification and Cure Suggestion
Developers : B Venkata Anil Kumar & A Pooja Samanvitha
Institution: SRM University, Chennai — B.Tech 3rd Year


"""

from flask import Flask, render_template, request, jsonify
import os
import base64
import io
import json
import uuid
import datetime
import urllib.parse as urlparse
from PIL import Image, ImageOps
import numpy as np
from model_handler import DiseaseDetector
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['HISTORY_FILE']       = 'static/data/history.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['HISTORY_FILE']), exist_ok=True)

detector = DiseaseDetector(model_path='best.pt')


# ══════════════════════════════════════════════════════════════════

DATABASE_URL = os.environ.get('DATABASE_URL')   # automatically set on Render

if DATABASE_URL:
    # ── Running on Render.com ─────────────────────────────────────
    url = urlparse.urlparse(DATABASE_URL)
    DB_CONFIG = {
        'host':     url.hostname,
        'port':     url.port or 5432,
        'database': url.path[1:],
        'user':     url.username,
        'password': url.password
    }
    USE_DATABASE = True
    print("[Config] Render PostgreSQL detected via DATABASE_URL")

else:
    # ── Running locally ───────────────────────────────────────────
    USE_DATABASE = True   # set False to use JSON file only

    DB_CONFIG = {
        'host':     'localhost',
        'port':     5432,
        'database': 'crop_disease_db',
        'user':     'postgres',
        'password': os.environ.get('DB_PASSWORD', 'your_password_here')
        #                                           ↑
        #            Replace 'your_password_here' with your real password
        #            OR set the DB_PASSWORD environment variable (safer)
    }
    print(f"[Config] Local storage: {'PostgreSQL' if USE_DATABASE else 'JSON file'}")


# ══════════════════════════════════════════════
# Storage helpers — JSON (default)
# ══════════════════════════════════════════════

def load_history():
    try:
        if os.path.exists(app.config['HISTORY_FILE']):
            with open(app.config['HISTORY_FILE'], 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[Storage] Error loading history: {e}")
    return []


def save_history(history):
    try:
        with open(app.config['HISTORY_FILE'], 'w') as f:
            json.dump(history, f, indent=4)
        return True
    except Exception as e:
        print(f"[Storage] Error saving history: {e}")
        return False


# ══════════════════════════════════════════════
# Storage helpers — PostgreSQL (when enabled)
# ══════════════════════════════════════════════

def get_db_connection():
    import psycopg2
    return psycopg2.connect(**DB_CONFIG)


def save_to_db(history_item):
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO detections (
                id, detected_at, disease_name, display_name,
                confidence, plant_type,
                description, symptoms, treatment, prevention,
                image_path
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            history_item['id'],
            history_item['date'],
            history_item.get('disease_name'),
            history_item.get('display_name', history_item.get('disease_name')),
            history_item['confidence'],
            history_item['plant_type'],
            history_item['description'],
            json.dumps(history_item['symptoms']),
            json.dumps(history_item['treatment']),
            json.dumps(history_item['prevention']),
            history_item['image_path']
        ))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Error saving: {e}")
        return False


def load_from_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT id, detected_at, disease_name, display_name,
                   confidence, plant_type,
                   description, symptoms, treatment, prevention,
                   image_path
            FROM detections
            ORDER BY detected_at DESC
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        history = []

        for row in rows:
            confidence_value = float(row[4])

            history.append({
                'id':               row[0],
                'date':             str(row[1]),
                'disease_name':     row[2],
                'display_name':     row[3],
                'confidence':       confidence_value,
                'confidence_color': confidence_color(confidence_value),
                'plant_type':       row[5],
                'description':      row[6],
                'symptoms':         json.loads(row[7]) if row[7] else [],
                'treatment':        json.loads(row[8]) if row[8] else [],
                'prevention':       json.loads(row[9]) if row[9] else [],
                'image_path':       row[10]
            })

        return history

    except Exception as e:
        print("[DB] Error loading:", e)
        return []


def delete_from_db(item_id):
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("DELETE FROM detections WHERE id = %s", (item_id,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Error deleting: {e}")
        return False


# ══════════════════════════════════════════════
# Unified storage interface
# ══════════════════════════════════════════════

def storage_save(history_item):
    # Always save to JSON
    history = load_history()
    history.append(history_item)
    save_history(history)

    # Also save to DB if enabled
    if USE_DATABASE:
        save_to_db(history_item)

    return True


def storage_load():
    if USE_DATABASE:
        try:
            data = load_from_db()
            if data:
                return data
        except Exception as e:
            print("[Storage] DB load failed:", e)

    return load_history()


def storage_delete(item_id, image_path=''):
    # Delete from JSON
    history = load_history()
    history = [h for h in history if h['id'] != item_id]
    save_history(history)

    # Delete from DB
    if USE_DATABASE:
        delete_from_db(item_id)

    # Delete image file
    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except:
            pass


# ══════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════

def pil_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.convert('RGB').save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def confidence_color(pct):
    if pct >= 80: return '#27ae60'
    if pct >= 60: return '#f39c12'
    return '#e74c3c'


def annotate_image(pil_image, result):
    img_np  = np.array(pil_image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    display_name = result.get('display_name', result.get('disease_name', ''))
    confidence   = result.get('confidence', 0) * 100
    label        = f"{display_name}  {confidence:.1f}%"

    banner_h = 40
    overlay  = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (img_bgr.shape[1], banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 2
    (tw, _), _ = cv2.getTextSize(label, font, scale, thick)
    tx = max(8, (img_bgr.shape[1] - tw) // 2)
    cv2.putText(img_bgr, label, (tx, 28), font, scale, (0, 255, 100), thick, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ══════════════════════════════════════════════
# Core: build result + AUTO-SAVE in one function
# ══════════════════════════════════════════════

def process_and_save(pil_image, result):
    """
    Called by both /upload and /capture.
    1. Saves image to disk
    2. Saves detection record to storage
    3. Returns JSON response to frontend
    """
    item_id    = str(uuid.uuid4())
    img_path   = os.path.join(app.config['UPLOAD_FOLDER'], f"{item_id}.png")
    confidence = result['confidence'] * 100

    pil_image.save(img_path)

    annotated   = annotate_image(pil_image, result)
    img_b64     = pil_to_base64(pil_image)
    img_ann_b64 = pil_to_base64(annotated)

    raw_name   = result['disease_name']
    plant_type = raw_name.split('___')[0].replace(',', '').replace('_', ' ').strip()

    history_item = {
        'id':               item_id,
        'date':             datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'disease_name':     raw_name,
        'display_name':     result.get('display_name', raw_name),
        'confidence':       round(confidence, 2),
        'confidence_color': confidence_color(confidence),
        'plant_type':       plant_type,
        'description':      result['info']['description'],
        'symptoms':         result['info']['symptoms'],
        'treatment':        result['info']['treatment'],
        'prevention':       result['info']['prevention'],
        'image_path':       img_path
    }

    saved = storage_save(history_item)
    if saved:
        print(f"[Storage] Saved → {history_item.get('display_name', raw_name)} ({confidence:.1f}%) id={item_id}")
    else:
        print(f"[Storage] WARNING: Failed to save id={item_id}")

    return jsonify({
        'success':          True,
        'saved_id':         item_id,
        'image':            img_b64,
        'image_with_boxes': img_ann_b64,
        'result': {
            'disease_name':  raw_name,
            'display_name':  result.get('display_name', raw_name),
            'confidence':    result['confidence'],
            'top5':          result.get('top5', []),
            'bounding_box':  None,
            'info':          result['info']
        }
    })


# ══════════════════════════════════════════════
# Routes — Pages
# ══════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/history')
def history():
    history_items = storage_load()
    return render_template('history.html', history_items=history_items)


@app.route('/history/<item_id>')
def history_detail(item_id):
    history_items = storage_load()
    item = next((i for i in history_items if i['id'] == item_id), None)

    if not item:
        return render_template('history_detail.html', item=None)

    related_items = [
        i for i in history_items
        if (i['disease_name'] == item['disease_name'] or
            i['plant_type']   == item['plant_type']) and i['id'] != item['id']
    ]
    related_items = sorted(related_items, key=lambda x: x['date'], reverse=True)[:4]
    all_related   = sorted(related_items + [item], key=lambda x: x['date'])

    return render_template('history_detail.html',
                           item=item,
                           related_items=all_related,
                           all_items=history_items)


@app.route('/history/<item_id>/data')
def history_item_data(item_id):
    history_items = storage_load()
    item = next((i for i in history_items if i['id'] == item_id), None)
    if not item:
        return jsonify({'success': False, 'error': 'Item not found'})
    return jsonify({'success': True, 'item': item})


# ══════════════════════════════════════════════
# Routes — Actions
# ══════════════════════════════════════════════

@app.route('/history/<item_id>/delete', methods=['POST'])
def delete_history_item(item_id):
    history_items = storage_load()
    item = next((i for i in history_items if i['id'] == item_id), None)
    if not item:
        return jsonify({'success': False, 'error': 'Item not found'})
    storage_delete(item_id, item.get('image_path', ''))
    return jsonify({'success': True})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        if img.size != (480, 480):
            img = ImageOps.fit(img, (480, 480), Image.LANCZOS)

        result = detector.detect_disease(img)

        if result is None:
            return jsonify({
                'success': False,
                'image':   pil_to_base64(img),
                'error':   'Could not classify the image. Please upload a clear plant leaf photo.'
            })

        return process_and_save(img, result)

    except Exception as e:
        print(f"[Upload] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/capture', methods=['POST'])
def capture():
    try:
        image_data = request.json.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image data'}), 400

        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]

        img = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
        if img.size != (480, 480):
            img = ImageOps.fit(img, (480, 480), Image.LANCZOS)

        result = detector.detect_disease(img)

        if result is None:
            return jsonify({
                'success': False,
                'image':   pil_to_base64(img),
                'error':   'Could not classify the image. Please capture a clear plant leaf photo.'
            })

        return process_and_save(img, result)

    except Exception as e:
        print(f"[Capture] Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/save_detection', methods=['POST'])
def save_detection():
    # Kept for backward compatibility — no longer needed
    return jsonify({
        'success': True,
        'message': 'Results are now auto-saved during detection.'
    })


if __name__ == '__main__':
    detector.load_model()
    print("\n" + "=" * 50)
    print("  Crop Disease Classification and Cure Suggestion")
    print(f"  Storage : {'PostgreSQL' if USE_DATABASE else 'JSON  →  static/data/history.json'}")
    print("  URL     : http://localhost:5000")
    print("=" * 50 + "\n")
    port       = int(os.environ.get('PORT', 5000))          # Render sets PORT automatically
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)