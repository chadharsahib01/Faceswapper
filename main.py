import cv2, shutil, uuid, os, base64, logging, sys
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import insightface
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from rembg import remove

# --- Security & Logging ---
API_KEY = "ishark_secret_key_2026" 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# FIX: Route logs directly to Hugging Face standard output so we can see errors in real-time
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- AI Models ---
analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
analyzer.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

# Super Resolution Setup
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("ESPCN_x2.pb")
sr.setModel("espcn", 2)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY: raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

def cleanup_files(file_paths: list):
    for path in file_paths:
        if os.path.exists(path): os.remove(path)

def color_transfer(source, target):
    """Seamless Skin Tone Blending via Lab Color Space"""
    try:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
        (l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = cv2.meanStdDev(source_lab)
        (l_mean_tar, l_std_tar, a_mean_tar, a_std_tar, b_mean_tar, b_std_tar) = cv2.meanStdDev(target_lab)
        
        # FIX: Prevent division by zero crashes
        l_std_src = max(l_std_src[0][0], 1e-5)
        a_std_src = max(a_std_src[0][0], 1e-5)
        b_std_src = max(b_std_src[0][0], 1e-5)

        source_lab[:,:,0] = ((source_lab[:,:,0] - l_mean_src[0][0]) * (l_std_tar[0][0] / l_std_src)) + l_mean_tar[0][0]
        source_lab[:,:,1] = ((source_lab[:,:,1] - a_mean_src[0][0]) * (a_std_tar[0][0] / a_std_src)) + a_mean_tar[0][0]
        source_lab[:,:,2] = ((source_lab[:,:,2] - b_mean_src[0][0]) * (b_std_tar[0][0] / b_std_src)) + b_mean_tar[0][0]
        source_lab = np.clip(source_lab, 0, 255).astype("uint8")
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.error(f"Color transfer failed, using original: {e}")
        return target

# FIX: Add conditional encoding. If background is removed, it MUST encode as PNG.
def img_to_b64(img, is_png=False):
    ext = '.png' if is_png else '.jpg'
    _, buffer = cv2.imencode(ext, img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/detect/")
@limiter.limit("10/minute")
async def detect_faces(request: Request, source_image: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    req_id = str(uuid.uuid4())
    src_path = f"temp_det_{req_id}.jpg"
    try:
        with open(src_path, "wb") as f: shutil.copyfileobj(source_image.file, f)
        img = cv2.imread(src_path)
        if img is None: raise HTTPException(status_code=400, detail="Corrupted image file.")
        
        faces = analyzer.get(img)
        if not faces: raise HTTPException(status_code=400, detail="No faces detected.")
            
        face_data = []
        for idx, face in enumerate(faces):
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img.shape[1], box[2]), min(img.shape[0], box[3])
            cropped = img[y1:y2, x1:x2]
            face_data.append({"index": idx, "thumbnail": f"data:image/jpeg;base64,{img_to_b64(cropped)}"})
            
        cleanup_files([src_path])
        return JSONResponse(content={"faces": face_data})
    except Exception as e:
        cleanup_files([src_path])
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swap/")
@limiter.limit("5/minute")
async def swap_face(
    request: Request,
    background_tasks: BackgroundTasks,
    source_image: UploadFile = File(...), 
    target_image: UploadFile = File(...),
    face_index: int = Form(...),
    remove_bg: bool = Form(False),
    api_key: str = Depends(verify_api_key)
):
    logger.info(f"Swap requested. IP: {request.client.host}")
    req_id = str(uuid.uuid4())
    src_path, tgt_path = f"src_{req_id}.jpg", f"tgt_{req_id}.jpg"
    
    try:
        with open(src_path, "wb") as f: shutil.copyfileobj(source_image.file, f)
        with open(tgt_path, "wb") as f: shutil.copyfileobj(target_image.file, f)
        
        target_img, scene_img = cv2.imread(tgt_path), cv2.imread(src_path)
        if target_img is None or scene_img is None: raise HTTPException(status_code=400, detail="Corrupted image file.")
        
        target_faces, scene_faces = analyzer.get(target_img), analyzer.get(scene_img)
        
        if not target_faces: raise HTTPException(status_code=400, detail="No face in target image.")
        if face_index >= len(scene_faces): raise HTTPException(status_code=400, detail="Invalid face selection.")
            
        # Seamless Skin Tone Blending
        target_img_blended = color_transfer(target_img, scene_img)
        target_faces_blended = analyzer.get(target_img_blended)
        use_target = target_faces_blended[0] if target_faces_blended else target_faces[0]

        # Execute Swap
        result_img = swapper.get(scene_img, scene_faces[face_index], use_target, paste_back=True)
        
        # Super Resolution (Upscaling)
        try:
            result_img = sr.upsample(result_img)
        except Exception as e:
            logger.error(f"SuperRes failed (Likely OOM): {e}")
            # Failsafe: Continue without upscaling instead of crashing
        
        # Deep Zoom Crop
        new_faces = analyzer.get(result_img)
        deep_zoom_b64 = None
        if new_faces:
            box = new_faces[0].bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(result_img.shape[1], box[2]), min(result_img.shape[0], box[3])
            deep_zoom = result_img[y1:y2, x1:x2]
            deep_zoom_b64 = img_to_b64(deep_zoom, is_png=False)

        # Background Removal
        if remove_bg:
            result_img = remove(result_img) # Returns RGBA

        # FIX: Encode as PNG if background was removed
        main_b64 = img_to_b64(result_img, is_png=remove_bg)
        
        background_tasks.add_task(cleanup_files, [src_path, tgt_path])
        return JSONResponse(content={
            "main_image": f"data:image/{'png' if remove_bg else 'jpeg'};base64,{main_b64}",
            "deep_zoom": f"data:image/jpeg;base64,{deep_zoom_b64}" if deep_zoom_b64 else None
        })
        
    except Exception as e:
        logger.error(f"Swap Failed: {str(e)}")
        cleanup_files([src_path, tgt_path])
        raise HTTPException(status_code=500, detail="Backend processing error. See logs.")

@app.get("/health")
def root(): return {"status": "Enterprise API Running.", "uptime": "OK"}
