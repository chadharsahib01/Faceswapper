import cv2, os, base64, logging, sys, uuid, gc
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
import insightface
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from rembg import remove

# --- Security & Logging ---
API_KEY = os.getenv("API_KEY", "ishark_secret_key_2026") 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Added Request ID tracking to logs
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["https://ishark-swap.vercel.app", "http://localhost:3000"], 
    allow_methods=["POST", "GET"], 
    allow_headers=["*"]
)

# --- AI Models ---
analyzer = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
analyzer.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("ESPCN_x2.pb")
sr.setModel("espcn", 2)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY: raise HTTPException(status_code=403, detail="Unauthorized API Key")
    return api_key

# FIX: Added dynamic resizing for 4K images to prevent "Ghosting" and memory OOM
def resize_if_large(img, max_size=1920):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

# FIX: Safely handles RGBA transparency now via cv2.IMREAD_UNCHANGED
async def load_image_from_bytes(file: UploadFile):
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024: 
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) 
    if img is None: raise HTTPException(status_code=400, detail="Corrupted or invalid image file.")
    
    # Standardize to 3 channels if it's 4 channels (we handle transparency separately later)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    return resize_if_large(img)

def color_transfer(source, target):
    try:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
        (l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = cv2.meanStdDev(source_lab)
        (l_mean_tar, l_std_tar, a_mean_tar, a_std_tar, b_mean_tar, b_std_tar) = cv2.meanStdDev(target_lab)
        
        # FIX: NaN protection for completely uniform backgrounds
        l_std_src_val, a_std_src_val, b_std_src_val = max(l_std_src.item(), 1e-5), max(a_std_src.item(), 1e-5), max(b_std_src.item(), 1e-5)

        source_lab[:,:,0] = ((source_lab[:,:,0] - l_mean_src.item()) * (l_std_tar.item() / l_std_src_val)) + l_mean_tar.item()
        source_lab[:,:,1] = ((source_lab[:,:,1] - a_mean_src.item()) * (a_std_tar.item() / a_std_src_val)) + a_mean_tar.item()
        source_lab[:,:,2] = ((source_lab[:,:,2] - b_mean_src.item()) * (b_std_tar.item() / b_std_src_val)) + b_mean_tar.item()
        
        # FIX: Prevent NaN infection in arrays
        np.nan_to_num(source_lab, copy=False, nan=0.0)
        source_lab = np.clip(source_lab, 0, 255).astype("uint8")
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logger.error(f"Color transfer skipped due to math error: {e}")
        return source

# FIX: Reduced Base64 payload bloat by applying JPEG compression quality (85%)
def img_to_b64(img, ext='.jpg'):
    if ext == '.jpg' or ext == '.jpeg':
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    else:
        _, buffer = cv2.imencode(ext, img)
    return base64.b64encode(buffer).decode('utf-8')

# Middleware to inject Request IDs
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = str(uuid.uuid4())
    request.state.req_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response

@app.post("/detect/")
@limiter.limit("10/minute")
async def detect_faces(request: Request, source_image: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    logger.info(f"[Req: {request.state.req_id}] Detection started.")
    try:
        img = await load_image_from_bytes(source_image)
        faces = analyzer.get(img)
        if not faces: 
            raise HTTPException(status_code=400, detail="NO_FACE_DETECTED") # Standardized error code
            
        face_data = []
        for idx, face in enumerate(faces):
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(img.shape[1], box[2]), min(img.shape[0], box[3])
            cropped = img[y1:y2, x1:x2]
            face_data.append({"index": idx, "thumbnail": f"data:image/jpeg;base64,{img_to_b64(cropped)}"})
            
        return JSONResponse(content={"faces": face_data})
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"[Req: {request.state.req_id}] Detection Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
    finally:
        gc.collect() # FIX: InsightFace Memory Leak cleanup

@app.post("/swap/")
@limiter.limit("5/minute")
async def swap_face(
    request: Request,
    source_image: UploadFile = File(...), 
    target_image: UploadFile = File(...),
    face_index: int = Form(...),
    remove_bg: str = Form("false"), 
    api_key: str = Depends(verify_api_key)
):
    logger.info(f"[Req: {request.state.req_id}] Swap requested.")
    remove_bg_bool = remove_bg.lower() == "true"
    
    try:
        scene_img = await load_image_from_bytes(source_image)
        target_img = await load_image_from_bytes(target_image)
        
        target_faces, scene_faces = analyzer.get(target_img), analyzer.get(scene_img)
        if not target_faces: raise HTTPException(status_code=400, detail="No face in target image.")
        if face_index >= len(scene_faces): raise HTTPException(status_code=400, detail="Invalid face selection.")
            
        target_img_blended = color_transfer(target_img, scene_img)
        target_faces_blended = analyzer.get(target_img_blended)
        use_target = target_faces_blended[0] if target_faces_blended else target_faces[0]

        result_img = swapper.get(scene_img, scene_faces[face_index], use_target, paste_back=True)
        
        try:
            result_img = sr.upsample(result_img)
        except Exception as e:
            logger.warning(f"[Req: {request.state.req_id}] SuperRes bypassed: {e}")
        
        new_faces = analyzer.get(result_img)
        deep_zoom_b64 = None
        if new_faces:
            box = new_faces[0].bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(result_img.shape[1], box[2]), min(result_img.shape[0], box[3])
            deep_zoom = result_img[y1:y2, x1:x2]
            deep_zoom_b64 = img_to_b64(deep_zoom, '.jpg')

        mime_type = "image/jpeg"
        ext = '.jpg'
        if remove_bg_bool:
            logger.info(f"[Req: {request.state.req_id}] Executing BG Removal...")
            _, buffer = cv2.imencode('.png', result_img)
            bg_removed_bytes = remove(buffer.tobytes())
            nparr = np.frombuffer(bg_removed_bytes, np.uint8)
            result_img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) 
            mime_type = "image/png"
            ext = '.png'

        main_b64 = img_to_b64(result_img, ext)
        
        return JSONResponse(content={
            "main_image": f"data:{mime_type};base64,{main_b64}",
            "deep_zoom": f"data:image/jpeg;base64,{deep_zoom_b64}" if deep_zoom_b64 else None
        })
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"[Req: {request.state.req_id}] Swap Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Processing error. Image may be too complex.")
    finally:
        gc.collect() # FIX: Prevent Out Of Memory crashes

@app.get("/health")
def root(): return {"status": "Enterprise API Running.", "uptime": "OK"}
