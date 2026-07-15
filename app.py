import os
import io
import base64
import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Import the U-Net model from our unet.py file
from unet import UNet

app = FastAPI(title="Crack Segmentation API")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration matching our training setup
IMG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
MODEL_PATH = 'best_model.pth'

# Global variable to hold the model
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Loading model on {DEVICE}...")
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"WARNING: Model file {MODEL_PATH} not found. Ensure it is trained or downloaded.")

def image_to_base64(img_np):
    """Convert numpy array (RGB) to base64 encoded PNG string."""
    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(status_code=500, content={"error": "Model not loaded. Ensure best_model.pth is in the directory."})
        
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(image)
        
        # --- PRE-PROCESS: Scale down huge images to max 1024px to prevent OOM ---
        max_dim = 1024
        h, w = img_np.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_process = cv2.resize(img_np, (new_w, new_h))
        else:
            img_process = img_np.copy()
            
        h_orig, w_orig = img_process.shape[:2]
        
        # --- PATCH-BASED INFERENCE (Native Resolution) ---
        # 1. Pad image so dimensions are multiples of IMG_SIZE
        pad_h = (IMG_SIZE - (h_orig % IMG_SIZE)) % IMG_SIZE
        pad_w = (IMG_SIZE - (w_orig % IMG_SIZE)) % IMG_SIZE
        
        img_padded = cv2.copyMakeBorder(img_process, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        h_pad, w_pad = img_padded.shape[:2]
        
        # Initialize full probability map
        full_pred = np.zeros((h_pad, w_pad), dtype=np.float32)
        
        # 2. Iterate through patches
        for y in range(0, h_pad, IMG_SIZE):
            for x in range(0, w_pad, IMG_SIZE):
                patch = img_padded[y:y+IMG_SIZE, x:x+IMG_SIZE]
                
                # Normalize patch
                patch_norm = patch.astype(np.float32) / 255.0
                for i in range(3):
                    patch_norm[:, :, i] = (patch_norm[:, :, i] - MEAN[i]) / STD[i]
                
                # To tensor
                patch_tensor = torch.from_numpy(patch_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                
                # Inference on patch
                with torch.no_grad():
                    output = model(patch_tensor)
                    pred_patch = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Stitch back
                full_pred[y:y+IMG_SIZE, x:x+IMG_SIZE] = pred_patch
                
        # 3. Crop back to original dimensions
        pred = full_pred[:h_orig, :w_orig]
        
        # Create binary mask
        mask = (pred > 0.5)
        mask_uint8_raw = (mask.astype(np.uint8)) * 255
        
        # --- MULTI-FACTOR STRUCTURAL HEALTH ALGORITHM ---
        # 1. Original Resolution
        original_resolution = f"{w_orig}x{h_orig}"
        
        # 2. Confidence Score
        if mask.sum() > 0:
            confidence_score = float(pred[mask].mean() * 100)
        else:
            confidence_score = float((1.0 - pred[~mask]).mean() * 100)
        
        # 3. Multi-Factor Structural Integrity Score (SIS)
        total_pixels = pred.shape[0] * pred.shape[1]
        crack_pixels = int(mask.sum())
        crack_ratio = (crack_pixels / total_pixels) * 100
        
        # Factor A: Crack Area Ratio (0-20 pts)
        # 5% coverage is catastrophic for thin cracks
        area_score = min(crack_ratio / 5.0, 1.0) * 20
        
        # Factor B: Number of Distinct Cracks via Connected Components (0-15 pts)
        num_labels, labels = cv2.connectedComponents(mask_uint8_raw)
        num_cracks = num_labels - 1  # subtract the background label
        crack_count_score = min(num_cracks / 10.0, 1.0) * 15
        
        # Factor C: Maximum Crack Length (0-25 pts)
        # Use the FULL image diagonal as reference
        max_crack_length = 0
        img_diagonal = np.sqrt(w_orig**2 + h_orig**2)
        for label_id in range(1, num_labels):
            component = (labels == label_id).astype(np.uint8)
            coords = cv2.findNonZero(component)
            if coords is not None:
                x_coord, y_coord, w_coord, h_coord = cv2.boundingRect(coords)
                diag = np.sqrt(w_coord**2 + h_coord**2)
                max_crack_length = max(max_crack_length, diag)
        length_score = min(max_crack_length / img_diagonal, 1.0) * 25
        
        # Factor D: Average Crack Width (0-20 pts)
        # Use morphological skeleton to estimate width = area / skeleton_length
        avg_width = 0.0
        if crack_pixels > 0:
            skeleton = cv2.ximgproc.thinning(mask_uint8_raw) if hasattr(cv2, 'ximgproc') else mask_uint8_raw
            skeleton_pixels = (skeleton > 0).sum()
            if skeleton_pixels > 0:
                avg_width = crack_pixels / skeleton_pixels
            else:
                avg_width = 1.0
        width_score = min(avg_width / 8.0, 1.0) * 20
        
        # Factor E: Branching / Intersection Penalty (0-20 pts)
        # Only fires when multiple SIGNIFICANT cracks exist
        branching_score = 0.0
        if num_cracks >= 3:
            large_cracks = 0
            for label_id in range(1, num_labels):
                component_area = (labels == label_id).sum()
                if component_area > total_pixels * 0.005:  # must be > 0.5% of image
                    large_cracks += 1
            branching_score = min(large_cracks / 4.0, 1.0) * 20
        
        # Composite Damage Score (0-100)
        damage_score = area_score + crack_count_score + length_score + width_score + branching_score
        integrity_score = max(0, 100 - damage_score)  # 100 = perfect, 0 = destroyed
        
        # Debug logging
        print(f"\n--- SIS Debug ---")
        print(f"  Crack Ratio: {crack_ratio:.3f}%  |  Crack Pixels: {crack_pixels}/{total_pixels}")
        print(f"  Factor A (Area):      {area_score:.1f}/20")
        print(f"  Factor B (Count={num_cracks}):    {crack_count_score:.1f}/15")
        print(f"  Factor C (Length={max_crack_length:.0f}px): {length_score:.1f}/25")
        print(f"  Factor D (Width={avg_width:.1f}px):  {width_score:.1f}/20")
        print(f"  Factor E (Branch):    {branching_score:.1f}/20")
        print(f"  DAMAGE SCORE: {damage_score:.1f}/100  |  INTEGRITY: {integrity_score:.0f}/100")
        
        # Classify based on composite score
        if crack_pixels == 0:
            health_status = "Healthy"
            health_color = "green"
        elif integrity_score >= 75:
            health_status = "Minor Wear"
            health_color = "yellow"
        elif integrity_score >= 45:
            health_status = "Moderate Damage"
            health_color = "orange"
        else:
            health_status = "Critical Damage"
            health_color = "red"
        # ------------------------------------------------
        
        # Create overlay (Cracks in red)
        overlay = img_process.copy().astype(float) / 255.0
        overlay[mask, 0] = 1.0  # Red
        overlay[mask, 1] = 0.0  # Green
        overlay[mask, 2] = 0.0  # Blue
        
        # Blend original and overlay (60% original, 40% red mask)
        img_process_float = img_process.astype(float) / 255.0
        blended = 0.6 * img_process_float + 0.4 * overlay
        blended = np.clip(blended, 0, 1)
        blended_uint8 = (blended * 255).astype(np.uint8)
        
        # Create Probability Heatmap (JET colormap on raw sigmoid output)
        heatmap_gray = (pred * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        # Blend heatmap with original for context
        heatmap_blend = (0.4 * img_process_float + 0.6 * heatmap_rgb.astype(float) / 255.0)
        heatmap_blend = np.clip(heatmap_blend, 0, 1)
        heatmap_uint8 = (heatmap_blend * 255).astype(np.uint8)
        
        # Convert mask to visual format
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_rgb = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2RGB)
        
        # Encode to base64
        original_b64 = image_to_base64(img_process)
        overlay_b64 = image_to_base64(blended_uint8)
        mask_b64 = image_to_base64(mask_rgb)
        heatmap_b64 = image_to_base64(heatmap_uint8)
        
        return {
            "success": True,
            "original": f"data:image/png;base64,{original_b64}",
            "overlay": f"data:image/png;base64,{overlay_b64}",
            "mask": f"data:image/png;base64,{mask_b64}",
            "heatmap": f"data:image/png;base64,{heatmap_b64}",
            "resolution": original_resolution,
            "confidence": f"{confidence_score:.1f}%",
            "confidence_raw": round(confidence_score, 1),
            "health_status": health_status,
            "health_color": health_color,
            "crack_ratio": f"{crack_ratio:.2f}%",
            "integrity_score": f"{integrity_score:.0f}/100",
            "integrity_raw": round(integrity_score),
            "num_cracks": num_cracks,
            "max_crack_length_px": int(max_crack_length),
            "avg_width_px": f"{avg_width:.1f}",
            "factors": {
                "area": round(area_score, 1),
                "count": round(crack_count_score, 1),
                "length": round(length_score, 1),
                "width": round(width_score, 1),
                "branching": round(branching_score, 1)
            }
        }
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Mount static files for the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
