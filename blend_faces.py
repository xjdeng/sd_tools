import os
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
import torch
import safetensors.torch
from typing import List

def blend_faces_to_file(
    image_paths: List[str],
    output_file: str,
    compute_method: str = "mean",   # Options: "mean", "median", or "mode"
    shape_check: bool = False,
    ctx_id: int = 0
):
    """
    Blends multiple face images into one face model file that includes the blended embedding 
    and all required metadata for ReActor. The output file will contain:
      - "embedding": [512] float32
      - "bbox": [4] float32
      - "age": scalar int64
      - "det_score": scalar float32
      - "gender": scalar int64
      - "kps": [5,2] float32
      - "landmark_2d_106": [106,2] float32
      - "landmark_3d_68": [68,3] float32
      - "pose": [3] float32

    If any attribute is missing from the first detected face, a default (zero) value is used.
    
    Parameters:
      - image_paths: list of file paths containing face images.
      - output_file: destination path for the .safetensors file; if the extension is missing, it is appended.
      - compute_method: method for blending embeddings ("mean", "median", or "mode").
      - shape_check: if True, only include embeddings close to the first one.
      - ctx_id: GPU index (e.g. 0) or CPU (-1).
    """
    # Ensure output file ends with .safetensors
    if not output_file.endswith(".safetensors"):
        output_file += ".safetensors"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize the face analysis model (using the same settings as ReActor)
    face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=ctx_id)

    embeddings = []
    first_face = None

    # Process each input image
    for i, path in enumerate(image_paths):
        img = np.array(Image.open(path).convert("RGB"))
        faces = face_app.get(img)
        if not faces:
            raise ValueError(f"No face detected in {path}")
        face = faces[0]
        if i == 0:
            first_face = face
        embeddings.append(face.embedding)

    if shape_check:
        ref = embeddings[0]
        filtered = [e for e in embeddings if np.linalg.norm(e - ref) < 0.6]
        if not filtered:
            raise ValueError("All embeddings rejected by shape check")
        embeddings = filtered

    # Compute the blended embedding
    stack = np.stack(embeddings)
    if compute_method == "mean":
        blended = np.mean(stack, axis=0)
    elif compute_method == "median":
        blended = np.median(stack, axis=0)
    elif compute_method == "mode":
        from scipy.stats import mode
        blended = mode(stack, axis=0).mode[0]
    else:
        raise ValueError(f"Unsupported compute method: {compute_method}")
    
    # Normalize and create a 1D tensor of shape [512]
    blended /= np.linalg.norm(blended)
    embedding_tensor = torch.tensor(blended.astype(np.float32))  # shape: [512]

    # --- Build metadata from the first face (or defaults if missing) ---
    def default_tensor(shape, dtype=torch.float32):
        return torch.zeros(shape, dtype=dtype)
    
    # bbox: expected shape [4]
    bbox_tensor = torch.tensor(getattr(first_face, "bbox", [0, 0, 0, 0]), dtype=torch.float32) if first_face is not None else default_tensor([4])
    
    # age: scalar int64
    age_tensor = torch.tensor(getattr(first_face, "age", 0), dtype=torch.int64) if first_face is not None else torch.tensor(0, dtype=torch.int64)
    
    # det_score: scalar float32
    det_score_tensor = torch.tensor(getattr(first_face, "det_score", 1.0), dtype=torch.float32) if first_face is not None else torch.tensor(1.0, dtype=torch.float32)
    
    # gender: scalar int64
    gender_tensor = torch.tensor(getattr(first_face, "gender", 0), dtype=torch.int64) if first_face is not None else torch.tensor(0, dtype=torch.int64)
    
    # kps: expected shape [5, 2]
    kps = getattr(first_face, "kps", None)
    if kps is None or np.array(kps).size != 10:
        kps_tensor = default_tensor((5, 2))
    else:
        kps_tensor = torch.tensor(np.array(kps), dtype=torch.float32)
    
    # landmark_2d_106: expected shape [106,2]
    lm2d = getattr(first_face, "landmark_2d_106", None)
    if lm2d is None or np.array(lm2d).size != 106 * 2:
        landmark_2d_106_tensor = default_tensor((106, 2))
    else:
        landmark_2d_106_tensor = torch.tensor(np.array(lm2d), dtype=torch.float32)
    
    # landmark_3d_68: expected shape [68,3]
    lm3d = getattr(first_face, "landmark_3d_68", None)
    if lm3d is None or np.array(lm3d).size != 68 * 3:
        landmark_3d_68_tensor = default_tensor((68, 3))
    else:
        landmark_3d_68_tensor = torch.tensor(np.array(lm3d), dtype=torch.float32)
    
    # pose: expected shape [3]
    pose = getattr(first_face, "pose", None)
    if pose is None or np.array(pose).size != 3:
        pose_tensor = default_tensor((3,))
    else:
        pose_tensor = torch.tensor(np.array(pose), dtype=torch.float32)

    # --- Assemble the final model data ---
    model_data = {
        "embedding": embedding_tensor,         # shape: [512]
        "bbox": bbox_tensor,                   # shape: [4]
        "age": age_tensor,                     # scalar int64
        "det_score": det_score_tensor,         # scalar float32
        "gender": gender_tensor,               # scalar int64
        "kps": kps_tensor,                     # shape: [5,2]
        "landmark_2d_106": landmark_2d_106_tensor,  # shape: [106,2]
        "landmark_3d_68": landmark_3d_68_tensor,    # shape: [68,3]
        "pose": pose_tensor                    # shape: [3]
    }

    safetensors.torch.save_file(model_data, output_file)
    print(f"✔ Saved blended face model to: {output_file}")
