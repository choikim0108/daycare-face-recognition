import os
import shutil
import numpy as np
from deepface import DeepFace
from PIL import Image

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"

def get_largest_face_embedding(img_path):
    """
    이미지에서 가장 큰 얼굴의 임베딩을 반환합니다.
    (프로필 사진에 여러 얼굴이 있을 경우를 대비)
    """
    try:
        faces = DeepFace.represent(
            img_path=img_path, 
            model_name=MODEL_NAME, 
            detector_backend=DETECTOR_BACKEND, 
            enforce_detection=True
        )
        if not faces:
            return None
        
        # 가장 큰 얼굴 찾기 (가로 * 세로 넓이 기준)
        largest_face = max(faces, key=lambda f: f['facial_area']['w'] * f['facial_area']['h'])
        return np.array(largest_face['embedding'])
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def build_profiles(profile_dir, progress_callback=None):
    """
    프로필 폴더를 순회하여 각 아동별 평균 임베딩을 계산합니다.
    반환: { '아동이름': numpy_array_평균임베딩, ... }
    """
    profiles = {}
    
    if not os.path.exists(profile_dir):
        return profiles

    child_folders = [f for f in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, f)) and f != "미분류(Unknown)"]
    total_folders = len(child_folders)
    
    for i, child_name in enumerate(child_folders):
        child_dir = os.path.join(profile_dir, child_name)
        image_files = [f for f in os.listdir(child_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(child_dir, img_file)
            emb = get_largest_face_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
        
        if embeddings:
            # 아동의 모든 사진 임베딩을 평균내어 단일 마스터 임베딩 생성
            avg_embedding = np.mean(embeddings, axis=0)
            # Normalize (ArcFace 등 Cosine Similarity 사용을 위해)
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            profiles[child_name] = avg_embedding
            
        if progress_callback:
            progress_callback("profiles", i + 1, total_folders, f"{child_name} 프로필 학습 완료")
            
    return profiles

def get_cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def process_photos(input_dir, profile_dir, profiles, progress_callback=None, threshold=0.32):
    """
    원본 사진들을 분석하여 일치하는 아동 폴더로 복사합니다.
    threshold: ArcFace 모델의 일반적인 코사인 유사도 임계값 (거리 임계값이 0.68이므로 유사도는 약 0.32 이상)
    """
    if not os.path.exists(input_dir):
        return
        
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    
    unknown_dir = os.path.join(profile_dir, "미분류(Unknown)")
    os.makedirs(unknown_dir, exist_ok=True)
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_dir, img_file)
        
        try:
            # 단체 사진일 수 있으므로 enforce_detection=False (얼굴 없으면 빈 리스트 리턴할 수 있도록)
            faces = DeepFace.represent(
                img_path=img_path, 
                model_name=MODEL_NAME, 
                detector_backend=DETECTOR_BACKEND, 
                enforce_detection=False
            )
            
            matched_children = set()
            
            if isinstance(faces, list) and len(faces) > 0 and 'embedding' in faces[0] and faces[0]['embedding'] is not None:
                for face in faces:
                    emb = np.array(face['embedding'])
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                        
                    best_match = None
                    best_score = -1
                    
                    for child_name, profile_emb in profiles.items():
                        sim = get_cosine_similarity(emb, profile_emb)
                        if sim > best_score:
                            best_score = sim
                            best_match = child_name
                            
                    if best_match and best_score > threshold:
                        matched_children.add(best_match)
            
            # 매칭된 아동이 있으면 각각의 폴더로 복사 (단체 사진 중복 분배)
            if matched_children:
                for child in matched_children:
                    target_dir = os.path.join(profile_dir, child)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy2(img_path, os.path.join(target_dir, img_file))
            else:
                # 얼굴이 없거나 매칭된 아동이 없으면 미분류
                shutil.copy2(img_path, os.path.join(unknown_dir, img_file))
                
        except Exception as e:
            # 탐지 오류 (얼굴 없음 등) 발생 시 미분류로 처리
            print(f"Failed to process {img_file}: {e}")
            shutil.copy2(img_path, os.path.join(unknown_dir, img_file))
            
        if progress_callback:
            progress_callback("process", i + 1, total_images, f"{img_file} 처리 완료")
