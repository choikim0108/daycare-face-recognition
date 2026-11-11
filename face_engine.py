import os
import numpy as np
from photo_storage import (validate_directories, child_names, child_directory,
                           copy_unique, review_source, move_review_photo, overlaps, resolved,
                           mark_results, reject_results_as_source)

def represent(**kwargs):
    from deepface import DeepFace
    return DeepFace.represent(**kwargs)

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"

def get_largest_face_embedding(img_path):
    """
    이미지에서 가장 큰 얼굴의 임베딩을 반환합니다.
    (프로필 사진에 여러 얼굴이 있을 경우를 대비)
    """
    try:
        faces = represent(
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
    reject_results_as_source(profile_dir)

    if not os.path.exists(profile_dir):
        return profiles

    child_folders = child_names(profile_dir)
    total_folders = len(child_folders)

    for i, child_name in enumerate(child_folders):
        child_dir = child_directory(profile_dir, child_name)
        image_files = [f for f in os.listdir(child_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(child_dir, f)) and not os.path.islink(os.path.join(child_dir, f))]

        # 사진이 너무 많아지면 학습 시간이 기하급수적으로 늘어나는 것을 방지 (최신 20장만 사용)
        if len(image_files) > 20:
            # 파일 수정 시간(최신순)으로 정렬하여 상위 20개만 추출
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(child_dir, x)), reverse=True)
            image_files = image_files[:20]

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

def process_photos(input_dir, output_dir, profiles, progress_callback=None, threshold=0.32, *, profile_dir):
    """
    원본 사진들을 분석하여 일치하는 아동 폴더로 복사합니다.
    threshold: ArcFace 모델의 일반적인 코사인 유사도 임계값 (거리 임계값이 0.68이므로 유사도는 약 0.32 이상)
    """
    validate_directories(input_dir, profile_dir, output_dir)
    for child in profiles:
        child_directory(output_dir, child)
    mark_results(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.isfile(os.path.join(input_dir, f)) and not os.path.islink(os.path.join(input_dir, f))]
    total_images = len(image_files)

    unknown_dir = str(child_directory(output_dir, "미분류(Unknown)"))
    os.makedirs(unknown_dir, exist_ok=True)

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_dir, img_file)

        try:
            # 얼굴이 탐지되지 않으면 예외를 통해 미분류로 보존합니다.
            faces = represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True
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

        except Exception as e:
            # 탐지 오류 (얼굴 없음 등) 발생 시 미분류로 처리
            print(f"Failed to process {img_file}: {e}")
            matched_children = set()

        # 저장 실패는 호출자에게 전달합니다. 이미 보존한 결과는 유지됩니다.
        if matched_children:
            for child in sorted(matched_children):
                copy_unique(img_path, child_directory(output_dir, child))
        else:
            copy_unique(img_path, unknown_dir)

        if progress_callback:
            progress_callback("process", i + 1, total_images, f"{img_file} 처리 완료")

def approve_reference_photo(source, output_dir, profile_dir, child):
    """Explicit approval only; group photos cannot become references."""
    reject_results_as_source(profile_dir)
    if overlaps(resolved(output_dir), resolved(profile_dir)):
        raise ValueError("결과와 기준 폴더는 분리해야 합니다.")
    source = review_source(source, output_dir)
    if child not in child_names(profile_dir) or source.parent.name != child:
        raise ValueError("올바른 아동의 결과 폴더로 먼저 이동해주세요.")
    faces = represent(img_path=str(source), model_name=MODEL_NAME,
                      detector_backend=DETECTOR_BACKEND, enforce_detection=True)
    if not isinstance(faces, list) or len(faces) != 1 or faces[0].get('embedding') is None:
        raise ValueError("기준에는 얼굴이 정확히 하나인 사진만 추가할 수 있습니다.")
    return copy_unique(source, child_directory(profile_dir, child))
