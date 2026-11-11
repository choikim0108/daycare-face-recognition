"""Lossless storage operations; no model or GUI dependency."""
import os
import shutil
from pathlib import Path

UNKNOWN = "미분류(Unknown)"
RESULT_MARKER = '.daycare-classification-results'

def reject_results_as_source(path):
    root = resolved(path)
    if any((p / RESULT_MARKER).exists() for p in (root, *root.parents)):
        raise ValueError("이 폴더는 분류 결과입니다. 원본이나 기준으로 재사용할 수 없습니다.")

def mark_results(output_dir):
    root = resolved(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / RESULT_MARKER
    try:
        with marker.open('x', encoding='utf-8') as file:
            file.write('Classification results only. Do not use as input or reference.\n')
    except FileExistsError:
        pass

def resolved(path):
    if not path:
        raise ValueError("기준, 원본, 결과 폴더를 모두 선택해주세요.")
    return Path(path).resolve()

def overlaps(a, b):
    return a == b or a in b.parents or b in a.parents

def validate_directories(input_dir, profile_dir, output_dir):
    paths = [resolved(p) for p in (input_dir, profile_dir, output_dir)]
    reject_results_as_source(paths[0])
    reject_results_as_source(paths[1])
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            if overlaps(a, b):
                raise ValueError("원본·기준·결과는 서로 포함되지 않는 별도 폴더여야 합니다.")
    if not paths[0].is_dir() or not paths[1].is_dir():
        raise ValueError("원본과 기준 폴더가 존재해야 합니다.")
    return paths

def child_names(root, include_unknown=False):
    if not root or not Path(root).is_dir():
        return []
    return sorted(p.name for p in Path(root).iterdir()
                  if p.is_dir() and not p.is_symlink()
                  and (include_unknown or p.name != UNKNOWN))

def child_directory(root, child):
    base = resolved(root)
    if not child or child in (".", "..") or '/' in child or '\\' in child:
        raise ValueError("잘못된 아동 폴더 이름입니다.")
    target = (base / child).resolve()
    if target.parent != base:
        raise ValueError("외부 폴더 연결은 사용할 수 없습니다.")
    return target

def review_source(source, output_dir):
    source = resolved(source)
    root = resolved(output_dir)
    if source.parent.parent != root or not source.is_file():
        raise ValueError("선택한 파일이 결과 폴더의 아동 폴더에 없습니다.")
    return source

def copy_unique(source, target_dir):
    """Reserve destination exclusively, preserving every existing file."""
    source = Path(source)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        name = source.name if index == 0 else f"{source.stem}_{index}{source.suffix}"
        target = target_dir / name
        try:
            dest = target.open('xb')
        except FileExistsError:
            index += 1
            continue
        try:
            with source.open('rb') as src, dest:
                shutil.copyfileobj(src, dest)
        except BaseException:
            dest.close()
            target.unlink(missing_ok=True)
            raise
        return target

def move_review_photo(source, output_dir, child):
    source = review_source(source, output_dir)
    target_dir = child_directory(output_dir, child)
    if source.parent == target_dir:
        return source
    target = copy_unique(source, target_dir)
    # Only remove the reviewed result after a complete copy. Input is untouched.
    source.unlink()
    return target
