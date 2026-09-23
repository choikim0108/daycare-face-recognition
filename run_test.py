import os
import shutil

base_dir = r"C:\Users\user\.gemini\antigravity\scratch\daycare-face-recognition\test_data"
artifact_dir = r"C:\Users\user\.gemini\antigravity\brain\48d9da27-ce50-47a3-ad44-759b86ef01f2"

profiles_dir = os.path.join(base_dir, "Profiles")
input_dir = os.path.join(base_dir, "Input")

os.makedirs(os.path.join(profiles_dir, "Alice"), exist_ok=True)
os.makedirs(os.path.join(profiles_dir, "Bob"), exist_ok=True)
os.makedirs(input_dir, exist_ok=True)

# Find the generated images
images = [f for f in os.listdir(artifact_dir) if f.endswith(".jpg")]

alice_img = next((f for f in images if "profile_alice" in f), None)
bob_img = next((f for f in images if "profile_bob" in f), None)
group_img = next((f for f in images if "group_alice_bob" in f), None)
unknown_img = next((f for f in images if "unknown_charlie" in f), None)

if alice_img:
    shutil.copy(os.path.join(artifact_dir, alice_img), os.path.join(profiles_dir, "Alice", "1.jpg"))
    shutil.copy(os.path.join(artifact_dir, alice_img), os.path.join(input_dir, "alice_test.jpg"))
if bob_img:
    shutil.copy(os.path.join(artifact_dir, bob_img), os.path.join(profiles_dir, "Bob", "1.jpg"))
if group_img:
    shutil.copy(os.path.join(artifact_dir, group_img), os.path.join(input_dir, "group_test.jpg"))
if unknown_img:
    shutil.copy(os.path.join(artifact_dir, unknown_img), os.path.join(input_dir, "unknown_test.jpg"))

print("Test data setup complete.")

import sys
sys.path.append(r"C:\Users\user\.gemini\antigravity\scratch\daycare-face-recognition")

import face_engine

print("Building profiles...")
profiles = face_engine.build_profiles(profiles_dir)
print(f"Profiles built: {list(profiles.keys())}")

print("Processing photos...")
face_engine.process_photos(input_dir, profiles_dir, profiles, threshold=0.50) # Lower threshold slightly for generated images which might have variations

print("Process complete. Checking results:")
for root, dirs, files in os.walk(profiles_dir):
    print(f"\nFolder: {os.path.basename(root)}")
    for file in files:
        print(f" - {file}")

