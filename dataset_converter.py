import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

DATASET_PATH = "dataset/asl_alphabet_train"
OUTPUT_FILE = "landmarks.csv"

START_FROM = "J"  # 👈 change this if needed

# Handle nested folder automatically
subfolders = os.listdir(DATASET_PATH)
if len(subfolders) == 1:
    inner_path = os.path.join(DATASET_PATH, subfolders[0])
    if os.path.isdir(inner_path):
        DATASET_PATH = inner_path

print(f"Using dataset path: {DATASET_PATH}")

start_collecting = False

with open(OUTPUT_FILE, "a", newline="") as f:
    writer = csv.writer(f)

    for label in sorted(os.listdir(DATASET_PATH)):
        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        # Start from J
        if label == START_FROM:
            start_collecting = True

        if not start_collecting:
            continue

        print(f"Processing {label}...")

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    data = []

                    for lm in hand_landmarks.landmark:
                        data.extend([lm.x, lm.y, lm.z])

                    data.append(label)
                    writer.writerow(data)

print("✅ Additional data appended to landmarks.csv")