import imutils
import numpy as np
import cv2
from math import ceil
from model_answer import CNN_Model
from collections import defaultdict
import os
import pandas as pd

# Tạo thư mục lưu ảnh nếu chưa tồn tại
output_dir = "new_dataset"
os.makedirs(output_dir, exist_ok=True)

def crop_image(img):
    """Detect and crop answer blocks with area > 10% of total image area."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape[:2]
    total_area = height * width
    
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_canny = cv2.Canny(blurred, 100, 200)
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0
    min_area = 0.1 * total_area  # 10% of total image area

    if len(cnts) > 0:
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        for c in cnts:
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
            if w_curr * h_curr > min_area:
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)
                if len(ans_blocks) == 0 or (check_xy_min > 20000 and check_xy_max > 20000):
                    ans_blocks.append((gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    x_old, y_old, w_old, h_old = x_curr, y_curr, w_curr, h_curr

        sorted_ans_blocks = sorted(ans_blocks, key=lambda b: b[1][0])
        print(f"Found {len(sorted_ans_blocks)} answer blocks.")
        return sorted_ans_blocks
    return []

def process_ans_blocks(ans_blocks):
    """Process answer blocks into a list of answer regions."""
    list_answers = []
    for ans_block in ans_blocks:
        ans_block_img = ans_block[0]  # Đã là numpy array
        offset1 = ceil(ans_block_img.shape[0] / 6)
        for i in range(6):
            box_img = ans_block_img[i * offset1:(i + 1) * offset1, :]
            height_box = box_img.shape[0]

            # Tính toán để cắt 1.5/19 của chiều cao ở trên và 1/19 ở dưới
            cut_top = ceil(height_box * 1.5 / 19)
            cut_bottom = ceil(height_box * 1 / 19)
            box_img = box_img[cut_top:height_box - cut_bottom, :]  # Cắt phần không cần thiết

            offset2 = ceil(box_img.shape[0] / 5)

            # Lặp qua từng dòng trong box
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])
    return list_answers

def process_list_ans(list_answers):
    """Process answer regions into individual bubble choices."""
    list_choices = []
    img_width = list_answers[0].shape[1]
    offset = (img_width // 7) * 6 // 4
    start = (2 * img_width // 11)

    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 127, 255, cv2.THRESH_BINARY)[1]
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        raise ValueError(f"Length of list_choices must be 480, got {len(list_choices)}")
    return list_choices

def save_list_ans(list_ans, prefix="bubble"):
    """Save list_ans as individual images or a NumPy file."""
    for idx, choice in enumerate(list_ans):
        filename = os.path.join(output_dir, f"{prefix}_{idx:03d}.png")
        cv2.imwrite(filename, choice.squeeze())  # Squeeze để bỏ chiều 1
    print(f"Saved {len(list_ans)} bubble choices to {output_dir}")
    # Lưu dưới dạng file NumPy (tùy chọn)
    np.save(os.path.join(output_dir, "list_ans.npy"), np.array(list_ans))

def get_answers(list_answers):
    """Predict filled choices using CNN model."""
    results = defaultdict(list)
    model = CNN_Model('weight.keras').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    
    for idx, score in enumerate(scores):
        question = idx // 4
        if score[1] > 0.99:
            chosed_answer = ["A", "B", "C", "D"][idx % 4]
            results[question + 1].append(chosed_answer)
    return results

def annotate_answers(ans_blocks, answers, answer_key, questions_per_block=30, total_questions=120, img=None):
    if img is None:
        annotated_img = cv2.imread('output_resized.jpg')
    else:
        annotated_img = img

    ans_blocks = sorted(ans_blocks, key=lambda b: b[1][0])
    for block_idx, ans_block in enumerate(ans_blocks):
        block_img, (x, y, w, h) = ans_block
        offset1 = ceil(h / 6)
        question_offset = block_idx * questions_per_block

        for i in range(6):
            box_y = y + i * offset1
            box_h = offset1

            cut_top = ceil(box_h * 1 / 19)
            cut_bottom = ceil(box_h * 1.5 / 19)
            effective_box_h = box_h - cut_top - cut_bottom
            offset2 = ceil(effective_box_h / 5)

            for j in range(5):
                line_y = box_y + cut_top + j * offset2
                offset = (w // 7 * 6) // 4
                start = w // 7

                question = question_offset + i * 5 + j + 1
                if question > total_questions:
                    continue

                student_ans = answers.get(question, [])
                correct_ans = answer_key[question - 1]  # Adjust to 0-based index

                for k in range(4):
                    choice_x = x + start + k * offset
                    choice_w = offset

                    if student_ans and ["A", "B", "C", "D"][k] in student_ans:
                        color = (0, 255, 0) if ["A", "B", "C", "D"][k] == correct_ans else (0, 0, 255)
                        cv2.rectangle(annotated_img, (choice_x, line_y),
                                    (choice_x + choice_w, line_y + offset2), color, 2)
                    elif ["A", "B", "C", "D"][k] == correct_ans:
                        cv2.rectangle(annotated_img, (choice_x, line_y),
                                    (choice_x + choice_w, line_y + offset2), (0, 255, 0), 2)
    return annotated_img

import cv2
import os
import numpy as np

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs("newdataset/choice", exist_ok=True)
os.makedirs("newdataset/unchoice", exist_ok=True)

# Hàm lưu ảnh
def save_dataset(list_ans, folder):
    """Lưu danh sách ảnh vào thư mục tương ứng."""
    for idx, choice in enumerate(list_ans):
        filename = os.path.join(folder, f"bubble_{idx:03d}.png")
        cv2.imwrite(filename, choice.squeeze())  # Squeeze để bỏ chiều 1
    print(f"Saved {len(list_ans)} bubble choices to {folder}")

# Đọc hai ảnh đầu vào
img_choice = cv2.imread('Image_choice.jpg')
img_unchoice = cv2.imread('Image_unchoice.jpg')

# Kiểm tra ảnh có tồn tại không
if img_choice is None or img_unchoice is None:
    raise FileNotFoundError("One or both images not found. Please check file paths.")

# Cắt các ô trả lời từ mỗi ảnh
list_ans_boxes_choice = crop_image(img_choice)
list_ans_boxes_unchoice = crop_image(img_unchoice)

if not list_ans_boxes_choice or not list_ans_boxes_unchoice:
    print("No answer blocks found in one or both images. Exiting.")
else:
    # Xử lý danh sách ô trả lời từ mỗi ảnh
    list_ans_choice = process_ans_blocks(list_ans_boxes_choice)
    list_ans_unchoice = process_ans_blocks(list_ans_boxes_unchoice)

    # Xử lý từng lựa chọn trong danh sách
    bubbles_choice = process_list_ans(list_ans_choice)
    bubbles_unchoice = process_list_ans(list_ans_unchoice)

    # Lưu ảnh vào thư mục
    save_dataset(bubbles_choice, "new_dataset/choice")
    save_dataset(bubbles_unchoice, "new_dataset/unchoice")
