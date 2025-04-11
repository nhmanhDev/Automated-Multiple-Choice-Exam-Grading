import imutils
import numpy as np
import cv2
from math import ceil
from model_answer import CNN_Model
from collections import defaultdict
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def crop_image(img):
    logger.debug("Starting crop_image")
    if img is None:
        logger.error("Input image is None")
        raise ValueError("Input image is None")
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape[:2]
    total_area = height * width
    
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    img_canny = cv2.Canny(blurred, 100, 200)
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0
    min_area = 0.10 * total_area

    if len(cnts) > 0:
    # Sắp xếp contours theo x
        cnts = sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])
        for c in cnts:
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
            # Chỉ lấy contour có diện tích > 10%
            if w_curr * h_curr > min_area:
                block_img = gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr]
                # Kiểm tra xem contour này có quá gần contour trước đó không
                is_duplicate = False
                for _, prev_coords in ans_blocks:
                    x_prev, y_prev, w_prev, h_prev = prev_coords
                    # Nếu vị trí và kích thước gần giống contour trước thì bỏ qua
                    if (abs(x_curr - x_prev) < 10 and 
                        abs(y_curr - y_prev) < 10 and 
                        abs(w_curr - w_prev) < 10 and 
                        abs(h_curr - h_prev) < 10):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    ans_blocks.append((block_img, [x_curr, y_curr, w_curr, h_curr]))

        logger.info(f"Found {len(ans_blocks)} answer blocks")
        return ans_blocks  # Đã được sắp xếp theo x
    logger.warning("No answer blocks found")
    return []

def process_ans_blocks(ans_blocks):
    logger.debug(f"Processing {len(ans_blocks)} answer blocks")
    list_answers = []
    for idx, ans_block in enumerate(ans_blocks):
        ans_block_img = ans_block[0]
        if ans_block_img is None or ans_block_img.size == 0:
            logger.error(f"Answer block {idx} is invalid")
            raise ValueError(f"Answer block {idx} is invalid")
        
        offset1 = ceil(ans_block_img.shape[0] / 6)
        for i in range(6):
            box_img = ans_block_img[i * offset1:(i + 1) * offset1, :]
            if box_img.size == 0:
                logger.error(f"Box {i} in block {idx} is empty")
                raise ValueError(f"Box {i} in block {idx} is empty")
            
            height_box = box_img.shape[0]
            cut_top = ceil(height_box * 1.5 / 19)
            cut_bottom = ceil(height_box * 1 / 19)
            box_img = box_img[cut_top:height_box - cut_bottom, :]

            offset2 = ceil(box_img.shape[0] / 5)
            for j in range(5):
                line_img = box_img[j * offset2:(j + 1) * offset2, :]
                if line_img.size == 0:
                    logger.error(f"Line {j} in box {i}, block {idx} is empty")
                    raise ValueError(f"Line {j} in box {i}, block {idx} is empty")
                list_answers.append(line_img)
    logger.debug(f"Processed {len(list_answers)} answer regions")
    return list_answers

def process_list_ans(list_answers):
    logger.debug("Processing list of answers")
    if not list_answers or len(list_answers) == 0:
        logger.error("List of answers is empty")
        raise ValueError("List of answers is empty")
    
    img_width = list_answers[0].shape[1]
    offset = (img_width // 7) * 6 // 4
    start = (2 * img_width // 11)

    list_choices = []
    for idx, answer_img in enumerate(list_answers):
        if answer_img.size == 0:
            logger.error(f"Answer image {idx} is empty")
            raise ValueError(f"Answer image {idx} is empty")
        
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            if bubble_choice.size == 0:
                logger.error(f"Bubble choice {i} in answer {idx} is empty")
                raise ValueError(f"Bubble choice {i} in answer {idx} is empty")
            
            bubble_choice = cv2.threshold(bubble_choice, 127, 255, cv2.THRESH_BINARY)[1]
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        logger.error(f"Length of list_choices must be 480, got {len(list_choices)}")
        raise ValueError(f"Length of list_choices must be 480, got {len(list_choices)}")
    logger.debug(f"Processed {len(list_choices)} bubble choices")
    return list_choices

# def save_list_ans(list_ans, prefix="bubble"):
#     """Save list_ans as individual images or a NumPy file."""
#     for idx, choice in enumerate(list_ans):
#         filename = os.path.join(output_dir, f"{prefix}_{idx:03d}.png")
#         cv2.imwrite(filename, choice.squeeze())  # Squeeze để bỏ chiều 1
#     print(f"Saved {len(list_ans)} bubble choices to {output_dir}")
#     # Lưu dưới dạng file NumPy (tùy chọn)
#     np.save(os.path.join(output_dir, "list_ans.npy"), np.array(list_ans))

def get_answers(list_answers):
    logger.debug("Getting final answers")
    try:
        model = CNN_Model('weight.keras').build_model(rt=True)  # Không cần truyền weight_path, dùng mặc định /src/weight.keras
        list_answers = np.array(list_answers)
        logger.debug(f"Predicting on {len(list_answers)} choices")
        scores = model.predict_on_batch(list_answers / 255.0)
        results = defaultdict(list)
        for idx, score in enumerate(scores):
            question = idx // 4
            if score[1] > 0.99:
                chosed_answer = ["A", "B", "C", "D"][idx % 4]
                results[question + 1].append(chosed_answer)
        logger.debug(f"Detected answers for {len(results)} questions")
        return results
    except Exception as e:
        logger.exception("Error in get_answers")
        raise

def annotate_answers(ans_blocks, answers, answer_key, questions_per_block=30, img=None):
    logger.debug("Annotating answers on image")
    if img is None:
        logger.error("No image provided for annotation")
        raise ValueError("No image provided for annotation")
    
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
            effective_box_hA = box_h - cut_top - cut_bottom
            offset2 = ceil(effective_box_hA / 5)

            for j in range(5):
                line_y = box_y + cut_top + j * offset2
                offset = (w // 7 * 6) // 4
                start = w // 7

                question = question_offset + i * 5 + j + 1
                if question > 120:  # Giả định 120 câu hỏi
                    continue

                student_ans = answers.get(question, [])
                correct_ans = answer_key[question]  # Không cần -1 vì key bắt đầu từ 1

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

if __name__ == '__main__':
    image_path = 'Exam/Test10diem.jpg'
    img = cv2.imread(image_path)
    
    list_ans_boxes = crop_image(img)
    if not list_ans_boxes:
        print("No answer blocks found. Exiting.")
    else:
        list_ans = process_ans_blocks(list_ans_boxes)
        list_ans_processed = process_list_ans(list_ans)
        
        # save_list_ans(list_ans_processed)
        
        answers = get_answers(list_ans_processed)
        print("Detected answers:")
        for question, ans in sorted(answers.items()):
            print(f"Question {question}: {ans}")
        
        answerkey = pd.read_excel('AnswerKey/FullC.xlsx', engine='openpyxl')
        answerkey = answerkey.iloc[:, 1].tolist()
       
        total_questions = len(answerkey)  # Dynamically set based on answer key length
        annotated_img = annotate_answers(list_ans_boxes, answers, answerkey, total_questions=total_questions, img=img)
        cv2.imwrite('annotated_full_image.jpg', annotated_img)