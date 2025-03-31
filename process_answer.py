import imutils
import numpy as np
import cv2
from math import ceil
from model_answer import CNN_Model
from collections import defaultdict


def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]


def crop_image(img):
    # convert image from BGR to GRAY to apply canny edge detection algorithm
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # remove noise by blur image
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # apply canny edge detection algorithm
    img_canny = cv2.Canny(blurred, 100, 200)

    # find contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in descending order
        cnts = sorted(cnts, key=get_x_ver1)

        # loop over the sorted contours
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:
                # check overlap contours
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # if list answer box is empty
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # update coordinates (x, y) and (height, width) of added contours
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

        # sort ans_blocks according to x coordinate
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        return sorted_ans_blocks


def process_ans_blocks(ans_blocks):
    """
        this function process 2 block answer box and return a list answer has len of 200 bubble choices
        :param ans_blocks: a list which include 2 element, each element has the format of [image, [x, y, w, h]]
    """
    list_answers = []

    # Loop over each block ans in
    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])

        offset1 = ceil(ans_block_img.shape[0] / 6)
        # Loop over each box in answer block
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            height_box = box_img.shape[0]

            box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)

            # loop over each line in a box
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])

    return list_answers


def process_list_ans(list_answers):
    list_choices = []
    offset = 44
    start = 32

    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        raise ValueError("Length of list_choices must be 480")
    return list_choices


def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    else:
        answer_circle = "D"
    return answer_circle


def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.keras').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 4

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.99:  # choiced confidence score > 0.9
            chosed_answer = map_answer(idx)
            results[question + 1].append(chosed_answer)

    return results

# def resize_image(input_path, output_path, target_size=(1056, 1500)):
#     """
#     Hàm resize ảnh về kích thước target_size và lưu lại.
    
#     Args:
#         input_path (str): Đường dẫn ảnh gốc.
#         output_path (str): Đường dẫn ảnh sau khi resize.
#         target_size (tuple): Kích thước đích (width, height). Mặc định (1056, 1500).
#     """
#     # Đọc ảnh
#     img = cv2.imread(input_path)
#     if img is None:
#         print(f"Lỗi: Không thể đọc ảnh {input_path}")
#         return

#     # Resize ảnh
#     img_resized = cv2.resize(img, target_size)

#     # Lưu ảnh đã resize
#     cv2.imwrite(output_path, img_resized)
#     print(f"Ảnh đã được resize và lưu tại {output_path}")

def annotate_answers(ans_blocks, answers, answer_key, questions_per_block=30, total_questions=120, img=None):
    if img is None:
        annotated_img = cv2.imread('output_resized.jpg')
    else:
        annotated_img = img  # Sử dụng ảnh đã vẽ SBD/MDT
    drawn_questions = set()

    ans_blocks = sorted(ans_blocks, key=lambda b: b[1][0])
    for block_idx, ans_block in enumerate(ans_blocks):
        block_img, (x, y, w, h) = ans_block
        offset1 = ceil(h / 6)
        question_offset = block_idx * questions_per_block
        
        for i in range(6):
            box_y = y + i * offset1
            box_h = offset1
            box_img = block_img[i * offset1:(i + 1) * offset1, :]
            box_img = box_img[14:box_h - 14, :]
            offset2 = ceil((box_h - 28) / 5)
            for j in range(5):
                question = question_offset + i * 5 + j + 1
                if question > total_questions or question in drawn_questions:
                    continue
                drawn_questions.add(question)

                line_y = box_y + 14 + j * offset2
                for k in range(4):
                    choice_x = x + 32 + k * 44
                    student_ans = answers.get(question, [])
                    correct_ans = answer_key.get(question, '')
                    if student_ans and student_ans[0] == map_answer(k):
                        color = (0, 255, 0) if student_ans[0] == correct_ans else (0, 0, 255)
                        cv2.rectangle(annotated_img, (choice_x, line_y), (choice_x + 44, line_y + offset2), color, 2)
                    elif map_answer(k) == correct_ans:
                        cv2.rectangle(annotated_img, (choice_x, line_y), (choice_x + 44, line_y + offset2), (0, 255, 0), 2)

    # cv2.imwrite('annotated_full_image.jpg', annotated_img)
    return annotated_img

def resize_image(input_path, output_path, target_size=(1056, 1500)):
    """Resize an image to target_size and save it."""
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Unable to read image at {input_path}")
        return
    img_resized = cv2.resize(img, target_size)
    cv2.imwrite(output_path, img_resized)


if __name__ == '__main__':
    image_path = 'Exam/Test003.jpg'

    resize_image(image_path, 'output_resized.jpg')

    img = cv2.imread('output_resized.jpg')
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)
    print(answers)
