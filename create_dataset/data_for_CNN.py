import os
import cv2
import numpy as np
import imutils
from math import ceil


# Các hàm hỗ trợ (không thay đổi)
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
    # Chuyển ảnh từ BGR sang grayscale để áp dụng Canny
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Lọc nhiễu bằng GaussianBlur
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Áp dụng Canny edge detection
    img_canny = cv2.Canny(blurred, 100, 200)
    # Tìm các contours
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    if len(cnts) > 0:
        cnts = sorted(cnts, key=get_x_ver1)
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)
            if w_curr * h_curr > 100000:
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)
                if len(ans_blocks) == 0:
                    ans_blocks.append((gray_img[y_curr:y_curr+h_curr, x_curr:x_curr+w_curr],
                                       [x_curr, y_curr, w_curr, h_curr]))
                    x_old, y_old, w_old, h_old = x_curr, y_curr, w_curr, h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append((gray_img[y_curr:y_curr+h_curr, x_curr:x_curr+w_curr],
                                       [x_curr, y_curr, w_curr, h_curr]))
                    x_old, y_old, w_old, h_old = x_curr, y_curr, w_curr, h_curr
        # Sắp xếp các block theo hoành độ
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        return sorted_ans_blocks
    else:
        return []

def process_ans_blocks(ans_blocks):
    """
    Hàm này xử lý các block đáp án (mỗi block chứa nhiều câu)
    và tách ra thành các vùng chứa câu hỏi riêng lẻ.
    """
    list_answers = []
    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])
        offset1 = ceil(ans_block_img.shape[0] / 6)
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            height_box = box_img.shape[0]
            # Cắt bỏ viền trên và dưới
            box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])
    return list_answers

def process_list_ans(list_answers):
    """
    Hàm này tách từng câu thành 4 ô đáp án (A, B, C, D).
    Code gốc mong đợi tổng số 480 ô (120 câu x 4 đáp án).
    Nếu ảnh của bạn có layout khác, hãy điều chỉnh giá trị start và offset.
    """
    list_choices = []
    offset = 44
    start = 32
    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset : start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)
    # Nếu không thu được 480 ô, bạn có thể bỏ kiểm tra này hoặc điều chỉnh theo layout ảnh của mình.
    if len(list_choices) != 480:
        print(f"Chú ý: Số ô thu được là {len(list_choices)} thay vì 480.")
    return list_choices

# Hàm lưu dataset từ list ảnh bubble cho mỗi loại (choice hoặc unchoice)
def save_dataset(list_bubbles, folder):
    if os.path.exists(folder):
        # Xóa thư mục cũ nếu có
        import shutil
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    for i, bubble in enumerate(list_bubbles):
        bubble_2d = bubble.reshape((28, 28))
        filename = os.path.join(folder, f"bubble_{i}.png")
        cv2.imwrite(filename, bubble_2d)
    print(f"Đã lưu {len(list_bubbles)} ô vào '{folder}'")

# MAIN: xử lý 2 ảnh
if __name__ == '__main__':

    
    img_choice = cv2.imread('Image_choice.jpg')
    if img_choice is None:
        raise ValueError("Không tìm thấy file 'choice.jpg'")
    ans_boxes_choice = crop_image(img_choice)
    ans_lines_choice = process_ans_blocks(ans_boxes_choice)
    bubbles_choice = process_list_ans(ans_lines_choice)
    save_dataset(bubbles_choice, "dataset/choice")

    # Ảnh chứa đáp án chưa được tô (unchoice)
    img_unchoice = cv2.imread('Image_unchoice.jpg')
    if img_unchoice is None:
        raise ValueError("Không tìm thấy file 'unchoice.jpg'")
    ans_boxes_unchoice = crop_image(img_unchoice)
    ans_lines_unchoice = process_ans_blocks(ans_boxes_unchoice)
    bubbles_unchoice = process_list_ans(ans_lines_unchoice)
    save_dataset(bubbles_unchoice, "dataset/unchoice")
