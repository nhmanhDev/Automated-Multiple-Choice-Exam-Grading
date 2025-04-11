import os
import numpy as np
import cv2
from math import ceil

# Set up output directory
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_image(image, filename):
    """Save an image to the output_images directory."""
    cv2.imwrite(os.path.join(output_dir, filename), image)

# def resize_image(input_path, output_path, target_size=(1056, 1500)):
#     """Resize an image to target_size and save it."""
#     img = cv2.imread(input_path)
#     if img is None:
#         print(f"Error: Unable to read image at {input_path}")
#         return
#     img_resized = cv2.resize(img, target_size)
#     cv2.imwrite(output_path, img_resized)

def detect_mid_contours(image_path):
    """Detect and extract two regions (student ID and test code) from an image."""
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Unable to read image. Check file path.")
        return None, None

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray_img.shape[:2]
    min_area = 0.01 * (width * height)
    max_area = 0.03 * (width * height)
    # print("Image size:", width, "x", height)
    # print("Minimum area:", min_area, "Maximum area:", max_area)

    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 250)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours), "contours found")

    mid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    # print(len(mid_contours), "contours in range")

    if len(mid_contours) != 2:
        print("⚠️ Expected 2 regions, but found a different number. Check input image!")
        return None, None

    mid_contours = sorted(mid_contours, key=lambda c: cv2.boundingRect(c)[0])
    x1, y1, w1, h1 = cv2.boundingRect(mid_contours[0])  # Student ID region
    x2, y2, w2, h2 = cv2.boundingRect(mid_contours[1])  # Test code region
    sbd_img = img[y1:y1+h1, x1:x1+w1]
    mdt_img = img[y2:y2+h2, x2:x2+w2]

    save_image(sbd_img, "sbd.png")
    save_image(mdt_img, "mdt.png")
    return sbd_img, mdt_img

def detect_mid_contours_with_coords(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Unable to read image. Check file path.")
        return None, None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape[:2]
    min_area = 0.01 * (width * height)
    max_area = 0.03 * (width * height)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 250)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    if len(mid_contours) != 2:
        print("⚠️ Expected 2 regions, but found a different number.")
        return None, None, None, None

    mid_contours = sorted(mid_contours, key=lambda c: cv2.boundingRect(c)[0])
    x1, y1, w1, h1 = cv2.boundingRect(mid_contours[0])
    x2, y2, w2, h2 = cv2.boundingRect(mid_contours[1])
    sbd_img = img[y1:y1+h1, x1:x1+w1]
    mdt_img = img[y2:y2+h2, x2:x2+w2]
    
    return sbd_img, mdt_img, (x1, y1, w1, h1), (x2, y2, w2, h2)


def process_sbd_id_block(id_block_img):
    """Split student ID block into 6 columns."""
    gray = cv2.cvtColor(id_block_img, cv2.COLOR_BGR2GRAY) if len(id_block_img.shape) == 3 else id_block_img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_image(blurred, "sbd_block_grayscale.png")

    offset_x = ceil(blurred.shape[1] / 6)
    offset_x_adjusted = int(offset_x * 0.97)  # Slightly reduce column width
    columns = []

    for i in range(6):
        column_img = blurred[:, i * offset_x_adjusted:(i + 1) * offset_x_adjusted]
        save_image(column_img, f"column_img_sbd{i+1}.png")
        columns.append(column_img)
    return columns

def process_mdt_block(mdt_block_img):
    """Split test code block into 3 columns."""
    gray = cv2.cvtColor(mdt_block_img, cv2.COLOR_BGR2GRAY) if len(mdt_block_img.shape) == 3 else mdt_block_img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_image(blurred, "mdt_block_grayscale.png")

    offset_x = ceil(blurred.shape[1] / 3)
    columns = []

    for i in range(3):
        column_img = blurred[:, i * offset_x:(i + 1) * offset_x]
        save_image(column_img, f"column_img_mdt{i+1}.png")
        columns.append(column_img)
    return columns

def process_image_column(column_img):
    """Process a column by splitting it into 10 cells, applying blur, thresholding, and resizing."""
    gray = cv2.cvtColor(column_img, cv2.COLOR_BGR2GRAY) if len(column_img.shape) == 3 else column_img
    offset_y = ceil(gray.shape[0] / 10)
    cells = []
    for i in range(10):
        cell_img = gray[i * offset_y:(i + 1) * offset_y, :]
        blurred = cv2.GaussianBlur(cell_img, (5, 5), 0)
        binary_img = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
        # save_image(binary_img, f"cell_img_{i+1}.png")
        resized_cell = cv2.resize(binary_img, (28, 28))
        cells.append(resized_cell)
    return cells

def process_all_columns(columns):
    """Process all columns into lists of cell images."""
    return [process_image_column(column) for column in columns]

def check_all_columns_filled(all_columns_cells):
    """Check which cells in all columns are filled."""
    filled_cells = []  # List of tuples: (col_index, row_index)
    for col_idx, column_cells in enumerate(all_columns_cells):
        for row_idx, cell_img in enumerate(column_cells):
            # _, binary_img = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            black_pixel_count = np.sum(cell_img == 0)
            if black_pixel_count > (cell_img.size * 0.15):
                filled_cells.append((col_idx, row_idx))
    return filled_cells

def convert_filled_to_numbers_per_column(filled_cells, num_columns, num_rows=10):
    """Convert filled cell positions to numbers per column."""
    result = [[] for _ in range(num_columns)]
    for col_idx, row_idx in filled_cells:
        result[col_idx].append(row_idx)
    return result

def annotate_block(columns, filled_cells, num_rows=10, label="sbd"):
    """
    Draw red bounding boxes on the cells that are filled.
    Args:
        columns: list of column images (grayscale).
        filled_cells: list of (col_index, row_index) tuples indicating filled cells.
        num_rows: number of cells per column.
        label: used for saving file name and window name (e.g., "sbd" or "mdt").
    Returns:
        Annotated block image (columns concatenated horizontally).
    """
    annotated_columns = []
    for col_idx, col_img in enumerate(columns):
        col_color = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
        height, width = col_color.shape[:2]
        cell_height = ceil(height / num_rows)
        for (f_col, f_row) in filled_cells:
            if f_col == col_idx:
                pt1 = (0, f_row * cell_height)
                pt2 = (width - 1, min((f_row + 1) * cell_height, height - 1))
                cv2.rectangle(col_color, pt1, pt2, (0, 0, 255), 2)
        annotated_columns.append(col_color)
    
    annotated_block = cv2.hconcat(annotated_columns)
    save_image(annotated_block, f"annotated_{label}_block.png")
    
    # Hiển thị ảnh với tên cửa sổ riêng biệt dựa trên label
    # window_name = f"Annotated {label.upper()} Block"
    # cv2.imshow(window_name, annotated_block)
    # cv2.waitKey(50000)  # Chờ 5 giây
    # cv2.destroyAllWindows()  # Tự động đóng cửa sổ sau 10 giây
    
    return annotated_block


# # Main execution
# image_path = 'Exam/Test678_FullA.jpg'
# resize_image(image_path, 'output_resized.jpg')
# sbd, mdt = detect_mid_contours('output_resized.jpg')

# if sbd is None or mdt is None:
#     print("Cannot proceed due to missing regions.")
# else:
#     sbd_columns = process_sbd_id_block(sbd)
#     mdt_columns = process_mdt_block(mdt)
    
#     all_sbd_cells = process_all_columns(sbd_columns)
#     all_mdt_cells = process_all_columns(mdt_columns)
    
#     filled_sbd = check_all_columns_filled(all_sbd_cells)
#     filled_mdt = check_all_columns_filled(all_mdt_cells)
    
#     result_sbd = convert_filled_to_numbers_per_column(filled_sbd, 6)
#     result_mdt = convert_filled_to_numbers_per_column(filled_mdt, 3)
    
#     print("📌 Filled cells for student ID (SBD):")
#     for i, col in enumerate(result_sbd):
#         print(f"Column {i+1}: {col}")
#     print("📌 Filled cells for test code (MDT):")
#     for i, col in enumerate(result_mdt):
#         print(f"Column {i+1}: {col}")
    
#     # Annotate blocks with unique window names
#     annotated_sbd = annotate_block(sbd_columns, filled_sbd, num_rows=10, label="sbd")
#     annotated_mdt = annotate_block(mdt_columns, filled_mdt, num_rows=10, label="mdt")