import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray_image

def display_image(image, title):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def display_gray_image(gray_image, title):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()

# Load the image
#image_path = 'images/bacterias.png'
image_path = 'images/area5.jpg'
image, gray_image = load_and_convert_image(image_path)
display_image(image, 'Original Image')
display_gray_image(gray_image, 'Grayscale Image')
def detect_edges(gray_image, threshold1=50, threshold2=150):
    edges = cv2.Canny(gray_image, threshold1, threshold2, apertureSize=3)
    return edges

edges = detect_edges(gray_image)
display_gray_image(edges, 'Edge Detection')

def detect_lines(edges, threshold=100, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

lines = detect_lines(edges)
if lines is not None:
    image_with_lines = np.copy(image)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    display_image(image_with_lines, 'Detected Lines')
    
def count_unit_squares(horizontal_lines, vertical_lines):
    horizontal_lines_excluding_top = horizontal_lines[1:]
    square_count = (len(horizontal_lines_excluding_top) - 1) * (len(vertical_lines) - 1)
    return square_count
import numpy as np

def merge_nearest_lines(lines, image, threshold=50):
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 10:  # Horizontal line
            horizontal_lines.append(y1)
        elif abs(x1 - x2) < 10:  # Vertical line
            vertical_lines.append(x1)

    horizontal_lines = sorted(set(horizontal_lines))
    vertical_lines = sorted(set(vertical_lines))

    def merge_lines(line_positions, threshold):
        merged_lines = []
        current_line = line_positions[0]

        for line in line_positions[1:]:
            if line - current_line <= threshold:
                continue
            else:
                merged_lines.append(current_line)
                current_line = line

        merged_lines.append(current_line)
        return merged_lines

    merged_horizontal_lines = merge_lines(horizontal_lines, threshold)
    merged_vertical_lines = merge_lines(vertical_lines, threshold)

    image_with_merged_lines = np.copy(image)

    for y in merged_horizontal_lines:
        cv2.line(image_with_merged_lines, (0, y), (image.shape[1], y), (0, 255, 0), 2)

    for x in merged_vertical_lines:
        cv2.line(image_with_merged_lines, (x, 0), (x, image.shape[0]), (0, 255, 0), 2)

    return image_with_merged_lines, merged_horizontal_lines, merged_vertical_lines

image_with_merged_lines, merged_horizontal_lines, merged_vertical_lines = merge_nearest_lines(lines, image)
display_image(image_with_merged_lines, 'Merged Lines')

unit_square_count = count_unit_squares(merged_horizontal_lines, merged_vertical_lines)
print(f'Number of smallest unit squares: {unit_square_count}')
def index_squares(image, horizontal_lines, vertical_lines):
    index = 1
    horizontal_lines_excluding_top = horizontal_lines[1:]

    for i in range(len(horizontal_lines_excluding_top) - 1):
        for j in range(len(vertical_lines) - 1):
            top_left = (vertical_lines[j], horizontal_lines_excluding_top[i])
            bottom_right = (vertical_lines[j + 1], horizontal_lines_excluding_top[i + 1])
            # Draw the index on the image
            cv2.putText(image, str(index), 
                        (top_left[0] + 5, top_left[1] + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            index += 1
    return image

image_with_indexed_squares = index_squares(image_with_merged_lines, merged_horizontal_lines, merged_vertical_lines)
display_image(image_with_indexed_squares, 'Indexed Squares')