import os
import io
import numpy as np
from PIL import Image
import cv2
import fitz  # PyMuPDF
from tqdm import tqdm

def get_footer_line(image, zoom, filename, num):
    """
    image: cv2 image
    """
    max_line_length = 150*zoom
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50*zoom, maxLineGap=0)
    
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if abs(y1 - y2) < 5 and line_length <= max_line_length: #horizontal check and length check
                filtered_lines.append(line)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2) #greenline
                
    if len(filtered_lines) == 0:
        return None,None
    
    lowest_line = None
    lowest_y = -float('inf')
    
    #filter lowest line
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        min_y = min(y1, y2)
        if min_y > lowest_y:
            lowest_y = min_y
            lowest_line = line
    
    if lowest_line is not None:
        x1, y1, x2, y2 = lowest_line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  #redline
    
    Image.fromarray(image[:, :, ::-1]).save(f'{filename}_page_{num}_lines.png') #log

    return lowest_line, lowest_y
    

def remove_footers(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)
    zoom = 3.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in tqdm(range(len(doc)), desc="Processing pages"):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=matrix)
        
        #img_bytes = pix.tobytes("png")
        #img = Image.open(io.BytesIO(img_bytes))
        #img.save(f"page_{page_num + 1}.png")
        
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        lowest_line, lowest_y= get_footer_line(img, zoom, os.path.splitext(os.path.basename(input_pdf))[0], page_num+1)
        if lowest_y is not None:
            l_x0, l_y0, l_x1, l_y1 = lowest_line[0]
            #white box on footer
            rect = fitz.Rect(l_x0/zoom, l_y0/zoom-1, page.rect.width, page.rect.height)
            page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
        
    doc.save(output_pdf)
    doc.close()


def process_pdfs(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok = True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            input_pdf = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_pdf = os.path.join(output_folder, f"{base_name}_no_footers.pdf")
            print(f"Processing {filename} . . .")
            remove_footers(input_pdf, output_pdf)
            print(f"Processed {filename}")


if __name__ == '__main__':
    input_folder = "input"
    output_folder = "output"
    process_pdfs(input_folder, output_folder)