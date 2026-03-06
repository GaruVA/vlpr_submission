import numpy as np

def smart_character_ordering(char_boxes, plate_shape):
    if not char_boxes:
        return [], []
    
    plate_height, plate_width = plate_shape[:2]
    row_threshold = plate_height * 0.3
    
    y_positions = [box['y'] + box['h']/2 for box in char_boxes]
    y_positions.sort()
    median_y = y_positions[len(y_positions)//2]
    
    top_row = [box for box in char_boxes if (box['y'] + box['h']/2) < median_y]
    bottom_row = [box for box in char_boxes if (box['y'] + box['h']/2) >= median_y]
    
    is_two_row = (len(top_row) >= 2 and len(bottom_row) >= 2 and 
                  len(top_row) + len(bottom_row) >= 5)
    
    if is_two_row:
        top_row.sort(key=lambda x: x['x'])
        bottom_row.sort(key=lambda x: x['x'])
        
        top_letters = sum(1 for box in top_row if box['is_letter'])
        bottom_numbers = sum(1 for box in bottom_row if not box['is_letter'])
        
        if top_letters >= len(top_row) * 0.6 and bottom_numbers >= len(bottom_row) * 0.6:
            ordered_boxes = top_row + bottom_row
        else:
            char_boxes.sort(key=lambda x: x['x'])
            ordered_boxes = char_boxes
    else:
        char_boxes.sort(key=lambda x: x['x'])
        ordered_boxes = char_boxes
    
    chars = [box['char'] for box in ordered_boxes]
    confidences = [box['conf'] for box in ordered_boxes]
    
    return chars, confidences
