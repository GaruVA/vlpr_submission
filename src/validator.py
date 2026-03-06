import re

class SriLankanPlateValidator:
    def __init__(self):
        self.class_to_char = {
            0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
            11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J',
            21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T',
            31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'
        }
        
        self.patterns = [
            r'^[A-Z]{2}-\d{4}$',
            r'^[A-Z]{3}-\d{4}$'
        ]
        
        self.invalid_patterns = [
            r'^[A-Z]{1}-.*',
            r'^.*-\d{1,3}$',
            r'^.*-\d{5,}$',
            r'^.*-\d{1,4}[A-Z]+.*',
            r'^[A-Z]{4,}-.*',
            r'^.*-[A-Z]{2,}$',
        ]
        
        self.valid_prefixes = []
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for first in letters:
            for second in letters:
                self.valid_prefixes.append(first + second)
        
        for first in letters:
            for second in letters:
                for third in letters:
                    prefix = first + second + third
                    self.valid_prefixes.append(prefix)
        
        self.letter_corrections = {
            '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', 
            '6': 'G', '7': 'T', '8': 'B', '9': 'P'
        }
        
        self.number_corrections = {
            'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 
            'G': '6', 'T': '7', 'B': '8', 'P': '9', 'D': '0', 'Q': '0'
        }
        
        self.letter_confusions = {
            'K': ['X', 'H'], 'X': ['K'], 'H': ['K'], 
            'B': ['8', 'P'], 'P': ['B'], '8': ['B'],
            'D': ['O', '0'], 'O': ['D', '0'], '0': ['O', 'D'],
            'C': ['G'], 'G': ['C'], 'F': ['E'], 'E': ['F']
        }
    
    def validate_and_correct(self, plate_text, confidence):
        if not plate_text or plate_text == "No text detected":
            return plate_text, confidence, False
        
        cleaned_text = plate_text.strip().upper()
        corrected_text = self.apply_positional_corrections(cleaned_text)
        formatted_text = self.format_sri_lankan_plate(corrected_text)
        
        is_valid = any(re.match(pattern, formatted_text) for pattern in self.patterns)
        is_invalid = any(re.match(pattern, formatted_text) for pattern in self.invalid_patterns)
        
        prefix_realistic = False
        dash_pos = formatted_text.find('-')
        if dash_pos > 0:
            prefix = formatted_text[:dash_pos]
            prefix_realistic = prefix in self.valid_prefixes
        
        is_valid = is_valid and not is_invalid and prefix_realistic
        adjusted_confidence = confidence * (1.2 if is_valid else 0.8)
        
        return formatted_text, min(adjusted_confidence, 1.0), is_valid
    
    def apply_positional_corrections(self, text):
        if not text:
            return text
        
        if '-' in text:
            parts = text.split('-')
            if len(parts) == 2:
                prefix, suffix = parts
                
                if len(prefix) >= 3:
                    last_char = prefix[-1]
                    if last_char.isalpha() and last_char in self.number_corrections:
                        corrected_char = self.number_corrections[last_char]
                        prefix = prefix[:-1]
                        suffix = corrected_char + suffix
                
                corrected_prefix = ""
                for char in prefix:
                    if char.isdigit():
                        corrected_prefix += self.letter_corrections.get(char, char)
                    else:
                        corrected_prefix += char
                
                corrected_suffix = ""
                for i, char in enumerate(suffix):
                    if char.isalpha():
                        corrected_suffix += self.number_corrections.get(char, char)
                    else:
                        corrected_suffix += char
                
                return f"{corrected_prefix}-{corrected_suffix}"
        
        clean_text = text.replace(' ', '').upper()
        corrected = []
        
        for i, char in enumerate(clean_text):
            if i < 3:
                if char.isdigit():
                    corrected.append(self.letter_corrections.get(char, char))
                else:
                    corrected.append(char)
            else:
                if char.isalpha() and i < len(clean_text) - 1:
                    corrected.append(self.number_corrections.get(char, char))
                else:
                    corrected.append(char)
        
        return ''.join(corrected)

    def format_sri_lankan_plate(self, text):
        if not text:
            return text
        
        clean_text = text.replace('-', '').replace(' ', '')
        match = re.match(r'^([A-Z]{2,3})(\d{1,4})([A-Z]?)$', clean_text)
        
        if match:
            letters, numbers, suffix = match.groups()
            formatted = f"{letters}-{numbers}{suffix}"
            return formatted
        
        return text

def is_reasonable_plate_text(text):
    if not text or len(text) < 5 or len(text) > 10:
        return False
    
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    
    if not (has_letters and has_numbers):
        return False
    
    for char in set(text):
        if text.count(char) > len(text) * 0.6:
            return False
    
    return True
