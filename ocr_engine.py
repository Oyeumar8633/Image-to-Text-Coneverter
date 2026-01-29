"""
OCR Engine Module
Handles image preprocessing, OCR text extraction, and formatting detection.
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Try to import EasyOCR, but make it optional
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

# Try to import PaddleOCR, but make it optional
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None


@dataclass
class TextBlock:
    """Represents a text block with formatting information."""
    text: str
    is_bold: bool = False
    is_italic: bool = False
    confidence: float = 0.0
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    alignment: str = "left"  # left, center, right


def preprocess_image(image: Image.Image, method: str = "adaptive") -> Image.Image:
    """
    Preprocess image to improve OCR accuracy with multiple strategies.
    
    Args:
        image: PIL Image object
        method: Preprocessing method ('adaptive', 'otsu', 'morphology', 'enhanced')
        
    Returns:
        Preprocessed PIL Image object
    """
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    
    # Convert to grayscale if not already
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Scale up image if too small (improves OCR accuracy)
    height, width = gray.shape
    if width < 600 or height < 600:
        scale_factor = max(600 / width, 600 / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    if method == "otsu":
        # Otsu's thresholding (good for clear documents)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed = thresh
    
    elif method == "morphology":
        # Morphological operations for better text separation
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    elif method == "enhanced":
        # Enhanced preprocessing with multiple steps
        # Bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        # Denoising
        processed = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    else:  # "adaptive" - default
        # Apply adaptive thresholding to get binary image
        # Using adaptive threshold for better results with varying lighting
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        # Apply denoising
        processed = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(processed)
    
    return processed_image


def detect_formatting_from_confidence(confidence: float, word_height: int, 
                                     avg_height: float) -> Tuple[bool, bool]:
    """
    Detect bold and italic formatting based on OCR confidence and word characteristics.
    
    Args:
        confidence: OCR confidence score for the word
        word_height: Height of the word bounding box
        avg_height: Average height of words in the document
        
    Returns:
        Tuple of (is_bold, is_italic)
    """
    is_bold = False
    is_italic = False
    
    # Bold text often has lower confidence due to thicker strokes
    # and may have slightly different height characteristics
    if confidence < 50 and word_height > avg_height * 0.9:
        is_bold = True
    
    # Italic text detection is more challenging with basic OCR
    # Lower confidence with normal height might indicate italic
    if 30 < confidence < 60 and word_height <= avg_height * 1.1:
        is_italic = True
    
    return is_bold, is_italic


def detect_alignment(left: int, width: int, page_width: int) -> str:
    """
    Detect text alignment based on position.
    
    Args:
        left: Left position of the text block
        width: Width of the text block
        page_width: Total width of the page/image
        
    Returns:
        Alignment string: 'left', 'center', or 'right'
    """
    center_x = left + width / 2
    page_center = page_width / 2
    
    # Define margins (20% of page width on each side)
    left_margin = page_width * 0.2
    right_margin = page_width * 0.8
    
    if left < left_margin:
        return "left"
    elif center_x > right_margin:
        return "right"
    elif abs(center_x - page_center) < page_width * 0.15:
        return "center"
    else:
        return "left"  # Default to left


def extract_text_with_formatting(image: Image.Image, psm_mode: int = 6) -> List[TextBlock]:
    """
    Extract text from image with formatting information.
    
    Args:
        image: PIL Image object (preprocessed)
        psm_mode: Tesseract PSM mode (Page Segmentation Mode)
            - 3: Fully automatic page segmentation (default)
            - 6: Assume uniform block of text
            - 11: Sparse text (for images with sparse text)
            - 13: Raw line (treat image as single text line)
        
    Returns:
        List of TextBlock objects with text and formatting info
    """
    # Get image dimensions
    img_width, img_height = image.size
    
    # Get detailed OCR data
    ocr_data = pytesseract.image_to_data(
        image, 
        output_type=pytesseract.Output.DICT,
        config=f'--psm {psm_mode}'  # Configurable PSM mode
    )
    
    text_blocks = []
    word_heights = []
    
    # First pass: collect all word heights to calculate average
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:  # Only process non-empty text
            height = ocr_data['height'][i]
            if height > 0:
                word_heights.append(height)
    
    avg_height = np.mean(word_heights) if word_heights else 0
    
    # Second pass: extract text with formatting
    current_line_top = None
    current_paragraph = []
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if not text:
            continue
        
        conf = float(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0.0
        left = ocr_data['left'][i]
        top = ocr_data['top'][i]
        width = ocr_data['width'][i]
        height = ocr_data['height'][i]
        
        # Detect formatting
        is_bold, is_italic = detect_formatting_from_confidence(
            conf, height, avg_height
        )
        
        # Detect alignment
        alignment = detect_alignment(left, width, img_width)
        
        # Group words into lines and paragraphs
        if current_line_top is None:
            current_line_top = top
        
        # Check if this is a new line (significant vertical difference)
        if abs(top - current_line_top) > avg_height * 0.5:
            # New line detected
            if current_paragraph:
                # Combine paragraph text
                para_text = ' '.join([block.text for block in current_paragraph])
                if para_text.strip():
                    # Use formatting from first word in paragraph (simplified)
                    first_block = current_paragraph[0]
                    text_blocks.append(TextBlock(
                        text=para_text,
                        is_bold=first_block.is_bold,
                        is_italic=first_block.is_italic,
                        confidence=first_block.confidence,
                        left=first_block.left,
                        top=first_block.top,
                        width=max(b.width for b in current_paragraph),
                        height=first_block.height,
                        alignment=first_block.alignment
                    ))
                current_paragraph = []
            current_line_top = top
        
        # Create text block for this word
        block = TextBlock(
            text=text,
            is_bold=is_bold,
            is_italic=is_italic,
            confidence=conf,
            left=left,
            top=top,
            width=width,
            height=height,
            alignment=alignment
        )
        current_paragraph.append(block)
    
    # Add remaining paragraph
    if current_paragraph:
        para_text = ' '.join([block.text for block in current_paragraph])
        if para_text.strip():
            first_block = current_paragraph[0]
            text_blocks.append(TextBlock(
                text=para_text,
                is_bold=first_block.is_bold,
                is_italic=first_block.is_italic,
                confidence=first_block.confidence,
                left=first_block.left,
                top=first_block.top,
                width=max(b.width for b in current_paragraph),
                height=first_block.height,
                alignment=first_block.alignment
            ))
    
    return text_blocks


def extract_text_simple(image: Image.Image, psm_mode: int = 6) -> str:
    """
    Simple text extraction without formatting (fallback).
    
    Args:
        image: PIL Image object
        psm_mode: Tesseract PSM mode (Page Segmentation Mode)
        
    Returns:
        Extracted text as string
    """
    text = pytesseract.image_to_string(image, config=f'--psm {psm_mode}')
    return text.strip()


def try_multiple_psm_modes(image: Image.Image) -> Tuple[str, int]:
    """
    Try multiple PSM modes and return the best result.
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (best_text, best_psm_mode)
    """
    psm_modes = [6, 3, 11, 13]  # Try different modes
    best_text = ""
    best_psm = 6
    max_words = 0
    
    for psm in psm_modes:
        try:
            text = pytesseract.image_to_string(image, config=f'--psm {psm}')
            # Count valid words (simple heuristic: words with 3+ characters)
            words = [w for w in text.split() if len(w) >= 3]
            if len(words) > max_words:
                max_words = len(words)
                best_text = text
                best_psm = psm
        except:
            continue
    
    return best_text.strip(), best_psm


# EasyOCR Functions
_easyocr_reader = None


def get_easyocr_reader():
    """
    Initialize and return EasyOCR reader (singleton pattern).
    
    Returns:
        EasyOCR reader object
    """
    global _easyocr_reader
    if not EASYOCR_AVAILABLE:
        raise ImportError("EasyOCR is not installed. Please install it with: pip install easyocr")
    
    if _easyocr_reader is None:
        # Initialize EasyOCR reader for English
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    
    return _easyocr_reader


def extract_text_easyocr(image: Image.Image, min_confidence: float = 0.5) -> str:
    """
    Extract text from image using EasyOCR (better for handwriting).
    
    Args:
        image: PIL Image object
        min_confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Extracted text as string
    """
    if not EASYOCR_AVAILABLE:
        raise ImportError("EasyOCR is not installed. Please install it with: pip install easyocr")
    
    try:
        reader = get_easyocr_reader()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize EasyOCR: {str(e)}")
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # EasyOCR works better with original images, but convert RGB to BGR if needed
    if len(img_array.shape) == 3:
        # Convert RGB to BGR for OpenCV compatibility
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    try:
        # Perform OCR with detail=0 for faster processing
        # detail=1 gives more info but is slower
        results = reader.readtext(img_array, detail=1, paragraph=False)
    except Exception as e:
        raise RuntimeError(f"EasyOCR processing failed: {str(e)}")
    
    # Extract text from results with better filtering
    text_lines = []
    for result in results:
        if len(result) >= 3:
            bbox, text, confidence = result[0], result[1], result[2]
            # Filter by confidence and text quality
            if confidence >= min_confidence and text.strip():
                # Additional filtering: remove very short single characters that are likely noise
                if len(text.strip()) > 1 or (len(text.strip()) == 1 and text.strip().isalnum()):
                    text_lines.append(text.strip())
    
    # Join with newlines, but also try to group by proximity
    if not text_lines:
        return ""
    
    return '\n'.join(text_lines)


def extract_text_with_formatting_easyocr(image: Image.Image, min_confidence: float = 0.5) -> List[TextBlock]:
    """
    Extract text from image using EasyOCR with formatting information.
    
    Args:
        image: PIL Image object
        min_confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        List of TextBlock objects with text and formatting info
    """
    if not EASYOCR_AVAILABLE:
        raise ImportError("EasyOCR is not installed. Please install it with: pip install easyocr")
    
    try:
        reader = get_easyocr_reader()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize EasyOCR: {str(e)}")
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    img_width, img_height = image.size
    
    # EasyOCR expects BGR format if image is color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    try:
        # Perform OCR
        results = reader.readtext(img_array, detail=1, paragraph=False)
    except Exception as e:
        raise RuntimeError(f"EasyOCR processing failed: {str(e)}")
    
    text_blocks = []
    word_heights = []
    
    # First pass: collect heights and filter by confidence
    valid_results = []
    for result in results:
        if len(result) >= 3:
            bbox, text, confidence = result[0], result[1], result[2]
            if confidence >= min_confidence and text.strip():
                # Filter out single character noise unless it's alphanumeric
                if len(text.strip()) > 1 or (len(text.strip()) == 1 and text.strip().isalnum()):
                    valid_results.append(result)
                    # Calculate bounding box dimensions
                    top_left = bbox[0]
                    bottom_right = bbox[2]
                    height = bottom_right[1] - top_left[1]
                    if height > 0:
                        word_heights.append(height)
    
    avg_height = np.mean(word_heights) if word_heights else 0
    
    # Second pass: create text blocks
    current_line_top = None
    current_paragraph = []
    
    for result in valid_results:
        bbox, text, confidence = result[0], result[1], result[2]
        
        # Extract bounding box coordinates
        top_left = bbox[0]
        bottom_right = bbox[2]
        left = int(top_left[0])
        top = int(top_left[1])
        width = int(bottom_right[0] - top_left[0])
        height = int(bottom_right[1] - top_left[1])
        
        # Detect alignment
        alignment = detect_alignment(left, width, img_width)
        
        # EasyOCR doesn't provide bold/italic info, so we'll use heuristics
        # Higher confidence and larger text might indicate emphasis
        is_bold = confidence > 0.8 and height > avg_height * 0.9 if avg_height > 0 else False
        is_italic = False  # EasyOCR doesn't detect italic
        
        # Group into lines and paragraphs
        if current_line_top is None:
            current_line_top = top
        
        # Check if this is a new line
        if abs(top - current_line_top) > avg_height * 0.5 if avg_height > 0 else 20:
            if current_paragraph:
                para_text = ' '.join([block.text for block in current_paragraph])
                if para_text.strip():
                    first_block = current_paragraph[0]
                    text_blocks.append(TextBlock(
                        text=para_text,
                        is_bold=first_block.is_bold,
                        is_italic=first_block.is_italic,
                        confidence=first_block.confidence,
                        left=first_block.left,
                        top=first_block.top,
                        width=max(b.width for b in current_paragraph),
                        height=first_block.height,
                        alignment=first_block.alignment
                    ))
                current_paragraph = []
            current_line_top = top
        
        # Create text block
        block = TextBlock(
            text=text,
            is_bold=is_bold,
            is_italic=is_italic,
            confidence=confidence * 100,  # Convert to 0-100 scale
            left=left,
            top=top,
            width=width,
            height=height,
            alignment=alignment
        )
        current_paragraph.append(block)
    
    # Add remaining paragraph
    if current_paragraph:
        para_text = ' '.join([block.text for block in current_paragraph])
        if para_text.strip():
            first_block = current_paragraph[0]
            text_blocks.append(TextBlock(
                text=para_text,
                is_bold=first_block.is_bold,
                is_italic=first_block.is_italic,
                confidence=first_block.confidence,
                left=first_block.left,
                top=first_block.top,
                width=max(b.width for b in current_paragraph),
                height=first_block.height,
                alignment=first_block.alignment
            ))
    
    return text_blocks

# PaddleOCR Functions
_paddleocr_reader = None


def get_paddleocr_reader():
    """
    Initialize and return PaddleOCR reader (singleton pattern).
    
    Returns:
        PaddleOCR reader object
    """
    global _paddleocr_reader
    if not PADDLEOCR_AVAILABLE:
        raise ImportError("PaddleOCR is not installed. Please install it with: pip install paddlepaddle paddleocr")
    
    if _paddleocr_reader is None:
        # Initialize PaddleOCR reader for English
        # lang='en' for English
        # Note: PaddleOCR 3.4.0+ doesn't support use_gpu or use_angle_cls parameters
        # GPU detection is automatic, and angle classification is built-in
        _paddleocr_reader = PaddleOCR(lang='en')
    
    return _paddleocr_reader


def extract_text_paddleocr(image: Image.Image, min_confidence: float = 0.5) -> str:
    """
    Extract text from image using PaddleOCR (excellent for both printed and handwritten text).
    
    Args:
        image: PIL Image object
        min_confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        Extracted text as string
    """
    if not PADDLEOCR_AVAILABLE:
        raise ImportError("PaddleOCR is not installed. Please install it with: pip install paddlepaddle paddleocr")
    
    try:
        ocr = get_paddleocr_reader()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PaddleOCR: {str(e)}")
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # PaddleOCR expects BGR format if image is color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    try:
        # Perform OCR
        # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
        # Note: cls parameter removed in newer versions
        results = ocr.ocr(img_array)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR processing failed: {str(e)}")
    
    # Extract text from results
    # PaddleOCR 3.4.0+ returns OCRResult objects with rec_texts, rec_scores, rec_polys
    text_lines = []
    if results and len(results) > 0:
        result = results[0]  # Get first result
        # OCRResult behaves like a dict but might not be isinstance dict
        if hasattr(result, 'get') or isinstance(result, dict):
            # New format: dict-like with rec_texts, rec_scores, rec_polys
            try:
                rec_texts = result.get('rec_texts', []) if hasattr(result, 'get') else result.get('rec_texts', [])
                rec_scores = result.get('rec_scores', []) if hasattr(result, 'get') else result.get('rec_scores', [])
                
                # Match texts with their confidence scores
                for i, text in enumerate(rec_texts):
                    if text and text.strip():
                        confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                        # Filter by confidence and text quality
                        if confidence >= min_confidence:
                            # Additional filtering: remove very short single characters that are likely noise
                            if len(text.strip()) > 1 or (len(text.strip()) == 1 and text.strip().isalnum()):
                                text_lines.append(text.strip())
            except (KeyError, IndexError, TypeError) as e:
                # If there's an error accessing the result, return empty
                pass
        else:
            # Old format fallback (shouldn't happen in 3.4.0+)
            try:
                for line in result:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line[0], line[1]
                        if confidence >= min_confidence and text.strip():
                            if len(text.strip()) > 1 or (len(text.strip()) == 1 and text.strip().isalnum()):
                                text_lines.append(text.strip())
            except (TypeError, IndexError):
                pass
    
    if not text_lines:
        return ""
    
    return '\n'.join(text_lines)


def extract_text_with_formatting_paddleocr(image: Image.Image, min_confidence: float = 0.5) -> List[TextBlock]:
    """
    Extract text from image using PaddleOCR with formatting information.
    
    Args:
        image: PIL Image object
        min_confidence: Minimum confidence threshold (0.0-1.0)
        
    Returns:
        List of TextBlock objects with text and formatting info
    """
    if not PADDLEOCR_AVAILABLE:
        raise ImportError("PaddleOCR is not installed. Please install it with: pip install paddlepaddle paddleocr")
    
    try:
        ocr = get_paddleocr_reader()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PaddleOCR: {str(e)}")
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    img_width, img_height = image.size
    
    # PaddleOCR expects BGR format if image is color
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    try:
        # Perform OCR
        # Note: cls parameter removed in newer versions
        results = ocr.ocr(img_array)
    except Exception as e:
        raise RuntimeError(f"PaddleOCR processing failed: {str(e)}")
    
    text_blocks = []
    word_heights = []
    valid_results = []
    
    # First pass: collect heights and filter by confidence
    # PaddleOCR 3.4.0+ returns OCRResult objects with rec_texts, rec_scores, rec_polys
    if results and len(results) > 0:
        result = results[0]  # Get first result
        # OCRResult behaves like a dict but might not be isinstance dict
        if hasattr(result, 'get') or isinstance(result, dict):
            # New format: dict-like with rec_texts, rec_scores, rec_polys
            rec_texts = result.get('rec_texts', []) if hasattr(result, 'get') else result.get('rec_texts', [])
            rec_scores = result.get('rec_scores', []) if hasattr(result, 'get') else result.get('rec_scores', [])
            rec_polys = result.get('rec_polys', []) if hasattr(result, 'get') else result.get('rec_polys', [])
            
            # Match texts with their confidence scores and bounding boxes
            for i, text in enumerate(rec_texts):
                if text and text.strip():
                    confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                    bbox = rec_polys[i] if i < len(rec_polys) else None
                    
                    if confidence >= min_confidence:
                        # Filter out single character noise unless it's alphanumeric
                        if len(text.strip()) > 1 or (len(text.strip()) == 1 and text.strip().isalnum()):
                            valid_results.append((bbox, text, confidence))
                            # Calculate bounding box dimensions
                            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] or None
                            if bbox is not None:
                                try:
                                    # Handle different bbox formats
                                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                        if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                                            # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                            top_left = bbox[0]
                                            bottom_right = bbox[2] if len(bbox) > 2 else bbox[1]
                                            height = abs(bottom_right[1] - top_left[1])
                                            if height > 0:
                                                word_heights.append(height)
                                except (IndexError, TypeError):
                                    pass
        else:
            # Old format fallback (shouldn't happen in 3.4.0+)
            for line in result:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line[0], line[1]
                    if confidence >= min_confidence and text.strip():
                        if len(text.strip()) > 1 or (len(text.strip()) == 1 and text.strip().isalnum()):
                            valid_results.append((bbox, text, confidence))
                            if bbox and len(bbox) >= 4:
                                top_left = bbox[0]
                                bottom_right = bbox[2]
                                height = bottom_right[1] - top_left[1]
                                if height > 0:
                                    word_heights.append(height)
    
    avg_height = np.mean(word_heights) if word_heights else 0
    
    # Second pass: create text blocks
    current_line_top = None
    current_paragraph = []
    
    for bbox, text, confidence in valid_results:
        # Extract bounding box coordinates
        left = 0
        top = 0
        width = 100
        height = 20
        
        if bbox is not None:
            try:
                # Handle different bbox formats
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                        # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        top_left = bbox[0]
                        bottom_right = bbox[2] if len(bbox) > 2 else bbox[1]
                        left = int(top_left[0])
                        top = int(top_left[1])
                        width = int(abs(bottom_right[0] - top_left[0]))
                        height = int(abs(bottom_right[1] - top_left[1]))
            except (IndexError, TypeError, ValueError) as e:
                # If bbox parsing fails, use default values
                pass
        
        # Detect alignment
        alignment = detect_alignment(left, width, img_width)
        
        # PaddleOCR doesn't provide bold/italic info, so we'll use heuristics
        # Higher confidence and larger text might indicate emphasis
        is_bold = confidence > 0.8 and height > avg_height * 0.9 if avg_height > 0 else False
        is_italic = False  # PaddleOCR doesn't detect italic
        
        # Group into lines and paragraphs
        if current_line_top is None:
            current_line_top = top
        
        # Check if this is a new line
        if abs(top - current_line_top) > avg_height * 0.5 if avg_height > 0 else 20:
            if current_paragraph:
                para_text = ' '.join([block.text for block in current_paragraph])
                if para_text.strip():
                    first_block = current_paragraph[0]
                    text_blocks.append(TextBlock(
                        text=para_text,
                        is_bold=first_block.is_bold,
                        is_italic=first_block.is_italic,
                        confidence=first_block.confidence,
                        left=first_block.left,
                        top=first_block.top,
                        width=max(b.width for b in current_paragraph),
                        height=first_block.height,
                        alignment=first_block.alignment
                    ))
                current_paragraph = []
            current_line_top = top
        
        # Create text block
        block = TextBlock(
            text=text,
            is_bold=is_bold,
            is_italic=is_italic,
            confidence=confidence * 100,  # Convert to 0-100 scale
            left=left,
            top=top,
            width=width,
            height=height,
            alignment=alignment
        )
        current_paragraph.append(block)
    
    # Add remaining paragraph
    if current_paragraph:
        para_text = ' '.join([block.text for block in current_paragraph])
        if para_text.strip():
            first_block = current_paragraph[0]
            text_blocks.append(TextBlock(
                text=para_text,
                is_bold=first_block.is_bold,
                is_italic=first_block.is_italic,
                confidence=first_block.confidence,
                left=first_block.left,
                top=first_block.top,
                width=max(b.width for b in current_paragraph),
                height=first_block.height,
                alignment=first_block.alignment
            ))
    
    return text_blocks
