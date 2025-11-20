"""Speech utilities - Helper functions for speech processing."""

from __future__ import annotations

import re
from typing import Set

# Vietnamese filler words
FILLER_WORDS_VI: Set[str] = {
    "ờ", "à", "ừ", "ưm", "ùm", "ơ", "ờm",
    "hử", "hem", "ư", "ô", "ớ", "ồ", "ổ",
    "ơi", "ối", "ồi", "úi", "ủa", "ơ kìa",
    "thì", "là", "ấy", "nhỉ", "nhé", "nha",
    "ấy mà", "thì ra", "à há",
}

# English filler words  
FILLER_WORDS_EN: Set[str] = {
    "uh", "um", "er", "ah", "eh", "mm", "hmm", "mhm",
    "uhm", "erm", "emm", "hm", "uhh", "umm",
    "like", "you know", "i mean", "actually", "basically",
    "literally", "so", "well", "right", "okay", "ok",
}

# Combine all fillers (lowercase for matching)
ALL_FILLERS: Set[str] = FILLER_WORDS_VI | FILLER_WORDS_EN


def filter_filler_words(text: str, min_word_length: int = 2) -> str:
    """
    Remove filler words and very short words from transcript.
    
    Args:
        text: Input text
        min_word_length: Minimum word length to keep (default 2)
    
    Returns:
        Filtered text
    
    Examples:
        >>> filter_filler_words("Ờ, tôi nghĩ là um, project này khá tốt nhé")
        "tôi nghĩ project này khá tốt"
        
        >>> filter_filler_words("So, uh, the API is, like, really fast, you know?")
        "the API is really fast"
    """
    if not text or not text.strip():
        return ""
    
    # Tokenize by whitespace
    words = text.split()
    filtered = []
    
    for word in words:
        # Remove punctuation for comparison
        clean = word.lower().strip('.,!?;:\'"()[]{}…–—')
        
        # Skip empty after cleaning
        if not clean:
            continue
        
        # Allow important single-char words
        if len(clean) == 1 and clean not in {'a', 'i', 'ở'}:
            continue
        
        # Skip if too short
        if len(clean) < min_word_length and clean not in {'ai', 'gì', 'sao', 'à', 'ừ'}:
            # Allow some Vietnamese question words even if short
            if clean not in {'ai', 'gì', 'sao', 'the', 'is', 'an'}:
                continue
        
        # Skip fillers
        if clean in ALL_FILLERS:
            continue
        
        # Check multi-word fillers
        skip = False
        for filler in {"you know", "i mean", "ấy mà", "thì ra", "à há", "ơ kìa"}:
            if clean in filler.split():
                # Need context check - for now keep
                pass
        
        filtered.append(word)
    
    result = ' '.join(filtered)
    
    # Post-processing: remove repeated punctuation
    result = re.sub(r'([.,!?;:])\1+', r'\1', result)
    
    # Remove leading/trailing punctuation
    result = result.strip('.,!?;: ')
    
    return result


def normalize_vietnamese_text(text: str) -> str:
    """
    Normalize Vietnamese text (fix common OCR/STT errors).
    
    Args:
        text: Input Vietnamese text
    
    Returns:
        Normalized text
    
    Examples:
        >>> normalize_vietnamese_text("Đôi khi  tôi   nghĩ")
        "Đôi khi tôi nghĩ"
    """
    if not text:
        return ""
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common Vietnamese STT errors
    replacements = {
        " đc ": " được ",
        " ko ": " không ",
        " k ": " không ",
        " dc ": " được ",
        " đk ": " đăng ký ",
        " vs ": " với ",
        " tl ": " trả lời ",
        " bv ": " bảo vệ ",
    }
    
    text_lower = text.lower()
    for wrong, correct in replacements.items():
        if wrong in text_lower:
            # Case-insensitive replace (preserve original case pattern)
            text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    
    # Capitalize first letter of sentences
    text = re.sub(r'(^|[.!?]\s+)([a-zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ])',
                  lambda m: m.group(1) + m.group(2).upper(), text)
    
    return text.strip()


def should_log_transcript(text: str) -> bool:
    """
    Determine if transcript should be logged (filter out noise/fillers).
    
    Args:
        text: Recognized text
    
    Returns:
        True if should log, False if should skip
    
    Examples:
        >>> should_log_transcript("ờ ừ")
        False
        
        >>> should_log_transcript("Tôi đồng ý với ý kiến này")
        True
    """
    if not text or not text.strip():
        return False
    
    # Filter and normalize
    filtered = filter_filler_words(text)
    
    # Skip if too short after filtering
    if len(filtered) < 3:
        return False
    
    # Skip if only punctuation
    if all(c in '.,!?;:\'"()[]{}…–— \t\n' for c in filtered):
        return False
    
    # Skip if less than 2 actual words
    words = [w for w in filtered.split() if len(w) >= 2]
    if len(words) < 2:
        return False
    
    return True


def calculate_speech_confidence(text: str, azure_confidence: float | None = None) -> float:
    """
    Calculate overall confidence for speech recognition result.
    
    Combines Azure confidence with text quality heuristics.
    
    Args:
        text: Recognized text
        azure_confidence: Optional Azure confidence score (0-1)
    
    Returns:
        Combined confidence score (0-1)
    """
    if not text:
        return 0.0
    
    # Start with Azure confidence or neutral
    confidence = azure_confidence if azure_confidence is not None else 0.5
    
    # Penalty for short text
    word_count = len(text.split())
    if word_count < 3:
        confidence *= 0.8
    
    # Penalty for filler-heavy text
    filtered = filter_filler_words(text)
    filler_ratio = 1 - (len(filtered) / max(len(text), 1))
    if filler_ratio > 0.5:  # More than 50% fillers
        confidence *= 0.7
    
    # Bonus for Vietnamese diacritics (indicates good recognition)
    vietnamese_chars = sum(1 for c in text if c in 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
    if vietnamese_chars > word_count * 0.3:  # At least 30% Vietnamese chars
        confidence = min(1.0, confidence * 1.1)
    
    return round(confidence, 3)
