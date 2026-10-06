import re

def preprocess_sms(text: str) -> str:
    """
    Normalizes noisy SMS text before it is exposed to the LLM context windows.
    """
    if not isinstance(text, str):
        return ""

    # Convert alphanumeric URLs to a uniform [URL] token placeholder
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.\S+')
    text = url_pattern.sub('[URL]', text)

    # Convert phone numbers (local & international) to [PHONE_NUMBER]
    # Matches formats like +23480..., 080..., etc. Minimum 7 digits.
    phone_pattern = re.compile(r'\+?\b\d[\d\-\s]{6,14}\b')
    # Convert shortcodes to [SHORTCODE] (typically 3-6 digits)
    shortcode_pattern = re.compile(r'\b\d{3,6}\b')

    # Apply phone number replacement first to avoid replacing parts of phone numbers with shortcodes
    text = phone_pattern.sub('[PHONE_NUMBER]', text)
    text = shortcode_pattern.sub('[SHORTCODE]', text)

    # Strip non-essential ASCII characters while preserving punctuation that denotes panic or structural markers (e.g., !!!, ??, *)
    # Allowed: alphanumeric, whitespace, and specific punctuation . , ! ? * ' " - _ # $ % & ( ) + = [ ] @
    allowed_chars_pattern = re.compile(r'[^\w\s\.\!\?\*\,\'\"\-\_\#\$\%\&\(\)\+\=\[\]\@]')
    text = allowed_chars_pattern.sub('', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
