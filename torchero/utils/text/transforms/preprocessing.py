import re
import unicodedata

RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
RE_NUMERIC = re.compile(r"[-+]?\d*\.\d+|\d+", re.UNICODE)

__all__ = ['convert_to_unicode',
           'strip_accents',
           'strip_tags',
           'strip_numeric']

def convert_to_unicode(text):
    """ Convert text to unicode

    Arguments:
        text (str, bytes): Input text

    Returns:
        str
            unicode text
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def strip_accents(text):
    """ Strips accents from a piece of text.

    Arguments:
        text (str): Input text

    Returns:
        str
            text with accents normalized
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def strip_tags(text):
    """ Remove HTML tags from given text

    Arguments:
        text (str): Input text

    Returns:
        str
            text without tags
    """
    return RE_TAGS.sub("", text).strip()

def strip_numeric(text):
    """ Remove numbers from given text

    Arguments:
        text (str): Input text

    Returns:
        str
            text without numbers
    """
    return RE_NUMERIC.sub("", text).strip()
