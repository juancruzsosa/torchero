import unicodedata

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
