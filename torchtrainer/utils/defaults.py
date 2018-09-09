def get_default_mode(meter):
    if hasattr(meter.__class__, 'DEFAULT_MODE'):
        return getattr(meter.__class__, 'DEFAULT_MODE')
    else:
        return ''
