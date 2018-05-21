class ZeroMeasurementsError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return "No measurements has been made"
