class MeterNotFound(Exception):
    METER_NOT_FOUND_MESSAGE = 'Meter \'{name}\' not found!'

    def __init__(self, meter_name):
        super(MeterNotFound, self).__init__(self.METER_NOT_FOUND_MESSAGE
                                                .format(name=meter_name))
