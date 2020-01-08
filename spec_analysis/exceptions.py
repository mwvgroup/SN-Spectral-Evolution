class NoInputGiven(Exception):
    """No input given from matplotlib input request"""
    pass


class FeatureOutOfBounds(Exception):
    """The requested feature was not observed"""
    pass
