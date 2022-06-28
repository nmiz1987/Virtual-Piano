from enum import Enum


class PianoEnum(Enum):

    @classmethod
    def to_dict(cls):
        return {e.name: e.value for e in cls}

    @classmethod
    def keys(cls):
        """Returns a list of all the enum keys."""
        return cls._member_names_

    @classmethod
    def values(cls):
        """Returns a list of all the enum values."""
        return list(cls._value2member_map_.keys())


class Variables(PianoEnum):
    ACTION = "ACTION"
    PLOT_GRAPHS = "PLOT_GRAPHS"


class Action(PianoEnum):
    RECORD = "record"
    PROCESS = "process"


class Landmarks(PianoEnum):
    HANDS = "hands"
    TIMESTAMP = "timestamp"
    LOC_X = "loc_x"
    LOC_Y = "loc_y"
    AXIS_X = "axis_x"
    AXIS_Y = "axis_y"
    COEFFICIENTS = "coefficients"
    COEFFICIENT_A = "coefficient_a"
    COEFFICIENT_B = "coefficient_b"
    COEFFICIENT_C = "coefficient_c"
    VELOCITY_X = "velocity_x"
    VELOCITY_Y = "velocity_y"
    TAP = "tap"


class Keyboard(PianoEnum):
    POLYGON = "polygon"
    SOUND_FILE = "sound_file"
