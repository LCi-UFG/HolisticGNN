def ATOMIC_NUMBER():
    return [
        # Alkali Metals (1 electron in the valence shell)
        1, 11, 19,
        # Alkaline Earth Metals (2 electrons in the valence shell)
        12, 20,
        # Transition Metals (various numbers of electrons)
        22, 23, 24, 25, 26, 27, 28, 29, 30,
        # Metalloids and elements with 3-5 electrons in the valence shell
        5, 13, 14, 15, 33,
        # Non-metals and elements with 6-7 electrons in the valence shell
        6, 7, 8, 16, 34, 9, 17,
        # Other halogens
        35, 53
        ]


METALS_LIST = {
    3, 4, 21, 31, 37, 38, 39, 40, 41, 
    42, 43, 44, 45, 46, 47, 48, 49, 
    50, 51, 52, 55, 56, 57, 58, 
    59, 60, 61, 62, 63, 64, 65, 66, 
    67, 68, 69, 70, 71, 72, 73, 74, 75, 
    76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86
    }


ATOMIC_SUBSTITUTIONS = {
    # Alkali Metals (1 electron in the valence shell)
    1: [1],
    # Alkaline Earth Metals (2 electrons in the valence shell)
    11: [12],
    12: [11, 20], 
    19: [11, 12],
    20: [12],
    # Transition Metals (various numbers of electrons)
    22: [23, 24, 25, 26],
    23: [22, 24, 25, 26], 
    24: [22, 23, 25, 26],
    25: [22, 23, 24, 26], 
    26: [22, 23, 24, 25],  
    # Metalloids and elements with 3-5 electrons in the valence shell
    5: [7],
    13: [5], 
    14: [6, 13], 
    15: [7, 33],
    33: [15], 
    # Non-metals and elements with 6-7 electrons in the valence shell
    6: [6, 7, 8, 9, 14],
    7: [6, 8, 16],
    8: [7, 9, 16, 17],
    9: [1, 8, 17, 35],
    16: [8, 7, 6, 34],
    17: [8, 9, 35, 53],
    34: [16],
    # Other halogens
    35: [6, 9, 17, 53],
    53: [6, 9, 17, 35],
    }


CHARGE_SUBSTITUTIONS = {
    7: [0, 1, 2], 
    8: [0, -1], 
    15: [0, 1], 
    16: [-1, 0, 1],
    34: [-1, 0],
    }


HYBRIDIZATION_SUBSTITUTIONS = {
    1: {'types': [0]}, 
    5: {'types': [1, 2]}, 
    6: {'types': [2, 1, 0]}, 
    7: {'types': [2, 1, 0]},
    8: {'types': [1, 2]}, 
    9: {'types': [2]}, 
    11: {'types': [0]}, 
    12: {'types': [0]},
    13: {'types': [2]}, 
    14: {'types': [2]}, 
    15: {'types': [2, 1]}, 
    16: {'types': [2, 1]},
    17: {'types': [2]}, 
    19: {'types': [0]}, 
    20: {'types': [0]}, 
    22: {'types': [3]}, 
    23: {'types': [3]},
    24: {'types': [3]}, 
    25: {'types': [3]}, 
    26: {'types': [3]}, 
    27: {'types': [3]},
    28: {'types': [3]}, 
    29: {'types': [3]}, 
    30: {'types': [3]}, 
    33: {'types': [2]},
    34: {'types': [2, 1]}, 
    35: {'types': [2]}, 
    53: {'types': [2]}
    }