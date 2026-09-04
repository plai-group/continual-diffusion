"""Vendored, unmodified from PLAIOmni's keypress_lookup.py (plaicraft-debug#80).

Upstream commit: bdc567e8b53ee96857b77126cf71d035b297ddbd. Byte-identical to
plaicraft-model-pi0/keypress_autoencoder/constants.py's FIXED_KEYS /
FIXED_MOUSE_BUTTONS (plaicraft-debug#74) -- the 6 debug keys and the two mouse
buttons sit at the same real indices in both, which is what lets
km_tokenizer/model.py's _RAW_POSITIONS transfer directly (see keypress_scatter.py).
"""
from typing import Final


FIXED_KEYS: Final[dict[str, str]] = {
    "32": "space", "65": "a", "66": "b", "67": "c", "68": "d", "69": "e", "70": "f",
    "71": "g", "72": "h", "73": "i", "74": "j", "75": "k", "76": "l", "77": "m",
    "78": "n", "79": "o", "80": "p", "81": "q", "82": "r", "83": "s", "84": "t",
    "85": "u", "86": "v", "87": "w", "88": "x", "89": "y", "90": "z", "48": "0",
    "49": "1", "50": "2", "51": "3", "52": "4", "53": "5", "54": "6", "55": "7",
    "56": "8", "57": "9", "256": "Escape", "257": "Return", "258": "Tab", "259": "BackSpace",
    "260": "Insert", "261": "Delete", "262": "Right", "263": "Left", "264": "Down",
    "265": "Up", "266": "Page_Up", "267": "Page_Down", "268": "Home", "269": "End",
    "340": "Shift_L", "341": "Control_L", "342": "Alt_L", "343": "Super_L", "344": "Shift_R",
    "345": "Control_R", "346": "Alt_R", "347": "Super_R", "348": "Menu", "91": "bracketleft",
    "93": "bracketright", "92": "backslash", "59": "semicolon", "39": "apostrophe",
    "44": "comma", "46": "period", "47": "slash", "45": "minus", "61": "equal", "96": "grave",
    "290": "F1", "292": "F3", "294": "F5", "280": "Caps_Lock"
}

FIXED_MOUSE_BUTTONS: Final[dict[str, str]] = {
    "left": "mouse_left",
    "right": "mouse_right",
    "scroll_up": "scroll_up",
    "scroll_down": "scroll_down",
}

KEYPRESS_IDS: Final[list[str]] = list(FIXED_KEYS.keys()) + list(FIXED_MOUSE_BUTTONS.keys())
KEYPRESS_LABELS: Final[list[str]] = list(FIXED_KEYS.values()) + list(FIXED_MOUSE_BUTTONS.values())
NUM_KEYPRESS_CLASSES: Final[int] = len(KEYPRESS_IDS)

assert NUM_KEYPRESS_CLASSES == 79, NUM_KEYPRESS_CLASSES

ID_TO_KEYPRESS_INDEX: Final[dict[str, int]] = {
    key_id: idx for idx, key_id in enumerate(KEYPRESS_IDS)
}

INDEX_TO_KEYPRESS_ID: Final[dict[int, str]] = {
    idx: key_id for key_id, idx in ID_TO_KEYPRESS_INDEX.items()
}

INDEX_TO_KEYPRESS_LABEL: Final[dict[int, str]] = {
    idx: label for idx, label in enumerate(KEYPRESS_LABELS)
}

KEYPRESS_ID_TO_LABEL: Final[dict[str, str]] = {
    **FIXED_KEYS,
    **FIXED_MOUSE_BUTTONS,
}


def build_keypress_lookup() -> dict[str, int]:
    lookup: dict[str, int] = {}

    for idx, (key_id, key_name) in enumerate(FIXED_KEYS.items()):
        lookup[str(key_id)] = idx
        lookup[str(key_name)] = idx
        lookup[str(key_name).lower()] = idx

    mouse_offset = len(FIXED_KEYS)
    for offset, (mouse_id, mouse_name) in enumerate(FIXED_MOUSE_BUTTONS.items()):
        idx = mouse_offset + offset
        lookup[str(mouse_id)] = idx
        lookup[str(mouse_id).lower()] = idx
        lookup[str(mouse_name)] = idx
        lookup[str(mouse_name).lower()] = idx

    return lookup


KEYPRESS_LOOKUP: Final[dict[str, int]] = build_keypress_lookup()