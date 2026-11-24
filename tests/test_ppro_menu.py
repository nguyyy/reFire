"""ppro/menu.py: Premiere's menu bar found by label (the win32 calls need Premiere, not tested)."""
import pytest

from refire.ppro import menu

TREE = [("&Edit", -1, [("Select &All\tCtrl+A", 57642, []), ("&Deselect All\tCtrl+Shift+A", 11303, [])]),
        ("&Window", -1, [("&Timelines", -1, [("reFire cut 1", 32486, []), ("refire_captions", 32477, [])])]),
        ("&Graphics and Titles", -1, [("Upgrade Caption to Graphic", 21827, [])])]


def test_menu_items_are_found_by_label_not_id():
    assert menu.label("Select &All\tCtrl+A") == "Select All"
    assert menu.find(TREE, ["Edit", "Select All"]) == 57642
    assert menu.find(TREE, ["Window", "Timelines", "refire_captions"]) == 32477
    assert menu.find(TREE, ["Graphics and Titles", "Upgrade Caption to Graphic"]) == 21827
    with pytest.raises(KeyError, match="Window > Timelines > nope"):
        menu.find(TREE, ["Window", "Timelines", "nope"])
