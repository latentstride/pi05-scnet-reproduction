"""Opt-in G2-E native horizon contract; legacy and ARX admission stay unchanged."""

PROTOCOL = "g2e_native_guarded_prefix_v1"
CONTACT_MODE_NATIVE = "g2e_native_measured_contact_v1"
WINDOWS = (1, 4, 26, 31)


def validate_window(window, protocol=None, embodiment="aloha-agilex"):
    if protocol is not None and (protocol != PROTOCOL or embodiment != "aloha-agilex"):
        raise ValueError("G2-E protocol requires native Aloha")
    admitted = WINDOWS if protocol == PROTOCOL else (1, 4)
    if type(window) is not int or window not in admitted:
        raise ValueError("invalid action window for the selected protocol")
