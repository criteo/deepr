"""Uuid"""

import struct
import uuid


def msb_lsb_to_str(msb, lsb):
    """Convert two 64 bit integers MSB and LSB to a 128 bit UUID."""
    return str(uuid.UUID(bytes=struct.pack(">qq", msb, lsb)))


def str_to_msb_lsb(el):
    """Convert a 128 bit UUID to two 64 bit integers MSB and LSB."""
    return struct.unpack(">qq", uuid.UUID(el).bytes)
