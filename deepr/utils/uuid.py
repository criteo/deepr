"""Uuid"""

import struct
import uuid


def msb_lsb_to_str(msb, lsb):
    return str(uuid.UUID(bytes=struct.pack(">qq", msb, lsb)))


def str_to_msb_lsb(el):
    return struct.unpack(">qq", uuid.UUID(el).bytes)
