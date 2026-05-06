
import struct
from this import d

from src.record import encode_record, decode_record

def test_encode_decode_record1():
    row = [42, "hello", None]
    
    encoded = encode_record(row)
    decoded = decode_record(encoded)
    
    assert decoded == row 