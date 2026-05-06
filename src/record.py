"""
This module defines the record structure used in the B+ tree.
Responsiblities: Packing a dictionary or tuple into fixed-length or variable byte string. 

key concept: Implementing a slotted page structure here is vital. Instead of writing records at arbitary offsets like 100,200, you store an array of offsets at the tope of the page. 


In sqlite row is stored as [header|payload]

[header_size (2 bytes)]
[num_cols (2 bytes)]
[type1][type2][type3]...   <-- 1 byte each
----------------------------------------
[value1][value2][value3]

"""

import struct  
from enum import Enum 

class ValueType(Enum):
    NULL = 0 
    INT = 1
    STRING = 2
    
# encode one column
def encode_value(value):

    if value is None:
        return ValueType.NULL, b''
    elif isinstance(value, int):
        return ValueType.INT, struct.pack('<i', value) # 4-byte int 
    elif isinstance(value, str):
        encoded = value.encode('utf-8')
        length = struct.pack('<H', len(encoded)) # 2-byte length
        return ValueType.STRING, length+encoded  
    else:
        raise ValueError(f'unsupported value type: {type(value)}')

# encode full row (all columns)
def encode_record(values):
    types = [] 
    payload = b''

    for v in values:
        t, p = encode_value(v)
        types.append(t)
        payload += p 

    num_cols = len(values)
    header = struct.pack('<H', num_cols) + b''.join([t.value.to_bytes(1, 'little') for t in types])
    header_size = len(header)

    return struct.pack('<H', header_size) + header + payload 

def decode_record(data: bytes):

    offset = 0 
    header_size = struct.unpack('<H', data[offset:offset+2])[0]
    offset += 2
    num_cols = struct.unpack('<H', data[offset:offset+2])[0]
    offset += 2
    types = [ValueType(t) for t in data[offset:offset+num_cols]]
    offset += num_cols

    values = []
    for t in types:
        if t == ValueType.NULL:
            values.append(None)
        elif t == ValueType.INT:
            values.append(struct.unpack('<i', data[offset:offset+4])[0])
            offset += 4
        elif t == ValueType.STRING:
            length = struct.unpack('<H', data[offset:offset+2])[0]
            values.append(data[offset+2:offset+2+length].decode('utf-8'))
            offset += 2 + length
        else:
            raise ValueError(f'unsupported value type: {t}')

    return values