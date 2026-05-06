import struct

PAGE_SIZE = 4096

class SlottedPage:
    def __init__(self):
        self.data = bytearray(PAGE_SIZE)

        # initialize header
        self._set_free_start(12)
        self._set_free_end(PAGE_SIZE)
        self._set_num_slots(0)

    # ---------------- HEADER ----------------

    def _get_free_start(self):
        return struct.unpack('<I', self.data[0:4])[0]

    def _set_free_start(self, val):
        self.data[0:4] = struct.pack('<I', val)

    def _get_free_end(self):
        return struct.unpack('<I', self.data[4:8])[0]

    def _set_free_end(self, val):
        self.data[4:8] = struct.pack('<I', val)

    def _get_num_slots(self):
        return struct.unpack('<I', self.data[8:12])[0]

    def _set_num_slots(self, val):
        self.data[8:12] = struct.pack('<I', val)

    def insert(self, record: bytes):
        free_start = self._get_free_start()
        free_end = self._get_free_end()
        num_slots = self._get_num_slots()
    
        record_with_len = struct.pack('<I', len(record)) + record
        size = len(record_with_len)
    
        if free_end - free_start < size + 4:
            raise Exception("Page full")
    
        # write record (right → left)
        free_end -= size
        self.data[free_end:free_end+size] = record_with_len
    
        # write slot (left → right)
        slot_offset = 12 + num_slots * 4
        self.data[slot_offset:slot_offset+4] = struct.pack('<I', free_end)
    
        # update header
        self._set_free_start(free_start + 4)
        self._set_free_end(free_end)
        self._set_num_slots(num_slots + 1)
    
        return num_slots

    def read(self, slot_id):
        num_slots = self._get_num_slots()
    
        if slot_id >= num_slots:
            raise IndexError("Invalid slot")
    
        slot_offset = 12 + slot_id * 4
        record_offset = struct.unpack('<I', self.data[slot_offset:slot_offset+4])[0]
    
        if record_offset == 0:
            return None  # deleted
    
        # read length
        length = struct.unpack('<I', self.data[record_offset:record_offset+4])[0]
    
        start = record_offset + 4
        end = start + length
    
        return self.data[start:end]

    def delete(self, slot_id):
        num_slots = self._get_num_slots()
    
        if slot_id >= num_slots:
            raise IndexError("Invalid slot")
    
        slot_offset = 12 + slot_id * 4
    
        # mark as deleted
        self.data[slot_offset:slot_offset+4] = struct.pack('<I', 0)

    def compact(self):
        num_slots = self._get_num_slots()
    
        new_data = bytearray(PAGE_SIZE)
    
        # copy header later
        new_free_start = 12 + num_slots * 4
        new_free_end = PAGE_SIZE
    
        # copy slots temporarily
        new_slots = []
    
        for i in range(num_slots):
            slot_offset = 12 + i * 4
            record_offset = struct.unpack('<I', self.data[slot_offset:slot_offset+4])[0]
    
            if record_offset == 0:
                new_slots.append(0)
                continue
    
            length = struct.unpack('<I', self.data[record_offset:record_offset+4])[0]
            total_size = 4 + length
    
            # move record (right → left)
            new_free_end -= total_size
    
            new_data[new_free_end:new_free_end+total_size] = \
                self.data[record_offset:record_offset+total_size]
    
            new_slots.append(new_free_end)
    
        # write slots back
        for i, offset in enumerate(new_slots):
            slot_pos = 12 + i * 4
            new_data[slot_pos:slot_pos+4] = struct.pack('<I', offset)
    
        # write header
        new_data[0:4] = struct.pack('<I', new_free_start)
        new_data[4:8] = struct.pack('<I', new_free_end)
        new_data[8:12] = struct.pack('<I', num_slots)
    
        self.data = new_data