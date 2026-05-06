"""
Responsibilities: find(key), insert(key, value), delete(key) and split_node()

Technical Details: It calls pager to get a page, uses record.py to parse it, and decides which child_page_id to follow. 

"""

from src.slotted_page import SlottedPage
import struct 


class BTreeLeaf:
    def __init__(self, page: SlottedPage):
        self.page = page 


    def _read_key(self, slot_id):
        raw = self.page.read(slot_id)
        if raw is None:
            return None 
        return struct.unpack('<I', raw[:4])[0]

    def find_slot(self, key):
        lo, hi = 0, self.page._get_num_slots() - 1
        
        while lo < hi:
            mid = (lo + hi + 1) // 2
            mid_key = self._read_key(mid)
            # if mid_key is None:
            #     hi = mid - 1
            if mid_key == key:
                return mid
            elif mid_key < key:
                lo = mid + 1
            else:
                hi = mid
        return lo # insertion position anta 

    def insert(self, key: int, record: bytes):
        new_cell = struct.pack('<I', key) + record

        entries = [] 

        for i in range(self.page._get_num_slots()):
            raw = self.page.read(i)
            if raw is None:
                continue 

            k = struct.unpack('<I', raw[:4])[0]
            entries.append((k, raw))
            
        # insert new 
        entries.append((key, new_cell)) 

        # sort by key 
        entries.sort(key=lambda x: x[0])

        # rebuild page 
        self.page = SlottedPage()
        for _, cell in entries:
            self.page.insert(cell)

    def search(self, key: int) -> bytes:

        pos = self.find_slot(key)

        if pos >= self.page._get_num_slots():
            return None 

        raw = self.page.read(pos)
        if raw is None:
            return None 

        k = struct.unpack('<I', raw[:4])[0]
        if k != key:
            return None 

        # extract record 
        length = struct.unpack('<I', raw[4:8])[0]
        return raw[8:8 + length]

    def split(self):
        entries = [] 

        for i in range(self.page._get_num_slots()):
            raw = self.page.read(i)
            if raw is None:
                continue 
            k = struct.unpack('<I', raw[:4])[0]
            entries.append((k, raw))

        mid = len(entries) // 2

        left_entries = entries[:mid]
        right_entries = entries[mid:]

        left_page = SlottedPage()
        for _, cell in left_entries:
            left_page.insert(cell)

        right_page = SlottedPage()
        for _, cell in right_entries:
            right_page.insert(cell)

        split_key = right_entries[0][0]

        return BTreeLeaf(left_page), BTreeLeaf(right_page), split_key
