
"""

This is the only module that talks to the filesystem 
- read_page(page_num)
- write_page(page_num, data)

In future this is where we implement page cache to keep frequently used pages in memory.

"""



class Pager:

    def __init__(self):
        self.pages = {}
        self.next_page_id = 0 

    def new_page(self) -> SlottedPage:
        page = SlottedPage()
        self.pages[self.next_page_id] = page
        self.next_page_id += 1
        return page

    def get_page(self, page_id: int) -> SlottedPage:
        return self.pages[page_id]

    def search(self, key):
        lo, hi = 0, self.page._get_num_slots() - 1
    
        while lo <= hi:
            mid = (lo + hi) // 2
            raw = self.page.read(mid)
    
            k = int.from_bytes(raw[:4], 'little')
    
            if k == key:
                length = int.from_bytes(raw[4:8], 'little')
                return raw[8:8+length]
            elif k < key:
                lo = mid + 1
            else:
                hi = mid - 1
    
        return None

class LeafNode:
    def __init__(self, pager, page_id):
        self.pager = pager
        self.page_id = page_id
        self.page = pager.get_page(page_id)