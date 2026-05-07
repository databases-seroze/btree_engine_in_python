"""
BPlusPager — page allocator for the B+Tree.

Keeps an in-memory dict of page_id → LeafPage | IndexPage.
Designed so that the same interface can later be backed by file I/O
(new_leaf_page / new_index_page write bytes; get_page deserialises them).
"""

from src.bplus_page import LeafPage, IndexPage


class BPlusPager:

    def __init__(self):
        self._pages:      dict[int, LeafPage | IndexPage] = {}
        self._next_id:    int = 0

    # ------------------------------------------------------------------
    # Page allocation
    # ------------------------------------------------------------------

    def new_leaf_page(self, max_keys: int = 200) -> LeafPage:
        page = LeafPage(self._next_id, max_keys=max_keys)
        self._pages[self._next_id] = page
        self._next_id += 1
        return page

    def new_index_page(self, max_keys: int = 4) -> IndexPage:
        page = IndexPage(self._next_id, max_keys=max_keys)
        self._pages[self._next_id] = page
        self._next_id += 1
        return page

    # ------------------------------------------------------------------
    # Page access
    # ------------------------------------------------------------------

    def get_page(self, page_id: int) -> LeafPage | IndexPage:
        if page_id not in self._pages:
            raise KeyError(f"Page {page_id} does not exist")
        return self._pages[page_id]

    def num_pages(self) -> int:
        return len(self._pages)
