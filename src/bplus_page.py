"""
B+Tree page types.

LeafPage  — wraps a SlottedPage, stores sorted (key, record) cells,
            has a next_page_id pointer for the leaf linked list.

IndexPage — stores (keys[], children[]) for internal tree routing.
            len(children) == len(keys) + 1 always.

Both accept a max_keys parameter so tests can force splits at small sizes
without needing to fill 4 KB pages.
"""

import struct
from enum import Enum

from src.slotted_page import SlottedPage


class PageType(Enum):
    INDEX = 0
    LEAF  = 1


# Every leaf cell on a SlottedPage is:  [key: 4B uint32][record bytes...]
_KEY_FMT  = '<I'
_KEY_SIZE = 4


# ---------------------------------------------------------------------------
# LeafPage
# ---------------------------------------------------------------------------

class LeafPage:
    """
    B+Tree leaf node.

    Wraps a SlottedPage whose cells are sorted by key.  The page_type is
    always PageType.LEAF.  next_page_id links adjacent leaves so range
    scans never need to go back up through index pages.
    """

    page_type = PageType.LEAF

    def __init__(self, page_id: int, max_keys: int = 200):
        self.page_id      = page_id
        self.next_page_id = None          # page_id of right sibling, or None
        self.max_keys     = max_keys
        self._page        = SlottedPage()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entries(self) -> list:
        """Return [(key, record_bytes), ...] in sorted key order."""
        result = []
        for i in range(self._page._get_num_slots()):
            raw = self._page.read(i)
            if raw is None:
                continue
            key    = struct.unpack(_KEY_FMT, raw[:_KEY_SIZE])[0]
            record = raw[_KEY_SIZE:]
            result.append((key, record))
        return result

    def _rebuild(self, entries: list):
        """Replace the underlying SlottedPage with exactly these entries."""
        self._page = SlottedPage()
        for key, record in entries:
            cell = struct.pack(_KEY_FMT, key) + record
            self._page.insert(cell)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def num_keys(self) -> int:
        return self._page._get_num_slots()

    def is_full(self) -> bool:
        """True when the page has exceeded its max_keys capacity."""
        return self.num_keys() > self.max_keys

    def search(self, key: int):
        """Return record bytes for key, or None if not found."""
        entries = self._entries()
        lo, hi  = 0, len(entries) - 1
        while lo <= hi:
            mid   = (lo + hi) // 2
            k, record = entries[mid]
            if k == key:
                return record
            elif k < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def insert(self, key: int, record: bytes):
        """Insert or update a key.  Maintains sorted order."""
        entries = self._entries()
        for i, (k, _) in enumerate(entries):
            if k == key:
                entries[i] = (key, record)   # update in place
                self._rebuild(entries)
                return
        entries.append((key, record))
        entries.sort(key=lambda x: x[0])
        self._rebuild(entries)

    def split(self, right: 'LeafPage') -> int:
        """
        Split this leaf into two halves.

        Left half (self) keeps entries[:mid].
        Right half (right) gets entries[mid:].

        Leaf links are updated:  self → right → old self.next

        Returns the split_key (first key of right), which is *copied* up
        to the parent index page.
        """
        entries = self._entries()
        mid     = len(entries) // 2

        left_entries  = entries[:mid]
        right_entries = entries[mid:]

        self._rebuild(left_entries)
        right._rebuild(right_entries)

        # stitch the linked list
        right.next_page_id = self.next_page_id
        self.next_page_id  = right.page_id

        return right_entries[0][0]   # split_key copied up


# ---------------------------------------------------------------------------
# IndexPage
# ---------------------------------------------------------------------------

class IndexPage:
    """
    B+Tree internal (index) node.

    Stores keys[] and children[] where len(children) == len(keys) + 1.

    Routing rule:
        key < keys[0]              → children[0]
        keys[i-1] <= key < keys[i] → children[i]
        key >= keys[-1]            → children[-1]
    """

    page_type = PageType.INDEX

    def __init__(self, page_id: int, max_keys: int = 4):
        self.page_id  = page_id
        self.max_keys = max_keys
        self.keys:     list[int] = []
        self.children: list[int] = []   # page_ids; len == len(keys) + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def num_keys(self) -> int:
        return len(self.keys)

    def is_full(self) -> bool:
        return self.num_keys() > self.max_keys

    def find_child(self, key: int) -> int:
        """Return the page_id of the child that should contain key."""
        for i, k in enumerate(self.keys):
            if key < k:
                return self.children[i]
        return self.children[-1]

    def insert_key(self, key: int, right_child_id: int):
        """
        Insert a key that was pushed up from a child split.
        right_child_id is the newly-created right sibling page.
        """
        i = 0
        while i < len(self.keys) and self.keys[i] < key:
            i += 1
        self.keys.insert(i, key)
        self.children.insert(i + 1, right_child_id)

    def split(self, right: 'IndexPage') -> int:
        """
        Split this index page into two halves.

        The middle key is *pushed* up to the parent (not copied — it does
        not remain in either child).

        Left half (self)  keeps keys[:mid]    and children[:mid+1].
        Right half (right) gets keys[mid+1:]  and children[mid+1:].

        Returns push_up_key (the key to insert into the parent).
        """
        mid          = len(self.keys) // 2
        push_up_key  = self.keys[mid]

        right.keys     = self.keys[mid + 1:]
        right.children = self.children[mid + 1:]

        self.keys     = self.keys[:mid]
        self.children = self.children[:mid + 1]

        return push_up_key
