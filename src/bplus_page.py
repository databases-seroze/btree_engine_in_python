"""
B+Tree page types.

Every page is exactly PAGE_SIZE (4096) bytes on disk.

LeafPage  — wraps a SlottedPage, stores sorted (key, record) cells,
            has a next_page_id pointer for the leaf linked list.

IndexPage — stores (keys[], children[]) for internal tree routing.
            len(children) == len(keys) + 1 always.

Both accept a max_keys parameter so tests can force splits at small sizes
without needing to fill 4 KB pages.

On-disk formats
---------------
LeafPage (4096 bytes total):

  Byte 0        : page_type = PageType.LEAF.value  (1)
  Bytes 1-4     : next_page_id  (uint32LE, NO_NEXT = 0xFFFFFFFF means no sibling)
  Bytes 5-4095  : SlottedPage body  (LEAF_BODY_SIZE = 4091 bytes)

IndexPage (4096 bytes total):

  Byte 0        : page_type = PageType.INDEX.value  (0)
  Bytes 1-4     : num_keys  (uint32LE)
  Bytes 5 …     : keys[0..n-1]        (n × uint32LE)
  …             : children[0..n]      ((n+1) × uint32LE)
"""

import struct
from enum import Enum

from src.slotted_page import SlottedPage, PAGE_SIZE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class PageType(Enum):
    INDEX = 0
    LEAF  = 1


# Each leaf cell on a SlottedPage is: [key: 4B uint32][record bytes...]
_KEY_FMT  = '<I'
_KEY_SIZE = 4

# LeafPage on-disk header occupies the first 5 bytes of the 4096-byte page.
LEAF_HEADER_SIZE = 5                          # type(1) + next_page_id(4)
LEAF_BODY_SIZE   = PAGE_SIZE - LEAF_HEADER_SIZE  # 4091 bytes for SlottedPage

# Sentinel stored in next_page_id when there is no right sibling.
NO_NEXT = 0xFFFF_FFFF


# ---------------------------------------------------------------------------
# LeafPage
# ---------------------------------------------------------------------------

class LeafPage:
    """
    B+Tree leaf node.

    Wraps a SlottedPage whose cells are sorted by key.  The page_type is
    always PageType.LEAF.  next_page_id links adjacent leaves so range
    scans never need to go back up through index pages.

    On-disk, the first 5 bytes are the B+Tree leaf header (type + next_page_id)
    and the remaining 4091 bytes are the SlottedPage body (LEAF_BODY_SIZE).
    """

    page_type = PageType.LEAF

    def __init__(self, page_id: int, max_keys: int = 200):
        self.page_id      = page_id
        self.next_page_id = None             # page_id of right sibling, or None
        self.max_keys     = max_keys
        self._page        = SlottedPage(size=LEAF_BODY_SIZE)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """
        Serialise this leaf page to exactly PAGE_SIZE (4096) bytes.

        Layout:
            [0]     page_type  (1 byte)
            [1:5]   next_page_id  (uint32LE; NO_NEXT when there is no sibling)
            [5:]    SlottedPage body  (LEAF_BODY_SIZE = 4091 bytes)
        """
        buf = bytearray(PAGE_SIZE)
        buf[0] = PageType.LEAF.value
        nxt    = NO_NEXT if self.next_page_id is None else self.next_page_id
        struct.pack_into('<I', buf, 1, nxt)
        buf[LEAF_HEADER_SIZE:] = self._page.data   # exactly LEAF_BODY_SIZE bytes
        return bytes(buf)

    @classmethod
    def from_bytes(cls, page_id: int, data: bytes | bytearray,
                   max_keys: int = 200) -> 'LeafPage':
        """
        Deserialise a LeafPage from a PAGE_SIZE byte buffer.

        Parameters
        ----------
        page_id  : int   — the page's logical ID (position in the file)
        data     : bytes — PAGE_SIZE bytes read from disk
        max_keys : int   — split threshold (must match the tree's order)
        """
        leaf = cls(page_id, max_keys)
        nxt  = struct.unpack_from('<I', data, 1)[0]
        leaf.next_page_id = None if nxt == NO_NEXT else nxt
        # Overwrite the fresh SlottedPage's data with the serialised body.
        # The body encodes its own header (free_start, free_end, num_slots),
        # so the SlottedPage is fully restored by this assignment.
        leaf._page.data[:] = data[LEAF_HEADER_SIZE : LEAF_HEADER_SIZE + LEAF_BODY_SIZE]
        return leaf

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
        """Replace the underlying SlottedPage with exactly these entries (must be sorted)."""
        self._page = SlottedPage(size=LEAF_BODY_SIZE)
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
            mid       = (lo + hi) // 2
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
                entries[i] = (key, record)
                self._rebuild(entries)
                return
        entries.append((key, record))
        entries.sort(key=lambda x: x[0])
        self._rebuild(entries)

    def delete_key(self, key: int) -> bool:
        """
        Remove key from this leaf.

        Returns True if the key existed and was removed, False if it was
        not present.  The underlying SlottedPage is rebuilt without the
        deleted entry.
        """
        entries     = self._entries()
        new_entries = [(k, v) for k, v in entries if k != key]
        if len(new_entries) == len(entries):
            return False   # key not found
        self._rebuild(new_entries)
        return True

    def split(self, right: 'LeafPage') -> int:
        """
        Split this leaf into two halves.

        Left half (self) keeps entries[:mid].
        Right half (right) gets entries[mid:].

        Leaf links are updated:  self → right → old self.next

        Returns the split_key (first key of right), which is *copied* up
        to the parent index page (B+Tree copy-up rule).
        """
        entries = self._entries()
        mid     = len(entries) // 2

        self._rebuild(entries[:mid])
        right._rebuild(entries[mid:])

        right.next_page_id = self.next_page_id
        self.next_page_id  = right.page_id

        return entries[mid][0]   # split_key copied up


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

    On-disk, the full keys and children lists are packed as uint32LE arrays
    right after a 5-byte header.  The total size is always PAGE_SIZE (4096)
    bytes regardless of how many keys are stored; unused bytes are zeros.
    """

    page_type = PageType.INDEX

    def __init__(self, page_id: int, max_keys: int = 4):
        self.page_id  = page_id
        self.max_keys = max_keys
        self.keys:     list[int] = []
        self.children: list[int] = []   # page_ids; len == len(keys) + 1

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """
        Serialise this index page to exactly PAGE_SIZE (4096) bytes.

        Layout:
            [0]      page_type  (1 byte)
            [1:5]    num_keys   (uint32LE)
            [5 …]    keys[0..n-1]      (n × uint32LE)
            [… ]     children[0..n]    ((n+1) × uint32LE)
            rest     zeros
        """
        buf    = bytearray(PAGE_SIZE)
        n      = len(self.keys)
        buf[0] = PageType.INDEX.value
        struct.pack_into('<I', buf, 1, n)
        offset = 5
        for k in self.keys:
            struct.pack_into('<I', buf, offset, k)
            offset += 4
        for c in self.children:
            struct.pack_into('<I', buf, offset, c)
            offset += 4
        return bytes(buf)

    @classmethod
    def from_bytes(cls, page_id: int, data: bytes | bytearray,
                   max_keys: int = 4) -> 'IndexPage':
        """
        Deserialise an IndexPage from a PAGE_SIZE byte buffer.

        Parameters
        ----------
        page_id  : int   — the page's logical ID
        data     : bytes — PAGE_SIZE bytes read from disk
        max_keys : int   — split threshold (must match the tree's order)
        """
        page = cls(page_id, max_keys)
        n    = struct.unpack_from('<I', data, 1)[0]
        if n > 0:
            offset       = 5
            page.keys    = list(struct.unpack_from(f'<{n}I',   data, offset))
            offset      += n * 4
            page.children = list(struct.unpack_from(f'<{n+1}I', data, offset))
        else:
            page.keys     = []
            page.children = []
        return page

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
        not remain in either child after the split).

        Left half (self)   keeps keys[:mid]    and children[:mid+1].
        Right half (right) gets  keys[mid+1:]  and children[mid+1:].

        Returns push_up_key (the key to insert into the parent).
        """
        mid         = len(self.keys) // 2
        push_up_key = self.keys[mid]

        right.keys     = self.keys[mid + 1:]
        right.children = self.children[mid + 1:]

        self.keys     = self.keys[:mid]
        self.children = self.children[:mid + 1]

        return push_up_key
