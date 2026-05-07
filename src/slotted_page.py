"""
slotted_page.py — fixed-size page with a slot directory.

On-disk / in-memory layout (default PAGE_SIZE = 4096 bytes)
------------------------------------------------------------

  Byte offset 0                                       PAGE_SIZE-1
  ┌──────────────────────────────────────────────────────────────┐
  │ HEADER (12 bytes)                                            │
  │   [0:4]  free_start  ← offset where the next slot is written │
  │   [4:8]  free_end    ← offset where the next record starts   │
  │   [8:12] num_slots   ← total slot count (includes deleted)   │
  ├──────────────────────────────────────────────────────────────┤
  │ SLOT ARRAY  (grows →)                                        │
  │   slot 0 : 4-byte record offset                              │
  │   slot 1 : 4-byte record offset                              │
  │   …                                                          │
  ├──────────────────────────────────────────────────────────────┤
  │              F R E E   S P A C E                             │
  ├──────────────────────────────────────────────────────────────┤
  │ RECORDS  (← grows)                                          │
  │   …  [4-byte length][record bytes]  …                       │
  │   [4-byte length][record bytes]                              │
  └──────────────────────────────────────────────────────────────┘

Slot semantics
--------------
* A slot stores the byte offset of its record inside the page.
* A deleted slot stores offset = 0 (a sentinel; offset 0 is always occupied
  by the header and can never be a valid record start).
* Slot IDs are stable: deleting slot 1 does not renumber slot 2.
* compact() physically reclaims the space freed by deleted records and
  rewrites records contiguously, but the slot array (and therefore slot IDs)
  does not change.

Record storage format
---------------------
Each record is stored as:
    [length : 4 bytes little-endian uint32] [record payload : <length> bytes]

The length prefix is written by insert() and stripped by read(), so callers
always work with the raw payload only.

Variable `size`
---------------
The default page size is PAGE_SIZE (4096), but any size can be passed via the
`size` constructor argument.  This is used by LeafPage in the B+Tree, which
reserves the first 5 bytes of each on-disk page for its own header, leaving
PAGE_SIZE - 5 = 4091 bytes for the SlottedPage body.
"""

import struct

PAGE_SIZE = 4096


class SlottedPage:
    """
    A fixed-size page that stores variable-length records using a slot
    directory.

    The slot directory grows from the left (low addresses) and record data
    grows from the right (high addresses).  A page is full when the two
    regions would overlap.

    Parameters
    ----------
    size : int
        Total byte capacity of this page.  Defaults to PAGE_SIZE (4096).
        Pass a smaller value when the page is embedded inside a larger
        on-disk structure that has its own header (e.g. LeafPage reserves
        5 bytes for the B+Tree page header, so it passes size=4091).
    """

    # Header field positions (byte offsets inside self.data)
    _FREE_START_OFFSET = 0
    _FREE_END_OFFSET   = 4
    _NUM_SLOTS_OFFSET  = 8
    _HEADER_SIZE       = 12

    def __init__(self, size: int = PAGE_SIZE):
        self.size = size
        self.data = bytearray(size)

        self._set_free_start(self._HEADER_SIZE)   # first slot goes right after header
        self._set_free_end(size)                   # first record goes at the far end
        self._set_num_slots(0)

    # ------------------------------------------------------------------
    # Header accessors — all values are 4-byte little-endian uint32s
    # ------------------------------------------------------------------

    def _get_free_start(self) -> int:
        """Offset just past the last slot entry (= where next slot will be written)."""
        return struct.unpack('<I', self.data[0:4])[0]

    def _set_free_start(self, val: int):
        self.data[0:4] = struct.pack('<I', val)

    def _get_free_end(self) -> int:
        """Offset of the first byte of the lowest record (= top of record region)."""
        return struct.unpack('<I', self.data[4:8])[0]

    def _set_free_end(self, val: int):
        self.data[4:8] = struct.pack('<I', val)

    def _get_num_slots(self) -> int:
        """Total number of slots ever allocated (includes deleted ones)."""
        return struct.unpack('<I', self.data[8:12])[0]

    def _set_num_slots(self, val: int):
        self.data[8:12] = struct.pack('<I', val)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, record: bytes) -> int:
        """
        Append a record to the page and return its slot ID.

        The record is stored right-to-left (growing toward lower addresses)
        prefixed with a 4-byte length.  A new slot entry pointing to the
        record is appended to the left-to-right slot array.

        Parameters
        ----------
        record : bytes
            Raw record payload.

        Returns
        -------
        int
            The slot ID that permanently identifies this record.

        Raises
        ------
        Exception
            If the page does not have enough free space.
        """
        free_start = self._get_free_start()
        free_end   = self._get_free_end()
        num_slots  = self._get_num_slots()

        # On-disk form: 4-byte length prefix + payload
        record_with_len = struct.pack('<I', len(record)) + record
        size = len(record_with_len)

        # Need room for the record itself AND a new 4-byte slot entry
        if free_end - free_start < size + 4:
            raise Exception("Page full")

        # Write record growing right-to-left
        free_end -= size
        self.data[free_end : free_end + size] = record_with_len

        # Write slot entry growing left-to-right
        slot_offset = self._HEADER_SIZE + num_slots * 4
        self.data[slot_offset : slot_offset + 4] = struct.pack('<I', free_end)

        # Update header
        self._set_free_start(free_start + 4)
        self._set_free_end(free_end)
        self._set_num_slots(num_slots + 1)

        return num_slots   # slot ID for this record

    def read(self, slot_id: int):
        """
        Return the payload stored under slot_id, or None if deleted.

        Parameters
        ----------
        slot_id : int
            The value returned by a prior insert() call.

        Returns
        -------
        bytes or None
            Record payload, or None if the slot has been deleted.

        Raises
        ------
        IndexError
            If slot_id >= num_slots (slot was never allocated).
        """
        num_slots = self._get_num_slots()

        if slot_id >= num_slots:
            raise IndexError("Invalid slot")

        slot_offset   = self._HEADER_SIZE + slot_id * 4
        record_offset = struct.unpack('<I', self.data[slot_offset : slot_offset + 4])[0]

        if record_offset == 0:
            return None   # deleted sentinel

        length = struct.unpack('<I', self.data[record_offset : record_offset + 4])[0]
        start  = record_offset + 4
        return bytes(self.data[start : start + length])

    def delete(self, slot_id: int):
        """
        Mark a slot as deleted (lazy deletion — space is NOT reclaimed).

        The slot entry is set to offset 0, which is the deleted sentinel.
        Call compact() afterwards if you need to reclaim the freed space.

        Parameters
        ----------
        slot_id : int
            Slot to delete.

        Raises
        ------
        IndexError
            If slot_id >= num_slots.
        """
        if slot_id >= self._get_num_slots():
            raise IndexError("Invalid slot")

        slot_offset = self._HEADER_SIZE + slot_id * 4
        self.data[slot_offset : slot_offset + 4] = struct.pack('<I', 0)

    def compact(self):
        """
        Reclaim space from deleted records and defragment the record region.

        Records are copied to a fresh page in their original order, skipping
        deleted ones.  Slot IDs are preserved (a deleted slot still maps to
        None after compaction).

        After compaction, free_end − free_start is maximised; the gap that
        existed between deleted records and live ones is closed.
        """
        num_slots = self._get_num_slots()

        new_data      = bytearray(self.size)
        new_free_end  = self.size
        new_slots: list[int] = []

        for i in range(num_slots):
            slot_offset   = self._HEADER_SIZE + i * 4
            record_offset = struct.unpack('<I', self.data[slot_offset : slot_offset + 4])[0]

            if record_offset == 0:
                new_slots.append(0)   # preserve deleted sentinel
                continue

            length     = struct.unpack('<I', self.data[record_offset : record_offset + 4])[0]
            total_size = 4 + length

            new_free_end -= total_size
            new_data[new_free_end : new_free_end + total_size] = (
                self.data[record_offset : record_offset + total_size]
            )
            new_slots.append(new_free_end)

        # Write updated slot array
        for i, offset in enumerate(new_slots):
            pos = self._HEADER_SIZE + i * 4
            new_data[pos : pos + 4] = struct.pack('<I', offset)

        # Write updated header
        new_free_start = self._HEADER_SIZE + num_slots * 4
        new_data[0:4]  = struct.pack('<I', new_free_start)
        new_data[4:8]  = struct.pack('<I', new_free_end)
        new_data[8:12] = struct.pack('<I', num_slots)

        self.data = new_data
