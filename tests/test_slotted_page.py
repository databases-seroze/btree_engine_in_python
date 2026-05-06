
from src.slotted_page import SlottedPage 

def make_record(s: str):
    return s.encode('utf-8')


def test_insert_read():
    page = SlottedPage()

    slot = page.insert(make_record("hello"))
    result = page.read(slot)

    assert result == b"hello"


def test_multiple_inserts():
    page = SlottedPage()

    data = [b"a", b"bb", b"ccc", b"dddd"]
    slots = []

    for d in data:
        slots.append(page.insert(d))

    for i, slot in enumerate(slots):
        assert page.read(slot) == data[i]


def test_delete():
    page = SlottedPage()

    s1 = page.insert(b"hello")
    s2 = page.insert(b"world")

    page.delete(s1)

    assert page.read(s1) is None
    assert page.read(s2) == b"world"

def test_compact():
    page = SlottedPage()

    slots = []
    slots.append(page.insert(b"aaa"))
    slots.append(page.insert(b"bbb"))
    slots.append(page.insert(b"ccc"))

    # delete middle
    page.delete(slots[1])

    before = page.read(slots[0]), page.read(slots[2])

    page.compact()

    after = page.read(slots[0]), page.read(slots[2])

    assert before == after
    assert page.read(slots[1]) is None


def test_compact_reclaims_space():
    page = SlottedPage()

    slots = []
    for _ in range(50):
        slots.append(page.insert(b"x" * 50))

    # delete half
    for i in range(0, 50, 2):
        page.delete(slots[i])

    free_before = page._get_free_end() - page._get_free_start()

    page.compact()

    free_after = page._get_free_end() - page._get_free_start()

    assert free_after > free_before


def test_invalid_slot():
    page = SlottedPage()

    try:
        page.read(0)
        assert False
    except IndexError:
        pass


def test_page_full():
    page = SlottedPage()

    try:
        while True:
            page.insert(b"x" * 100)
    except Exception as e:
        assert str(e) == "Page full"

def test_slot_stability_after_delete_and_compact():
    page = SlottedPage()

    s1 = page.insert(b"one")
    s2 = page.insert(b"two")
    s3 = page.insert(b"three")

    page.delete(s2)
    page.compact()

    # slot IDs should still map correctly
    assert page.read(s1) == b"one"
    assert page.read(s2) is None
    assert page.read(s3) == b"three"
