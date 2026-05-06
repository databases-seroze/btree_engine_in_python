import time
import json
import os
import zlib
import random 

class BloomFilter:
    """A simple, dependency-free Bloom Filter."""
    def __init__(self, size=1024, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = 0  # Using a large integer as a bit-array

    def _hashes(self, key):
        """Generates multiple hash positions for a key."""
        hashes = []
        for i in range(self.hash_count):
            # Using adler32 with different seeds to simulate multiple hash functions
            h = zlib.adler32(f"{key}:{i}".encode()) % self.size
            hashes.append(h)
        return hashes

    def add(self, key):
        for h in self._hashes(key):
            self.bit_array |= (1 << h)

    def exists(self, key):
        for h in self._hashes(key):
            if not (self.bit_array & (1 << h)):
                return False
        return True

class PersistentLSM:
    def __init__(self, mem_limit=3, data_dir="data"):
        self.data_dir = data_dir
        self.mem_limit = mem_limit
        self.memtable = {}
        self.index = [] 
        self.filters = {}      # Maps filename -> BloomFilter object
        self.generation = 0
        self.wal_path = os.path.join(self.data_dir, "wal.log")

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # RECOVERY: Rebuild state from disk
        self._recover()

    def _recover(self):
        """Scans disk to rebuild index and replay WAL."""
        print("--- Initializing Recovery ---")
        
        # 1. Rebuild Index and Generation from existing SSTables
        files = [f for f in os.listdir(self.data_dir) if f.startswith("sst_")]
        # Sort by generation number
        files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        for f in files:
            path = os.path.join(self.data_dir, f)
            self.index.append(path)
            gen = int(f.split('_')[2].split('.')[0])
            self.generation = max(self.generation, gen)
            
            # Rebuild Bloom Filter for each existing file
            bf = BloomFilter()
            with open(path, 'r') as sst:
                data = json.load(sst)
                for key in data:
                    bf.add(key)
            self.filters[path] = bf
        
        # 2. Replay WAL into Memtable
        if os.path.exists(self.wal_path):
            print(f"Replaying WAL: {self.wal_path}")
            with open(self.wal_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    # Restore into memtable
                    self.memtable[entry['key']] = entry['data']
            print(f"Recovered {len(self.memtable)} items from WAL.")

    def _write_to_wal(self, key, data):
        """Appends a single operation to the Write-Ahead Log."""
        with open(self.wal_path, 'a') as f:
            f.write(json.dumps({'key': key, 'data': data}) + "\n")

    def put(self, key, value):
        entry = {'val': value, 'ts': time.time(), 'del': False}
        self._write_to_wal(key, entry) # Step 1: Durability
        self.memtable[key] = entry      # Step 2: Performance
        if len(self.memtable) >= self.mem_limit:
            self.flush()

    def delete(self, key):
        entry = {'val': None, 'ts': time.time(), 'del': True}
        self._write_to_wal(key, entry)
        self.memtable[key] = entry
        if len(self.memtable) >= self.mem_limit:
            self.flush()

    def flush(self):
        if not self.memtable: return
        self.generation += 1
        fname = os.path.join(self.data_dir, f"sst_gen_{self.generation}.json")
        
        # Build Bloom Filter for the new file
        bf = BloomFilter()
        sorted_data = dict(sorted(self.memtable.items()))
        for key in sorted_data:
            bf.add(key)
        
        with open(fname, 'w') as f:
            json.dump(sorted_data, f)
            
        self.index.append(fname)
        self.filters[fname] = bf
        self.memtable = {}
        
        # Clear WAL after successful flush
        if os.path.exists(self.wal_path):
            os.remove(self.wal_path)
        print(f"--- Flushed {fname} and cleared WAL ---")

    def get(self, key):
        # 1. Check Memtable
        if key in self.memtable:
            e = self.memtable[key]
            return None if e['del'] else e['val']
        
        # 2. Check SSTables (using Bloom Filter to skip unnecessary reads)
        for fname in reversed(self.index):
            if self.filters[fname].exists(key):
                # Only open file if Bloom Filter says 'Maybe'
                with open(fname, 'r') as f:
                    data = json.load(f)
                    if key in data:
                        e = data[key]
                        return None if e['del'] else e['val']
        return "Not Found"

# --- Test Script ---
if __name__ == "__main__":
    # 1. First run: Add data and "crash" (stop script)
    db = PersistentLSM(mem_limit=5)
    for _ in range(100):
        rand_key = f"{random.randint(1, 100000)}_key"
        rand_val = f"{random.randint(1, 100000)}_val"
        db.put(rand_key, rand_val)   
    
    # Normally we would stop here. If you run the script again, 
    # the code below will show that 'session_1' was recovered from the WAL.
    print(f"Check session_1: {db.get('session_1')}")