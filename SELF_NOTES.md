
Milestones
[] Implement Page Header & Fixed-length Header and read/write at specified byte offset using seek()
[] 
[] 


The Data Flow

When you call db.get("user_1"):

    Engine asks the B-Tree to find "user_1".

    B-Tree asks the Pager for the Root Page (Page 0).

    B-Tree uses Record to parse the Root Page's keys.

    B-Tree identifies the child Page ID, asks the Pager for it, and repeats until it finds the Leaf Page.

    Record extracts the final value and returns it to the Engine.