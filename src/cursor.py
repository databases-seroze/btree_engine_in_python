"""
A cursor tracks the state of a query. If you are halfway through a table scan,  the cursor remembers which page and which cell you are 
currently looking at. 

Responsibilities: 
    - next()
    - prev()    
    - reset()
    
"""