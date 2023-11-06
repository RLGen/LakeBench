

class invert_index:

    def __init__(self):
        self.index = {}
    
    def add(self, grid, col):
        if grid not in self.index.keys():
            self.index[grid] = set()
        self.index[grid].add(col)
        
    def search(self, grid):
        return self.index.get(grid, set())