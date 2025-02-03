import random, string

class Build:
    def __init__(self,label,depth,parents = [None]):
        self.label = label
        self.depth = depth
        self.reward = depth * depth 
        self.children = []
        self.parents = parents

class BinaryComposition:
    def __init__(self,depth, num_syms, keep_n=2):
        self.depth = depth
        self.num_syms = num_syms
        self.symbols = random.sample(string.ascii_uppercase + string.digits,self.num_syms)
        self.winstate = None

        self.objlist = [Build(s, 1) for s in self.symbols]
        self.build()

    def build(self):
        # number of generations to evolve for
        for d in range(1, self.depth + 1):
            next_level = []
            for left in self.objlist:
                for right in self.objlist:
                    # allow for reverse, same element composed twice
                    next_level.append(Build(left.label + right.label, d, [left, right]))
            
            self.objlist += next_level
            
        # choose random winner from the last level
        self.winstate = random.choice(next_level)

    def backtrace_win_path(self):
        path = []
        queue = [self.winstate]
        while queue:
            current = queue.pop(0)
            path.append(current)
            queue.extend([parent for parent in current.parents if parent is not None])
        
        return path

    def print_win_path(self):
        path = self.backtrace_win_path()
        cur_level = 0
        level = []
        for node in path:
            if node.depth > cur_level:
                print(level)
                level = []
                cur_level = node.depth
            
            level.append(node.label)
        
        print(level)

    def parse_sequence(self, seq):
        best_match = ""
        best_reward = 0
        
        def dfs(node):
            nonlocal best_match, best_reward
            # Check if this node's label matches a prefix of seq
            if node.label and seq.startswith(node.label):
                if node.reward > best_reward:
                    best_reward = node.reward
                    best_match = node.label
            # Continue searching children
            for child in node.children:
                dfs(child)
        
        dfs(self.root)
        return best_match, best_reward
   
    def generate_random(self, objlist, exclude = None, include = None):
        if exclude is None: exclude = []
        if include is None: include = []
        if len(include) > 0:
            return random.choice([obj for obj in objlist if obj.label in include])
        else:
            return random.choice([obj for obj in objlist if obj.label not in exclude])

class NComposition:
    def __init__(self, composition_depth, n_unique_elements, n_syms, n_winstates, max_element_length):
        self.composition_depth = composition_depth
        self.n_unique_elements = n_unique_elements
        self.unique_elements = []

        self.n_syms = n_syms
        self.max_element_length = max_element_length
        self.symbols = random.sample(string.ascii_uppercase + string.digits, self.n_syms)
        
        self.n_winstates = n_winstates
        self.winstates = []
    
    def make_all_combinations(self):
        # make all element combinations of up to max_element_length
        combinations = []
        self.add_combinations(combinations, "", self.symbols, self.max_element_length)
        return combinations

    def add_combinations(self, combinations, sofar, symbols, remaining):
        # backtracking approach
        if remaining == 0:
            return
        
        for symbol in symbols:
            combinations.append(sofar + symbol)
            self.add_combinations(combinations, sofar + symbol, symbols, remaining - 1)
    
    def build(self):
        combinations = self.make_all_combinations()
        self.unique_elements = random.sample(combinations, self.n_unique_elements)
        # sample n_winstates from composing unique elements, allowing repeats
        for i in range(self.n_winstates):
            winstate = ""
            for j in range(self.composition_depth):
                winstate += random.choice(self.unique_elements)

            self.winstates.append(winstate)
    
    def check(self, input):
        # return longest substring of input that matches a winstate
        max_overlap = 0
        match = False
        for winstate in self.winstates:
            for i in range(len(winstate)):
                if winstate[i:] in input:
                    max_overlap = max(max_overlap, len(winstate[i:]) / len(winstate))
                    
                    # complete match
                    if i == 0:
                        return 1, True
        
        return max_overlap, False

if __name__ == "__main__":
   tree = NComposition(composition_depth = 3, n_unique_elements = 5, n_syms = 10, 
                       n_winstates = 10, max_element_length = 3)
   tree.build()
   print(tree.unique_elements)
   print(tree.winstates)
   print(tree.check("ABCD"), tree.check(tree.winstates[0]))