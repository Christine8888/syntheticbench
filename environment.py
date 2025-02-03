import model
import alchemy

class Environment():
    def __init__(self, composition_depth, n_unique_elements, n_syms, n_winstates, max_element_length):
        
        self.tree = alchemy.NComposition(composition_depth = composition_depth, 
                                         n_unique_elements = n_unique_elements, 
                                         n_syms = n_syms, n_winstates = n_winstates, 
                                         max_element_length = max_element_length)
        self.tree.build()

        print(self.tree.unique_elements)
        print(self.tree.winstates)
        
    def prompt(self):
        charstring = '{' + ', '.join(self.tree.symbols) + '}'
        return f"Generate a string of characters from {charstring}. <start> "

    def compute_reward(self, response):
        if "</start>" in response:
            response = response.split("<start>")[1].split("</start>")[0]
            response.strip()
            overlap, match = self.tree.check(response)
            if match:
                return 2
            else:
                return overlap - 0.5
            
        else: return -1
        
