from typing import Dict, Optional
from .prime_tree import build_prime_tree, PrimeTreeNode

class PrimeRecursiveDictEntry:
    def __init__(self, n: int):
        self.n = n
        self.tree: PrimeTreeNode = build_prime_tree(n)
        self.parentheses: str = self.tree.to_parentheses()
        self.children: Optional[Dict[int, 'PrimeRecursiveDictEntry']] = None
        # For composite numbers, recursively store children
        if len(self.tree.children) > 0:
            self.children = {}
            for child in self.tree.children:
                if child.value != n:  # Avoid self-loop
                    self.children[child.value] = PrimeRecursiveDictEntry(child.value)

    def __repr__(self):
        return f"PrimeRecursiveDictEntry(n={self.n}, parentheses='{self.parentheses}')"

class PrimeRecursiveDict:
    def __init__(self):
        self.entries: Dict[int, PrimeRecursiveDictEntry] = {}

    def lookup(self, n: int) -> PrimeRecursiveDictEntry:
        if n not in self.entries:
            self.entries[n] = PrimeRecursiveDictEntry(n)
        return self.entries[n]

# Example usage:
if __name__ == "__main__":
    rec_dict = PrimeRecursiveDict()
    entry = rec_dict.lookup(60)
    print(f"Entry for 60: {entry}")
    print(f"Children: {entry.children}")
    print(f"Parentheses encoding: {entry.parentheses}")
