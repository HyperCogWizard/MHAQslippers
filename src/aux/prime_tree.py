import math
from typing import List, Optional

class PrimeTreeNode:
    def __init__(self, value: int, children: Optional[List['PrimeTreeNode']] = None):
        self.value = value
        self.children = children or []

    def to_parentheses(self) -> str:
        if not self.children:
            return "()"
        return "(" + "".join(child.to_parentheses() for child in self.children) + ")"

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def prime_factors(n: int) -> List[int]:
    factors = []
    # Handle 2 separately
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    # Odd factors
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors.append(f)
            n //= f
        f += 2
    if n > 1:
        factors.append(n)
    return factors

def build_prime_tree(n: int) -> PrimeTreeNode:
    if is_prime(n):
        return PrimeTreeNode(n)
    factors = prime_factors(n)
    # For each factor, build subtree
    children = [build_prime_tree(f) for f in factors]
    return PrimeTreeNode(n, children)

# Example usage:
if __name__ == "__main__":
    n = 60
    tree = build_prime_tree(n)
    print(f"Prime factorization tree for {n}:")
    print(tree.to_parentheses())
