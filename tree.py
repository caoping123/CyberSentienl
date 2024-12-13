import numpy as np

cnt = 0;
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        global cnt
        cnt += 1

    def generate_child(self):
        self.left = Node(self.value+1)
        self.right = Node(self.value+2)


class Tree:
    def __init__(self, n):
        self.root = Node(1)
        self.height = np.log2(n) + 1
        self.leaf = []
        self.init_tree()

    def init_tree(self):
        def generate(root: Node, level: int):
            root.generate_child()
            if level + 1 == self.height:
                self.leaf.append(root.left)
                self.leaf.append(root.right)
                return
            generate(root.left, level + 1)
            generate(root.right, level + 1)
        generate(self.root, 1)


if __name__ == '__main__':
    tree = Tree(8)
