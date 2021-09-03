#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#HashMap 哈希表：
#hashmap：用Array + LinkedList(chianing)实现的能在平均O(1)时间内快速增删查的数据结构
#表内存储的数据结构需要实现equals()和hashcode()
        
                


# In[36]:


#二叉树：Binary Tree：A binary tree is a tree data structure in which each node has 
# at most two children, which are referred to as the left child and the right child

#Complete Binary Tree:
#In a complete binary tree every level, except possibly the last,
# is completely filled, and all nodes in the last level are as far left
# as possible 

#Full Binary Tree:
# A full binary tree(sometimes referred to as a proper or plane binary tree)
# is a tree in which every node has either 0 or 2 children. 




# In[57]:


#Create a Binary Tree
# Step 1: Create a Node class:

class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None 







        
        


# In[58]:


# Step 2: Create a BinaryTree class based on the class Node:
class BinaryTree(object):
    def _init_(self, value):
        self.root = Node(root)
        
# Step 3: Create a BinaryTree:
#        1
    #   / \
    #  2   3 
#     / \
#    4   5 
    

tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)


# In[95]:



#Binary Tree Traversal 二项树遍历：

# Tree Traversal: Process of visiting (checking and/or updating)
# each node in a tree data structure, exactly once
# Unlike linked lists, one-dimensional arrays etc., which are canonically
# traversed in linear order, trees may be travesed in multiple ways. they
# may be traversed in depth-first （深度优先遍历）
# or breadth-first order(宽度优先遍历)
# 3 common ways to traverse them in depth-first order:
# in-order
# pre-order
# post-order 

#pre-order:
# 1. check if the current node is empty/null
# 2. Display the data part of the root(or current node)
# 3. Traverse the left subtree by recursively calling the pre-order function
# 4. Traverse the right subtree by recursively calling the pre-order function

#in-order:
# 1. Check if the current node is empty/null
# 2. Traverse the left subtree by recursively calling in-order function 
# 3. Diaplay the data part of the root (or current node)
# 4. Traverse the right subtree by recursively calling in-order function


# Level-order traversal (层级遍历):
# level-order traversal: from the left to the right and the top to the bottom
# traversing every level of the tree 



# Calculate Height of Binary Tree:
# Height of Binary Tree: The ehight of a tree is the height of its root node.
# Height of Node:
# The height of a node is the number of edges on the longest path between
# that node and a leaf. 

# Create a new Queue class for level-order travesal:

class Queue(object):
    def __init__(self):
        self.items = []
    
    def enqueue(self,item):
        self.items.insert(0,item)# insert()函数不会覆盖原有值
        
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop()
    def is_empty(self):
        return len(self.items)==0
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1].value
        
    def __len__(self): #重载len()函数，__len__()为系统保留命名方式
        return self.size()
    
    def size(self):
        return len(self.items)
        
        
        

# Create a BinaryTree class based on the class Node:
class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)
    
    
    def print_tree(self, traversal_type):
        if traversal_type=="preorder":
            return self.preorder_print(self.root,"")
        if traversal_type =="inorder":
            return self.inorder_print(self.root, "")
        if traversal_type =="postorder":
            return self.postorder_print(self.root,"")
        
        if traversal_type =="levelorder":
            return self.levelorder_print(self.root)
        if traversal_type == "reverselevelorder":
            return self.reverselevelorder_print(self.root)
            
    def preorder_print(self, start, traversal):#preorder traversal function
        """Root -> Left -> Right"""
        if start:
            traversal += (str(start.value)+"-")
            traversal = self.preorder_print(start.left, traversal)
            traversal = self.preorder_print(start.right, traversal)
        return traversal
    def inorder_print(self, start, traversal):#inorder traversal function
        """left->Root->Right"""
        if start:
            traversal = self.inorder_print(start.left,traversal)
            traversal += (str(start.value)+"-")
            traversal = self.inorder_print(start.right, traversal)
            
        return traversal
    
    def postorder_print(self, start, traversal):
        """Left->Right->Root"""
        if start:
            traversal = self.postorder_print(start.left,traversal)
            traversal = self.postorder_print(start.right, traversal)
            traversal += (str(start.value)+"-")
            
        return traversal
            
    def levelorder_print(self, start):
        if start is None:
            return False
        queue = Queue()
        queue.enqueue(start)
        traversal = ""
        
        while len(queue)>0:
            traversal += str(queue.peek())+"-"
            node = queue.dequeue()
            if node.left:
                queue.enqueue(node.left)
                
            if node.right:
                queue.enqueue(node.right)
                
        return traversal
    
    def reverselevelorder_print(self, start):
        if start is None:
            return False
        
        queue = Queue()
        queue.enqueue(start)
        traversal = ""
        
        while len(queue)>0:
            traversal += "-" + str(queue.peek())
            node = queue.dequeue()
            if node.right:
                queue.enqueue(node.right)
                
            if node.left:
                queue.enqueue(node.left)
            
        return traversal[::-1]
        
    def height(self, node):
        if node is None:
            return -1
        left_height = self.height(node.left)
        right_height = self.height(node.right)
        
        return 1+max(left_height,right_height)
    
    def size_tree(self, start):
        if start is None:
             return 0
        queue = Queue()
        queue.enqueue(start)
        count = 0
        
        while len(queue) > 0:
            count += 1
            node = queue.dequeue()
            if node.left:
                queue.enqueue(node.left)
            
            if node.right:
                queue.enqueue(node.right)
        
        return count
    
    
    
    
            
        
        
        
        






# In[96]:


tree = BinaryTree(1)
tree.root.left = Node(2)
tree.root.right = Node(3)
tree.root.left.left = Node(4)
tree.root.left.right = Node(5)


print(tree.print_tree("preorder"))
print(tree.print_tree("inorder"))
print(tree.print_tree("postorder"))
print(tree.print_tree("levelorder"))
print(tree.print_tree("reverselevelorder"))
print(tree.height(tree.root))
print(tree.size_tree(tree.root))


# In[51]:


# Binary Search Tree:
# Binary search trees differ from binary trees in that the entries are ordered 

# Insertion:
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        
class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, data):
        if self.root is None:
            self.root = Node(data)
            
        else:
            self._insert(data, self.root)
    
    def _insert(self, data, cur_node):
        if data < cur_node.data:
            if cur_node.left is None:
                cur_node.left = Node(data)
                
            else:
                self._insert(data, cur_node.left)
        
        elif data > cur_node.data:
            if cur_node.right is None:
                cur_node.right = Node(data)
            
            else:
                self._insert(data, cur_node.right)
        
        else:
            print("The value exists in the tree.")
            
    
    def find(self, data):
        if self.root is None:
            print("The tree is empty.")
            
        else:
            is_find = self._find(data, self.root)
            if is_find:
                return True
            
            else:
                return False
            
        return None
            
    def _find(self, data, cur_node):
        if data == cur_node.data:
            return True
        elif data<cur_node.data:
            if cur_node.left is None:
                return False
            else:
                return self._find(data, cur_node.left)
                
        else:
            if cur_node.right is None:
                return False
            else:
                return self._find(data,cur_node.right)
         
            
        
        
        
        
                
            
                
        


# In[52]:


bst = BST()

bst.insert(4)
bst.insert(2)
bst.insert(8)
bst.insert(5)
bst.insert(10)


bst.find(11)


# In[105]:



#Linear search:
def linear_search(data, target):
    for i in range(len(data)):
        if data[i] == target:
            return True
        
    return False
        




# In[108]:


# Iterative Binary Search:
def binary_search_iterative(data, target):
    low = 0
    high = len(data) - 1

    while low <= high: 
        
        mid =(low+high)//2
        if target == data[mid]:
            return True
        elif target < data[mid]:
            high = mid - 1
        else:
            low = mid +1
    
    return False
   
        
data = [3,5,7,13,18,19,25,44,51,60,73,74,78,82]
print(binary_search_iterative(data,73))


# In[103]:


# Recursive Binary Search:
def binary_search_recursive(data, target, low, high):
    if low > high:
        return False
    mid = (low+high)//2
    if target == data[mid]:
        return True
    elif target < data[mid]:
        return binary_search_recursive(data, target, low, mid-1)
    else: 
        return binary_search_recursive(data, target, mid + 1, high)

data = [3,5,7,13,18,19,25,44,51,60,73,74,78,82]
print(binary_search_recursive(data,73,0,len(data)-1))


# In[110]:


# Binary Search in Python: Find Closest Number:

def find_closest_nu(data, target):
    min_diff = float("inf")
    low = 0
    high = len(data)-1
    closest_num = None
    
    
    if len(data) == 0:
        return None
    if len(data) ==1:
        return data[0]
    while low <= high:
        mid = (low + high)//2
        #Ensure we don't read beyond the bound of the list
        # And obtain the left and right difference values
        if mid +1 < len(data):
            min_diff_right = abs(data[mid+1]-target)
        if mid - 1 >-1:
            min_diff_left = abs(data[mid-1]-target)
        #Check if the absolute value between left and right
        # elements are smaller than any seen prior
        if min_diff_left < min_diff:
            min_diff = min_diff_left
            closest_num = data[mid-1]
        if min_diff_right < min_diff:
            min_diff = min_diff_right
            closest_num = data[mid+1]
        #Move the mid-point accordingly as is done
        #via binary search
        if data[mid]<target:
            low = mid+1
        elif data[mid]>target:
            high = mid - 1 
            
        else:
            return data[mid]
    
    return closest_num 
                
    

data = [3,5,7,13,18,19,25,44,51,60,73,74,78,82]   
print(find_closest_nu(data, 65))
        
        
    
    


# In[126]:


# Find fixed point:
# A fixed point in an array "data" is an index "i" such that data[i] 
# is equal to "i"

def find_fixed_point(data):
    low = 0
    high = len(data)-1
    while low <= high:
        mid = (low+high)//2
        if mid == data[mid]:
            return mid
        elif mid > data[mid]:
            low = mid+1
        else:
            high = mid -1
    return None
data = [-10,1,3,4,5]  


print(find_fixed_point(data))


# In[147]:


# find Bitonic Peak: example of bitonic peak: [1,2,3,4,5,4,3,2,1]
# find the larget element in Bitonic sequence:

def find_highest_number(data):
    low = 0
    high = len(data)-1
    
    if len(data) < 3:
        return None
    
    while low <= high:
        mid = (high+low)//2
        
       
        mid_left = data[mid-1] if mid - 1 > 0 else float("-inf")
        

        mid_right = data[mid+1] if mid + 1 <len(data) else float("inf")
        
        if mid_left < data[mid] and mid_right > data[mid]:
            low = mid + 1
        
        elif mid_left> data[mid] and mid_right < data[mid]:
            high = mid -1
        
        elif mid_left < data[mid] and mid_right < data[mid]:
            return data[mid]
        
data = [1,2,3,5,5,5,3,2,1]
print(find_highest_number(data))


# In[7]:


# Binary Search: Find First Entry in List with Duplicates:

def find_first_entry(data,target):
    if len(data) == 0:
        return False
    
    low = 0
    high = len(data) - 1
    
    while low <= high:
        mid = (high+low)//2
        
        if target < data[mid]:
            high = mid - 1
        
        if target > data[mid]:
            low = mid + 1
        
    
        if target == data[mid]:
            if target == data[mid-1] and mid-1 >0:
                high = mid - 1
                
            else:
                return mid
    
data = [-14,-10,2,108,108,243,285,285,285,401]
target = 0

print(find_first_entry(data,target))
    


    
    


# In[139]:


# Binary Search: Integer Square Root:
# find the largest integer square smaller than a given number:
def find_square_root(k):

    low = 1 
    high = k
    if k <=1:
        return False
    
    while low<=high:
        mid = (low+high)//2
        if mid*mid > k:
            high = mid - 1
        if mid* mid < k:
            low = mid + 1
        
        if mid*mid == k:
            return mid-1
    
    return mid

print(find_square_root(100))
        


# In[142]:


#Binary Search: Cyclically Shifted Array:
def find_smallest_num(data):
    low = 0
    high =len(data)-1
    

    
    while low<=high:
        mid = (low+high)//2
        
        if data[mid] > data[high]:
            low = mid+1
        elif data[mid]<=data[high]:
            high = mid - 1
            
    return low
            

data = [4,5,6,7,1,2,3]

print(find_smallest_num(data))
        
        
        


# In[162]:


# Greedy Algorithms: Optimal Task Assignment:
def optimal_task(data):
    if len(data) <=1:
        return False
    if len(data) == 2:
        return data[0]+data[1]
    
    low = 0
    high = len(data) - 1
    maxi = 0
    data = sorted(data)
    while low +1< high-1:
        a = data[low] + data[high]
        b = data[low +1] + data[high-1]
       
        maxi = max(a,b,maxi)

        low += 1
        high -= 1
        
    return maxi

data = [6,3,2,7,5,2,5]
print(optimal_task(data))

data = sorted(data)

for i in range(len(data)//2):
    print(data[i],data[~i])


# In[52]:


# Singly Linked List:
class Node(object):
    def __init__(self, value):
        self.value = value
        self.next = None
    def print_list(self):
        
        while self:
            print(self.value)
            self = self.next
        
class LinkedList(object):
    def __init__(self):
        self.head = None
    
    def print_list(self):
        cur_node = self.head
        while cur_node:
            print(cur_node.value)
            cur_node = cur_node.next
    
    def append(self, value):
        new_node = Node(value)
        
        if self.head is None:
            self.head = new_node
        else:
            needle = self.head
            while needle.next:
                needle = needle.next
            
            needle.next  = new_node
            
    def prepend(self, value):
        new_node = Node(value)
        
        if self.head is None:
            self.head = new_node
        else:
            new_node.next = self.head
            self.head = new_node
            
    def insert_after_node(self, prev_node, value):
        new_node = Node(value)
        
        if prev_node.next == None:
            prev_node.next = new_node
            
        else:
            new_node.next = prev_node.next
            prev_node.next = new_node
            
    def print_helper(self, node, name):
        if node is None:
            print(name + ": None")
        else:
            print(name + ":"+node.value)
            
    def  reverse_iterative(self):
        
        prev = None
        cur_node = self.head
        
        while cur_node:
            next_node = cur_node.next #先存储下一个节点地址
            
            cur_node.next = prev # 调换存储顺序
            self.print_helper(prev, "PREV")
            self.print_helper(cur_node,"CUR")
            self.print_helper(next_node,"NEX")

            prev = cur_node # 移动双指针
            cur_node = next_node # 将当前指针指向下一个节点地址
           
            
        
        self.head = prev
    
    def reverse_recursive(self, prev, cur):
        if cur is None:
            return 
        nex = cur.next
        cur.next = prev
        prev = cur
        cur = nex
        self.head=prev
        return self.reverse_recursive(prev, cur)
    
    def lucid_reverse_recursive(self):
        
        def _lucid_reverse_recursive(cur, prev):
            if not cur:
                return prev
            
            nex = cur.next
            cur.next = prev
            prev = cur
            cur = nex
            return _lucid_reverse_recursive(cur,prev)
        
        self.head = _lucid_reverse_recursive(cur = self.head,prev=None)
        
        
    
    def delete_node(self, key):
        cur_node = self.head
        
        if cur_node and cur_node.value == key:
            self.head = cur_node.next
            cur_node = None
            
            return 
            
        prev_node = self.head
        next_node = prev_node.next
        
        while next_node.next:
            if next_node.value == key:
                prev_node.next = next_node.next
                next_node = None
                return
                
            next_node = next_node.next
            prev_node = prev_node.next
                
                
        if next_node.next ==None:
            prev_node.next = None
            next_node = None
            
            return 
    def merge_lists(self, list_2):
        if list_2 is None:
            return 
        new_list = LinkedList()
        new_head = new_list
        p=self.head
        q=list_2.head
        
        while p and q:
            if p.value <= q.value:
                new_list.next = p
                p = p.next
                
            else:
                new_list.next = q
                q = q.next
                
            new_list = new_list.next
        if p :
            new_list.next = p
        
        else:
            new_list.next = q
        
        new_head = new_head.next
        return new_head
    
    
    def delete_node_atpos(self, pos):
        if pos == 0:
            self.delete_node(self.head.value)
            return
        
        prev = self.head
        
        for i in range(pos):
            prev = prev.next
            
        if prev == None:
            return
        self.delete_node(prev.value)
        return
   # remove duplicates will require Hash Table and Dictionary 
    def remove_duplicates(self):
        cur = self.head
        prev = None
        
        dup_values = dict()
        
        while cur:
            print(dup_values)
            if cur.value in dup_values:
                prev.next = cur.next
                cur = None
                cur = prev.next
            else:
                dup_values[cur.value] = 1
                prev = cur
                
                cur = prev.next
                
            
    def nth_to_last(self, n):
        
        
        ned_1 = self.head
        ned_2 = ned_1
        
        for i in range(n):
            if ned_1:
                ned_1 = ned_1.next
            else:
                return False
        while ned_1:
            ned_1 = ned_1.next
            ned_2 = ned_2.next
        
        return ned_2.value
            
    
    def count_occurences_iterative(self, data):
        cur = self.head
        
        count_dic = dict()
        count_dic[data]=0
        
        while cur:
            if cur.value == data:
                count_dic[cur.value] += 1
            
            cur = cur.next
            
        return count_dic[data]
            
        
    def count_occurances_recursive(self, node, data):
        count = 0
        curr = node
        
        if curr is None:
            return 0
        
        if curr:
            if node.value == data:
                count = 1  
                
            
            else:
                count=0
        
        return count + self.count_occurances_recursive(node.next,data)
            
    def sum_two_lists(self, llist_2):
        p_1 = self.head
        p_2 = llist_2.head
        llist = Node(0)
        h1 = llist
        while p_1 or p_2:
            if p_1 and p_2:
                k = h1.value
                h1.value = (p_1.value + p_2.value + h1.value)%10
            
                if p_1.value + p_2.value + k >=10:
                    h1.next = Node(1)
                    
            
            elif p_1 is None and p_2:
                p_1.value = 0
                k = h1.value
                h1.value = (p_1.value + p_2.value + h1.value)%10
            
                if p_1.value + p_2.value + k >=10:
                    h1.next = Node(1) 
            else:
                p_2.value = 0
                k = h1.value
                h1.value = (p_1.value + p_2.value + h1.value)%10
            
                if p_1.value + p_2.value + k >=10:
                    h1.next = Node(1) 
            h1 = h1.next
            p_1 = p_1.next
            p_2 = p_2.next
            
        return llist
             
    
    def rotate(self, n):
        cur = self.head
        i =0
        while cur and i < n-1:
            cur = cur.next
            
            i += 1
        
        if cur is None:
            return 
        
        q = cur
        while q.next:
            q = q.next
            
        
        q.next = self.head
        self.head = cur.next
        cur.next = None
        
    def is_palindrome(self):
        
        # method 1:
        #cur = self.head
        #strr = ""
        #while cur:
        #    strr += cur.value
        #    cur = cur.next
            
        #if strr == strr[::-1]:
        #   return True
            
        #else: return False
        
        #method 2:
        
        #p = self.head
        
        #s = []
        
        #while p :
        #    s.append(p.value)
        #    p = p.next
        #p = self.head    
        #while p:
        #    data = s.pop()
        #    if p.value != data:
        #        return False
        #    p = p.next
            
        #return True
        
        #method 3:
        p = self.head
        q = self.head
        prev = []
        
        i=0
        while q:
            prev.append(q)
            q = q.next
            i = i+1
            
        
        
        while p and i >=0:
            if p.value != prev[i-1].value:
                return False
            
            i -= 1
            p = p.next
            
        return True
        
    def move_head_to_tail(self):
        p = self.head
        
        prev = None
        
        while p.next:
            prev = p 
            p = p.next
        
        p.next = self.head
        self.head = p
        prev.next = None
        
        
    
            
    def length_iterative(self):
        cur_node = self.head
        count = 0
        while cur_node:
            count += 1
            cur_node = cur_node.next
            
        return count
            
    def length_recursive(self, cur_node):
        count = 0
        
        if cur_node:
            count += 1 
            count = count + self.length_recursive(cur_node.next)
            
        return count
    
    def swap_node(self,key_1, key_2):
        
        if key_1 == key_2:
            return self.head
        
        prev_1 = None
        curr_1 = self.head
        while curr_1 and curr_1.value != key_1:
            prev_1 = curr_1
            curr_1 = curr_1.next
            
        
        
        
        prev_2 = None
        curr_2 = self.head    
        
        while curr_2 and curr_2.value !=key_2:
            prev_2 = curr_2
            curr_2 = curr_2.next       
            
 #not quite understand 
        
        if not curr_1 or not curr_2:
            return 
        
        if prev_1:
            
            prev_1.next = curr_2
        else:
            self.head = curr_2
            
        if prev_2:
            prev_2.next = curr_1
        else:
            self.head = curr_1
            
        curr_1.next, curr_2.next = curr_2.next, curr_1.next #a, b = b, a
        # 等号右边返回一个元组[curr_2.next所指向的存储域, curr_1.next所指向的存储域]
        # 之后等号左边的两个指针分别指向元组中第一个存储域和第二个存储域
        # A ---> B ---> C ---> D ---> E
        
    
    
#LucidProgramming:
#    def delete_node(self,key):
#        cur_node = self.head
#        
#        if cur_node and cur_node.value == key:
#            self.head = cur_node.next
#            cur_node = None
#            
#            return
#        
#        prev = None
#        
#        while cur_node and cur_node.data != key:
#            prev = cur_node
#            cur_node = cur_node.next
#            
#        
#        if cur_node == None:
#            return
#        
#        prev.next = cur_node.next
#        cur_node = None
     
        
        
        
         
       
            
            
        
            
        
            
    

llist = LinkedList()
llist.append(1)
llist.append(2)
llist.append(5)
llist.append(5)
llist.append(2)
llist.append(8)
llist.append(1)
llist.append(10)
llist.append(8)

llist.rotate(4)
llist.print_list()


print(llist.nth_to_last(4))
print(llist.count_occurences_iterative(2))
print(llist.count_occurances_recursive(llist.head, 2))

llist_2 = LinkedList()
llist_2.append(1)
llist_2.append(1)
llist_2.append(1)



llist_3 = LinkedList()
llist_3.append(9)
llist_3.append(9)
llist_3.append(9)

new_list = LinkedList()
new_list = llist_2.sum_two_lists(llist_3)
print(365+248)
new_list.print_list()





# In[98]:


class CircularLinkedList:
    def __init__(self):
        self.head = None
    
    def prepend(self,value):
        node = Node(value)
        cur = self.head
        if self.head is None:
            self.head = node
            self.head.next = self.head
        else:
            while cur.next != self.head:
                cur = cur.next
            
            node.next = self.head
            self.head = node
            cur.next = self.head
            
    def append(self, value):
        node = Node(value)
        if self.head is None:
            self.head = node
            self.head.next = self.head
        else:
            needle = self.head
            while needle.next != self.head:
                needle = needle.next
                
            needle.next = node
            node.next = self.head
            
    def remove(self, key):
        if self.head.next == self.head:
            if self.head.value == key:
                return None
            else:
                return
        cur = self.head
        while cur:
            
            if cur.next.value == key:
                nxt = cur.next
                cur.next = nxt.next
                nxt = None
            else:
                cur = cur.next
            if cur.next == self.head:
                if cur.next.value == key:
                    nxt = cur.next
                    cur.next = nxt.next
                    self.head = nxt.next
                    nxt = None
                    
                    break
                else:
                    break
                    
    def lucid_remove(self, key):
        if self.head.data == key:
            cur = self.head
            while cur.next != self.head:
                cur = cur.next
            
            cur.next = self.head.next
            self.head = self.head.next
            
        else:
            cur = self.head
            prev = None
            while cur.next != self.head:
                prev = cur
                cur = cur.next
                if cur.data == key:
                    prev.next = cur.next
                    cur = cur.next
    
    def __len__(self):
        cur = self.head
        if self.head is None:
            return 0
        length = 1
        while cur.next != self.head:
            cur = cur.next
            length += 1
        
        return length
            
    
    
    def split_list(self):
        size = len(self)
        print(size)
        
        if size ==0:
            return None
        if size ==1:
            return self.head
        mid = size//2
        count = 0
        
        prev = None
        cur = self.head
        
        while cur and count < mid:
            count += 1
            prev = cur
            cur = cur.next
        
        prev.next = self.head
        
        split_list = CircularLinkedList()
        while cur.next != self.head:
            split_list.append(cur.value)
            cur = cur.next
            
        split_list.append(cur.value)
        
        self.print_list()
        print("\n")
        split_list.print_list()
        
    
        
        
        
    def node_remove(self, node):
        if self.head == node:
            cur = self.head
            while cur.next != self.head:
                cur = cur.next
            
            cur.next = self.head.next
            self.head = self.head.next
            
        else:
            cur = self.head
            prev = None
            while cur.next != self.head:
                prev = cur
                cur = cur.next
                if cur  == node:
                    prev.next = cur.next
                    cur = cur.next  
                    
    def josephus_circle(self, step):
        if step ==0:
            return 
        cur = self.head
        
        while len(self)>1:
            count = 1
            while count < step:
                cur = cur.next
                count += 1
            print("Remove:", cur.value)    
            self.node_remove(cur)
            cur = cur.next # node_remove 函数并没有将cur节点给删除，只是将指针移动位置
            
            
    def is_circular_linked_list(self, input_list):
        cur = input_list.head
        while cur:
            cur = cur.next
            if cur == input_list.head:
                return True
            
        if cur == None:
            return False 
        
            
            
        
            
        
    def print_list(self):
        cur = self.head
        
        while cur:
            print(cur.value)
            cur = cur.next
            if cur == self.head:
                break
                
        
    
cllist = CircularLinkedList()
cllist.append("A")
cllist.append("B")
cllist.append("C")
cllist.append("D")
#cllist.prepend("E")
#cllist.print_list()
cllist.josephus_circle(1)
cllist.print_list()

llist = LinkedList()
llist.append(1)
llist.append(2)
llist.append(5)
llist.append(5)
llist.append(2)

cllist.is_circular_linked_list(llist)


# In[192]:


# Doubly Linked List:

class DoublyNode:
    def __init__(self, value):
        self.value = value 
        self.next = None
        self.prev = None
        
class DoublyLinkedList(object):
    def __init__(self):
        self.head = None
        
        
    def append(self, value):
        node = DoublyNode(value)
        
        if self.head is None:
            self.head = node
            node.prev = None
        else:
            cur = self.head
            while cur.next :
                cur = cur.next
            cur.next = node
            node.prev = cur
            node.next = None
            
    def prepend(self, value):
        node = DoublyNode(value)
        
        if self.head is None:
            self.head = node
            node.prev = None
            
        else:
            self.head.prev = node
            node.next = self.head
            self.head = node
            node.prev = None
                
    
    def print_list(self):
        cur = self.head
        while cur:
            print(cur.value)
            cur = cur.next
            
    def add_after_node(self, value, value_after):
        cur = self.head
        while cur.next:
            if cur.value != value:
                cur = cur.next
                if cur.next is None and cur.value ==value:
                    node = DoublyNode(value_after)
                    cur.next = node
                    node.prev = cur
                    node.next = None        
                    return   
            else:        
                nxt = cur.next
                node = DoublyNode(value_after)
                cur.next = node
                node.next = nxt
                nxt.prev = node
                node.prev = cur           
                return       
        return False
             
            
            
    def add_before_node(self, value, value_before):
        cur = self.head 
        
        while cur:
            if self.head.value == value:
                node = DoublyNode(value_before)
                node.next = cur
                cur.prev = node
                node.prev = None
                self.head = node
                return
                                  
            if cur.value != value:
                cur = cur.next
            
            else:
                node = DoublyNode(value_before)
                prev = cur.prev
                
                prev.next = node
                node.next = cur
                cur.prev = node
                node.prev = prev
                return
            
        return False
                
    def delete(self, key):
        cur = self.head
        if cur.next is None:
            if cur.value == key:    
                self.head = None
                return
            else:
                return 
        while cur:
            if cur.value == key:
                if cur.next is None:
                    cur.prev.next = None
                    cur.prev = None
                    cur = None
                    return
                else:
                    if cur == self.head:
                        self.head = cur.next
                        cur.next.prev = None
                        cur.next = None
                        cur = None
                        return
                    else:
                        cur.prev.next = cur.next
                        cur.next.prev = cur.prev
                        cur = None
                        return
                    
            cur = cur.next
    # **一个节点，只能控制它自身的next和prev指针**    
    def reverse(self):
        tmp = None
        cur = self.head
        while cur:
            tmp = cur.prev
            cur.prev = cur.next
            cur.next = tmp
            cur = cur.prev
            
        
        if tmp :
            self.head = tmp.prev
            
        else:
            return False
            
    def remove_duplicates(self):
        prev = None
        cur = self.head
        dic = dict()
        
        while cur:
            
            if cur.value in dic:
                if cur.next is None:
                    
                    cur.prev.next = None
                    
                    return 
                else:
                    tmp = cur
                    cur.prev.next = cur.next
                    cur.next.prev = cur.prev
            
                    
                    cur = cur.next
                    tmp = None
                    
                
            else:
                dic[cur.value] = 1
                cur = cur.next
                
    def pairs_with_sum(self, summ):
        cur = self.head
        arr = []
        if self.head.next is None:
            if self.head.value == summ:
                return self.head.value
            else:
                return False
        
        while cur:
            nxt = cur.next 
            while nxt:
                if nxt.value +cur.value == summ:
                    arr.append([nxt.value, cur.value])
                    
                nxt=nxt.next
                
            cur = cur.next
        print(arr)
            
            
                
                
            
        
            
            
            
        
        


dllist = DoublyLinkedList()
dllist.append(1)
dllist.append(1)
dllist.append(3)
dllist.append(11)
dllist.append(3)
dllist.append(2)
dllist.append(3)
dllist.append(2)
dllist.append(3)
dllist.append(11)
dllist.append(11)
dllist.prepend(4)
dllist.print_list()
dllist.remove_duplicates()
dllist.pairs_with_sum(5)





    
    


# In[218]:



def unique(strr):
    leng = len(strr)
    
    dic = dict()
    i = 0
    while i<leng:
        if strr[i] in dic:
            print(strr[i])
            return False
    
        else:
            dic[strr[i]] = 1
            print(strr[i])
            i +=1
    return True

strr = input("string: ")

print(unique(strr))

#method 2: set(string) 可以自动将重复字符剔除，只保留下出现的每一个字符一次，并组成一个集合

def is_unique(s):
    return len(s) ==len(set(s))

s1="unique"

print(is_unique(s1))


# In[221]:


#method 2: set(string) 可以自动将重复字符剔除，只保留下出现的每一个字符一次，并组成一个集合

def is_unique(s):
    return len(s) ==len(set(s))

s1="unique"
s2 = "bear"

print(is_unique(s2))


# In[210]:


#look-and-say: In mathematics, the look-and-say sequence is the sequence of integers beginning as follows:

#1, 11, 21, 1211, 111221, 312211, 13112221, 1113213211, ... /
def next_number(s):
    result = []
    i = 0
    while i< len(s):
        count =1 # 每次循环后count置1，重新计数
        while i+1 < len(s) and s[i]==s[i+1]:#统计有多少个相同数字
            i += 1
            count += 1
        result.append(str(count)+s[i])
        i += 1
            
    return ''.join(result)

s="1"
n = 5


for i in range(n):
    s = next_number(s)
    print(s)
        


# In[32]:


#Palindrome: 回文字符串
# i is a char, i.isalpha() 检测i是否是字母, i.isalnum(), 判断i是否为字母a-z, 0-9。 string.lower()字母取小写
#　reverse of a string: string[::-1]
s = "Was it a cat I saw?"
#method 1: using extra space proportional to size of string "s"
s = ''.join([i for i in s if i.isalpha()]).replace(' ','').lower()
print(s)
print(s==s[::-1])

#method 2: linear Time solution:
def is_palindrome(s):
    i = 0
    j = len(s)-1
    while i < j:
        while not s[i].isalnum() and i<j:
            i += 1
        while not s[j].isalnum() and i<j:
            j -= 1
        if s[i].lower()!=s[j].lower():
            return False 
        
        i += 1
        j -= 1
    if s[i]==s[j]:
        return True
    
print(is_palindrome(s))
        
    
            


# In[31]:


# Array Advance Game: Given an arry of non-negative integers, each number represents the maximjum you can 
#advance in the array.
#Question: Is it possible to advance from the start of the array to the last element? 

#solution: 每个元素都遍历，找到每个元素可以走的最远距离，即元素下标+元素值，得到最远路径，
#如果该路径值大于数组长度，即可以到达最后一个，否则不可以.

def furthest_path(a):
    i = 0
    furthest = 0 
    while i <= furthest and furthest<len(a)-1:
        if a[0]==0:
            return False
        
        furthest = max(furthest, a[i]+i)
        print(furthest)
        i +=1
        
    return furthest >=len(a)-1

a1 = [3,3,1,0,2,0,1]

print(furthest_path(a1))


# In[ ]:


#Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        pos = 0 
        i = 0
        while i < len(nums):
            if target <= nums[i]:
                position = i
                return position
            else:
                i = i+1 
        if i >= len(nums):
            return i

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/search-insert-position
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# In[ ]:



#The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

#countAndSay(1) = "1"
#countAndSay(n) is the way you would "say" the digit string from countAndSay(n-1), which is then converted into a different digit string.
#To determine how you "say" a digit string, split it into the minimal number of groups so that each group is a contiguous section all of the same character. Then for each group, say the number of characters, then say the character. To convert the saying into a digit string, replace the counts with a number and concatenate every saying.

class Solution:
    def countAndSay(self, n: int) -> str:
        result = ""
        k = 0
        s = "1"
        if n ==1:
            return "1"
        while k < n-1:
            i = 0
            while i < len(s):
                count = 1
                while i+1<len(s) and s[i] == s[i+1]:
                    count += 1
                    i +=1

                result = result+ str(count) + s[i]

                i = i+1
                 
            result_tmp = result
            k  = k + 1
            s = result_tmp
            result = ""

        return result_tmp
    
#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/count-and-say
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# In[ ]:


#Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        result = -100000000
        maxi = 0
        if len(nums) ==1:
            return nums[0]
        
        for i in range(len(nums)):
            maxi = max(maxi + nums[i], nums[i]) #从第一个元素开始不断的累加，直到当加到下一个元素使得之前累加结果的总值小于当前元素，再重新开始累加
            result = max(maxi, result)
        return result


#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/maximum-subarray
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# In[ ]:


#Given a string s consists of some words separated by spaces, return the length of the last word in the string. If the last word does not exist, return 0.

#A word is a maximal substring consisting of non-space characters only. (length of last word)
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if len(s) == 0:
            return 0
        
        k = len(s)-1
        count = 0
        while s[k] == " " and k >= 0:
            k -= 1
        while s[k] != " " and k >= 0:
            count += 1
            k -= 1

        return count 
    
#method 2:
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return (s.strip().split(" "))[-1]



#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/length-of-last-word
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# In[ ]:


#Given a non-empty array of decimal digits representing a non-negative integer, increment one to the integer.

#The digits are stored such that the most significant digit is at the head of the list, and each element in the array contains a single digit.

#You may assume the integer does not contain any leading zero, except the number 0 itself.


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        i = len(digits)-1
        s = set(digits)
        j = 0
        if s == {9}:
            digit = [1]
            while j < len(digits):
                digit.append(0)
                j += 1
                print(digit)
            return digit

        else:
            while i-1 >= 0 and digits[i] + 1 == 10:
                digits[i] = 0
                i -= 1

            digits[i] += 1

        return digits

# method 2:
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits)-1,-1,-1):
            digits[i] += 1
            digits[i] = digits[i] % 10
            if digits[i] != 0: return digits
            
        dig = [0]*(len(digits)+1)
        dig[0] = 1
        return dig
#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/plus-one
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# In[ ]:


#Given two binary strings a and b, return their sum as a binary string.
#基本思路：从每个字符串的尾部进行取值， 尾部值相加，逢二逢三进一，设置一个tmp全局变量，记录上一位结果是否进一，若进一则tmp=1 并在计算下一位时算入。
#最后将得到的字符串序列反转

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        len_a = len(a)
        len_b = len(b)
        i = len_a -1
        j = len_b -1
        result = ""
        tmp = 0
        while i >= 0 and j >= 0:
            
            if int(a[i]) + int(b[j])+ tmp == 0:
                result += "0"
                i -= 1
                j -= 1
                tmp = 0

            elif int(a[i]) + int(b[j])+ tmp == 1:
                result += "1"
                i -= 1
                j -= 1
                tmp = 0 
            elif int(a[i]) + int(b[j]) + tmp == 2 :
                result += "0"
                tmp = 1
                i -= 1
                j -= 1
            else: 
                result += "1"
                tmp =1
                i -= 1
                j -= 1
        if j < 0 and i >=0:
            while i>= 0:
                if int(a[i]) +tmp == 0:
                    result += "0"
                    i -= 1
                    tmp = 0 
                elif int(a[i]) + tmp == 1:
                    result += "1"
                    i -= 1
                    tmp = 0
                elif int(a[i]) + tmp == 2 :
                    result += "0"
                    tmp = 1
                    i -= 1
                else: 
                    result += "1"
                    tmp =1
                    i -= 1
            if tmp == 1:
                result += "1"
            return result[::-1]
        if j >= 0 and i <0:
            while j>= 0:
                if int(b[j]) +tmp == 0:
                    result += "0"
                    j -= 1
                    tmp = 0
                elif int(b[j]) + tmp == 1:
                    result += "1"
                    j -= 1
                    tmp = 0

                elif int(b[j]) + tmp == 2 :
                    result += "0"
                    tmp = 1
                    j -= 1
                else: 
                    result += "1"
                    tmp =1
                    j -= 1
            if tmp == 1:
                result += "1"
            return result[::-1]
        
        if i < 0 and j < 0:
            if tmp ==1:
                result += "1"
                return result[::-1]
            else:
                return result[::-1]
        


# In[ ]:


#Given a non-negative integer x, compute and return the square root of x.

#Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/sqrtx
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
# method 1:

class Solution:
    def mySqrt(self, x: int) -> int:
        square_rt = sqrt(x)
        if square_rt < int(square_rt):
            return int(square_rt)-1
        else:
            return int(int(square_rt))
#method 2 Algorithm using Newton's method: x(k+1) = (x(k) + n/x(k))/2

class Solution:
    def mySqrt(self, x: int) -> int:
        if x <= 1:
            return x
        r = x
        while r*r > x :
            r = (r + x/r)//2 #需要向下取余，否则会无限趋近于square root

        return int(r)


# In[223]:


#You are climbing a staircase. It takes n steps to reach the top.
#思路： f(x) = f(x-1) + f(x-2) f为到第x阶梯的总共的方法

class Solution:
    def climbStairs(self, n: int) -> int:
        i = 2
        st_0 = 1
        st_1 = 1
        ways = 0
        if n < 2 :
            return 1
        while i <= n:
            ways = st_0 + st_1
            st_0 = st_1
            st_1 = ways
            i += 1
        return ways 

#Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/climbing-stairs
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# In[ ]:


#Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dic = dict()
        prev = None
        curr = head 
        while curr:
            if curr.val not in  dic:
                dic[curr.val] = 1
                prev= curr
                curr = curr.next
            else:
                prev.next = curr.next
                tmp = curr
                curr = curr.next
                tmp = None 

        return head 


# In[226]:


#Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

#The number of elements initialized in nums1 and nums2 are m and n respectively. You may assume that nums1 has a size equal to m + n such that it has enough space to hold additional elements from nums2.



#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/merge-sorted-array
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i = m+n-1
        j = m-1
        k = n-1
        if n == 0:
            return
      
        if len(nums1) == len(nums2):
            i = 0
            while i < len(nums1):
                nums1[i] = nums2[i]
                i+=1
            return  
        while i >= 0 and k >= 0 and j >= 0 :
            
            if nums2[k] >= nums1[j]:
                nums1[i] = nums2[k]

                k -= 1
                i -= 1
            else:
                nums1[i] = nums1[j]
 
                j -= 1
                i -= 1
        while j < 0 and k>=0 and i >= 0:
            nums1[i] = nums2[k]
            k -= 1
            i -= 1
            
            

class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        if  nums1 == [0]:
            nums1[0] = nums2[0]
            return 
        if n == 0 and m != 0:
            return 
        if m == n ==0:
            return 
        for i in range(m, m+n):
            nums1[i] = nums2[i - m]
            prev = i-1
            curr = i
            while nums1[prev] > nums1[curr] and prev >= 0 and curr >= 0 :
                nums1[prev], nums1[curr] = nums1[curr], nums1[prev]
                prev -= 1
                curr -= 1
        


# In[ ]:


#Given the roots of two binary trees p and q, write a function to check if they are the same or not.

#Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/same-tree
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if p is None and q is None:
            return True
        elif q is None or  p is None :
            return False
        elif p.val != q.val:
            return False 
                
        bool_1 = self.isSameTree(p.left, q.left) 
        bool_2 = self.isSameTree(p.right, q.right)
        return bool_1 and bool_2


# In[ ]:


'''*'''#Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).



#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/symmetric-tree
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if root.left is None and  root.right is None:
            return True 
        if root is None:
            return True

        if root.left is None  or root.right is None:
            return False 
        left_node = root.left
        right_node = root.right
        def division(left_node, right_node):
            if left_node is None and right_node is None:
                return True 
            elif left_node is None or right_node is None:
                return False
            elif left_node.val != right_node.val:
                return False 
            
            return division(left_node.left, right_node.right) and division(left_node.right, right_node.left)

        return division(left_node, right_node)


# In[ ]:


#Given the root of a binary tree, return its maximum depth.

#A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/maximum-depth-of-binary-tree
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if root is None :
            return 0
        right_height = self.maxDepth(root.right)
        left_height = self.maxDepth(root.left)

        return 1 + max(right_height, left_height)


# In[ ]:


'''*'''#Given an integer array nums where the elements are sorted in ascending order, convert it to a height-balanced binary search tree.

#A height-balanced binary tree is a binary tree in which the depth of the two subtrees of every node never differs by more than one.

#平衡二叉树，递归思想

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        def helper(left, right):
            if left > right:
                return None
            mid = (left+right)//2
            root = TreeNode(nums[mid])

            root.left = helper(left, mid-1)
            root.right = helper(mid+1, right)

            return root
        
        return helper(0, len(nums)-1)


# In[ ]:


'''*'''#Given a binary tree, determine if it is height-balanced.

#For this problem, a height-balanced binary tree is defined as:

#a binary tree in which the left and right subtrees of every node differ in height by no more than 1.
#求树的高度的延伸版


#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/balanced-binary-tree
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def diff_height(node):
            if node is None:
                return 0
            left_height = diff_height(node.left)
            right_height = diff_height(node.right)
            return max(left_height, right_height)+1


        if root is None:
            return True
        return abs(diff_height(root.left) - diff_height(root.right)) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
## abs(diff_height(root.left) - diff_height(root.right)) <= 1判断当前根节点到叶子结点是否为平衡树
#  self.isBalanced(root.left) and self.isBalanced(root.right) 递归判断根节点的下一节点到叶子结点是否为平衡树。 
    
    


# In[ ]:


'''*'''#Given a binary tree, find its minimum depth.

#The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

#Note: A leaf is a node with no children.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/minimum-depth-of-binary-tree
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。



# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        left_height = self.minDepth(root.left)
        right_height = self.minDepth(root.right)
        if left_height ==0 and right_height !=0:
            
            return 1+ right_height
        if right_height == 0 and left_height != 0:
            return 1 + left_height

        return 1 + min(left_height, right_height)
    
    
    
    
    
    
    
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        def helper(node, cont):
            if node is None: #判断是否为空节点，为空节点则深度为0
                return 0
                
            if node.left is None and node.right is None: #判断节点的左右两侧为空，若为空，则说明该节点为叶子结点，将该层的深度1加入
                                                        #不需要再往下遍历因为该节点下面已经无子树
                cont = 1

            if node.left and node.right is None: # 判断该节点是否有左子树而无右子树， 若是，则首先算入该节点的深度1，之后遍历左子树
                cont = 1 
                cont = cont + helper(node.left, cont)

            if node.right and node.left is None: # 判断该节点是否有右子树而无左子树， 若是，则首先算入该节点的深度1，之后遍历右子树
                cont = 1
                cont = cont + helper(node.right, cont)

            if node.right and node.left: #判断该节点是否左右两侧均有子节点，若有，首先该节点的深度加1， 因为要求最小深度，cont值为遍历
                                            #左子树和右子树后，取两个子树的最小深度，return
                cont = 1
                cont = cont + min(helper(node.right, cont), helper(node.left, cont))

            return cont 

        return helper(root, 0)


# In[1]:


'''*'''#Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

#A leaf is a node with no children.



#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/path-sum
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

#method 1: 广度优先遍历


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections

class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if root is None:
            return False 
        
        node_deque = collections.deque([root])
        val_deque = collections.deque([root.val])

        while node_deque: #广度优先搜索遍历方法：通过队列实现，先将根节点放入队列，然后不断将该节点的子节点从左到右依次放入队列，并且同时将路径上的数字和相加放入对应的数字队列，放入速度会比取出速度快，但是没关系，我们的判断条件是队列不为空
            node_tmp = node_deque.popleft() #不断从队列左侧取出
            val_tmp = val_deque.popleft()
            if node_tmp.left is None and node_tmp.right is None: #当一个路径到达尽头时，判断是否为target，是则直接返回，如果不是，后面的存放步骤就不需要，直接进行下一个取出
                if val_tmp == targetSum:
                    return True
                else:
                    continue
            if node_tmp.left:
                node_deque.append(node_tmp.left)
                val_deque.append(val_tmp + node_tmp.left.val)

            if node_tmp.right:
                node_deque.append(node_tmp.right)
                val_deque.append(val_tmp + node_tmp.right.val)

        return False 
    
#method 2: recursion:

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if root is None: # 判断是否为空节点，若是，则返回false
            return False 
        if not root.left and not root.right: # 判断该节点是否为叶子结点，因为我们要判断路径和是否为target， 必须是从根节点到叶子结点
            if targetSum == root.val:        # 若是叶子结点，则判断target - 该节点之前路径上的所有节点值加起来 的余值是否为该节点的值。 
                
                return True                  #若是，则找到了，若不是则进行下一次遍历，下一次遍历会遍历到空节点，会return false
        
        return self.hasPathSum(root.left, targetSum-root.val) or self.hasPathSum(root.right, targetSum-root.val) #分别判断根节点左侧右侧
    
#自己的想法:正向去做这个题，每到一个节点，累计该节点的值, 判断到叶子结点后，路径和是否为target 
    
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        def helper(node, summ, target):
            if node is None:  #判断该节点是否为空节点，若是，则路径和值加0
                return 0

            if node.left is None and node.right is None: # 判断该节点是否为均无左右节点，若是，说明已经到了叶子结点，则summ加上该节点值
                summ = node.val + summ                   # 并且不用继续遍历，判断summ是否为target值，返回bool判断结果
                return summ == target

            if node.left and node.right is None:         # 判断节点左侧是否有节点并且节点右侧无节点，若是，则summ加上该节点值后遍历左侧节点
                return helper(node.left, summ+node.val,target) 

            if node.right and node.left is None:        # 判断该节点右侧是否有节点并且左侧无节点，如是，则summ加上该节点值后遍历右侧节点
                return  helper(node.right, summ+node.val,target)

            if node.right and node.left:                # 判断该节点左右两侧是否均有节点，若是，则分别判断该节点左右两侧的路径 用or 关联因为只要左右两侧有一侧路径和为target则是true
                return helper(node.right,summ+node.val,target) or helper(node.left, summ + node.val,target)
                
        if root is None:
            return False
        return helper(root, 0, targetSum)


# In[ ]:


#Given an integer numRows, return the first numRows of Pascal's triangle.

#In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/pascals-triangle
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        result = [[1]]
        i = 1
        while i < numRows:
            j =0
            nums = [1]
            print(nums)
            while j+1< len(result[i-1]):
                nums.append(result[i-1][j]+result[i-1][j+1])
                j += 1
            
            nums.append(1)
            result.append(nums)
            print(result)
            i += 1

        return result
    


# In[ ]:


#Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

#In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/pascals-triangle-ii
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        i = 0
        curr = [1]
        while i < rowIndex:
            prev = curr
            j = 0
            curr = [1]
            while j + 1  < len(prev):
                curr.append(prev[j]+prev[j+1])
                j += 1 
            curr.append(1)

            i += 1

        return curr


# In[ ]:


'''*'''#You are given an array prices where prices[i] is the price of a given stock on the ith day.

#You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

#Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

#暴力解法：

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0
        i = 0 
        maxx = 0 
        while i+1 < len(prices):
            j = i+1
            while j < len(prices):
                maxx = max(prices[j]- prices[i], 0,  maxx)
                j += 1
                print(maxx)
            i += 1 
        return maxx 
    
    
    



#you are given an array prices where prices[i] is the price of a given stock on the ith day.

#You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

#Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = 100000000000
        max_profit = 0

        for i in range(len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            elif prices[i] - min_price > max_profit:
                max_profit = prices[i] - min_price
            else:
                continue

        return max_profit
    

    
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0] #先设置最小值股价为第一天的价格，初始化
        i = 1
        profit = 0
        while i  < len(prices): # 遍历队列
            min_price = min(min_price, prices[i]) #每遍历一个price，则更新当前的历史上的最小价格
            profit = max(profit, prices[i]- min_price) #计算当前价格与历史最低价的差值，为利润，判断是否比历史最高利润还要高
            i += 1 

        return profit
    


# In[ ]:


#You are given an array prices where prices[i] is the price of a given stock on the ith day.

#Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

#Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

class Solution:
    def maxProfit(self, prices: List[int]) -> int: 
        summ = 0                         # 初始化利润值为0
        if len(prices) < 2:             # 若队列长度小于2， 则返回0
            return 0
        for i in range(len(prices)-1):  # 遍历队列
            if prices[i+1] > prices[i]: # 若下一个价格高于前一个价格，则买低卖高， 把sum累计上去
                summ = summ +prices[i+1]- prices[i]

        return summ 


# In[ ]:


#Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/valid-palindrome
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

class Solution:
    def isPalindrome(self, s: str) -> bool:
        strr = '' 
        for i in range(len(s)):
            if s[i].isalnum():
                strr = strr + s[i]

        strr = strr.replace(" ", "").lower()
        print(strr)
        if strr[::-1] == strr:
            return True
        else:
            return False 
        


# In[ ]:


#Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

#Follow up: Could you implement a solution with a linear runtime complexity and without using extra memory?



#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/single-number
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

#排序法：

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        nums.sort()
        print(nums)
        i = 0 
        j = i + 1
        while i + 1 < len(nums):
            if nums[i] == nums[j]:
                del nums[i:j+1]
            else:
                i += 1
                j += 1
            print(nums)

        return nums[0]
    
#字典法：

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        dic = dict()
        for i in range(len(nums)):
            if nums[i] in dic:
                dic[nums[i]] += 1
            else:
                dic[nums[i]] = 1

        for key in dic:
            if dic[key] == 1:
                return key
            
            


# In[ ]:


'''*'''#Given head, the head of a linked list, determine if the linked list has a cycle in it.

#There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

#Return true if there is a cycle in the linked list. Otherwise, return false.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/linked-list-cycle
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#哈希表/字典
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        curr = head
        dic = dict()
        while curr:
            if curr in dic:
                return True
                break 
            else:
                dic[curr] = 1 
                
            curr=curr.next

        return False 
    
#快慢指针 （高效）：如果该链表存在循环链表，则快指针一定会与慢指针相遇

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        fast = head
        slow = head 
        
        while fast and slow:
            if fast.next is None:
                return False 
            if fast.next.next == head:
                return True
            if fast.next.next == None:
                return False
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True

        return False 




    


# In[ ]:


'''*'''#Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

#Implement the MinStack class:

#MinStack() initializes the stack object.
#void push(val) pushes the element val onto the stack.
#void pop() removes the element on the top of the stack.
#int top() gets the top element of the stack.
#int getMin() retrieves the minimum element in the stack.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/min-stack
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

#method 1:

class MinStack:

    def __init__(self):
        self.stack = []


    def push(self, val: int) -> None:
        self.stack.append(val)
        print(self.stack)


    def pop(self) -> None:
        del self.stack[-1]
        print(self.stack)


    def top(self) -> int:
        return self.stack[-1]
        print(self.stack)


    def getMin(self) -> int:
        minn = 10000000000000000000
        for i in range(len(self.stack)):
            minn = min(minn, self.stack[i])
        print(self.stack)

        return minn



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

#method 2: 构建一个新栈

class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]


    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))#每一次都更新一下当前位置的一下的最小值，栈里面每个位置都有对应的值



    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()#每吐出一个值，最小值数组也要吐出来一个
        


    def top(self) -> int:
        return self.stack[-1]


    def getMin(self) -> int:
        return self.min_stack[-1]#当要求当前最小值时直接返回最小值数组的最后一个



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


# In[ ]:


#Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. 
#If the two linked lists have no intersection at all, return null.



#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/intersection-of-two-linked-lists
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# method 1:
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        dic = dict()
        curr = headA
        while curr:
            dic[curr] = 1
            curr = curr.next

        curr = headB
        while curr:
            if curr in dic:
                return curr
            curr = curr.next

        return None
    
    
'''*'''# method 2: 双指针法： 思想：两个指针分别从两个链表头开始遍历链表，当一个指针走完时，将该指针指向另一个链表头，另一个指针同理，如果这两个指针有共同节点，
#                     两个指针会在共同节点相遇，如没有共同节点，两个指针会同时指向空节点。
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None and headB is None:
            return None

        currA = headA
        currB = headB
        while currA != currB :
            if currA is None:
                currA = headB
            else:
                currA = currA.next

            if currB is None:
                currB = headA
            else:
                currB = currB.next

        return currA


# In[ ]:


#Given an array of integers numbers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.

#Return the indices of the two numbers (1-indexed) as an integer array answer of size 2, where 1 <= answer[0] < answer[1] <= numbers.length.

#You may assume that each input would have exactly one solution and you may not use the same element twice.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/two-sum-ii-input-array-is-sorted
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

#思路：因为该数组已经是升序排列，我们将两个指针分别置于该数组的头尾，取和，如果和大于目标数，则右指针向左平移，若小于目标树，则左指针向右平移
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i = 0 
        j =  len(numbers)-1
       
        while i < j:
            if numbers[i] +numbers[j] > target:
                j -= 1
            elif numbers[i] +numbers[j] < target:
                i += 1 
            else:
                return [i+1,j+1]


# In[ ]:


#Given an array nums of size n, return the majority element.

#The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/majority-element
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
#method 1: 哈希表
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        dic = dict()
        if len(nums) == 0 :
            return 
        if len(nums) == 1 :
            return nums[0]
        for i in range(len(nums)):
            if nums[i] in dic:
                if dic[nums[i]] >= int(len(nums)/2):
                    return nums[i]
                else:
                    dic[nums[i]] += 1 
            else:
                dic[nums[i]] = 1 

'''*'''# method 2： 排序法， 排序后，因为要寻找的是出现次数最多的元素，所以下标为n/2 处的元素一定是众数。
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[int(len(nums)//2)]
                


# In[ ]:


#Given a string columnTitle that represents the column title as appear in an Excel sheet, return its corresponding column number.


#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/excel-sheet-column-number
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。o

class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        result = 0 
        if columnTitle == "":
            return 0
        for i in range(len(columnTitle)):
            result = result*26 + (ord(columnTitle[i]) - 65) + 1 

        return result
    


# In[ ]:


#Given an integer n, return the number of trailing zeroes in n!.

#Follow up: Could you write a solution that works in logarithmic time complexity?

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/factorial-trailing-zeroes
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
            
#暴力法：            
class Solution:
    def trailingZeroes(self, n: int) -> int:
        prod = 1
        i = n
        while i >=1:
            prod = i * prod
            i -= 1 
            print(prod)

        if prod %10 !=0:
            print(prod)
            return 0

        else:
            count = 0
            while prod % 10==0:
                count += 1 
                prod = prod//10
                
        return count 

'''*'''#其实该题只要找n！中有多少个因子5， 因为只有当一个因子2与一个因子5匹配时，才能出现10，但因为因子2的倍数的个数一定大于因子5的倍数的个数
#因此我们只要计算因子5的倍数的个数即可。

class Solution:
    def trailingZeroes(self, n: int) -> int:
        k = 5 
        count = 0
        while k <= n:
            i = k
            while i % 5 == 0:
                count += 1 
                i = i / 5
            k += 5
        return count 
    


# In[ ]:


#Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

#Note:

#Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
#In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.

#Input: n = 00000000000000000000000000001011
#Output: 3
#Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.


#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/number-of-1-bits
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

class Solution:
    def hammingWeight(self, n: int) -> int:
        dic = dict()
        dic[1] = 0
        strr = bin(n)[2:]
        for i in range(len(strr)):
            if strr[i] == "1":
                dic[1] += 1 

        return dic[1]
'''*'''# method 2: 位运算优化 n&(n-1) 的解释为将n的二进制中最后一个 1 变为 0 的操作，通过while循环直到没有1为止  
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n = n & (n-1)
            count += 1
        return count 


# In[ ]:


'''*'''#Write an algorithm to determine if a number n is happy.

#A happy number is a number defined by the following process:

#Starting with any positive integer, replace the number by the sum of the squares of its digits.
#Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
#Those numbers for which this process ends in 1 are happy.
#Return true if n is a happy number, and false if not.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/happy-number
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
#method 1: 哈希表
class Solution:
    def isHappy(self, n: int) -> bool:
        dic = dict()
        m = n
        while m not in dic and m != 1:
            dic[m] = 1
            sq = 0
            while m >0:
                k = m % 10
                sq = sq + k*k 
                m = m//10
            m = sq
        if m in dic:
            return False
        if m == 1:
            return True
# method 2：双指针法 （类似于判断循环链表）

class Solution:
    def isHappy(self, n: int) -> bool:
        def next_num(num):    #定义一个函数用于求解下一个值
            summ = 0
            while num > 0:
                num, digit = divmod(num, 10) #divmod()用于同时返回 除数 和 余数
                summ = summ + digit ** 2 

            return summ
        
        slow_num = n               # 设置慢指针初始值，为n
        fast_num = next_num(n)     # 设置快指针为下一个值
        while fast_num != 1 and slow_num != fast_num: #只要快指针不等于1 并且慢指针快指针不等，就接着循环
            slow_num = next_num(slow_num)
            fast_num = next_num(next_num(fast_num)) #快指针要遍历快一点
        
        return fast_num == 1 #返回判断是否快指针是否等于1


# In[ ]:


'''*'''#Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/remove-linked-list-elements
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        def delete(head, val):     
            if head is None:   #判断是否头指针为空，空则直接返回
                return head
            while head.val == val: #判断头指针的值是否就等于目标值，循环的目的是考虑从头指针之后连续几个节点的值都是目标值
                if head.next:
                    head = head.next
                else:
                    return None
            prev = head        
            if prev.next:   #下面考虑正常情况
                curr = prev.next # 双指针
                while curr: 
                    if curr.val == val:#当前指针等于目标值时，前一个指针指向当前指针下一个节点，将当前节点置为空，将当前节点指针移到下一个节点位置
                        prev.next = curr.next
                        curr = None
                        curr = prev.next

                    else:
                        curr= curr.next
                        prev = prev.next
                return head
            else:
                return head
            
        return delete(head, val)
    


# In[6]:



class Solution:
    def countPrimes(self, n: int) -> int:
        n = n-1
        lst_prime = 2
        if n == 0 or n == 1:
            return 0
        lst = []
        for i in range(2, n+1):
            lst.append(i)
        tmp_2 = lst_prime
        i = 2 
        while tmp_2 * i <= n:
            lst.remove(tmp_2*i)
            print(lst)
            i += 1 
        while n > lst_prime:
            lst_prime += 1 
            tmp = lst_prime
            i = 2 
            while tmp * i <= n:
                if tmp * i in lst:
                    lst.remove(tmp * i)
                i += 1 
                print(lst)
            n = lst[-1]
        return len(lst)
#method 2: 埃氏筛选法
class Solution:
    def countPrimes(self, n: int) -> int:
        is_prime = []
        i = 1
        is_prime = list(range(1, n)) #创建一个 从 1 到 n-1 的序列
        cnt = 0
        for i in range(2, n):
            if is_prime[i-1]:
                cnt += 1 
                if i * i < n:
                    for j in range(i*i, n , i):
                        is_prime[j-1] = 0
                       

        return cnt


# In[ ]:


'''*'''#Given two strings s and t, determine if they are isomorphic.

#Two strings s and t are isomorphic if the characters in s can be replaced to get t.

#All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character, but a character may map to itself.


#此题关键是要判断两个字符串的每个对应位置字符是否为1-1的映射关系

class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        dic_s = dict()                          #首先定义两个字典，用于分别记录对方的对应关系，key值为该字符串的对应字符，value为另一个字符串对应位置的字符
        dic_t = dict()

        for i in range(len(s)):                 #遍历字符串
            if (s[i] in dic_s and dic_s[s[i]] != t[i]) or (t[i] in dic_t and dic_t[t[i]] != s[i]): #当当前字符在字典中时，判断该字符对应的value是不是另一个字符相应位置对应的value，如果是，则继续，不是则返回false，不是isomorphic的
                return False 
            dic_s[s[i]] = t[i]     #若没有出现在字典里，则将当前字符加到字典里
            dic_t[t[i]] = s[i]
         

        return True  #若成功遍历完整个字符串，则说明是isomorphic的
    
    


# In[ ]:


'''*'''#You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

#You may assume the two numbers do not contain any leading zero, except the number 0 itself.

#来源：力扣（LeetCode）
#链接：https://leetcode-cn.com/problems/add-two-numbers
#著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        curr1 = l1
        curr2 = l2 
        tmp = 0     # tmp为临时变量，用于标记上一位是否有1加到下一位
        while curr1.next or curr2.next: # 同时循环遍历两条链， 结束条件为两条链的下个节点都为空节点
            if curr1.next is None and curr2.next: # 如果其中一条链的下个节点为空，另一个不是，则给为空的那个补一个节点
                curr1.next = ListNode(0)

            elif curr2.next is None and  curr1.next:# 如果其中一条链的下个节点为空，另一个不是，则给为空的那个补一个节点
                curr2.next = ListNode(0)

            if curr1.val + curr2.val + tmp >= 10: #我们任意选择一条链为累计值链，判断对应位的数字相加并且加上tmp是否>=10， 若是，则对和取余，并覆盖
                                                # curr1的原有的值
                curr1.val = (curr1.val + curr2.val + tmp) % 10 
                tmp = 1                        #将tmp置为1，因为这一位上有进一

            else:
                curr1.val = curr1.val + curr2.val  + tmp # 若小于10，则直接将三个数字相加覆盖curr1 的值
                tmp = 0                                  #并将tmp置为0，因为没有进一

            curr1 = curr1.next
            curr2 = curr2.next

        if tmp + curr1.val + curr2.val >= 10:        # 当退出循环时，我们还未累加最后一位数字， 累加最后一位对应数字和tmp， 判断是否大于等于10，若是，附加一个节点并赋值为1，返回curr1链表
            curr1.val = (curr1.val + curr2.val + tmp) % 10 
            curr1.next = ListNode(1)
            return l1
        else:
            curr1.val = curr1.val + curr2.val + tmp # 否则，直接将三个值相加，返回curr1 
            return l1 


# In[ ]:


#Given a string s, find the length of the longest substring without repeating characters.

#自己的思考：哈希表， 从头开始记录，每个key对应的val为字符的下标，遇到重复字符，把整个表删除，从重复字符的下标位下一位重新计数
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = dict() #创立一个字典
        i = 0 
        cont = 0  #设置一个记录无重复长度变量
        prev = 0  # 设置一个之前的最长度无重复长度变量
        while i < len(s): # 循环遍历字符串
            if s[i] not in dic: #如果字符不在字典中，则将该字符作为key， 该字符的下标作为value存入字典， 并且计数加一，i往后移动一位
                dic[s[i]] = i
                cont += 1 
                i += 1 
            else:              #如果字符在字典中，则首先找到该字符上一次出现在字符串中的位置（value），将i设置为该位置的下一个位置，从这开始往后遍历
                i = dic[s[i]]+1 
                dic.clear()    # 将整个字典清空
                cont = max(prev, cont) # 记录当前长度与上一个长度的最大值，并赋值给cont
                prev = cont # 将cont赋值给prev
                cont = 0 # cont 归0 用于记录下一组长度
        cont = max(prev, cont) #取prev 与 cont最大值并返回
        return cont
            
'''*'''# 滑动窗口：

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        widw = set()
        rk , length= 0
        for i in range(len(s)):      # 第一层循环，遍历字符串
            if i != 0 :              # 如果不是第一个字符，则左指针每次向后移一位，也就是窗口内最左侧的元素被删除
                widw.remove(s[i-1])
            while rk  < len(s) and s[rk] not in widw: # 右指针向右滑动，直到在窗口内出现重复元素为止
                widw.add(s[rk])
                rk = rk + 1 

            length = max(length, rk-i) # 比较当前窗口长度与历史最长窗口长度，取最大值

        return length 


# In[ ]:


#Given a string s, return the longest palindromic substring in s. *Medium

#暴力法：

class Solution:
    def longestPalindrome(self, s: str) -> str:
        def palindromic(left, right, s):
            while left < right:
                if s[left] != s[right]:
                    return False 
                left += 1 
                right -= 1 
            return True
        
        if len(s)<=1:
            return s
        if len(s) == 2:
            if s[0] == s[1]:
                return s
            else:
                return s[0]
        max_length = 1
        left_index = 0 

        for i in range(len(s)):
            j = i +1
            while j <len(s):
                if j - i + 1 > max_length and palindromic(i,j,s):
                    max_length =  j-i+1
                    left_index = i

                j += 1 
        return s[left_index : left_index + max_length]
    
#中心扩散法：

class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) < 2:                           # 若字符串长度小于2，则直接返回s因为长度为一或0的字符串一定是回文序列
            return s
        def centerexpand(s, i, j):               #定义一个函数：该函数的实现方法为给定中心值，当时单一中心值时，i=j，当为双中心值时，i = i+1 
            length = len(s)                      
            while i >= 0 and j < length:         #由中心值遍历字符串，当两侧的字符不一样时则停止遍历
                if s[i] ==s[j]:
                    i -= 1
                    j += 1 
                else:
                    break
            return [i+1, j-1]                    # 通过返回一个数组，记录下回文序列的左右两个下标
        li = 0
        ri = 0
        max_len = 1
        for i in range(len(s) -1):               # 遍历字符串
            odd = centerexpand(s,i,i)            # 当遍历到每一个字符时，由该字符向两侧进行遍历，通过调动之前定义的函数，返回单中心回文字符两侧下标
            even = centerexpand(s, i, i+1)       # 返回双中心回文字符两侧下标

            if odd[1]-odd[0] > even[1]- even[0] and odd[1]-odd[0]+1> max_len: # 判断单中心回文串的长度和双中心回文串的长度哪个大，并与之前的最长回文串长度进行比较
                li = odd[0]
                ri = odd[1]
                max_len = odd[1]-odd[0] +1
            if odd[1]-odd[0] < even[1]- even[0] and even[1]-even[0]+1> max_len:
                li = even[0]
                ri = even[1]
                max_len = even[1] - even[0]  +1 
        return s[li : ri+1]                                     #返回找到的最长回文串
    
    


# In[ ]:


#Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

#Symbol       Value
#I             1
#V             5
#X             10
#L             50
#C             100
#D             500
#M             1000

#For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

#Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

#    I can be placed before V (5) and X (10) to make 4 and 9. 
#    X can be placed before L (50) and C (100) to make 40 and 90. 
#    C can be placed before D (500) and M (1000) to make 400 and 900.

#Given an integer, convert it to a roman numeral.



class Solution:
    def intToRoman(self, num: int) -> str: #创立一个字典，按照由大到小的顺序，按序排列特殊数字对应的字符
        dic = {1000 :'M', 900 : 'CM', 500 : 'D', 400 : 'CD', 100 : 'C', 90 : 'XC', 50 :'L', 40:'XL', 10 :'X', 9:'IX', 5:'V', 4:'IV', 1:'I'}
        ans = ""
        for key, value in dic.items(): #按序遍历字典里的所有元素
            while num >= key:          # 当数字大于特殊key值时，字符串需要加上对应的字符，并将数字减去那个特定的key值
                num = num - key
                ans = ans + value

        return ans
    
    
    
#Given a roman numeral, convert it to an integer. 罗马数字到阿拉伯数字

class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {'M':1000, 'CM':900,'D':500,'CD':400,'C':100,'XC':90,'L':50,'XL':40, 'X':10,'IX':9,'V':5, 'IV':4, 'I':1}
        res = 0
    
        while s :                           #遍历字符串
            if len(s) >=2 and s[0:2] in dic:  #判断是两个罗马字母表达一个数字的情况
                res = res +dic[s[0:2]]
                s = s[2:]                    #将判断过的字符删除
                
            
            else:
                res = res + dic[s[0]]         #否则就是一个罗马字母的情况
                s = s[1:]

        return res
    
    

    
    


# In[ ]:


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if nums1 ==[]:
            nums1 = nums2
        if nums2  == []:
            nums2 = nums1
        if nums1 != [] and nums2 !=[]:
            med1 = nums1[len(nums1)//2]
            med2 = nums2[len(nums2)//2]
        f1 = len(nums1)//2
        f2 = len(nums2)//2
        print(med1)
        print(med2)
        
        while med1 != med2:
            k1 = f1
            k2 = f2
            prev_med1 = med1
            prev_med2 = med2
            if med1 < med2:
                while k1 < len(nums1) and nums1[k1]< med2:
                    k1 += 1 

                k1 -= 1 
                while k2 >= 0 and nums2[k2] >med1:
                    k2 -= 1 
                
                k2 += 1 
              print("this is k1:", k1) 
              print("this is k2", k2) 
            
            if med1 > med2:
                while k2 < len(nums2) and nums2[k2]< med1:
                    k2 += 1 

                k2 -= 1 
                while k1 >= 0 and nums1[k1] >med2:
                    k1 -= 1 

                k1 += 1 
                print("this is k1:", k1) 
                print("this is k2", k2) 




            med1 = nums1[k1]
            med2 = nums2[k2]

            f1 = k1
            f2 = k2
            if prev_med1 == med1 and prev_med2 ==med2:
                if (len(nums1)+len(nums2)) %2 !=0:
                    return (med1+med2)//2
                return (med1+med2)/2


        return med1

      


# In[ ]:


#There is an integer array nums sorted in ascending order (with distinct values).

#Prior to being passed to your function, nums is rotated at an unknown pivot index k (0 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

#Given the array nums after the rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

#You must write an algorithm with O(log n) runtime complexity.

#Example 1:

#Input: nums = [4,5,6,7,0,1,2], target = 0
#Output: 4

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) ==1:
            if nums[0] == target:
                return 0

            else:
                return -1
        if nums[0]< nums[len(nums)-1]: #判断是否是在第一位翻转，如果是，则数组顺序为递增
            i = 0
            while i < len(nums):
                if target== nums[i]:
                    return i
                i += 1 

            return -1 


        print("this is nums[0]", nums[0])
        if target >= nums[0]: #当目标值大于第一个数，说明我们应该在翻转数列的前一段寻找
            i = 0
            while i+1< len(nums) :
                if nums[i]> nums[i+1]: #当循环到第一段的结尾时结束
                    if target == nums[i]:
                        return i
                    else:
                        break
                if target == nums[i]:
                        return i #如果找到，则返回下标，没找到返回-1
                i += 1

            return -1 

        else:
            i = len(nums)-1 #若目标数小于第一个数，我们应该在第二段搜索
            while i-1 >= 0 :
                if nums[i] < nums[i-1]: 
                    if target == nums[i]:
                        return i
                    else:
                        break
                if target == nums[i]:
                        return i
                i -= 1
                print("this is i: ", i)

            return -1


# In[ ]:


#Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

#The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.


#It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

#方法一：回溯法，**画出树形图**， 该题特点是可重复的从一个样本中取相同值
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        def helper(candidates, begin, size, path, res, target): # 定义一个递归函数因为我们要找不同的组合方法，所以遍历顺序没有意义
            if target < 0:       #如果遍历到叶子结点为小于0，则该组合不等于target # 所以我们要定义一个begin变量，每一个根节点的分支下每个分支的元素是左边树枝的可用元素包含了往右走的树枝的可用元素
                return 
            if target == 0:      # 如果target 剪到最后等于0，我们得到了一个组合，并且结束该路径的遍历
                res.append(path)
                return

            for index in range(begin, size): #内部的一个for循环用于产生由一个节点的多个分支---》用于定义母树的宽度
                helper(candidates, index, size, path+[candidates[index]], res, target - candidates[index]) # 由每个节点向下延伸子树，用于定义子树的深度和宽度


        size = len(candidates) 
        if size == 0 :
            return []

        res = [] #这里定义结果集，将该结果集传递到helper中，可以对这个数组本身进行操作
        helper(candidates,0, size, [], res, target) # 引用helper函数

        return res


# In[ ]:


#*Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

#Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
'''
 *               9   9   9
 *         ×     6   7   8
 *  ----------------------
 *              72  72  72
 *          63  63  63
 *      54  54  54
 *  ----------------------
 *      54 117 189 135  72
 *  ----------------------
 *      54 117 189 142   2
 *  -----------------------
 *      54 117 203   2   2
 *  -----------------------
 *      54 137   3   2   2
 *  -----------------------
 *      67   7   3   2   2
 *  -----------------------
 *   6   7   7   3   2   2
 */ '''

class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        mom = ""                              
        son = ""
        if num1 == "0" or num2 == "0":
            return "0"
        if len(num1) > len(num2):     #将长度较长的那个放在顶部，长度较短的那个放在底部进行乘法运算
            mom = num1[::-1]
            son = num2[::-1]
        else:
            mom = num2[::-1]
            son = num1[::-1]
        cont = len(mom)+(len(son)-1) #计算总体的数组长度：总体的长度为 被乘数的位数 +（乘数的位数-1）
        arr = [0]*cont               
        for i in range(0, len(son)): #从乘数的个位开始与mom的每一位分别相乘，每一位乘的结果直接保留到一个数组里， 注意：这里是son的每一位都会和mom的每一位相乘
            for j in range(0, len(mom)): #所以当son每升一位，当前位的son值与mom相乘的结果直接与son的上一位相加，但是需要向左错开一位
                arr[j+i] = int(mom[j]) * int(son[i]) + arr[j+i]
        tmp = 0
        for i in range(len(arr)): #下面开始从数组的个位开始，如果元素大于等于10， 则该元素保留个位上的数字，将十位的数字传给下一位
            if arr[i]+tmp >= 10:
                k = arr[i]+tmp
                arr[i] = str((arr[i]+tmp) % 10)
                tmp =  k//10    
            else:
                arr[i] = str(arr[i] +tmp)
                tmp = 0
        if tmp != 0:         #直到结束，如果退出循环后，tmp不是0，说明最后一位还是大于等于10，需要数组最后补一位，将tmp 值附加到数组后面
            arr.append(str(tmp)) 
        arr = arr[::-1]
        st = ''
        st = st.join(arr)
        return st


# In[ ]:


'''Given an array of non-negative integers nums, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

You can assume that you can always reach the last index.'''

#method: greedy algorithmic 贪心算法： 从数组开头，以当前数组值为搜索边界，找出在该边界内，哪个位置的元素可以走得最远
#找到后跳到该节点，并以该节点为起始点继续找下一个最佳节点，直到最后一个节点，每跳一次加一

class Solution:
    def jump(self, nums: List[int]) -> int:
        index = 0
        count = 1
        if len(nums) == 1: #判断长度是否为1，如果是1，则不用跳跃
            return 0
        while index < len(nums): # 循环条件为下标小于数组长度
            bound = index + nums[index] #根据当前值设置边界，边界为当前值的下标加上该元素值，即该元素可以跨越得最远距离
            if bound >= len(nums)-1: # 当边界值大于数组最后一个下标时，结束遍历，因为到这个时候我们一定可以到达最后一个节点，我们不需要
                break 
            maxx = 0
            mark = 0
            count += 1 
            for i in range(index+1, bound+1): # 在这个边界范围内，搜索跳的最远的点，跳的最远，即当前位置的最远能抵达的距离
                if nums[i]+i >= maxx:
                    mark = i
                    maxx = nums[i] + i
            index = mark    # 找到其中的最远距离点后跳到该点        
        return count 


# In[ ]:


''' 46. Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.'''

#自己的回溯思路，深度优先遍历
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def helper(nums, arr, res): #回溯函数
            if len(nums) < 1: #当数组中没有元素时，说明已经全部遍历完了
                res.append(arr) #将一种结果放到res中并返回结束遍历
                return 
            else:
                for i in range(len(nums)): # 每次遍历将当前元素放入数组，深度优先遍历
                    helper(nums[0:i]+nums[i+1:], arr +[nums[i]], res)


        res = []
        helper(nums, [], res)

        return res
    
#大神的回溯思路：
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def helper(nums, depth, used, res, path, size): #回溯函数
            print(path)
            if depth == size: #当深度与数组大小相等时结束遍历该路径，并将该路径的结果保留到res数组中
                res.append(path[:])
                return #终止该回溯并返回，确保函数后的程序可以执行到

            for i in range(size): #循环遍历数组
                if not used[i]: #通过布尔数组判断当前元素是否已经被使用过了

                    used[i] = True #将当前元素置为已经使用过
                    path.append(nums[i]) #将该路径添加新的元素，回溯函数上面的这两行是在做往下走的动作
                    helper(nums, depth + 1, used, res, path, size) #深度加一后接着遍历
                    print(path) 
                        
                    used[i] = False #在回溯函数后的这一段是在做往回走的动作，需要将当前用过的元素还回来，变成没用过的，以便后面遍历再次使用
                    path.pop() #将path上的最后一个元素撤回，是回到上一级节点的动作
            

        size = len(nums) # 计算数组大小
        used = [False for _ in  range(size)] #设置一个布尔数组
        res = [] 
        path = []

        depth = 0
        helper(nums, depth, used, res, path, size)

        return res 


# In[ ]:


'''47. Adcanced 46: Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order.'''

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def helper(nums, depth, used, res, path, size):
            print(path)
            if depth == size:
                res.append(path[:])
                return

            
                
            for i in range(size):
                if not used[i]:
                    if i > 0 and nums[i] == nums[i-1] and used[i-1] == False: #该步骤用于剪枝，去掉相同的元素，首先我们需要先将数组排序，排序后如果前后两个相同，并且前一个没有被使用过
                        continue                                              # 说明该元素是从上一个分支回溯时退回来的。退回来的，但是下一个元素和这个退回来的相同，如果再次计算则会重复，于是需要把这个剪去
                    used[i] = True
                    path.append(nums[i])
                    helper(nums, depth + 1, used, res, path, size)
                    print(path)
                        
                    used[i] = False
                    path.pop()
            
        nums.sort()
        size = len(nums)
        used = [False for _ in  range(size)]
        res = []
        path = []

        depth = 0
        helper(nums, depth, used, res, path, size)

        return res
    


# In[ ]:


'''Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.'''

#自己的庸俗思维：每到一个字符串，判断该字符串对应的字典是否已经出现在我们的字典数组中，若是，则在结果数组中根据下标定位放进去，若不在，则先将该字符串字典放到字符串数组中，并同时将该字符串放到结果数组中
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        i = 0 
        
       
        dic = [] # 创立一个数组，用于存放不同的字典
        mark = [] # 创建一个结果数组，因为我们是同时操作字典数组和结果数组，他们的每个位置是一一对应的
        tmp_dic = dict()#创立一个字典变量用于记录当前字符串
        while i < len(strs): #遍历字符串数组
            j = 0
            while j < len(strs[i]): #创立当前字符串的临时字典

                if strs[i][j] in tmp_dic:
                    tmp_dic[strs[i][j]] += 1 
                else:
                    tmp_dic[strs[i][j]] = 1 
                
                j += 1 

            if tmp_dic not in dic: #判断如果该字符串字典不在字典数组中，则把该字典附加到字典数组尾部， 同时在结果数组尾部加上一个新的数组与之对应
                dic.append(tmp_dic.copy())
                mark.append([strs[i]])
            else:
                mark[dic.index(tmp_dic)].append(strs[i]) #如果在了，那就根据字典元素在字典数组中的位置下标，在结果数组的对应位置的子数组添加该字符串

            i += 1 
            tmp_dic.clear() # 每个循环结束后要清空临时字典以便下个循环使用


        return mark #返回结果数组
    
#大神的普遍思想：将每个字符串排序，排序后的字符是唯一确定的，注意⚠️这里用sorted(string)返回的是一个列表，需要用join()将字符合并为一个字符串

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = dict()
        i = 0
        res = []
        for i in range(len(strs)): #遍历字符串数组
            tmp_strs = strs[i]
            tmp_strs = "".join(sorted(tmp_strs))  #将当前字符串里的字母排序然后合并为新的字符串   
            if tmp_strs in dic: # 如果该字符串已经在字典里出现，则找到对应的value数组，把该字符串添加进去
                dic[tmp_strs].append(strs[i])

            else:
                dic[tmp_strs] = [strs[i]] #否则在字典中添加一个新元素


        for key in dic: #遍历字典，将所有的值都放入一个数组中输出
            res.append(dic[key])


        return res

    

    


# In[ ]:


'''56. Merge Interval: Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
and return an array of the non-overlapping intervals that cover all the intervals in the input.'''



#自己愚蠢的方法：

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort() # 我们先把数组排序，排序的结果是，以每个子数组的第一个元素为排序对标进行排序
        print(intervals)
        res = []
        i = 0 
        left = 0
        right = 1
        if len(intervals) == 1: # 如果数组长度是1， 直接返回
            return intervals
        if len(intervals) == 2 : #如果长度是2，我们也要单独判断下因为下面的循环，数组长度必须大于2
            if intervals[0][right] >= intervals[1][left]: #当第一个子数组的右侧总是大于等于第二个子数组的左侧，则这两个数组可以合并，合并时，合并数组的右侧由这两个数组右侧的最大值决定
                res = [[intervals[0][left], max(intervals[0][right], intervals[1][right])]]
            else:
                res = intervals #如果不大于或等于，则无法合并，直接返回原数组
            return res

        while i < len(intervals)-1 and len(intervals)>2: #正常情况：数组长度大于2
            tmp_arr = [intervals[i][left]] #设置一个临时数组变量，将第一个无法和之前合并区间的左侧放入，如果没有之前的区间也就直接放入了
            maxx = intervals[i][right] # 初始化区间右侧，默认为当前区间的右边界
         
            if maxx >= intervals[i][left]: # 判断当上一个左边界大于下一个右边界时，一直遍历直到为否为止
                while maxx >= intervals[i+1][left]:
                    maxx = max(maxx, intervals[i+1][right]) #不断更新最大右边界
                    i += 1 
                    if i == len(intervals)-1: #当i到了最后一个时停止
                        break
                
                print(maxx)
                tmp_arr.append(maxx) # 将最大右边界赋上
            else: 
                tmp_arr.append(maxx) #未经过遍历的话，直接把当前的右边界赋上
            
            res.append(tmp_arr) #将临时数组给结果数组
            
            tmp_arr = []
            i += 1 
        if i == len(intervals) -1 :#因为我们到了最后一个数组时就没有进行判断，我们需要下面再进行判断
            if intervals[i][left] <= res[-1][right]:
                res[-1][right] = max(res[-1][right], intervals[i][right])

            else:
                res.append(intervals[i])
        return res 
    

# 大神们的方法：
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x: x[0])'''该语句的意义是以每个子数组的开头元素作为对标进行排序'''
        res = []
        for interval in intervals:
            if not res or res[-1][1] < interval[0]: '''一定要注意这里的判空语句是not res，判空以后用not res。如果是空数组或者结果数组的尾部值小于下一个元素的头部值，则在结果数组尾部附加上'''
                res.append(interval)

            else:
                res[-1][1] = max(res[-1][1], interval[1])'''如果末尾值大于下个元素开头值，则说明可以合并，末尾值改为末尾值与下个数组的右边界的最大值'''

        return res  #返回结果
                


# In[10]:



#回溯法，可行但是严重超时

def helper(index_1, index_2, grid, path, minn, m, n):
    if index_1 > m-1  or index_2 > n-1:
        return 0
    if index_1 == m-1 and index_2 == n-1 :
        path = grid[index_1][index_2]
                
                
           
          

    if index_2 == n-1 and index_1 < m-1:
        path = grid[index_1][index_2]
                
        path = path + helper(index_1+1, index_2, grid, 0,minn, m,n)
        if path > minn:
            return 
       

    if index_1 == m-1 and index_2 < n-1:
        path = grid[index_1][index_2]
        path = path + helper(index_1, index_2+1,grid,0,minn, m,n)
        if path > minn:
            return 
              

    if index_1 < m-1 and index_2 < n-1:
        path = grid[index_1][index_2]
        path = path + min(helper(index_1, index_2+1, grid, 0,minn, m,n), helper(index_1+1, index_2, grid, 0,minn,m,n))             
        if path > minn:
            return 
        else:
            minn = path
                
                


    return path
grid = [[7,1,3,5,8,9,9,2,1,9,0,8,3,1,6,6,9,5],[9,5,9,4,0,4,8,8,9,5,7,3,6,6,6,9,1,6],[8,2,9,1,3,1,9,7,2,5,3,1,2,4,8,2,8,8],[6,7,9,8,4,8,3,0,4,0,9,6,6,0,0,5,1,4],[7,1,3,1,8,8,3,1,2,1,5,0,2,1,9,1,1,4],[9,5,4,3,5,6,1,3,6,4,9,7,0,8,0,3,9,9],[1,4,2,5,8,7,7,0,0,7,1,2,1,2,7,7,7,4],[3,9,7,9,5,8,9,5,6,9,8,8,0,1,4,2,8,2],[1,5,2,2,2,5,6,3,9,3,1,7,9,6,8,6,8,3],[5,7,8,3,8,8,3,9,9,8,1,9,2,5,4,7,7,7],[2,3,2,4,8,5,1,7,2,9,5,2,4,2,9,2,8,7],[0,1,6,1,1,0,0,6,5,4,3,4,3,7,9,6,1,9]]
path = grid[0][0]
path = helper(0,0,grid,path,10000000, len(grid),len(grid[0]))
print(path)
    
# 动态规划：第一行和第一列的走法是固定的，只能一直往下或往右走，把每个走的值累计起来，其他方块的和为这个方块的上面和这个方块的左边的值的最小值
#空间换时间的思想
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = [[0 for i in range(len(grid[0]))] for j in range(len(grid))]
        dp[0][0] = grid[0][0]
        for i in range(1, len(dp[0])): #先计算第一行的每个空位的累计和
            dp[0][i] = dp[0][i-1] + grid[0][i]

        for i in range(1, len(dp)): #再计算第一列的每个空位的累计和
            dp[i][0] = dp[i-1][0]+grid[i][0]

        print(dp)
        for j in range(1,len(dp[0])): #最后计算剩余部分的空位的累计和， 累计和为该空位的上一个位置累计和和左边位置的累计和中的最小值
            for i in range(1, len(dp)):
                dp[i][j] = min(dp[i-1][j]+ grid[i][j], dp[i][j-1]+grid[i][j])

        return dp[len(grid)-1][len(grid[0])-1] #返回计算出的最小值


# In[ ]:


#Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

#We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

#You must solve this problem without using the library's sort function.


#自己的思考：奇数位偶数位轮流调换顺序
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        def even_ex(nums,size):#定义两个函数，第一个函数是偶数对遍历，从下标0开始
            if size <2: #当长度小于2时我们直接返回
                return nums
            first = 0
            second = 1
            while second < size:
                if nums[first]>nums[second]: #每当找到一对数前面的大于后面的，则调换顺序
                    nums[first], nums[second] = nums[second], nums[first]

                first += 2 #跳两位继续循环
                second +=2
            return nums
        def odd_ex(nums,size): # 第二个为奇数对遍历，从下标1开始
            if size <= 2: #当长度小于等于2时，我们不用进行奇数位遍历
                return nums
            first = 1
            second = 2 
            while second < size:
                if nums[first] > nums[second]:
                    nums[first], nums[second] = nums[second], nums[first]
                first += 2
                second += 2 
            return nums
        size = len(nums)
        count = 0
        while True: #循环遍历
            nums = odd_ex(even_ex(nums, size),size) #不断进行偶数位遍历再进行奇数位遍历，这两个遍历不分前后顺序，直到排序完成位置
            count += 1 
            nums_c = nums.copy() #注意⚠️这里判断时需要copy数组，而不是直接等于，直接等于还是指向原内存地址，再次调用函数肯定是相等的
            if nums_c == odd_ex(even_ex(nums, size),size): #如果相等说明已经排好序，退出返回nums
                break
        
        return nums 
    
# 方法二： 先把0放到前面，再把1放到0后面，即完成遍历
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        head = 0 #设置一个头边界，一开始为0，因为头边界还没有任何值
        for i in range(len(nums)): 
            if nums[i] == 0: #每当找到一个0 就把0放到head的位置，并同时head向后走一位
                nums[head], nums[i] = nums[i], nums[head]
                head += 1 

            

        if head < len(nums):
            for i in range(head, len(nums)): #找1，从head开始往后遍历，每当找到1就把1 放到head的位置，并往后遍历
                if nums[i] == 1:
                    nums[head], nums[i] = nums[i], nums[head]
                    head += 1 

                

            
        return nums


# In[ ]:


#Given an m x n grid of characters board and a string word, return true if word exists in the grid.

#The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

#回溯+标记重置
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        direction = [(0,1),(0,-1),(1,0),(-1,0)] #direction变量定义一个方向变量，用于记录下面向哪走
        def helper(i, j, k): 
            if board[i][j] != word[k]: # 如果指定元素不等于当前元素word[k]， 返回false
                return False
            if k == len(word)-1 : # 如果k已经达到了word最后一位， 返回true因为我们已经判断过这一位的值一定等于word[k]
                return  True
            result = False #置result为false
            mark[i][j] = True #当前位置为true 代表现在遍历了这一个位置
            for di, dj in direction: #遍历direction这个数组
                new_i, new_j = i + di, j + dj #将方向数组传递给当前位置
                if 0 <= new_i < len(board) and 0 <= new_j < len(board[0]):# 判断下一个位置是否在数组范围内
                    if mark[new_i][new_j] is False: #判断该位置是否之前已经被访问过
                        if helper(new_i,new_j,k+1):#判断该位置的值是否为对应的字母
                            result = True # 如果是则返回true
                            break
            mark[i][j] = False #回溯时，将当前位置的标记置为false，表示将其置为未访问过，因为如果运行到这一步，说明上一步不是对应到相应字母
            return result #返回result
        mark = [[False for i in range(len(board[0]))] for i in range(len(board))] #创立mark数组
        for i in range(len(board)): #遍历数组每个元素，将每个元素作为开头试一下
            for j in range(len(board[0])):
                if helper(i, j, 0):
                    return True 
        return False
    


# In[ ]:


#Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None or head.next == None: # 如果是空链表或者是只有一个节点的链表，就直接返回head
            return head
        mark = ListNode(None) # 在head前放一个空节点指向head
        mark.next = head 
        cur= head #创建一个指针指向head
        while cur:#遍历链表
            while cur.next and cur.next.val == mark.next.val:#当下一个链表值与前一个值相等时，并且cur有下一个链表值时，cur就一直往下走
                cur = cur.next
            if cur != mark.next: # 当上一个循环结束时，判断是否cur的位置有变化，如果有变化，则说明mark的下一个到cur节点中的所有节点都相等
                if cur == mark: # 如果cur 和 mark 相等，则说明已经到最后一位了，则直接返回head
                    return head
                if mark.next == head: # 如果mark还在头节点，我们需要改变头指针
                    if cur.next is None: #在这种情况下，如果cur的下个节点是空节点，则说明这个链表将会被删除，返回空
                        return None
                    else:
                        head = cur.next # 否则head指针指向cur的下一个节点并且把mark指针移动到head
                        mark.next = head
                elif not cur: # 如果mark不是head，并且cur是空，则说明从mark下一个到最后的元素都一样，把mark后面全部删除
                    mark.next = None
                else:
                    mark.next = cur.next #否则就是最正常的情况，中间段是重复的，则把mark指向cur下一个节点
                    cur = mark.next #cur指向mark
            else:
                mark = mark.next #如果没遇到相同节点，则Mark和cur一起向后移
                cur = cur.next
        return head #返回head


# In[ ]:


""""""#A message containing letters from A-Z can be encoded into numbers using the following mapping:
'''A' -> "1"
'B' -> "2"
...
'Z' -> "26"'''
'''动态规划'''
class Solution:
    def numDecodings(self, s: str) -> int:
        f_0 = 1
        i = 1
        f_1 = 0
        if len(s)== 0: #当该字符串为空时，解码对应的也是空，这也算一种解码方法，所以返回1
            return 1
        if len(s) == 1 : #当该字符串长度为1，如果该字符是0，则返回0因为无对应解码方法，如果该字符不是0，那一定有且仅有一种解法方式 
            if s[0] == "0":
                return 0
            else:
                return 1 
        if len(s) >=2: # 当字符串长度大于等于2，如果该字符串第一个元素是0，则f_1为0 否则为1
            if s[0] == "0":
                f_1 = 0
            else:
                f_1 = 1 
        f_next = 0
        while i < len(s):
            if s[i] != "0" and int(s[i-1:i+1]) <= 26 and s[i-1] != "0": #从第二个字符开始判断，如果第二个字符不是0并且前一个与本字符代表的数字小于等于26并且前一个字符不是0，则我们有取一个字符的情况也有取两个字符的情况
                f_next = f_1+f_0 #取一个字符的情况的解码方式为f(x-1)，取两个字符的情况的解码方式为f(x-2),即为f_0 和 f_1,将两个相加
                f_0 = f_1
                f_1 = f_next
            if (int(s[i-1:i+1]) > 26 or  s[i-1] == "0") and s[i] != "0": #当如果前一个字符和该字符代表的数字大于26或者前一个字符为0，则我们没有取两个字符的情况，如果该字符不是0，我们可以取一个字符
                f_next = f_1 #所以这一位的总体情况个数位f(x-1)
                f_0 = f_1
                f_1 = f_next
            if int(s[i-1:i+1]) <= 26 and s[i-1] != "0" and s[i] == "0": #如果是有存在两种字符的情况，但是当前字符为0，则我们不能只取一个字符，只能同时解码两个字符
                f_next = f_0 #所以这一位的总体解码方式为f(x-2)
                f_0 = f_1
                f_1 = f_next
            if (int(s[i-1:i+1]) > 26 or  s[i-1] == "0") and s[i] == "0": #如果该字符是0且前一位字符要么是0要么两个字符表达的数字>26则返回 0，因为这种情况下，该字符串无法表示任何一种解码方式
                return 0
            i += 1 
        return f_next #返回f_next


# In[ ]:


#Given an array of meeting time intervals where intervals[i] = [starti, endi], determine if a person could attend all meetings.
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:#保证没有overlap的数组
        intervals.sort() #先排序
        i = 0
        while i+1 < len(intervals):
            if intervals[i][1]> intervals[i+1][0]: #如果前一个时间段尾部大于后一个时间段头部，则返回false
                return False

            i += 1 


        return True
    
'''时间复杂度：O(n)'''
    


# In[ ]:


'''Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.'''

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int: 
        intervals.sort()
        i = 0 
        count = 0 
        while i+1 < len(intervals):
            if intervals[i][1]>intervals[i+1][0]: #在前一个右边界大于后一个左边界时，因为我们要删除元组，我们不需要移动数组指针
                if intervals[i][1] > intervals[i+1][1] : #如果前一个后边界大于后一个后边界时，将前一个数组删除，个数加一
                    intervals.pop(i)
                    count += 1 
                elif intervals[i][1] <=intervals[i+1][1]: #如果前一个右边界小于等于后一个左边界，删除后一个数组，个数加一
                    intervals.pop(i+1)
                    count += 1 
            else: #否则我们需要移动数组指针
                i+= 1 
        return count
    


'''时间复杂度：O(n)'''

'''改进，不用pop():'''

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        left = 0 
        right = 1  
        count = 0 
        while right < len(intervals):
            if intervals[left][1]>intervals[right][0]:
                if intervals[left][1] > intervals[right][1]:
                    left = right
                    right += 1 
                    count += 1 
                elif intervals[left][1] <= intervals[right][1]:
                    right += 1 
                    count += 1 
            else:
                left = right
                right += 1 
        return count


# In[11]:


'''There are some spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter, and hence the x-coordinates of start and end of the diameter suffice. The start is always smaller than the end.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with xstart and xend bursts by an arrow shot at x if xstart ≤ x ≤ xend. There is no limit to the number of arrows that can be shot. An arrow once shot keeps traveling up infinitely.

Given an array points where points[i] = [xstart, xend], return the minimum number of arrows that must be shot to burst all balloons.

'''
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort()
        i = 0 
        count = 0 
        k = -1 
        if len(points) ==1 : #如果就一个ballon，返回一
            return 1 
        while i+1 < len(points):#遍历数组
            if points[i][1]>=points[i+1][0]: #当前一个ballon和后一个有重合时
                k = i+1  #从当前ballon下一个开始遍历
                min_end = points[k][1]
                while k < len(points) and points[i][1]>=points[k][0]: #遍历条件为之后的ballon的左边界都要小于遍历头部的ballon的右边界
                    min_end = min(min_end,points[k][1]) #这里我们需要考虑虽然overlap就可以一个arrow 解决，但是要注意overlap只能解决所有元组交集的部分，所以overlap的决定因素是所有进入该循环的最小右边界要大于所有元组的左边界
                    if min_end < points[k][0]:#如果遇到有小于左边界的，直接退出
                        break
                    k += 1 
                i = k #将i指针指向overlap的后一个ballon
                count += 1  #再该情况下，只需要一发arrow
            else:
                count +=1  #否则每次都要一发
                i += 1  
        if (i < len(points) and points[i-1][1]< points[i][0]) or k < len(points) :#因为上一个遍历结束时，如果最后一个ballon和之前的没有overlap，则我们需要再次考虑一下最后一个ballon
            count += 1    
        return count 
    
    
    
'''时间复杂度：O(n^2)'''

'''降低时间复杂度: O(n)，空间复杂度: O(1)'''
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort()
        print(points)
        i = 0
        if len(points) <2 :
            return 1
        max_right = points[0][1]
        i = 1 
        count = 0 
        while i < len(points):
            if max_right >= points[i][0]:
                max_right = min(max_right, points[i][1])
                if i == len(points)-1:
                    count += 1   
                i += 1 
            else:
                if i == len(points) - 1:
                    count += 2 
                    i += 1 
                else:
                    count += 1 
                    max_right = points[i][1]
                    i += 1 
        return count 


# In[ ]:


#Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head1 = l1
        head2 = l2 
        res_node = ListNode()
        head = res_node
        while head1 and head2:
            if head1.val < head2.val:
                head.next = head1
                head1 = head1.next
                head = head.next
            else:
                head.next = head2
                head2 = head2.next
                head = head.next
        if head1 and not head2:
            head.next = head1
        elif head2 and not head1:
            head.next = head2
        return res_node.next
    
'''时间复杂度：O(n)'''


# In[ ]:


#You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

#Merge all the linked-lists into one sorted linked-list and return it.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def merger(list_1, list_2):
            new_node = ListNode()
            h1 = list_1
            h2 = list_2
            h = new_node
            while h1 and h2:
                if h1.val < h2.val:
                    h.next, h1 = h1, h1.next
                else:
                    h.next, h2 = h2, h2.next
                h = h.next          
            if not h1:
                h.next = h2
            if not h2:
                h.next = h1
            return new_node.next 
        if lists == []:
            return None
        if len(lists) == 1:
            return lists[0]   
        while len(lists) != 1:
            lists[0] = merger(lists[0],lists[1])
            lists.pop(1)
        return lists[0]

'''时间复杂度：O(n^2)'''


# In[ ]:


'''Your country has an infinite number of lakes. Initially, all the lakes are empty, but when it rains over the nth lake, the nth lake becomes full of water. If it rains over a lake which is full of water, there will be a flood. Your goal is to avoid the flood in any lake.

Given an integer array rains where:

rains[i] > 0 means there will be rains over the rains[i] lake.
rains[i] == 0 means there are no rains this day and you can choose one lake this day and dry it.
Return an array ans where:

ans.length == rains.length
ans[i] == -1 if rains[i] > 0.
ans[i] is the lake you choose to dry in the ith day if rains[i] == 0.
If there are multiple valid answers return any of them. If it is impossible to avoid flood return an empty array.

Notice that if you chose to dry a full lake, it becomes empty, but if you chose to dry an empty lake, nothing changes. (see example 4)
'''

'''Input: rains = [1,2,3,4]
Output: [-1,-1,-1,-1]
Explanation: After the first day full lakes are [1]
After the second day full lakes are [1,2]
After the third day full lakes are [1,2,3]
After the fourth day full lakes are [1,2,3,4]
There's no day to dry any lake and there is no flood in any lake.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/avoid-flood-in-the-city
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。'''


class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        wait_dic = dict() #定义一个字典用于记录注满水的湖号，键值对中，键为注满水的湖号，值为下雨日期
        mark = [] #标记数组用于记录哪一天是晴天可以dry
        ans = [1] * len(rains) #定义一个与rains数组等长的结果数组，先将每个位置设为1，目的是，如果有多余的晴天，可以不用再去赋值
        i = 0  
        while i < len(rains):#遍历rains数组
            if rains[i] != 0: #当当天不是晴天
                ans[i] = -1 # 当天不能dry，所以对应结果数组中是-1
                if rains[i] in wait_dic: # 如果当天要注水的湖泊已经有水
                    if mark != [] and mark[-1]>wait_dic[rains[i]]:#我们需要判断一下晴天数组，首先要满足晴天数组不能为空，另一方面，至少要满足晴天数组的最后一个元素的日期是要在第一次当前湖泊注满水的日期之后，如果在之前，湖泊还没注满水是没法晾干的
                        j = 0 #若以上条件均满足，我们需要找到该数组中离第一次湖泊注满水后的最近的晴天日期，通过一个loop实现
                        while j < len(mark): 
                            if mark[j] > wait_dic[rains[i]]:#当找到该晴天日期时，在那个晴天把该湖泊晒干，因为晴天用完，晴天数组要把那天给去掉
                                ans[mark[j]] = rains[i]
                                mark.pop(j)
                                wait_dic[rains[i]] = i #同时要更新新的该湖泊注水日期，即为当前下标
                                break #终止循环
                            j += 1 
                    else:
                        return [] #如果没有这样一个晴天日期，我们无法避免flood，直接返回[]
                else:
                    wait_dic[rains[i]] = i #如果当天注水湖泊没水，我们把它注满水，并且将湖泊编号与日期加入字典
                     
            else:            #如果那天是晴天，则记录到晴天数组
                mark.append(i)
            i += 1 
        return ans #返回结果数组
    
'''时间复杂度：O(n^2)'''


class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        dic = dict()
        dic_pos_lft = dict()
        dic_pos_rgt = dict()
        arr= []
        heap = []
        for i in range(len(rains)):
            if rains[i] == 0:
                continue
            if rains[i] in dic:
                tmp = [dic[rains[i]], i]
                heapq.heappush(heap, i)
                dic_pos_lft[i] = dic[rains[i]]
                dic_pos_rgt[dic[rains[i]]] = i
                dic[rains[i]] = i      
                arr.append(tmp)
            else:
                dic[rains[i]] = i
        arr.sort()
        dic_mark = dict()
        res = [-1 for i in range(len(rains))] 
        for i in range(len(rains)):
            if rains[i] != 0:
                if rains[i] in dic_mark:
                    if dic_mark[rains[i]] == 0:
                        dic_mark[rains[i]] += 1 
                    else:
                        return []
                else:

                    dic_mark[rains[i]] = 1 
                    continue
            elif rains[i] == 0:
                while heap and heap[0] not in dic_pos_lft:
                    heapq.heappop(heap)
                if not heap:
                    res[i] = 1
                else:
                    if i < dic_pos_lft[heap[0]]:
                        minn = 1000000
                        for j in range(i):
                            if j in dic_pos_rgt:
                                minn = min(minn, dic_pos_rgt[j])
                        if minn != 1000000:
                       
                            res[i] = rains[minn]
                            dic_pos_rgt.pop(dic_pos_lft[minn])
                            dic_pos_lft.pop(minn)
                            dic_mark[rains[minn]] -= 1 
                        else:
                            res[i] = 1
                    else:
                        res[i] = rains[heap[0]]
                        tmp = heapq.heappop(heap)
                        dic_pos_rgt.pop(dic_pos_lft[tmp])
                        dic_pos_lft.pop(tmp)
                        dic_mark[rains[tmp]] -= 1 
        return res
    

'''清晰的步骤'''
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        dic_raindays = defaultdict(deque) #创建一个字典，里面存放队列，该字典用于存放每个湖泊下雨的日期，用字典的形式存储
        dic_full = defaultdict(None) #创建一个字典用于标记当前湖泊是否已经满了
        for i in range(len(rains)): #遍历一遍数组
            if rains[i] in dic_raindays: # 每次遇到某个湖泊下雨就把该湖泊的下雨日期加到对应的湖泊编号的的队列中去
                dic_raindays[rains[i]].append(i)
            else:
                dic_raindays[rains[i]].append(i)
            dic_full[rains[i]]= False #同时将湖泊默认为没满，放到dic_full字典中去

        res = [-1 for i in range(len(rains))] # 创建一个结果数组先初始化为全是-1
        heap_nextrain = [] # 创建一个堆，用于存放下次最早下雨的湖泊，用一个tuple存放（下次下雨日期，对应湖泊编号）
        for i in range(len(rains)): # 再遍历一遍 rains
            if rains[i] != 0: #如果当前天不是晴天
                if dic_full[rains[i]] == True: #如果该湖泊已经满了，则我们直接返回空
                    return []
                else: #如果该湖泊还么满
                    dic_raindays[rains[i]].popleft() #我们首先pop掉这一次该湖泊下雨的
                    if dic_raindays[rains[i]]: #如果该湖泊下一次还有雨的话，我们记录下来放到heap中
                        heapq.heappush(heap_nextrain,(dic_raindays[rains[i]][0], rains[i]))
                    dic_full[rains[i]] = True #同时标记这个湖泊已经满了
            else:#如果是晴天
                if heap_nextrain and heap_nextrain[0][0] > i and dic_full[heap_nextrain[0][1]] == True: #我们看堆里面是否有元素，并且该元素对应的湖泊是否已经满了
                    res[i] = heap_nextrain[0][1]#把这个胡晒干
                    dic_full[heap_nextrain[0][1]] = False  #改为false，并且从堆中弹出
                    heapq.heappop(heap_nextrain)
                else:
                    res[i] = 1 #否则我们这天随便晒一个湖

        return res


# In[ ]:


#Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

import queue
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        q = queue.Queue() #创建一个先进先出的队列
        if root: #如果root不是空，则将root添加到队列中
            q.put(root)
        res = [] 
        while q.qsize() > 0: #当队列不为空
            traversal = [] #新建一个层级遍历数组
            n = q.qsize() #取队列元素个数，即当前层节点个数
            for i in range(n): #遍历循环当前层
                node = q.get() #利用先进先出，不断得到第一个第一个节点
                traversal.append(node.val) #将第一个节点传递给层级遍历数组
                if node.left: #如果当前遍历节点有左右节点，按先左再右原则添加节点
                    q.put(node.left)
                if node.right:
                    q.put(node.right)
            res.append(traversal) #将层级遍历结果给res
        return res


# In[ ]:


'''Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.'''

#自己的想法：
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool: #mark =1 是右子树，mark = 0 是左子树
        def dfs(node, limit, mark):#递归函数中有三个参数，第一个为当前的节点，第二个为当前树的节点值，第三个为标记是左子树还是右子树，目的是如果是左子树，它的所有右节点的值都要小于根节点的值
            #如果是右子树，它所有的左节点的值都要大于根节点
            if not node.left  and not node.right: #如果节点左右两边都为空，则直接返回true，因为说明我们已经遍历到叶子结点了，之前的所有节点都满足条件
                return True
            elif node.left and not node.right: #如果左边有节点，右面没有，则取左节点的值
                node_left = node.left.val
                if node_left < node.val: #如果左节点值小于他的父亲节点
                    if node.val != limit: #并且如果父亲节点不是根节点的话
                        if mark == 1: #因为这是父子节点的左节点，我们比较关心这个节点是不是右子树上面的，所以判断mark是不是1
                            if node_left > limit: #如果是右子树，我们需要判断该节点的值是不是杨哥大于根节点的值，是的进入下一层递归
                                return dfs(node.left, limit,1)
                            else: #否则直接返回false
                                return False
                        else:
                            return dfs(node.left,limit, 0) #如果不是右子树，那就是左子树，那就无所谓了，直接进入下一层递归
                    else:
                        return dfs(node.left,limit, 0) #如果父子节点是根节点，我们不需要进行左右子树判断，并且这个时候mark一定是0，因为我们这个是左节点，就意味着下面的树都是左子树
                else:
                    return False #如果左子树大于父亲节点，返回false
                
            elif node.right and not node.left: #右子树情况与左子树相同
                node_right = node.right.val
                if node_right > node.val:
                    if node.val != limit:
                        if mark ==0:
                            if node_right < limit:
                                return dfs(node.right,limit, 0)
                            else:
                                return False
                        else:
                            return dfs(node.right,limit, 1)
                    else:
                        return dfs(node.right,limit, 1)
                else:
                    return False 
            elif node.right and node.left: #如果父亲节点有左右子树
                node_left = node.left.val #取左子树与右子树的值
                node_right = node.right.val
                if node_left < node.val < node_right: #如果父亲节点的值是在左儿子节点和右儿子节点之间
                    if node.val != limit: #我们要判断父亲节点是不是根节点，如果是根节点，它的左儿子节点就会形成左子树，右儿子节点就会形成右子树
                        if mark == 0: #如果不是根节点，如果是左子树下的节点
                            if node_right < limit: #左子树右侧的节点是比较重要的，需要严格小于根节点， 如果小于根节点，我们就继续遍历，注意mark 都是0
                                return dfs(node.right,limit, 0) and dfs(node.left, limit,0)
                            else:
                                return False #否则返回false
                        else:
                            if node_left > limit: #如果是右子树，所有的左节点都要严格大于根节点
                                return dfs(node.right, limit,1) and dfs(node.left, limit,1) #如果是的，则继续遍历右子树，注意mark = 1 
                            else:
                                return False #否则返回false
                    else:
                        if mark == 0: #如果是根节点，那就开始创建左子树右子树，分别遍历
                            return dfs(node.left, limit,0)
                        else:
                            return dfs(node.right,limit, 1) 
                else:
                    return False 
        if root is None:
            return True
        return dfs(root, root.val, 0) and dfs(root, root.val, 1) and self.isValidBST(root.left) and self.isValidBST(root.right)
    
    
# 大神的方法：method 2
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def dfs(node, lower, upper):
            if not node:
                return True

            val = node.val
            if val <= lower or val >= upper:
                return False 

            if not dfs(node.right, val, upper):
                return False
            if not dfs(node.left, lower, val):
                return False 

            return True

        return dfs(root, -1000000000000, 100000000000000)


# In[ ]:


#You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

#Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

'''动态规划！遇到需要从一个数列头走到尾，并且求某种走的方法的问题，都会涉及动态规划'''

'''该问题打劫问题，当房屋数量为1个时只能打劫那一个，如果有两个，就打劫金额最大的那个，这两个情况将会作为边界条件下面进行应用

当两家以上的房屋时，一路打劫到第i家的最大金额可以如下计算：

1.如果我们要打劫第i家，那我们不能打劫第i-1家，所以这种情况下，我们的打劫金额为前i-2家的最大打劫金额加上第i家的金额

2. 如果我们不打劫第i家，我们的最大打劫金额为前i-1家的最大打劫金额。

我们可以创建一个最大金额打劫数组，用于记录到每一家为止的最大打劫金额，记为dp数组'''

class Solution:
    def rob(self, nums: List[int]) -> int:
        f_0 = nums[0]
        if len(nums)<2:#如果数组长度小于2，则直接返回第一家的金额
            return nums[0]
        f_1 = max(nums[0], nums[1]) #如果只有两家，最大金额为其中最大的那个

        n = len(nums) 
        i = 2
        dp = [0]*len(nums)  # 创建一个最大打劫金额数组，长度和房屋数组长度相同
        dp[0] = f_0
        dp[1] = f_1
        while i < n: #遍历到最后一家
            dp[i] = max(dp[i-2]+ nums[i], dp[i-1]) #到第i家时最大打劫金额为打劫第i家和前i-2家最大打劫金额 和不打劫第i家，打劫到第i-1家时最大打劫金额之和中的最大值
            i += 1 

        print(dp)
        return dp[n-1]


# In[ ]:


#Given the head of a singly linked list, reverse the list, and return the reversed list.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head is None:
            return None
        if head.next is None:
            return head
        cur = head.next
        head.next = None #这里一定要先将头部的next置为空，因为循环中会跳过头部节点
        while cur.next:
            tmp = cur.next
            cur.next = head
            head = cur
            cur = tmp
           

        cur.next = head
        head = cur
        cur = head
     
        return head
     


# In[ ]:


'''A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.'''



class Trie:

    def __init__(self):
        self.dic = dict()

    def insert(self, word: str) -> None:
        if word not in self.dic:
            self.dic[word] = 1

    def search(self, word: str) -> bool:
        if word in self.dic:
            return True

        else:
            return False

    def startsWith(self, prefix: str) -> bool:
        for key, value in self.dic.items():
            if prefix[0] == key[0]:
                i = 0 
                while i < len(prefix) and i < len(key):
                    if prefix[i] != key[i]:
                        break
                    i += 1 

                if i>= len(prefix):
                    return True

        return False 
    

'''**字典树** ：'''
class Trie:
    def __init__(self):
        self.children = [None] *26
        self.isEnd = False
    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ch = ord(ch)-ord("a")
            if not node.children[ch]:
                node.children[ch] = Trie() #children数组中每个元素存储的是下个字母的地址，每个children其实就存储了一个字母的地址，其他的都是空的
            node = node.children[ch] #将node指针移动到下个字母的地址位置
        node.isEnd = True #当到一个字符串最后时，将结尾标记置为true
    def search(self, word: str) -> bool:
        node = self
        for ch in word:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return False
            node = node.children[ch]
        if node.isEnd:
            return True
        else:
            return False
    def startsWith(self, prefix: str) -> bool:
        node = self
        for ch in prefix:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return False
            node = node.children[ch]
        return True


# In[ ]:


#two sum 
'''Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.'''
'''哈希表，时间复杂度O(n), 空间复杂度O(n)'''

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}

        for index, value in enumerate(nums):
            if target - value in dic:
                return [dic[target-value], index]

            dic[value] = index


# In[ ]:


#three sum:
#Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

#Notice that the solution set must not contain duplicate triplets.

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        print(nums)
        if  nums == []:
            return []
        if nums[0] >0 or nums[-1] < 0:
            return []
        res = []
        n = len(nums)
        for i in range(n):
            if i ==0 or nums[i] != nums[i-1]:
                third = n-1
                target = - nums[i]
                for j in range(i+1, n):
                    if j == i+1 or nums[j] != nums[j-1]:
                        while j < third and nums[third]+nums[j] > target:
                            third -= 1 
                        if j == third:
                            break
                        if nums[third]+nums[j] == target:
                            res.append([-target, nums[j], nums[third]])
        return res
    
'''个人感觉更好理解：'''    

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        l = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i]>0:
                break
            if i>0 and nums[i]== nums[i-1]:
                continue
            j=i+1
            k = len(nums)-1
            while k>j:
                if nums[j]+nums[k]+nums[i]==0:
                    l.append([nums[j],nums[k],nums[i]])
                    while j<k and nums[j] == nums[j+1]:
                        j = j+1
                    while j<k and nums[k] == nums[k-1]:
                        k = k-1
                
                    j = j+1
                    k = k-1
                elif nums[j] +nums[k]<-nums[i]:
                    j=j+1
                else:
                    k=k-1
        return l
    
'''哈希实现：'''

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        dic = dict() #创建字典
        dic_dup_j = set()#创建集合用于存储重复tuple
        nums.sort() #能否不用sort？
        if len(nums) <3:#长度小于三不用判断了
            return []
        for i in range(len(nums)): #先把所有元素放入数组
            dic[nums[i]] = i 
        res = []  
        for j in range(0, len(nums)-1):
            tmp = [] #创建一个临时数组，存放没个结果
            if j > 0 and nums[j] == nums[j-1]: #如果下个和之前一个一样不遍历，投机取巧了
                continue
            for k in range(j+1,len(nums)):
                if k > j+1 and nums[k] == nums[k-1]:#如果下个和之前一个一样不遍历，投机取巧了
                    continue
                if -nums[j]-nums[k] in dic  and   j != dic[-nums[j]-nums[k]] and k != dic[-nums[j]-nums[k]] : #去重，如果之前已经遍历过同样组合则跳过
                    tmp = [-nums[j]-nums[k], nums[j], nums[k]]#创造一个结果
                    tmp.sort()#排个序，判断该结果是否已经在res中，能否优化？
                    if tmp  not in res:#O(n/3)时间复杂度
                        res.append(tmp)
        return res


# In[ ]:


'''Given an unsorted integer array nums, find the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses constant extra space.'''

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        dic = dict()
        for i in range(len(nums)): #将与数组等大的从1开始的连续整数放入字典中
            dic[i+1] = False 
        for i in range(len(nums)): #一个for循环
            if nums[i] in dic: #如果该数字在字典中，则把value置为true
                dic[nums[i]] = True

        for key, value in dic.items(): #遍历字典，如果找到了第一个值为false，则返回，停止遍历
            if value == False:
                return key
        return len(nums)+1 #否则返回长度的下一个数字
'''时间复杂度：O(n),空间复杂度：O(n)'''


# In[ ]:


'''*Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?'''


class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        summ = int((n+1)*n/2)
        arr_sum = 0 
        for i in range(n):
            arr_sum = arr_sum + nums[i]

        return summ - arr_sum
'''时间复杂度：O(n),空间复杂度：O(1)'''


# In[ ]:


'''*On a 2D plane, there are n points with integer coordinates points[i] = [xi, yi]. Return the minimum time in seconds to visit all the points in the order given by points.

You can move according to these rules:

In 1 second, you can either:
move vertically by one unit,
move horizontally by one unit, or
move diagonally sqrt(2) units (in other words, move one unit vertically then one unit horizontally in 1 second).
You have to visit the points in the same order as they appear in the array.
You are allowed to pass through points that appear later in the order, but these do not count as visits.'''

class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        direction = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (1,1),(-1,1),(-1,-1)]
        n = len(points)
        if n == 1:
            return 0
        i = 0 
        summ = 0 
        while i+1< n:
            diag_step =  min(abs(points[i][0] - points[i+1][0]), abs(points[i][1]-points[i+1][1]))
            dis_x = abs(points[i][0]-points[i+1][0])
            dis_y = abs(points[i][1]-points[i+1][1])
            step = abs(dis_x-dis_y)
            summ = summ + diag_step + step
            i += 1 

        return summ
'''时间复杂度：O(n),空间复杂度：O(1)'''


# In[ ]:


'''*You are given a map of a server center, represented as a m * n integer matrix grid, where 1 means that on that cell there is a server and 0 means that it is no server. Two servers are said to communicate if they are on the same row or on the same column.

Return the number of servers that communicate with any other server.'''

class Solution:
    def countServers(self, grid: List[List[int]]) -> int:
        dic = dict()
        direction = [(0,1), (1,0)]
        for row in range(len(grid)): #循环遍历网格表
            for column in range(len(grid[0])):
                tmp_row = row
                tmp_col = column
                if grid[tmp_row][tmp_col] == 1:# 某个元素是1，则开始围绕这个圆元素寻找所在行所在列还有多少个1
                    tmp = 0    #标记变量，用于标记是否这个元素是个孤立sever，所在行所在列没有其他1，如果没有的话，我们这个sever不能算入
                    for d_row, d_col in direction: # 遍历方向数组
                        if d_row == 1: #如果是行数加1，下面是遍历所在列
                            length = len(grid)
                        else:
                            length = len(grid[0]) #否则则是遍历所在行
                        for n in range(1, length):#遍历当前列或行
                            new_row, new_col = tmp_row + n*d_row, tmp_col + n*d_col
                            if 0<= new_row <len(grid) and 0 <= new_col < len(grid[0]): #判断元素是否在区间范围内
                                if grid[new_row][new_col] == 1: #如果元素是1，并且该元素不在字典中，则说明可以communicate，放入数组
                                    if (new_row, new_col) not in dic: 
                                        dic[(new_row, new_col)] = 1 
                                    if tmp == 0: #如果是第一次进入该循环则tmp +1
                                        tmp += 1 
                    if tmp != 0 : #判断是否tmp有变化，有变化说明这个不是孤立的
                        if (tmp_row, tmp_col) not in dic:
                            dic[(tmp_row, tmp_col)] = 1

        return len(dic) 
'''时间复杂度：O(n^4), 空间复杂度：O(n)'''

class Solution:
    def countServers(self, grid: List[List[int]]) -> int:
        dic_row = dict()
        dic_col = dict()
        for i in range(len(grid)):
            count = 0
            for col in grid[i]:
                if col == 1:
                    count += 1 
            dic_row[i] = count
        for j in range(len(grid[0])):
            count = 0
            for i in range(len(grid)):
                if grid[i][j] == 1:
                    count += 1 
            dic_col[j] = count
        summ = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (dic_row[i] > 1 or dic_col[j] > 1) and grid[i][j] ==1:
                    summ += 1 
        return summ
    
'''时间复杂度O(n^2), 空间复杂度O(n)'''


# In[ ]:


'''Tic-tac-toe is played by two players A and B on a 3 x 3 grid.

Here are the rules of Tic-Tac-Toe:

Players take turns placing characters into empty squares (" ").
The first player A always places "X" characters, while the second player B always places "O" characters.
"X" and "O" characters are always placed into empty squares, never on filled ones.
The game ends when there are 3 of the same (non-empty) character filling any row, column, or diagonal.
The game also ends if all squares are non-empty.
No more moves can be played if the game is over.
Given an array moves where each element is another array of size 2 corresponding to the row and column of the grid where they mark their respective character in the order in which A and B play.

Return the winner of the game if it exists (A or B), in case the game ends in a draw return "Draw", if there are still movements to play return "Pending".

You can assume that moves is valid (It follows the rules of Tic-Tac-Toe), the grid is initially empty and A will play first.
'''

class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        grid = [[-1 for _ in range(3)] for j in range(3)]
        print(grid)
        def judge(grid, index):
            if index == [0,0]:
                if grid[0][0] == grid[1][1] ==grid[2][2] or grid[0][0] == grid[1][0] == grid[2][0] or grid[0][0] == grid[0][1] == grid[0][2]:
                    return True
                else:
                    return False
            if index == [0, 1]:
                if grid[0][1] == grid[0][0] == grid[0][2] or grid[0][1] == grid[1][1] == grid[2][1]:
                    return True
                else:
                    return False
            if index == [0,2]:
                if grid[0][2] == grid[0][1] == grid[0][0] or grid[0][2] == grid[1][1] == grid[2][0] or grid[0][2] == grid[1][2] == grid[2][2]:
                    return True
                else:
                    return False
            if index == [1,0]:
                if grid[1][0] == grid[0][0] == grid[2][0] or grid[1][0] == grid[1][1] == grid[1][2]:
                    return True
                else:
                    return False
            if index == [1, 1]:
                if grid[1][1] == grid[0][0] == grid[2][2] or grid[0][1] == grid[1][1] == grid[2][1] or grid[1][1] == grid[0][2] == grid[2][0] or grid[1][1] == grid[1][0] == grid[1][2]:
                    return True
                else:
                    return False
            if index == [1,2]:
                if grid[1][2] == grid[1][1] == grid[1][0] or grid[1][2] == grid[0][2] == grid[2][2]:
                    return True

                else:
                    return False

            if index == [2,0]:
                if grid[2][0] == grid[1][0] == grid[0][0] or grid[2][0] == grid[2][1] == grid[2][2] or grid[2][0] == grid[1][1] == grid[0][2]:
                    return True
                else:
                    return False
                
            if index == [2,1]:
                if grid[2][1] == grid[2][0] == grid[2][2] or grid[2][1] == grid[1][1] == grid[0][1]:
                    return True
                else:
                    return False
            if index == [2,2]:
                if grid[2][2] == grid[1][1] == grid[0][0] or grid[2][2] == grid[1][2] == grid[0][2] or grid[2][0] == grid[2][1] == grid[2][2]:
                    return True
                else:
                    return False
            return False 

        player = 0
        judger = False
        for i in range(len(moves)):
            grid[moves[i][0]][moves[i][1]] = player
            judger = judge(grid, moves[i])
            if judger == True:
                if player ==0:
                    return "A"
                else:
                    return "B"

            if player == 0:
                player = 1
            else:
                player = 0
        if len(moves) < len(grid)*3:
            return "Pending"

        if len(moves) == len(grid)*3:
            return "Draw"
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''You are playing the Bulls and Cows game with your friend.

You write down a secret number and ask your friend to guess what the number is. When your friend makes a guess, you provide a hint with the following info:

The number of "bulls", which are digits in the guess that are in the correct position.
The number of "cows", which are digits in the guess that are in your secret number but are located in the wrong position. Specifically, the non-bull digits in the guess that could be rearranged such that they become bulls.
Given the secret number secret and your friend's guess guess, return the hint for your friend's guess.

The hint should be formatted as "xAyB", where x is the number of bulls and y is the number of cows. Note that both secret and guess may contain duplicate digits.'''


'''自己想'''
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        count_A = 0
        count_B = 0
        secret = list(secret)
        guess = list(guess)
        for i in range(len(secret)):
            for j in range(len(guess)):
                if i == j and secret[i] == guess[j]:
                    count_A += 1
                    secret[i] = "s"
                    guess[i] = "g"
        for i in range(len(secret)):
            for j in range(len(guess)):
                if i != j and secret[i] == guess[j]:
                    count_B += 1 
                    secret[i] = "s"
                    guess[j] = "g"
        res = ""+ str(count_A) + "A" + str(count_B) + "B"
        return res
'''时间复杂度：O(n^2), 空间复杂度：O(n)'''
    
'''看答案'''   
#哈希表： 先遍历一边字符串, 对应相等的则说明是bull，否则，将元素存入字典，如果有重复元素则加1，将guess中的没有对应上的元素存入新的数组，之后再一次遍历这个新数组，看看字典里面有没有，有则字典里面数量-1，同时cow +1   
class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        count_A = 0
        count_B = 0
        new_guess = []
        dic = dict()
        for i in range(len(secret)):
            if secret[i] == guess[i]:
                count_A += 1
            else:
                if secret[i] in dic:
                    dic[secret[i]] += 1 
                else:
                    dic[secret[i]] = 1 
                new_guess.append(guess[i])
                
        for i in range(len(new_guess)):
            if new_guess[i] in dic:
                if dic[new_guess[i]] != 0:
                    dic[new_guess[i]] -= 1 
                    count_B += 1 
        res = ""+ str(count_A) + "A" + str(count_B) + "B"
        return res
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''**Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.
'''
'''看答案''' # 快速排序修改版
class Solution: [7,6,5,4,3,2,1]
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quicksort(arr, left, right): #定义一个快速排序函数，左指针指向目标段的开头，右指针指向目标段结尾
            j = right
            i = left
            while True:
                while j > left and arr[j] >= arr[left]: #当在j在左指针右侧的前提下，如果j所对应的元素大于我们的flag值则向左走，因为我们想要找第一个小于flag值的元素
                    j -= 1 
                while i < right and arr[i] <= arr[left]: # 当i在右指针左侧，如果i所对应的元素小于我们的flag值则向右走，因为我们想要找到第一个大于flag值的元素
                    i += 1 
                if i >= j: #如果左指针走到了右指针及右指针右侧，则中断
                    break
                arr[i], arr[j] = arr[j], arr[i] #期间不断交换左右指针满足条件的值
            arr[left], arr[j] = arr[j], arr[left] #最后遍历完一定是i>= j， 所以我们要交换flag与当前所指的值， 问题：为什么是j？
            return j #返回flag的最后的位置
        tmp = -1
        target = len(nums) - k  
        right = len(nums)-1
        left = 0 
        while tmp != target:
            tmp = quicksort(nums, left, right)
            if tmp < target: #当flag的位置在目标的左侧时，说明目标在右侧数组中，我们只再次遍历右侧数组
                left = tmp + 1  
            elif tmp > target: #否则我们遍历左侧数组
                right = tmp -1  

        return nums[tmp]

'''时间复杂度：O(n^2), 空间复杂度：O(1)'''



import random
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        target = len(nums) - k+1
        def quickselect(arr, target): #快搜是快排的特殊版本，只搜索一侧
            if arr == []:
                return []
            if len(arr) <= 1:  
                return arr[0]
            pivot = random.choice(arr) #随机选一个元素为pivot
            left = [x for x in arr if x <pivot] #将当前数组中全部小于该元素的放入到一个列表中
            mid = [x for x in  arr if x == pivot] #将当前数组中全部等于该元素的放入的一个列表中
            right = [ x for x in arr if x> pivot] #将当前数组中全部大于该元素的放入到一个列表中
            if target <= len(left): #如果目标位置是在左侧列表，则下面遍历左侧列表
                area = left
            elif len(left) < target <= len(left) + len(mid): #如果目标位置在中间列表，则说明我们已经找到了，直接返回
                return pivot
            else:
                target = target - len(left) -len(mid) #否则我们遍历右侧列表
                area = right 
            return quickselect(area, target) #递归遍历选择的列表
        return quickselect(nums, target)   
    
'''时间复杂度：O(n), 空间复杂度：O(n)'''


import random
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quickselect(n, arr, left, right):
            if left <= right:  #当左侧下标小于右侧下标则一直递归
                index = partition(arr, left, right) #随机的将其中一个元素放置到它所属于的位置，并得到它的下标
                if n < index: #如果目标位置小于该下标，则递归数组的该下标的左侧
                    return quickselect(n, arr, left, index-1) 
                elif n == index: #如果该下标就是index，则返回该元素
                    return arr[index]
                else: #如果目标位置大于该下标，则递归数组的该下标的右侧
                    return quickselect(n, arr, index+1,right)
        def partition(arr,left, right): #该函数用于排序
            pivot = random.randint(left,right)#随机找一个pivot，我们需要随机找一个，而不是设定左侧右侧为pivot，否则最坏的时间复杂度将会为O(n^2),平均时间复杂度为O(nlogn)
            arr[pivot], arr[right] = arr[right], arr[pivot] #将该pivot放到最右侧
            i = left - 1
            pivot = right 
            for j in range(left, right): #遍历从左侧到倒数第二个
                if arr[j] < arr[pivot]: #将所有的比pivot小的都往前放，永远保证i下标的下一个位置的元素一定是大于pivot的
                    i += 1 
                    arr[j], arr[i] = arr[i], arr[j]
            arr[pivot], arr[i+1] = arr[i+1], arr[pivot] #最后需要将pivot换到它自己的位置
            return i + 1 #返回pivot的位置
        return quickselect(len(nums)-k, nums, 0, len(nums)-1) 

'''时间复杂度：O(n), 空间复杂度：O(1)'''


# In[18]:


'''**快速排序算法'''

def quicksort(arr, left, right):
    if left < right:
        index = partition(arr, left, right)
        quicksort(arr,left, index-1)
        quicksort(arr,index + 1, right)
    
    return arr
    

    
def partition(arr, left, right):
    pivot = right
    i = left -1
    for j in range(left, right):
        if arr[j]<arr[pivot]:
            i += 1
            swap(arr,i,j)
        
    swap(arr,pivot,i+1)
    return i+1
    
    
def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
    

    
arr = [4,3,7,1,9,10,3,2,7,7,6,3,21,5,34,2,5,6,221,6,4,32,8,2,1,4,23,6,7,444,2,378,543,2312,7,453,2]
print(quicksort(arr,0,len(arr)-1))
    


# In[ ]:


#MINHEAP 最小堆的构建：
class Heap(object):
    def MinHeapFixup(array, index):
        father = (index-1)//2
        while index >= 0:
            if array[father] > array[index]:
                array[father], array[index] = array[index], array[father]
                
            else:
                break
                
            index = father
            father = (index-1)//2


# In[ ]:


'''*23. Merge k Sorted Lists. You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.'''

#方法：可以通过库函数heapq 来实现，效率更高

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        n = len(lists)
        new_heap = [] #创建一个堆的列表
        for i in range(n):
            head = lists[i] #取每一个链表
            while head:
                heapq.heappush(new_heap, head.val) #遍历该链表，将每个节点的值push到到堆里面，堆会自动排序
                head = head.next 


        new_head = ListNode(None) #创建一个新链表
        cur = new_head
        while new_heap: #遍历这个堆
            cur.next = ListNode(heapq.heappop(new_heap)) #不断pop元素，每次都是pop出当前堆的最小值， 放入到新链表里面
            cur = cur.next


        return new_head.next #返回新链表
'''时间复杂度：O(n^2), 空间复杂度：O(n)'''


# In[12]:


'''*Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.'''

#二分查找
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        i = 0 
        n = len(nums) - 1 
        l = 1 
        res = -1 
        while l <= n: # 当右指针 >= 左指针
            mid = (l+n)//2 #找到中间位置
            cont = 0 #初始化数数
            for i in range(len(nums)): #判断在数组中有多少个数字小于mid，我们的思想是重复元素的值一定是数组中第一个数字满足如下条件：数组中所有小于等于该数字的个数大于该数字的值
                if nums[i] <= mid: #不断累加
                    cont += 1 
            if cont <= mid: #如果cont 比 中间数小，左指针向右走
                l = mid + 1 
            else:         # 否则先记录下该数字，再检测该数字前面的数字是否满足我们的条件
                res = mid
                n  = mid - 1 #右指针向左走
        return res


# In[ ]:


'''Given an array, rotate the array to the right by k steps, where k is non-negative.'''
#方法一：
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k = k % len(nums)
        left = len(nums) - k   
        for i in range(left):
            nums.append(nums[0]) #数组右侧添加第一个元素，同时数组最左边删除第一个元素
            nums.pop(0)

'''时间复杂度：O(n), 空间复杂度：O(1)'''
#方法二：全翻+自翻：
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k = k % len(nums)
        nums.reverse() #先全部翻转
        nums[:k] = list(reversed(nums[:k])) #再局部翻转，局部翻转的长度由k决定
        nums[k:] = list(reversed(nums[k:]))
'''时间复杂度：O(1), 空间复杂度：O(1)'''

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k = k % len(nums)
        def rev_array(left, right):#我们自己定义一个翻转数组
            mid = (right-left+1)//2 #我们只要遍历一半的数组就行
            for i in range(mid): #不断交换前后顺序
                nums[left+i], nums[right-i] = nums[right-i], nums[left+i]
                
        rev_array(0,len(nums)-1) #整体先翻转一边
        rev_array(0,k-1) #左侧翻转
        rev_array(k, len(nums)-1)#右侧翻转


# In[ ]:


'''You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).'''

# method 1: 严重超时
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        def single_max(left, right, prices):
            min_price = 100000000
            max_profit = 0 
            for i in range(left, right+1):
                if prices[i]< min_price:
                    min_price = prices[i]
                if prices[i] - min_price > max_profit:
                    max_profit = prices[i] - min_price
            return max_profit
        if len(prices) ==1:
            return 0
        if len(prices)==2:
            if prices[1] - prices[0] >0:
                return prices[1] - prices[0]
            else:
                return 0
        cut = 1
        max_profit_left = 0
        max_profit_right = 0
        total_profit = 0
        while cut < len(prices):
            if cut == len(prices) -1:        
                max_profit_left = max_profit_right = single_max(0,cut, prices)
                total_profit = max(total_profit, max_profit_left)
            while cut < len(prices) and prices[cut-1] >= prices[cut]:
                cut += 1 
            if cut == len(prices) :
                return total_profit
            while cut < len(prices) and prices[cut-1] < prices[cut]:
                cut += 1 
            max_profit_left = single_max(0,cut-1, prices)
            max_profit_right = single_max(cut-1, len(prices)-1, prices)
            total_profit = max(total_profit, max_profit_left+max_profit_right)
            cut += 1 
        return total_profit
    

    


# In[ ]:


'''*Given an integer n, return the least number of perfect square numbers that sum to n.

A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.'''
#动态规划问题：动态转移方程为当前的整数由小到大的依次减去square integer 后的对应的最少square integer 组合 +1. dp[i] = min(dp[i], dp[i-j*j]+1)
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [i for i in range(n+1)] #首先创造一个n+1的数组，每个数组的元素初始化为最多需要squre integer 的个数。相当于最坏情况
        for i in range(n+1): #从第一个元素开始计算当前的最小需要的square integer。 
            j = 1 
            while i - j*j >= 0:    # 后面的每一个数字的最小square integer数量是由前面的决定的
                dp[i] = min(dp[i], dp[i-j*j] + 1)
                j += 1 

        return dp[-1]


# In[ ]:


'''*
318. Maximum Product of Word Lengths
Given a string array words, return the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. If no such two words exist, return 0.'''


class Solution:
    def maxProduct(self, words: List[str]) -> int:
        maxx = 0 
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                l1 = set(words[i])   #取集合操作可以去重
                l2 = set(words[j])
                l3 = set(words[i]+words[j])   #通过比较两个字符串分别去重后的串长与两个字符串和在一起后去重后的字符串长度
                if len(l1) + len(l2) == len(l3): #如果两个长度相等，则说明这两个字符串没有公共字符
                    maxx = max(maxx, len(words[i]) * len(words[j])) #则进入比较两个字符串长度乘积是否是最大的
        return maxx 




# In[ ]:


'''*92. Reverse Linked List II
Given the head of a singly linked list and two integers left and right where left <= right, reverse the nodes of the list from position left to position right, and return the reversed list.'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        if head.next is None: #判断链表是不是只有一个节点
            return head
        
        if right - left ==0: #如果要反转的链表长度为1，则直接返回
            return head
        i = 1
        cur = head
        while i < left - 1:#找到要反转的区间的前一个节点
            cur = cur.next
            i += 1
        prev_start = cur #标记该节点
        if left == 1: #判断一下，我们是不是要从第一个节点翻转，是的话则翻转的第一个节点及前一个节点
            start = prev_start
        else:
            start = cur.next #否则标记翻转的第一个节点
        prev = start #双指针遍历翻转区间
        cur = start.next
        prev.next = None #首先设置第一个节点指向空，因为在循环时第一个节点指向的方向单独设置比较费事
        i = 0
        while i < right - left: #遍历翻转区域时，将cur指向prev，同时prev 和 cur一起往下移，tmp/cur在最后将自动指向翻转区域的下一个节点
            tmp = cur.next
            cur.next = prev
            prev = cur
            cur = tmp 
            i += 1 
        if start == head: # 如果开始节点就是头节点，需要将头指针指向翻转节点的最后一个节点，同时头节点指向翻转区域后的第一个节点
            head = prev
            start.next = cur
        else:
            start.next = cur #如果开始节点不是头节点，头节点指向翻转区域后的第一个节点
            prev_start.next = prev #翻转区域前一个节点指向翻转区域区域最后一个节点
        return head


# In[ ]:


'''*Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.'''
"*岛屿数量"
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, index_row, index_col,count): 
            if not 0<= index_row < len(grid) or  not 0 <= index_col < len(grid[0]):#递归函数退出条件为下标超出范围，或者当前元素为0
                return 
            if grid[index_row][index_col] == "0":
                return
            grid[index_row][index_col] = "0" #如果该元素是1，我们把它变为0，之后递归该元素周围元素
            dfs(grid, index_row+1,index_col,count)
            dfs(grid, index_row-1,index_col,count)
            dfs(grid, index_row,index_col+1,count)
            dfs(grid, index_row,index_col-1,count)
        count = 0
        for i in range(len(grid)): #我们遍历grid数组，每找到一个1就进入递归，并且岛屿数量加一，因为在递归时我们已经把岛屿上所有的陆地都变为0了，所以在外层遍历时每找到一次1就说明找到一个岛屿
            for j in range(len(grid[0])):
                if grid[i][j] =="1":
                    count += 1
                    dfs(grid,i,j,count)
                            
        return count
    
    
    
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def helper(index_x, index_y):
            if index_x >= len(grid) or index_x < 0 or index_y < 0 or index_y >= len(grid[0]):
                return 
            if grid[index_x][index_y] == "0":
                return 
            grid[index_x][index_y] = "0"
            helper(index_x+1, index_y)
            helper(index_x-1, index_y)
            helper(index_x, index_y+1)
            helper(index_x, index_y-1)
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    count += 1 
                    helper(i, j)
        return count

'''*岛屿最大面积'''
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def dfs(grid, i,j,area):
            if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == 0: #同样的逻辑递归，不同的是递归函数里需要计算陆地的数量，每遇到一个1，area就是1，返回递归后的area的累计值
                return 0 
            area +=  1 
            grid[i][j] = 0
            return area + dfs(grid, i+1,j,0) +dfs(grid, i,j+1,0)+dfs(grid, i-1,j,0)+dfs(grid, i,j-1,0)
        maxx = 0
        for i in range(len(grid)): #遍历grid
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    maxx = max(maxx, dfs(grid, i, j, 0))
        return maxx
    
'''*岛屿周长问题：You are given row x col grid representing a map where grid[i][j] = 1 represents land and grid[i][j] = 0 represents water.

Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes", meaning the water inside isn't connected to the water around the island. One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.'''


class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        perimeter = 0
        direction = [(0,1),(0,-1),(1,0),(-1,0)]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] ==1:
                    count = 0
                    for ad_i,ad_j in direction:
                        new_i, new_j = ad_i +i, ad_j+j
                        if not 0<= new_i < len(grid) or not 0<=new_j<len(grid[0]) or grid[new_i][new_j] ==0:
                            count +=1 

                    perimeter += count


        return perimeter
    
''''''


# In[13]:


'''You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.'''

#严重超时
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        start = 0 
        if len(s) ==1:
            return 1
        maxx = 0
        mark = 0 
        while start < len(s):
            count = k 
            cur = start + 1
            length = 1
            length_2 = 0
            while  cur < len(s) and (count > 0 or s[cur] == s[start]):
                if s[cur] == s[start]: 
                    length += 1 
                else:
                    if count == k:
                        mark = cur 
                        length += 1
                        count -= 1 
                    elif count >0:
                        if count == 1:
                            length_2 = length +1
                        length += 1 
                        count -= 1 
                if cur == len(s)-1 and count > 0 and start != 0:
                    length_2 = length +1
                cur += 1 
            maxx= max(maxx, length, length_2)
            if k == 0 :
                if cur == len(s)-1:
                    break
                else:
                    start = cur
            elif  s[start] == s[mark] and mark == len(s)-1:
                break
            elif cur == len(s):
                start = cur
            else:
                start = mark 
        return maxx


# In[ ]:


'''*Given the head of a linked list, remove the nth node from the end of the list and return its head.

Follow up: Could you do this in one pass?'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
'''哈希表'''
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dic = dict()
        cur = head 
        i =1 
        if cur.next is None:
            return None
        while cur:
            dic[i] = cur 
            i += 1 
            cur = cur.next
        i -= 1 
        m = i-n +1 
        if m == i:
            dic[m-1].next = None
        elif m == 1:
            head = dic[m+1]
        else:
            dic[m-1].next = dic[m+1]
        return head
    
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''*Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        slow = head
        if head is None or head.next is None:
            return False

        fast = head.next
        while True:
            
            if fast is None or fast.next is None :
                return False

            if slow == fast:
                return True

            slow = slow.next
            fast = fast.next.next
            
'''时间复杂度：O(n), 空间复杂度：O(1)'''






# In[ ]:


'''Given an integer number n, return the difference between the product of its digits and the sum of its digits.'''

class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        prod = 1
        summ = 0 
        while n>0:
            tmp = n %10
            n = n //10
            prod = prod * tmp 
            summ = summ + tmp 


        return prod - summ
    
'''时间复杂度：O(n), 空间复杂度O(1)'''


# In[ ]:


'''**There are n people that are split into some unknown number of groups. Each person is labeled with a unique ID from 0 to n - 1.

You are given an integer array groupSizes, where groupSizes[i] is the size of the group that person i is in. For example, if groupSizes[1] = 3, then person 1 must be in a group of size 3.

Return a list of groups such that each person i is in a group of size groupSizes[i].

Each person should appear in exactly one group, and every person must be in a group. If there are multiple answers, return any of them. It is guaranteed that there will be at least one valid solution for the given input.'''
#题目的意思就是要得出一个数组，满足，每个数组的长度是groupsize数组的元素值，同时该数组的每个值是groupsize中数组的元素下标
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        dic = dict() #创建一个哈希表，我们将groupsize中的元素值作为key，对应数组作为value
        res = [] #初始化结果数组
        for i in range(len(groupSizes)): #遍历数组
            if groupSizes[i] in dic:  #如果该元素已经在字典中，则判断一下对应的数组长度是否已经等于该元素的值，如果等于则把该数组添加到结果数组中去，并把当前要添加的元素的添加到一个空数组中
                if len(dic[groupSizes[i]]) == groupSizes[i]:
                    res.append(dic[groupSizes[i]])
                    dic[groupSizes[i]] = [i]
                else: #否则我们直接将该元素添加到该数组
                    dic[groupSizes[i]].append(i)
            else:
                if groupSizes[i] == 1: #如果该元素还不在字典中，则添加该元素进字典，但如果该元素的值是一，我们不需要添加到字典了，直接将其存入结果数组
                    res.append([i])
                else:   
                    dic.setdefault(groupSizes[i],[]).append(i) 

        for key in dic.keys(): #最后还需要再遍历一边字典将未添加进结果数组的所有数组添加进去。
            res.append(dic[key])
        return res


# In[ ]:


'''*Given an array of integers nums and an integer threshold, we will choose a positive integer divisor, divide all the array by it, and sum the division's result. Find the smallest divisor such that the result mentioned above is less than or equal to threshold.

Each result of the division is rounded to the nearest integer greater than or equal to that element. (For example: 7/3 = 3 and 10/2 = 5).

It is guaranteed that there will be an answer.'''

#二分法
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        maxx = max(nums)
        left = 1
        right  = maxx
        def summ_fct(val, nums): #定义一个函数，用于计算每个情况的除数结果之和
            summ = 0
            i = 0
            while i < len(nums):
                summ = summ + math.ceil(nums[i]/val)
                i += 1 
            return summ
        res = 0  
        while left <= right: #二分法，左右指针
            mid = (right + left) //2 
            if summ_fct(mid, nums) > threshold: #大于threshold， 左指针到mid后面
                left = mid + 1
            if summ_fct(mid,nums) <= threshold: #小于threshold， 右指针到mid前面
                res = mid
                right = mid -1 
        return res #返回res
''' 时间复杂度：O(nlogn), 空间复杂度：O(1)'''


# In[ ]:


'''Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        cur = head
        s = ""
        while cur:
            s = s + str(cur.val)
            cur = cur.next

        n = int(s,2)
        return n


# In[ ]:


'''An integer has sequential digits if and only if each digit in the number is one more than the previous digit.

Return a sorted list of all the integers in the range [low, high] inclusive that have sequential digits.'''

class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        i = 1 
        j = 2 
        res = []
        while i <10:
            seq = i 
            while j < 10:
                seq = seq *10 + j 
                j = j + 1
                if low <= seq <= high:
                    res.append(seq)
                elif seq > high:
                    break
                else:
                    continue
            i += 1 
            j = i+1
        res = sorted(res)
        return res
#滑动窗口    
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        s = "123456789"
        res = []
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if low <= int(s[i:j+1]) <= high :
                    res.append(int(s[i:j+1]))
                elif int(s[i:j+1])  > high:
                    break

        return sorted(res)
'''时间复杂度：O(9+8+7+6+5+4+...+1), 空间复杂度：O(9+8+7+6+5+4+...+1))'''


# In[ ]:


'''*Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).'''


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        new_arr = []
        i = 0
        j = 0 
        while i < max(len(nums1), len(nums2)) or j < max(len(nums1), len(nums2)):
            if i >= len(nums1) and j < len(nums2):
                new_arr.append(nums2[j])
                j += 1 
            elif i < len(nums1) and j >= len(nums2):
                new_arr.append(nums1[i])
                i += 1 
            elif nums1[i] < nums2[j]:
                new_arr.append(nums1[i])
                i += 1 
            else:
                new_arr.append(nums2[j])
                j += 1 
            if len(new_arr) == (len(nums1) + len(nums2))//2 + 1 :
                if (len(nums1) + len(nums2) )  % 2 == 0:
                    return (new_arr[-2]+new_arr[-1])/2
                else:
                    return new_arr[-1]


# In[ ]:


#Maintain K Largest (需要面向对象的知识, 不熟悉的话可以跳过)
## 现在有一个长度为n的list, 要求你输出这个list前k大的数; 之后, 我们会慢慢地往这个list里面加数, 要求每次加一个数之后, 输出这个list第k大的数. 请补全下列代码. 
import heapq
class maintain_k_largest():
    def __init__(self, lst, k):
        return heapq.nlargest(self.k, self.lst) #堆实现，nlargest取前k个最大值
        
    def insert_number(self, new_number):
        heapq.heappush(self.lst,new_number) #将新的元素加入堆
        
    def print_kth_largest(self):
        return heapq.nlargest(self.k, self.lst)[0] #返回第k大的元素


# In[ ]:


'''*There are some spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter, and hence the x-coordinates of start and end of the diameter suffice. The start is always smaller than the end.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with xstart and xend bursts by an arrow shot at x if xstart ≤ x ≤ xend. There is no limit to the number of arrows that can be shot. An arrow once shot keeps traveling up infinitely.

Given an array points where points[i] = [xstart, xend], return the minimum number of arrows that must be shot to burst all balloons.

'''

'''降低时间复杂度O(n) 空间复杂度：O(1)'''
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort()
        i = 0
        if len(points) <2 :
            return 1
        max_right = points[0][1]
        i = 1 
        count = 0 
        while i < len(points):
            if max_right >= points[i][0]:
                max_right = min(max_right, points[i][1])
                if i == len(points)-1:
                    count += 1   
                i += 1 
            else:
                if i == len(points) - 1:
                    count += 2 
                    i += 1 
                else:
                    count += 1 
                    max_right = points[i][1]
                    i += 1 
        return count 


# In[ ]:


'''Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.'''

#字典➕递归
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        def dfs(count, arr, num, dic): 
            if count == 3:  #count变量作为标记是否我们已经找到了三个符合条件的元素
                return True
            if len(arr) == 0:  #如果最后数组已经变成空数组了说明走到最后都没有找到这样三个数，则返回空
                return 
            res = False #先把res置为false
            tmp_count = count #记录count当前值
            for i in range(len(arr)): #再次遍历后面的的所有数组元素
                if arr[i] > num: #如果找到一个元素是大于前面一个符合条件的元素的，则进行下一步判断
                    if arr[i] in dic and count == dic[arr[i]]: #如果该元素在字典里并且字典中的该元素的在三个数中的位置是和递归中另一个分支的位置是一样的，则我们这里需要剪枝，不要重复遍历
                        continue
                    else:
                        dic[arr[i]] = count #否则将该元素在三个元素中的位置记录下来
                    count += 1  #位数加1
                    res = dfs(count, arr[i+1:], arr[i],dic) # 之后进行下一层递归
                count = tmp_count #在找下一个arr[i]的时候，要把count恢复到之前一个值，因为我们这个loop只是循环遍历这一层递归     
                if res == True: #如果res是true，则返回true
                    return True
        res = False
        if len(nums) < 3: #如果数组长度小于3，直接返回false
            return False
        dic = dict() #创建一个字典
        tmp_dic = dict() #创建另一个字典
        for j in range(len(nums)): #遍历数组
            if nums[j] in dic: #如果当前元素在字典中，则继续，我们不需要在同一个位置上重复遍历同一个元素，但是注意我们可以在不同的位置上遍历同一个元素
                continue
            else:            #否则将该元素放入字典中
                dic[nums[j]] = 1 
            res = dfs(1,nums[j+1:], nums[j],tmp_dic) #如果不在数组中，则深度优先遍历，输入变量为从数组的下个元素开始的所有元素组成的数组和当前元素，同时输入一个空的字典，这里的空字典是因为不同的位置元素可以是相同的
            if res == True: #dfs函数返回的是bool，如果是true，则是true
                return True
        return False #如果啥都没找到则返回false
    
    
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        maxx_1 = -1000000000
        maxx_2 = -1000000000
        if len(nums) <3:
            return False
        for i in range(len(nums)-1, -1,-1): #基本思想：维护最大的两个值，从后向前遍历，从后向前的过程中，如果遇到一个比最大的初始值还要大的数，我们更新数字，遇到在中间数和最大数之间的数我们更新中间数，在过程中如果遇到比两个数都小的数，直接返回true
            if nums[i] > maxx_1: 
                maxx_1 = nums[i]
            elif maxx_2 <nums[i] < maxx_1:
                maxx_2 = nums[i]
            elif nums[i] < maxx_1 and nums[i]< maxx_2:
                return True
        return False  #如果没有找到，返回false
'''时间复杂度：O(n), 空间复杂度：O(1)'''
            


# In[ ]:


'''Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.'''

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        i = 1
        maxx = 0
        if nums == []:
            return 0
        heapq.heapify(nums) #利用小顶堆
        if len(nums) == 1: #如果数组长度为1，则返回1
            return 1 
        init = heapq.heappop(nums) #先弹出堆中的最小值
        while len(nums)>0:#如果数组长度大于0，则进入循环
            cur = heapq.heappop(nums) # 吐出下一个最小元素
            if cur == init: #如果前一个最小元素和后一个一样，则跳过此元素，继续向下遍历
                continue
            if   cur != init + 1: #如果找到了下一个元素不等于上一个加一，则计算挡墙连续数的长度，与之前的做比较，取最大值
                maxx = max(maxx, i)
                init = cur #初始化init
                i = 1
            else: 
                i += 1  #如果下一个是连续数，则连续数个数加一
                init = cur #将init指针移到下一个
        maxx = max(maxx, i)
        return maxx
    
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        def search(pivot):
            count = 1
            tmp = pivot 
            while pivot+1 in s:
                s.remove(pivot+1)
                count += 1
                pivot += 1 
            while tmp - 1 in s:
                s.remove(tmp-1)
                count += 1
                tmp -= 1 
            return count 
        s = set(nums)
        start = s.pop()
        s.add(start)    
        count = 1 
        while s:
            count = max(count, search(start))
            s.remove(start)
            if s:
                start = s.pop()
                s.add(start)     
        return count
'''时间复杂度： O(N),空间复杂度：O(n)'''


# In[ ]:


'''*Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

Follow-up: Could you solve the problem in linear time and in O(1) space?'''


class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        dic = dict()
        res = []
        if len(set(nums))==1:
            return list(set(nums))
        if len(nums)<3:
            return list(set(nums))
        for i in range(len(nums)):
            if nums[i] in dic:
                dic[nums[i]] += 1 
                if dic[nums[i]] >len(nums)//3 and nums[i] not in res:
                    res.append(nums[i])
                    if len(res) ==2:
                        return res

            else:
                dic[nums[i]] = 1 

        return res

#   ***摩尔投票法*** 
'''从数组开头开始计票，这里开始设定会有最多两个获胜者，而不是一般摩尔投票法的一个获胜者，
1. 第一个获胜者默认开始为第一个元素，第二个获胜者为下一个不是第一个获胜者的元素，开始遍历数组。
2. 当遍历到与第一个或者第二个获胜者时，对应的获胜者票数加一。
3. 当遍历到与两个获胜者不同的其他候选人时，这两个当前获胜者的对应票数减一。当其中一个获胜者的票数已经达到了0，注意如果该获胜者目前票数还有1票，
还可以减去1，当前位置不需要替换获胜者。只有当达到0票时，需要替换获胜者，换成当前元素，并且将其票数设为1票，另一个获胜者在此位置票数不受任何影响。
4. 遍历结束后，判断是否两个计数器值为大于0，再次遍历一遍数组，用以确认计数器不是0的那个获胜者的票数是否大于n//3。'''
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        if len(nums) == 1:#如果数组长度是1，则直接返回
            return nums
        num_1 = nums[0] #我们先把第一个元素作为获胜者
        count_1 = 1
        num_2 = -100000000000 #第二个元素我们先设为任意小的值
        count_2 = 0
        for i in range(1,len(nums)):#遍历数组
            if nums[i] != num_1 and num_2 == -100000000000: #当遇到第一个和之前的获胜者不同的元素时，将该元素设为获胜者
                num_2 = nums[i]
                count_2 += 1
            elif nums[i] != num_1 and nums[i] != num_2: #当遍历中遇到元素和两个获胜者都不一样时进一步分情况判断
                if count_1 >= 1 and count_2 >= 1: #如果两个获胜者的票数都大于等于1，则都-1
                    count_1 -= 1  
                    count_2 -= 1 
                elif count_1 < 1 and count_2 >= 1: #有一个小于1，另一个大于等于1，则把小于1的那个获胜者替换为当前元素，另一个获胜者保持不动
                    num_1 = nums[i]
                    count_1 = 1 
                elif count_1 >= 1 and count_2 < 1:
                    num_2 = nums[i]
                    count_2 = 1 
                else: #如果两个获胜者的票数都到0了，则将两个获胜者位置初始化，将当前元素作为其中一个获胜者，另一个元素设为任意小
                    num_1 = nums[i]
                    count_1 = 1 
                    num_2 = -100000000000
                    count_2 = 0
            elif nums[i] != num_1 and nums[i] == num_2: #如果当前元素和其中一个获胜者一样，则对应获胜者票数加1
                count_2 += 1 
            else:
                count_1 += 1 
        if count_1 != 0 and count_2 != 0: #结束后我们需要判断两个获胜者的票数是否不为0，如果其中有一个不为0，则再次遍历一边数组，认证确实是众数
            check_1 = 0
            check_2 = 0 
            for i in range(len(nums)):
                if nums[i] == num_1:
                    check_1 += 1 
                if nums[i] == num_2:
                    check_2 += 1 
            if check_1 > len(nums)//3 and check_2 > len(nums)//3:
                return [num_1,num_2]
            elif check_1 <= len(nums)//3 and check_2 > len(nums)//3:
                return [num_2]
            elif check_1 > len(nums)//3 and check_2 <= len(nums)//3:
                return [num_1]
            else:
                return []
        if count_1 == 0 and count_2 != 0:
            check_2 = 0
            for i in range(len(nums)):
                if nums[i] == num_2:
                    check_2 += 1 
            if check_2 >len(nums)//3:
                return [num_2]
            else:
                return []
        if count_1 != 0 and count_2 == 0:
            check_1 = 0 
            for i in range((len(nums))):
                if nums[i] == num_1:
                    check_1 += 1 
            if check_1 > len(nums)//3:
                return [num_1]
            else:
                return []
        else:
            return []


 


# In[ ]:


'''80. *Remove Duplicates from Sorted Array II

Given a sorted array nums, remove the duplicates in-place such that duplicates appeared at most twice and return the new length.

Do not allocate extra space for another array; you must do this by modifying the input array in-place with O(1) extra memory.'''

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        i = 0 
        if len(nums) < 3: #先判断数组长度是否小于3，如果小于三直接返回长度，因为这个数组一定满足条件
            return len(nums)
        j = 1 
        while j<len(nums): #否则遍历数组，三指针
            if nums[i] == nums[j]:#右指针如果和左指针相等，则开始进入判断右指针的下一位
                k = j + 1  
                while k < len(nums) and nums[k] == nums[j]: #第三个指针一直往下走直到找到了一个元素值不为前两个指针，过程中不断的删除其中的元素
                    nums.pop(k)
                if k >= len(nums) -1 : #推出循环时，判断如果第三个指针已经走到了最后一位，或者已经超出数组范围，则直接返回长度，因为最后一位一个元素一定满足条件
                    return len(nums)
                else:   #否则左右指针到第三指针和第三指针的下一位
                    i = k 
                    j = i + 1 
            else:#如果左右指针不等，则一直往后走
                i += 1 
                j += 1 
        return len(nums) 
'''时间复杂度：O(n^2),空间复杂度：O(1)'''


# In[ ]:


'''Suppose you are at a party with n people (labeled from 0 to n - 1), and among them, there may exist one celebrity. The definition of a celebrity is that all the other n - 1 people know him/her, but he/she does not know any of them.

Now you want to find out who the celebrity is or verify that there is not one. The only thing you are allowed to do is to ask questions like: "Hi, A. Do you know B?" to get information about whether A knows B. You need to find out the celebrity (or verify there is not one) by asking as few questions as possible (in the asymptotic sense).

You are given a helper function bool knows(a, b) which tells you whether A knows B. Implement a function int findCelebrity(n). There will be exactly one celebrity if he/she is in the party. Return the celebrity's label if there is a celebrity in the party. If there is no celebrity, return -1.'''

# The knows API is already defined for you.
# return a bool, whether a knows b
# def knows(a: int, b: int) -> bool:

class Solution:
    def findCelebrity(self, n: int) -> int:
        candidate = 0
        for x in range(n):
            if knows(candidate, x): #遍历数组，先假定名人是0号，如果候选人认识除了他自己以外的任何一个人，则候选人放弃名人，名人成为认识的那个人
                candidate = x

        for x in range(n): #再一次遍历
            if candidate == x:
                continue
            if knows(candidate, x): #如果最终的候选人认识任何一个除自己以外的人，则表明没有名人
                return -1
            if not knows(x, candidate): #如果有人不认识最终的候选人则也说明没有名人
                return -1

        return candidate #否则返回名人编号
                 


# In[ ]:


'''*There are some spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter, and hence the x-coordinates of start and end of the diameter suffice. The start is always smaller than the end.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with xstart and xend bursts by an arrow shot at x if xstart ≤ x ≤ xend. There is no limit to the number of arrows that can be shot. An arrow once shot keeps traveling up infinitely.

Given an array points where points[i] = [xstart, xend], return the minimum number of arrows that must be shot to burst all balloons.

'''

'''时间复杂度: O(nlogn)，空间复杂度: O(n)'''
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points_copy = sorted(points)
        i = 0
        if len(points_copy) <2 :
            return 1
        max_right = points_copy[0][1]
        i = 1 
        count = 0 
        while i < len(points_copy):
            if max_right >= points_copy[i][0]:
                max_right = min(max_right, points_copy[i][1])
                if i == len(points_copy)-1:
                    count += 1   
                i += 1 
            else:
                if i == len(points_copy) - 1:
                    count += 2 
                    i += 1 
                else:
                    count += 1 
                    max_right = points_copy[i][1]
                    i += 1 
        return count


# In[ ]:


'''*274. H-Index
Given an array of integers citations where citations[i] is the number of citations a researcher received for their ith paper, return compute the researcher's h-index.'''

class Solution:
    def hIndex(self, citations: List[int]) -> int:
        heapq._heapify_max(citations)
        i = 1
        maxx= 0 
        while citations:
            tmp = heapq._heappop_max(citations)
            if tmp >= i:
                maxx = max(maxx, i)
            i += 1 
        return maxx
    
'''时间复杂度：O(n), 空间复杂度：O(1)'''

class Solution:
    def hIndex(self, citations: List[int]) -> int:
        def quickselect(citations, left, right):
            if left > right:
                if left < len(citations) and citations[left] >= left+1:
                    return left +1 
                else:
                    return right + 1 
            if left <= right:
                index = partition(citations, left, right)
                if citations[index] >= index+1:
                    return quickselect(citations, index+1, right)

                elif citations[index] < index+1:
                    return quickselect(citations, left, index-1)

        def partition(citations,  left, right):
            pivot = random.randint(left, right)
            citations[pivot], citations[right] = citations[right], citations[pivot]
            i = left -1 
            pivot = right
            for j in range(left, right):
                if citations[j] > citations[pivot]:
                    i += 1 
                    citations[j], citations[i] = citations[i], citations[j]

            citations[i+1], citations[pivot] = citations[pivot], citations[i+1]

            return i+1

        return quickselect(citations,0, len(citations)-1)


# In[ ]:


'''归并排序'''

def merge_two_lsts(lst1,lst2):
    if len(lst1) ==0:
        return lst2
    if len(lst2) == 0:
        return lst1
    res = []
    i = 0 
    j = 0
    while True:
        if i == len(lst1):
            res += lst2[j:]
            break
        elif j == len(lst2):
            res += lst1[i:]
            break
        elif lst1[i] < lst2[j]:
            res.append(lst1[i])
            i += 1 
        else:
            res.append(lst2[j])
            j += 1 
            
    return res


def merge_sort(lst):
    if len(lst) < 2:
        return lst
    mid = (len(lst))//2
    return merge_two_lsts(merge_sort(lst[:mid]),merge_sort(lst[mid:]))


# In[ ]:


'''Given an unsorted integer array nums, find the smallest missing positive integer.

You must implement an algorithm that runs in O(n) time and uses constant extra space.'''

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        still = 0 
        while still < len(nums):
            if nums[still] <=0 or nums[still] > len(nums):#如果遇到元素不是在合理的范围内，就跳过该元素，第一个迷失的最小正整数一定是小于等于数组长度，大于0
                still += 1
                continue
            if nums[still] != still+1:#如果遇到当前位置上的元素不等于该位置的下标的+1（一个位置的下标+1即是该位置应该有的元素）
                move = nums[still]-1 #我们就要移动这个元素，move来记录这个元素会移动到哪里，会移动到该元素-1的下标位置
                still_ele = nums[still] #我们需要把这个当前元素取出来
                tmp_still= still_ele
                while nums[move] != tmp_still: #我们需要进入一个循环，循环的退出条件是我们要放入的位置的元素和我们要放入的元素不等，因为如果相等我们放不放就无所谓了
                    tmp = nums[move] #我们把要放入位置的元素拿起来
                    nums[move] = tmp_still #我们把元素放到这个位置
                    tmp_still = tmp #我们把拿起来的元素传值给tmp_still用于下一次循环做判断，就是下一次循环，拿起来的元素将要放入它应该在的位置
                    if  tmp_still <= 0 or tmp_still > len(nums): #这里对下一次要放的元素进行一次判断，如果元素不在合理范围内，我们就跳过该元素，终止循环
                        break
                    move = tmp_still -1 #我们更新move，更新到下一次要跳的位置
            still += 1
        for i in range(len(nums)): #判断first missing number
            if nums[i] != i + 1:#找到第一个当前元素不是下标+1的，即是missing的元素
                return i+1
        return len(nums) + 1 #没有的话，则返回数组长度的下一个元素
    
'''时间复杂度O(n),空间复杂度O(1), 时间复杂度分析：最好情况是所有元素都已经在他们该在的位置了，那我们就从头到尾走一遍O(n),最坏情况是，所有元素都不在该在的地方
静指针每走一次，动指针就要走2步，一共走了2n步，所以时间复杂度还是O(n)'''





def get_different_number(arr):
    i = 0 
    while i <len(arr):
    if arr[i] != i:
        if arr[i] >= len(arr):
            i += 1 
            continue 
        tmp = arr[arr[i]]
        index = arr[i]
        mark = 0
        while index != arr[index]:
        if index >= len(arr) and mark != 0:
            break 
        mark = 1 
        arr[index] = index
        index = tmp 
        if index >= len(arr):
            break
        tmp = arr[index] 
    i += 1     
    i = 0 
    while i <len(arr):

        if arr[i] !=i:
            return i
        i += 1 
    return len(arr)
        


# In[ ]:


'''128. Longest Consecutive Sequence''' 
'''最坏时间复杂度：O(n)， 空间复杂度：O(n)'''


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        arr = list(set(nums))
        dic = dict()
        for i in range(len(arr)):
            dic[arr[i]] = 1 
        maxx = 0
        for i in range(len(arr)): #最多循环2n次，2n次的情况是没有连续的序列，每次遍历到一个元素都要再向前遍历一个元素，2n还有可能是整个数列就是连续的，但是是倒着排的，要遍历到最后一个之后再向前遍历一次
            if arr[i] -1 in dic:
                continue 
            count = 0
            key = arr[i]
            while key in dic.keys(): 
                count += 1 
                key = key + 1 
            maxx = max(maxx, count)
        return max


# In[ ]:


'''146. LRU Cache
Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.'''

class DoubleLink:
    def __init__(self, x , next= None , prev = None):
        self.val = x
        self.next = None
        self.prev = None

class LRUCache:

    def __init__(self, capacity: int):
        i = 1
        self.dic = dict()
        self.dic_ads = dict()
        self.head = DoubleLink(-1)
        self.count = 0
        self.length = capacity
        self.top = self.head
        while i < capacity:
            self.top.next = DoubleLink(-1)
            self.top.next.prev = self.top
            self.top = self.top.next
            i += 1 

    def get(self, key: int) -> int:
        if key in self.dic:
            if self.dic_ads[key] == self.head:
                self.top.next = self.head
                self.head.prev = self.top
                self.head = self.head.next
                self.head.prev = None
                self.top = self.top.next
                self.top.next = None
                
            else:
                if not self.dic_ads[key].next:
                    return self.dic[key]
                else:
                    self.dic_ads[key].next.prev = self.dic_ads[key].prev
                    self.dic_ads[key].prev.next = self.dic_ads[key].next
                    self.dic_ads[key].prev, self.top.next = self.top, self.dic_ads[key]
                    self.dic_ads[key].next = None
                    self.top = self.top.next
                    self.top.next = None
            return self.dic[key]
        else:
            return -1
        
    def put(self, key: int, value: int) -> None:
        if key not in self.dic:
            if self.count < self.length:
                self.head.val = key
                self.dic_ads[key] = self.head
                self.count += 1  
                self.dic[key] = value
                self.top.next= self.head
                self.head.prev = self.top
                self.head = self.head.next
                self.head.prev = None
                self.top = self.top.next
                self.top.next = None
            else:
                self.top.next = self.head
                self.head.prev = self.top
                self.dic.pop(self.head.val)
                self.dic_ads.pop(self.head.val)
                self.dic_ads[key] = self.head
                self.head = self.head.next
                self.head.prev = None
                self.top = self.top.next
                self.top.next = None
                self.top.val = key
                self.dic[key] = value
        else:

            if self.dic_ads[key] == self.head:
                self.top.next = self.head
                self.head.prev = self.top
                self.head = self.head.next
                self.head.prev = None
                self.top = self.top.next
                self.top.next = None
                self.dic[key] = value
          
            else:

                if not self.dic_ads[key].next:
                    self.dic[key] = value
               

                else:
                    self.dic_ads[key].next.prev = self.dic_ads[key].prev
                    self.dic_ads[key].prev.next = self.dic_ads[key].next
                    self.dic_ads[key].prev, self.top.next = self.top, self.dic_ads[key]
                    self.dic_ads[key].next = None
                    self.top = self.top.next
                    self.top.next = None
                    self.dic[key] = value
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value

'''put(),get()函数时间复杂度：O(1)，空间复杂度：O(n)'''


# In[1]:


'''find mean and std from data stream'''
import math
class maintain_mean_std():
    
    def __init__(self): #时间复杂度：O(1), 空间复杂度：O(1)
        self.nums = []
        self.mean = None
        self.std = None
    def insert_number(self, number):#时间复杂度：O(n), 空间复杂度：O(1)
        self.nums.append(number)
        if self.mean == None:
            self.mean = number
            self.std = 0
        else:

            summ = self.mean *(len(self.nums)-1) + number
            length = len(self.nums)
            self.mean = summ / length
            var = 0
            for x in self.nums:
                var += (x - self.mean)**2
            self.std = math.sqrt(var)

    def print_mean_std(self):#时间复杂度：O(1),空间复杂度：O(1)
        print(self.mean)
        print(self.std)

obj = maintain_mean_std()
obj.insert_number(10)
obj.insert_number(1)
obj.insert_number(9)
obj.insert_number(109)
obj.insert_number(76)
obj.print_mean_std()


# In[ ]:


'''
977. Squares of a Sorted Array
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
'''

class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        new_arr = []
        if len(nums) == 1:
            new_arr.append(nums[0]**2)
            return new_arr

        if nums[0] >=0:
            for i in range(len(nums)):
                nums[i] = nums[i] **2

            return nums

        if nums[-1] <=0:
            left = 0
            right = len(nums) -1 
            mid = (left+right+1)//2
            for i in range(mid):
                nums[i], nums[len(nums)-1-i]= nums[i]**2, nums[len(nums)-1-i]**2
                nums[i], nums[len(nums)-1-i] = nums[len(nums)-1-i], nums[i]
            if nums[mid]<0:
                nums[mid] = nums[mid]**2
            return nums
        left = 0
        right = len(nums) -1 
        while nums[left]<0:
            left += 1 

        right = left
        left = left - 1 
        
        i = 0 
        while i < len(nums):
            nums[i] = nums[i] **2
            i += 1 
        while left >=0 or right <len(nums):  
            if left < 0 and right < len(nums):
                new_arr.append(nums[right])
                right += 1 
                
            elif right >= len(nums) and left >=0:
                new_arr.append(nums[left])
                left -= 1 
            elif nums[left] < nums[right]:
                new_arr.append(nums[left])
                left -= 1
            else:
                new_arr.append(nums[right])
                right += 1 
        return new_arr
            
'''时间复杂度：O(n), 空间复杂度：O(n)'''  


# In[ ]:


'''16. 3Sum Closest
Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.'''

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        new_nums = sorted(nums)
        summ = 100000000000
        for i in range(len(new_nums)):
            if i == len(new_nums) -1:
                break
            j = i + 1 
            k = len(new_nums) - 1
            while k >j:
                if abs(summ - target) > abs(new_nums[i]+new_nums[j]+new_nums[k] - target):
                    summ = new_nums[i]+new_nums[j]+new_nums[k]
                if new_nums[i]+new_nums[j]+new_nums[k] > target:
                    while j <k and new_nums[k] == new_nums[k-1]:
                        k -= 1 
                    k -= 1 

                elif new_nums[i]+new_nums[j]+new_nums[k] < target:
                    while j < k and new_nums[j] == new_nums[j+1]:
                        j += 1 

                    j += 1 

                else:
                    return target

        return summ
              


# In[ ]:


'''713. Subarray Product Less Than K. Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is strictly less than k.'''


class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if len(nums) <2:
            if nums[0] <k:
                return 1 
            else:
                return 0
        prod = 1 
        count = 0
        left = 0 
        for right in range(0,len(nums)):
            prod = prod * nums[right]
            while prod >= k and left < right :
                prod = prod // nums[left]
                left += 1 
            if prod < k:
                count = count + right - left + 1 

        return count


# In[ ]:


'''The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.'''

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def buildboard():
            board = []
            for i in range(n):
                each_row[pos[i]] = "Q"
                board.append(''.join(each_row))
                each_row[pos[i]] = "."
            return board

        def backtrack(row):
            if row == n:
                board = buildboard()
                res.append(board)
            for i in range(n):
                if i in col or row - i in diag_1 or i + row in diag_2:
                    continue
                pos[row] = i
                col.add(i)
                diag_1.add(row - i)
                diag_2.add(row + i)
                backtrack(row+1)
                col.remove(i)
                diag_1.remove(row-i)
                diag_2.remove(row+i)
                
        pos = [-1] * n
        col = set()
        diag_1 = set()
        diag_2 = set()
        res = []
        each_row = ["."] * n
        backtrack(0)
        return res 


# In[ ]:


'''The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.'''

#用堆，创建一个大顶堆，创建一个小顶堆，大顶堆用于存放较小的一半的数，小顶堆用于存放较大的一半的数
class MedianFinder:

    def __init__(self):
        self.upperheap = []
        self.lowerheap = []


    def addNum(self, num: int) -> None:
        if self.upperheap == [] and self.lowerheap ==[]:#当两个堆里面都没有元素时
            heapq.heappush(self.lowerheap, -num) #先默认将第一个元素放入较小的大顶堆里，注意创建大顶堆我们需要将数字加-号，用小顶堆存储
        elif self.upperheap == []: #如果小顶堆还没有元素，在这里我们需要判断加进去的第一个元素和现在的元素比大小，如果num较小的话，我们需要将大顶堆的元素与num交换,并将替换的元素放入小顶堆
            if num < - self.lowerheap[0]: 
                tmp = - heapq.heapreplace(self.lowerheap,-num)
                heapq.heappush(self.upperheap, tmp)
            else:
                heapq.heappush(self.upperheap, num) #否则直接将num放入小顶堆

        else:
            lower = - self.lowerheap[0] #如果两个都不为空，我们取出大顶堆的最大值，小顶堆的最小值
            upper = self.upperheap[0]
            size_lower = len(self.lowerheap) #取出两个堆的大小
            size_upper = len(self.upperheap)
            if num < lower and (size_upper - size_lower >=0):#如果num小于最小值，并且大顶堆要比小顶堆大，我们可以直接将num放入小顶堆
                heapq.heappush(self.lowerheap, -num)
            elif num < lower and size_upper < size_lower: #如果num小于最小值，并且大顶堆比小顶堆小，我们需要弹出大顶堆的最大值，并且放入小顶堆中，然后插入num进大顶堆
                tmp = -heapq.heapreplace(self.lowerheap,-num)
                heapq.heappush(self.upperheap, tmp)
            elif upper >num >= lower  and size_upper <= size_lower: #如果num是在最小值与最大值之间，并且大顶堆要比小顶堆小，我们将该元素放入大顶堆
                heapq.heappush(self.upperheap,num)
            elif upper >num >= lower and size_upper > size_lower:#如果num是在最小值与最大值之间，并且大顶堆要比小顶堆大，我们将该元素放入小顶堆
                heapq.heappush(self.lowerheap, -num)
            elif num >= upper and size_upper <= size_lower: #如果num大于最大值，并且小顶堆比大顶堆小，直接插入小顶堆
                heapq.heappush(self.upperheap,num)
            elif num >= upper and size_upper > size_lower:#如果num大于最大值，并且小顶堆比大顶堆大，弹出小顶堆最小值将其放入大顶堆，把num放入小顶堆
                tmp = heapq.heapreplace(self.upperheap,num)
                heapq.heappush(self.lowerheap, -tmp)



    def findMedian(self) -> float:
        size_lower = len(self.lowerheap)
        size_upper = len(self.upperheap)
        if size_lower == size_upper:
            lower = - self.lowerheap[0]
            upper = self.upperheap[0]
            return (lower + upper)/2

        elif size_lower < size_upper:
            return self.upperheap[0] 
        else:
            return -self.lowerheap[0]




# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()


# In[ ]:





# In[ ]:


'''You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if head.next is None or head.next.next is None:
            return
        count = 0
        cur = head
        while cur:
            count += 1 
            cur = cur.next
        dvd = count//2
        cut = head
        i = 0
        while i < dvd:
            cut = cut.next
            i += 1 

        head_2 = cut.next
        cut.next = None
        if head_2.next:
            if head_2.next.next is None:
                cur_2 = head_2.next
                cur_2.next = head_2
                head_2.next = None
                head_2 = cur_2
            else:
                prev_2 = head_2
                cur_2 = head_2.next
                nxt_2 = cur_2.next
                prev_2.next = None
                while nxt_2:
                    cur_2.next = prev_2
                    prev_2 = cur_2
                    cur_2 = nxt_2
                    nxt_2 = nxt_2.next
                cur_2.next = prev_2
                head_2 = cur_2
        else:
            cur_2 = head_2
        prev = head
        cur = prev.next
        
        while cur_2:
            prev.next = cur_2
            cur_2 = cur_2.next
            prev.next.next = cur
            prev = cur
            cur = prev.next


# In[ ]:


'''781. Rabbits in Forest

There is a forest with an unknown number of rabbits. We asked n rabbits "How many rabbits have the same color as you?" and collected the answers in an integer array answers where answers[i] is the answer of the ith rabbit.

Given the array answers, return the minimum number of rabbits that could be in the forest.'''

'''时间复杂度: O(nlogn),空间复杂度：O(n) '''
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        new_ans = sorted(answers)
        if len(new_ans) == 1:
            return 1 
        count = 0 
        i = 0 
        while i < len(new_ans):
            if new_ans[i] == 0:
                count += 1 
                i += 1 
                continue
            steps = new_ans[i]
            count = count + new_ans[i]+1
            j = i+1  
            while j < steps+i+1 and j < len(new_ans):
                if new_ans[j] != new_ans[i]:
                    break
                else:
                    j += 1 

            if j >= len(new_ans):
                break

            i = j
        return count
    
    

'''时间复杂度：O(n),空间复杂度: O(n)'''
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        dic = dict()
        for i in range(len(answers)):
            if answers[i] in dic:
                dic[answers[i]] += 1 

            else:
                dic[answers[i]] = 1 
        count = 0
        for key, value in dic.items():
            if key == 0:
                count += value

            elif value != 1:
                num = math.ceil(value/(key+1))
                count = count + (key+1)*num
            elif  value == 1:
                count = count + key+1

        return count


# In[ ]:


'''Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.'''

#递归法
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic = {"2" :"abc", "3":"def", "4":"ghi", "5":"jkl", "6":
        "mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
        
        def dfs(digits, index, prod):
            if index >= len(digits):#当index已经到了数字字符串尾部是，我们已经遍历该种情况结束，添加到结果数组，结束该路径
                res.append(prod)
                return 

            tmp = prod #先将当前组合存储起来
            s = dic[digits[index]] #取出我们正在遍历的数字的特定位置得数字的对应字母字符串
            for i in range(len(s)):#遍历这个字符串
                prod = prod + s[i] #每遍历到一个字符就是一种可能，我们把这个添加到我们的prod字符串中
                dfs(digits,index+1, prod) #添加一个字母可能后我们需要进入下一层递归，即判断下一个数字对应的字母，我们这个for循环只是判断当前这一层的所有字母
                prod = tmp #回溯操作，回溯回去，添加下一个可能的字母
        if digits == "":
            return []
        res = []
        dfs(digits,0,"")
        return res
    
'''时间复杂度：，空间复杂度：O(1)'''


# In[ ]:


'''Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def helper(node, lower, higher):
            if not node :
                return True

            if node.val >= lower or node.val <= higher:
                return False 

            if not helper(node.left, node.val, higher):
                return False

            if not helper(node.right,lower, node.val):
                return False 

            return True
        return helper(root, 10000000000, -10000000000)
    '''时间复杂度：, 空间复杂度：O(1)'''


# In[ ]:


'''Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted from left to right.
The first integer of each row is greater than the last integer of the previous row.'''


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix) == 1:
            if target in matrix[0]:
                return True
            else:
                return False 

        for i in range(len(matrix)-1):
            if matrix[i][0]<=target < matrix[i+1][0]:
                if target in matrix[i]:
                    return True
                else:
                    return False 
            if i+1 == len(matrix)-1:
                if  target in matrix[i+1]:
                    return True
                else:
                    return False 
        return False
    
'''时间复杂度：O(n+m),空间复杂度：O(1)'''


# In[ ]:


'''Given the root of a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

As a reminder, a binary search tree is a tree that satisfies these constraints:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.'''

'''中序遍历'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:
        self.prev = 0
        def in_order(node):
            if node is None:
                return 
            in_order(node.right)
            node.val = node.val + self.prev
            self.prev = node.val
            in_order(node.left)

        in_order(root)
        return root
    


# In[ ]:


'''A super ugly number is a positive integer whose prime factors are in the array primes.

Given an integer n and an array of integers primes, return the nth super ugly number.

The nth super ugly number is guaranteed to fit in a 32-bit signed integer.'''

#用堆做，利用的是heap本质上是一个数组，同时又能做到排序的目的，这题我们是想尽可能用小的prime来组成目标数字
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        prime_candidates = [1] #首先创造一个素数候选列表，初始为[1]，因为1的prime factor 是数组中任意数字的 0 次方
        res = [] #设立一个结果数组
        dic = set()
        while len(res) < n: #我们不断往结果数组里放ugly number
            num = heapq.heappop(prime_candidates)  #我们把候选数组中的最小的ugly number拿出来，在将其放入结果数组之前，我们需要生产出下面的ugly numbers
            for  prime in primes: #遍历primes数组
                if num * prime not in dic:
                    heapq.heappush(prime_candidates, num * prime) #将数组中每个元素*当前的最小ugly number 就会产生下一组最小ugly numbers，将结果放入候选heap
                    dic.add(num*prime)
            res.append(num) #该次循环结束时将最小ugly number 放入结果数组

        return res[n-1] #输出最小结果数组中第n个数


'''时间复杂度：O(n*m*logm), 空间复杂度：O(n)'''


# In[ ]:


'''The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.'''

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def buildboard():
            board = []
            for i in range(n):
                each_row[pos[i]] = "Q"
                board.append(''.join(each_row))
                each_row[pos[i]] = "."
            return board

        def backtrack(row):
            if row == n:
                board = buildboard()
                res.append(board)
            for i in range(n):
                if i in col or row - i in diag_1 or i + row in diag_2:
                    continue
                pos[row] = i
                col.add(i)
                diag_1.add(row - i)
                diag_2.add(row + i)
                backtrack(row+1)
                col.remove(i)
                diag_1.remove(row-i)
                diag_2.remove(row+i)
                
        pos = [-1] * n
        col = set()
        diag_1 = set()
        diag_2 = set()
        res = []
        each_row = ["."] * n
        backtrack(0)
        return res 
    
    
def nKnights(m = 8, n = 8):
    
    


# In[ ]:


'''*
4Sum:
Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.'''

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        dic = dict()
        res = []
        nums.sort() #先给数组排个序
        if len(nums) <4: #如果数组长度已经小于4，则直接不可能
            return []
        for j in range(len(nums)-3): #我们先把四数之和中所有较小的两个数之和的可能放入到一个字典中便于后续查找
            if j-1 >=0 and nums[j-1] == nums[j]: # 要进行剪枝，如果前后元素是一样的，则跳过后面的元素，我们要的是所有重复元素中的第一个重复元素下标
                continue
            if 4 * nums[j] > target: #如果当前元素的4倍已经比target大了，我们不用接着遍历了，因为越往后数字越大
                break 
            if nums[j] + 3*nums[len(nums)-1] < target: #如果当前元素加上数组中最大元素的3倍，即包含当前元素的最大可能的4sum都比target小，我们不用判断下个固定位置，直接进入下一轮循环
                continue
            for k in range(j+1, len(nums)-2): #开始固定第二个数
                if k-1 >= j+1 and nums[k-1] == nums[k]: #同样的剪枝操作
                    continue
                if nums[j] + 3*nums[k] > target:#这时候有两个固定元素，所以当前情况的最小4sum即当前元素的三倍加上之前的固定元素
                    break
                if nums[j] + nums[k] + 2*nums[len(nums)-1] < target:#同样有两个固定元素
                    continue
                dic.setdefault(nums[j]+nums[k], []).append((j,k))#如果全部满足上述条件，我们将该组合下标以tuple的形式存入字典
        
        for j in range(len(nums)-1, 2, -1):#下面我们要进行4sum中较大两位的判断，从后向前
            if j+1 <len(nums) and nums[j+1] == nums[j]:#同样的如果之前的一个和当前的一样的，我们则直接跳过
                continue
            if 4 * nums[j] < target:#当前位置的最大可能值如果还小于target，我们也不用遍历了，之后的元素组合一定比这个小
                break
            if nums[j] + nums[0] *3 > target:#当前位置的最小可能值如果还比target大，我们不用进行下一位的固定，直接j指针往下走
                continue
            for k in range(j-1, 1, -1):#开始固定第二个指针
                if k +1 <= j-1 and nums[k+1] == nums[k]:#同样的剪枝操作
                    continue
                if nums[j]+3*nums[k] < target: #这里有两个固定的元素
                    break
                if nums[j] + nums[k] + 2*nums[0] > target:
                    continue

                for index_left, index_right in dic.get(target-nums[j]-nums[k], []): #如果通过了以上的判断，我们如果找到了和为target的4sum
                    if k > index_right: #确保较大两个位置的较左边的位置一定是在较小两个元素组合的较右边的的位置的右侧
                        res.append([nums[j],nums[k],nums[index_left], nums[index_right]]) #添加结果到数组
        return res
    
'''时间复杂度：O(n^2), 空间复杂度：O(n^2)'''

def fourSum(nums, target):
    from collections import defaultdict
    result, d = set(), defaultdict(list) #对于4sum， 我们需要使用dict来判断下标是否有重复， 因为我们有重复存放相同位置元素的过程
    for k in range(len(nums)):#两层for循环遍历数组
        for l in range(k+1, len(nums)):
            if target - nums[k] - nums[l] in d: #当遇到target- nums[k] - nums[l] 在字典中时
                for i, j in dic[target-nums[k] - nums[l]]: #我们循环判断该key对应的每个 tuple value
                    if len(set(i,j,k,l)) == 4: #只有当四个元素都是不同的位置的元素时我们才能进行添加，因为一个元素不能用两次
                        result.add(tuple(sorted([nums[i], nums[j]], nums[k], nums[l]))) #我们进行sorted操作用于统一化我们的数组样子，再以tuple形式存储，并添加进set中用于去重
                        
            d[nums[k] + nums[l]].append((k,l)) #我们把和为nums[k] + nums[l]的该tuple对放入字典对应的key中
            
    return [list(x) for x in result] #结果以list返回

'''时间复杂度：O(n^3*4*log4), 空间复杂度：O(n^2)'''


class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        dic = set()
        record = set()
        res = []
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                for k in range(j+1,len(nums)):
                    if target - nums[i]-nums[j]-nums[k] in dic:
                        tmp = [target - nums[i]-nums[j]-nums[k],nums[i], nums[j],nums[k]]
                        tmp.sort()
                        if tuple(tmp) not in record:
                            record.add(tuple(tmp))
                            res.append(tmp)

                    
            dic.add(nums[i])

        return res


# In[ ]:


'''*3sum
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets. '''

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        dic = dict()
        res = []
        nums.sort()
        if len(nums) < 3:
            return []
        print(nums)
        for j in range(len(nums)-2):
            if j-1 >= 0 and nums[j-1] == nums[j]:
                continue
            if 3*nums[j] > 0:
                break
            if nums[j] + 2*nums[-1] < 0:
                continue
            for k in range(j+1, len(nums)-1):
                if k-1 >= j+1 and nums[k-1] == nums[k]:
                    continue
                if nums[j] + 2*nums[k] > 0:
                    break
                if nums[j] + nums[k] + nums[-1] < 0:
                    continue 
                dic.setdefault(nums[j]+nums[k], []).append((j, k))

        for i in range(len(nums)-1, 0, -1):
            if i + 1 < len(nums) and nums[i+1] == nums[i]:
                continue
            if 3 * nums[i] < 0:
                break
            if nums[i] + nums[0]*2 > 0:
                continue
            for index_left, index_right in dic.get(-nums[i], []):
                if i > index_right:
                    res.append([nums[index_left], nums[index_right], nums[i]])

        return res
'''时间复杂度：O(n^2), 空间复杂度：O(n^2)'''

def threeSum(nums):
    result, d = set(), set() #对于3sum 我们设置两个集合即可，不用设置字典，原因是每次我们只把nums[j]放入set， 也就是从数组头走到尾的所有元素，这是不会有重复下标的
    for j in range(len(nums)): #两层for循环遍历数组
        for k in range(j+1, len(nums)):
            if - nums[j] - nums[k] in d: #判断0-nums[j] - nums[k] 是否在集合d内， 在的话，我们不用判断下标是否重复，因为我们没有重复的将同一下表放入d的过程
                result.add(tuple(sorted([-nums[j] - nums[k], nums[j], nums[k]]))) #我们sorted操作，并将结果以tuple保存，放入result集合，也就顺便去重
                
        d.add(nums[j]) #仅添加j对应的元素

    return [list(x) for x in result] #结果以list输出

'''时间复杂度：O(n^2*3*log3), 空间复杂度：O(n)'''


# In[ ]:


'''Given a string num which represents an integer, return true if num is a strobogrammatic number.

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).'''

class Solution:
    def isStrobogrammatic(self, num: str) -> bool:
        dic = {"6":"9", "8":"8", "0":"0", "1":"1","9":"6" } #先把所有可能的正反看一样的数字存入字典，注意6,9两个数字是需要以中心对称的，其他数字可以自对称
        left = 0
        right = len(num) -1
        while left <= right: #双指针从字符串前后一起遍历
            if num[left] not in dic or num[right] not in dic: #首先如果该字符串中出现不是dic里的数字一律不符合要求，返回false
                return False 
            elif dic[num[right]] != num[left]: #如果right处的元素在字典中，但该元素在字典中对应的字符不是left字符，说明他们也不对称，返回false
                return False 
            left += 1
            right -= 1 
        return True #如果全部满足，就返回true
    
'''时间复杂度：O(n), 空间复杂度：O(1)'''


# In[ ]:


'''Given an integer n, return all the strobogrammatic numbers that are of length n. You may return the answer in any order.

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).'''

class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]: #该题要求找到所有的满足长度为n的strobogrammatic numbers组合
        dic = {"1" :"1", "6":"9", "9":"6", "0":"0", "8":"8"} #同样的我们把所有的正反看一样的字符放入dic中
        s_left = ["" for i in range(n//2)] #我们的策略是将结果字符串一分为二，左半边右半边分别存放在两个不同数组中，左侧的数组要小，右侧的数字大一点
        s_right = ["" for i in range(n-n//2)]#首先先确定数组长度，并且每个元素初始化为空

        def helper(s_left,s_right,index_left,index_right): #递归的思想解决， left数组从前向后，right数组从后向前
            if index_left >= len(s_left) and index_right <0: #如果left right数组的下标都达到最后和第一个，说明两个数组每个位置上都有元素了，把两个数组合并并且转化为字符串放到结果数组后面
                tmp = "".join(s_left)+"".join(s_right)#把两个数组合并并且转化为字符串放到结果数组后面
                res.append(tmp)
                return 
            elif index_left >=len(s_left) and index_right >=0: #因为左侧数组在长度为奇数的情况下，要比右侧数组短，该情况下，如果左侧数组到达尾部，右侧数组还会剩一个位置没放
                tmp = s_right #先存储目前的右侧数组
                for key, value in dic.items(): #从dic里面取正反相同数，因为只能再加一个元素，并且在正中间，该元素必须要是自对称元素，所以不能是6,9
                    if key != "6" and key != "9": #如果不是6,9，我们将该字符放入右侧数组第一个位置，进入递归
                        s_right[index_right] = key
                        helper(s_left,s_right,index_left,index_right -1)
                        s_right = tmp #然后进行回溯放入下一个符合条件的字符
            else: #如果两个指针都没有走到各自终点
                tmp_left = s_left #先提前记录下目前的两个数组，用于后续回溯
                tmp_right = s_right 
                for key, value in dic.items(): #从字典中取字符
                    if index_left != 0 and index_right != len(s_right) -1: #因为这要是一个数字，第一个元素不能是0，如果目前不是第一个元素，我们没有任何限制
                        s_left[index_left] = key #存储key在left数组里，value在right数组里，key value 放哪个无所谓，因为我们是遍历整个字典
                        s_right[index_right] = value
                        helper(s_left,s_right,index_left+1, index_right-1)
                        s_left = tmp_left #递归后回溯操作，返回当前层
                        s_right = tmp_right
                    else: #如果是首尾位置，我们要跳过0，进行同样的操作
                        if key != "0":
                            s_left[index_left] = key
                            s_right[index_right] = value
                            helper(s_left,s_right,index_left+1, index_right-1)
                            s_left = tmp_left
                            s_right = tmp_right
        res = []
        helper(s_left,s_right,0, len(s_right)-1)
        return res
    

    '''时间复杂度：O(5^N), 空间复杂度：O(n)'''


# In[ ]:


'''***You are playing a Flip Game with your friend.

You are given a string currentState that contains only '+' and '-'. You and your friend take turns to flip two consecutive "++" into "--". The game ends when a person can no longer make a move, and therefore the other person will be the winner.

Return true if the starting player can guarantee a win, and false otherwise.'''

class Solution:
    def canWin(self, currentState: str) -> bool:#博弈论，如果对手输了我就赢了
        for i in range(1, len(currentState)):
            if currentState[i] == currentState[i-1] == "+":#每遇到连续的两个+
                tmp = currentState[:i-1] + "--" + currentState[i+1:] #将这两个+变为-同时递归进行下一位的判断
                if not self.canWin(tmp): #当判断到对方不能赢时，说明我方赢，返回true
                    return True
        return False  #否则返回false
    
    
'''时间复杂度：O(n^n), 空间复杂度：O(n)'''


# In[ ]:


'''*Write an efficient algorithm that searches for a target value in an m x n integer matrix. The matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.'''
'''
1<2<3<4<5
^ ^ ^ ^ ^
3<4<5<6<7
^ ^ ^ ^ ^
5<6<7<8<9
^ ^ ^ ^ ^
7<8<9<10<11

'''


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:#策略是从右上角或者左下角进行判断，因为这两个角的元素大于（小于）所在行的，小于（大于）所在列的
        m = len(matrix[0])#取一行的元素个数#右上角为例，如果目标数字小于该数字，我们可以排除所在列的所有元素，返之我们可以排除所在行的所有元素，同时缩小搜索范围
        n = len(matrix)#取一列的元素个数
        i = 0
        j = m-1
        step = m + n -1 #step是元素所在行和列总元素个数，随着越往里层走，元素个数会逐次减1
        cur = matrix[i][j] #先从右上角开始
        while step > 0: #如果还没到左下角外侧
            if target < cur: #如果目标值小于目前的拐角值
                if j == 0: #如果该拐角值已经到了最左侧了，说明数组里面没有目标值，返回false
                    return False 
                j -= 1 #否则我们可以排除所在列，列数减1
                step -= 1  
                cur = matrix[i][j] #cur为下一个拐角值
            elif target > cur: #如果目标值大于该拐角值
                if i == len(matrix) -1: #如果已经到最后一行了，我们直接返回false
                    return False 
                i += 1  #否则我们排除所在行
                step -= 1 
                cur = matrix[i][j] #更新拐角值
 
            else: #如果等于拐角值，我们直接返回true
                return True 
        return False #如果到最后的左下角了，返回false
'''时间复杂度：O(m*n), 空间复杂度：O(1)'''


# In[ ]:


'''Given an array nums of integers, return how many of them contain an even number of digits'''

class Solution:
    def findNumbers(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            tmp = str(nums[i])
            if len(tmp) %2 ==0:
                count += 1 
        return count


# In[ ]:


'''*Given an array of integers nums and a positive integer k, find whether it's possible to divide this array into sets of k consecutive numbers
Return True if it is possible. Otherwise, return False.'''


class Solution:
    def isPossibleDivide(self, nums: List[int], k: int) -> bool:
        dic = dict()
        for i in range(len(nums)): #先将所有元素放入字典中，每个key的val是元素出现的次数
            if nums[i] in dic:
                dic[nums[i]] += 1 
            else:
                dic[nums[i]] = 1 
        arr = sorted(nums) #将数组排序
        for i in range(len(arr)): #从排序后的列表开头进行遍历，排序的目的是，确保我们所遍历到的元素为每组可能元素的第一个元素
            if arr[i] in dic: #因为后面当dic中key对应的val到0时我们会清除该key，所以这里我们需要判断是否arr[i]在字典中

                if dic[arr[i]] == 1: #如果在，但是只有一个没用时，我们删除该key，表明我们已经用完了数组中所有该key
                    dic.pop(arr[i])
                else:
                    dic[arr[i]] -= 1 #否则该key的数量减 1 代表用了一个
            else:
                continue #如果遍历到的元素已经用完了，我们不能用这个做开头，进入下一个遍历
            count = 1  #初始化长度为1，因为我们已经有了开头第一个元素
            tmp = arr[i] #初始化临时变量 = 第一个元素
            while count < k: #count大于等于k的时候结束遍历因为我们要的是一组 k个连续数
                tmp += 1  # 判断下一个连续数是否在dic中，在的话说明还有剩余的元素可以用
                if tmp  in dic:
                    if dic[tmp] == 1: #如果就剩一个，我们pop 该key
                        dic.pop(tmp)
                    else:
                        dic[tmp] -= 1  #否则我们用一个该key， 个数减一
                else:
                    return False  #如果没有下个连续数，我们无法形成 k 个连续数的数组，返回false
                count += 1 
        if not dic: #所有循环遍历完后，如果dic中还有剩余元素，说明还有多余的元素没用，这些元素也没法形成k个连续数的数组， 返回false
            return True  #如果字典是空，则返回true
        else:
            return False 
    
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''**Given a string s, return the maximum number of ocurrences of any substring under the following rules:

The number of unique characters in the substring must be less than or equal to maxLetters.
The substring size must be between minSize and maxSize inclusive.'''

class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:#这里的maxsize形同虚设，如果一个长度为5的字符串满足最多有maxletter个不一样的字符，那么长度比5小的一定也满足，而我们要找的又是最多有多少个满足条件的，我们尽量要选字符长度较小的，于是我们只看minsize
        def helper(size): #定义一个函数用于判断特定size的字符串的个数
            maxx = 0 
            dic = dict() #创立一个字典
            for i in range(len(s)-size+1):#遍历该字符串
                substr = s[i:size+i]#取出特定字符串
                tmp = set(list(substr)) #转化为set看看是否满足特点字符不超过maxletter
                if len(tmp) >maxLetters: #如果大于，我们跳过这个
                    continue
                if substr in dic: #否则如果该串字符串已经在dic中，个数加一
                    dic[substr] += 1 
                    maxx = max(maxx, dic[substr])  #同时判断目前为止的最多的满足情况的字符串       
                else:
                    dic[substr] = 1  #如果不在放入字典，并判断下maxx防止最多只有一个
                    maxx = max(maxx, dic[substr])         
            return maxx
        maxx = 0
        maxx = max(maxx, helper(minSize))
        return maxx
    
'''时间复杂度：O(n^2), 空间复杂度: O(n)'''

class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
        def helper(size):
            maxx = 0 
            dic = dict()
            substr = s[0:size]
            tmp = dict()
            for i in range(len(substr)):
                if substr[i] in tmp:
                    tmp[substr[i]] +=1 
                else:
                    tmp[substr[i]] = 1
            for i in range(len(s)-size+1):
                if i != 0:
                    if tmp[s[i-1]] == 1:
                        tmp.pop(s[i-1])
                    else:
                        tmp[s[i-1]] -= 1 

                    if s[i+size-1] in tmp:
                        tmp[s[i+size-1]] += 1
                    else:
                        tmp[s[i+size-1]] = 1 
                    substr = s[i:i+size]
                if len(tmp) >maxLetters:
                    continue 
                if substr in dic:
                    dic[substr] += 1 
                    maxx = max(maxx, dic[substr])        
                else:
                    dic[substr] = 1 
                    maxx = max(maxx, dic[substr])     
            return maxx
        maxx = 0
        maxx = max(maxx, helper(minSize))
        return maxx
'''时间复杂度：O(minSize*n), 空间复杂度: O(n)'''

class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
        dic_substr = dict()
        tmp = dict()
        substr = 0
        for i in range(minSize):
            if ord(s[i]) - ord('a')+1 in tmp:
                tmp[ord(s[i]) - ord('a')+1] += 1
            else:
                tmp[ord(s[i]) - ord('a')+1] = 1
            substr = (substr*26 + (ord(s[i]) - ord('a')+1))% (10**9+7)
        maxx = 0 
        for i in range(len(s)-minSize+1):
            if i != 0:
                if tmp[ord(s[i-1])-ord('a')+1] == 1:
                    tmp.pop(ord(s[i-1])-ord('a')+1)
                else:
                    tmp[ord(s[i-1])-ord('a')+1] -= 1
                substr = substr - ((26**(minSize-1))*(ord(s[i-1])-ord('a')+1))%(10**9+7)
                if ord(s[i+minSize-1])-ord('a')+1 in  tmp:
                    tmp[ord(s[i+minSize-1])-ord('a')+1] += 1 
                else:
                    tmp[ord(s[i+minSize-1])-ord('a')+1] = 1
                substr = (substr*26+(ord(s[i+minSize-1])-ord('a')+1))%(10**9+7)
            if len(tmp) >maxLetters:
                continue
            if substr in dic_substr:
                dic_substr[substr] += 1 
                maxx =max(maxx, dic_substr[substr])
            else:
                dic_substr[substr] = 1
                maxx = max(maxx, dic_substr[substr])
        return maxx
'''时间复杂度：O(n), 空间复杂度: O(n)'''


# In[ ]:


'''Given an array of strings products and a string searchWord. We want to design a system that suggests at most three product names from products after each character of searchWord is typed. Suggested products should have common prefix with the searchWord. If there are more than three products with a common prefix return the three lexicographically minimums products.

Return list of lists of the suggested products after each character of searchWord is typed. '''


class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        prod = products 
        heapq.heapify(prod) #将列表转化为堆，第一个元素就是ascii最小的元素
        res = []
        for i in range(len(searchWord)):# 从头遍历要搜索的单词
            tmp = [] #创建临时列表，用于存放每一位的搜索结果
            count = 0
            while  prod: #因为我们一直在往外pop元素，退出条件是prod为空时，我们相当于从小到大依次判断了每个字符串对应位置的字符是否与我们要搜索的字符一样
                if (i < len(prod[0]) and prod[0][0:i+1] != searchWord[0:i+1]) or (i >= len(prod[0])): #当我们搜到一个单词前 i个和目标单词前i个不等， 或者虽然全部相等，但是当前判断的字符串长度要小于搜索的目标字符串时，这个也不满足条件,直接跳过
                    heapq.heappop(prod) #将不符合条件的元素直接扔掉
                elif i < len(prod[0]) and prod[0][0:i+1] == searchWord[0:i+1]: #如果前i个字符都一样，那么说明当我们输到这一位字符时，是满足条件的，可以放入tmp中
                    if count == 3: #当我们已经找到三个时，退出该循环
                        break
                    count += 1 #否则count +1 
                    word = heapq.heappop(prod) #我们把满足的也弹出来，放入tmp中
                    tmp.append(word)
            res.append(tmp) #将tmp放入res， 如果是空的也要放
            for j in range(len(tmp)): #最后我们还要把上一轮弹出来但满足条件的再扔回堆里面， 下一轮还要用
                heapq.heappush(prod, tmp[j])
        return res
    
'''时间复杂度：O(n^2logn), 空间复杂度：O(n)'''    

'''Given an array of strings products and a string searchWord. We want to design a system that suggests at most three product names from products after each character of searchWord is typed. Suggested products should have common prefix with the searchWord. If there are more than three products with a common prefix return the three lexicographically minimums products.

Return list of lists of the suggested products after each character of searchWord is typed. '''


class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        arr= sorted(products)
        left = 0
        right = len(arr)-1
        res = []
        for i in range(len(searchWord)):
            tmp = []
            while (left <= right and i >= len(arr[left])) or (left <= right and i < len(arr[left]) and ord(arr[left][i]) < ord(searchWord[i])):
                left += 1 
            while (right >= left and i >= len(arr[right])) or (right >= left and i < len(arr[right]) and ord(arr[right][i]) > ord(searchWord[i])):
                right -= 1 

            if left <= right:
                k = left
                count = 0
                while k<= right:
                    tmp.append(arr[k])
                    k += 1 
                    count += 1 
                    if count == 3:
                        break
                res.append(tmp)
            else:
                res.append([])
        return res
'''时间复杂度：O(nlogn), 空间复杂度：O(1)'''


# In[ ]:


'''There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
Return the minimum number of candies you need to have to distribute the candies to the children.'''

class Solution:
    def candy(self, ratings: List[int]) -> int:
        length = len(ratings)
        res = [1 for i in range(length)]
        mark = []
        if length == 1:
            return 1
        def helper(index):
            for j in range(mark[0], index):
                if j == mark[0] and res[mark[0]] > 1+ index - j:
                    continue
                res[j] = 1+ index - j


        for i in range(1, length):
            if ratings[i] > ratings[i-1]:
                if len(mark) == 1:
                    helper(i-1)
                    mark = []
                res[i] = res[i-1] +1
            elif ratings[i]== ratings[i-1]:
                if len(mark) == 1:
                    helper(i-1)
                    mark = []
            else:
                if res[i-1] == 1:
                    if  mark == []:  
                        mark.append(i-1)
                    if i == length -1 :
                        helper(i)
                        mark = []
                else:
                    if i == length  - 1:
                        break
                    if mark == []:
                        mark.append(i-1)
        summ = sum(res)
        return summ
'''时间复杂度：O(N^2), 空间复杂度：O(N)'''

class Solution:
    def candy(self, ratings: List[int]) -> int:
        res = [1 for i in range(len(ratings))]
        for i in range(len(ratings)-1):
            if ratings[i+1] > ratings[i]:
                res[i+1] = res[i] + 1 
        for j in range(len(ratings)-1, 0, -1):
            if ratings[j-1] > ratings[j] and res[j-1] <= res[j]:
                res[j-1] = res[j] + 1 
        summ = sum(res)
        return summ
    
'''时间复杂度：O(N), 空间复杂度：O(N)'''


# In[ ]:


'''We can shift a string by shifting each of its letters to its successive letter.

For example, "abc" can be shifted to be "bcd".
We can keep shifting the string to form a sequence.

For example, we can keep shifting "abc" to form the sequence: "abc" -> "bcd" -> ... -> "xyz".
Given an array of strings strings, group all strings[i] that belong to the same shifting sequence. You may return the answer in any order.'''


class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        dic = dict()
        def same(string_1, string_2):
            for i in range(1, len(string_1)):
                tmp_1 = ord(string_1[i]) - ord(string_1[i-1])
                tmp_2 = ord(string_2[i]) - ord(string_2[i-1])
                if tmp_1 <0:
                    tmp_1 = 26 + tmp_1
                if tmp_2 <0:
                    tmp_2 = 26 + tmp_2
                if tmp_1 != tmp_2:
                    return False 
            return True 
        res = []     
        for i in range(len(strings)):
            if strings[i] == "*":
                continue
            tmp = [strings[i]]
            for j in range(i+1, len(strings)):
                if strings[j] == "*":
                    continue
                if strings[i] == strings[j] :
                    tmp.append(strings[j])
                    strings[j] = "*"
                elif len(strings[i]) == len(strings[j]) and same(strings[i], strings[j]):
                    tmp.append(strings[j])
                    strings[j] = "*"
                else:
                    continue
            strings[i] = "*"
            res.append(tmp)
        return res


# In[ ]:


'''
Given two stings ransomNote and magazine, return true if ransomNote can be constructed from magazine and false otherwise.

Each letter in magazine can only be used once in ransomNote.'''

class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        dic = dict()

        for i in range(len(magazine)):
            if magazine[i] in dic:
                dic[magazine[i]] += 1 
            else:
                dic[magazine[i]] = 1 
        for i in range(len(ransomNote)):
            if ransomNote[i] not in  dic or (ransomNote[i] in dic and  dic[ransomNote[i]]== 0):
                return False 

            elif ransomNote[i] in dic and  dic[ransomNote[i]]!= 0:
                dic[ransomNote[i]] -= 1 


        return True 


# In[ ]:


'''Given a string s, return the first non-repeating character in it and return its index. If it does not exist, return -1.'''


class Solution:
    def firstUniqChar(self, s: str) -> int:
        dic = dict()
        for i in range(len(s)):
            if s[i] in dic:
                dic[s[i]] = -1 
            else:
                dic[s[i]] = i
        minn = 1000000
        for value in dic.values():
            if value != -1:
                minn = min(value,minn)
        if minn ==1000000:
            return -1 
        return minn


# In[ ]:


'''Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Notice that you should not modify the linked list.'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        dic = dict()
        cur = head
        if not cur:
            return None
        while cur:
            if cur not in dic:
                dic[cur] = 1 
            else:
                dic[cur] += 1 
                break
            cur = cur.next
            if not cur:
                return None

        cur = head
        while cur:
            if dic[cur] == 2:
                return cur
            cur = cur.next

'''空间复杂度：O(1)'''            
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return None
        fast = head.next
        prev = head
        slow = head
        while True:
            if not fast or not fast.next:
                return None
            if fast == slow:
                break
            fast = fast.next.next
            slow = slow.next
            prev = prev.next.next
        mark = fast
        cur = None
        while True:
            if fast == prev:
                if prev == head:
                    return head
                cur = head
                tmp = head
                while cur.next != prev:
                    cur = cur.next
                prev = cur 
            elif cur and cur.next == fast:
                return fast
            fast = fast.next 


# In[ ]:



'''According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

The board is made up of an m x n grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

Any live cell with fewer than two live neighbors dies as if caused by under-population.
Any live cell with two or three live neighbors lives on to the next generation.
Any live cell with more than three live neighbors dies, as if by over-population.
Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Given the current state of the m x n grid board, return the next state.'''
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        board_copy = copy.deepcopy(board)
        
        directions = {(1,0), (-1,0),(0,1), (0,-1), (1,1), (1,-1),(-1,1),(-1,-1)}

        for i in range(len(board)):
            for j in range(len(board[0])):
                count = 0
                for x, y in directions:
                    new_x, new_y = x + i, y + j
                    if 0 <= new_x < len(board) and 0 <= new_y <len(board[0]):
                        if board_copy[new_x][new_y] == 1:
                            count += 1 

                if board_copy[i][j] == 1:
                    if count < 2 or count > 3:
                        board[i][j] = 0 

                    else:
                        board[i][j] == 1 


                else:
                    if count == 3:
                        board[i][j] = 1
                        

                        
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        dic = dict()
        for i in range(len(board)):
            for j in range(len(board[0])):
                dic[(i,j)] = board[i][j]
        directions = {(1,0), (-1,0),(0,1), (0,-1), (1,1), (1,-1),(-1,1),(-1,-1)}
        for i in range(len(board)):
            for j in range(len(board[0])):
                count = 0
                for x, y in directions:
                    new_x, new_y = x+i, y+j
                    if 0<= new_x <len(board) and 0 <= new_y <len(board[0]):
                        if dic[(new_x,new_y)] == 1:
                            count += 1 


                if board[i][j] == 1:
                    if count < 2 or count > 3:
                        board[i][j] = 0 

                    else:
                        board[i][j] == 1 


                else:
                    if count == 3:
                        board[i][j] = 1


# In[ ]:


'''Given an integer array nums, reorder it such that nums[0] <= nums[1] >= nums[2] <= nums[3]....

You may assume the input array always has a valid answer.'''

class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        max_heap = []
        for i in range(len(nums)):
            heapq.heappush(max_heap, -nums[i])
        min_heap = []
        for j in range(len(nums)):
            heapq.heappush(min_heap, nums[j])
        for i in range(len(nums)):
            if i %2 != 0:
                nums[i] = -heapq.heappop(max_heap)
            else:
                nums[i] = heapq.heappop(min_heap)
        
            


# In[ ]:


'''You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.'''

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        def helper(row):
            summ = len(matrix[0])-1 
            for i in range(row-1, len(matrix[0])-(row-1)-1):
                count = 1
                index_x = row-1
                index_y = i
                tmp = matrix[index_x][index_y]
                while count <=4:
                    tmp_2 = matrix[index_y][summ-index_x]
                    matrix[index_y][summ-index_x] = tmp
                    index_y, index_x = summ - index_x, index_y
                    count += 1
                    tmp = tmp_2

        for j in range(1, len(matrix)//2+1):
            helper(j)


# In[1]:



def nKnights(m = 8, n = 8):
    moves = {(1,2), (2,1), (1,-2),(-1,2),(-1,-2), (2,-1),(-2,1),(-2,-1)}
    buildboard = [[0 for i in range(8)] for j in range(8)]
    dic = set()
    for i in range(len(buildboard)):
        for j in range(len(buildboard)):
            dic.add((i,j))
    def dfs(buildboard, count, maxx, dic):
        
        if len(dic) == 0:
            
            if count >maxx:
                maxx = count 
                res = buildboard
            return 

        for i, j in dic:
            
                if (i,j) in dic:
                    continue

                buildboard[i][j] = 1 
                dic.add((i,j))
                dfs(buildboard, count+1,maxx,dic)
                buildboard[i][j] = 0
                dic.remove((i,j))

    res = []
    maxx = -1 
    dfs(buildboard, 0, maxx, dic)
    print(res)
    print(maxx)
          
nKnights()


# In[ ]:


'''Given a sorted integer array nums and an integer n, add/patch elements to the array such that any number in the range [1, n] inclusive can be formed by the sum of some elements in the array.

Return the minimum number of patches required.'''
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        count = 0 
        start = 0
        if nums[0] >1:
            count += 1
            start = 0
        else:
            start = 1 
        total = 1 
        i = start  
        while i < len(nums):
            while total <nums[i]-1 and total < n:
                total = (total + 1) * 2 -1 
                count += 1 
            total = nums[i] + total
            if total >= n:
                break   
            i += 1 

        while total < n and (i == len(nums)  or len(nums) == 1):
            total = (total + 1) * 2 -1 
            count += 1 
        return count
'''时间复杂度：O(n),  空间复杂度：O(1)'''


# In[2]:


'''find mean and std from data stream'''
import math
class maintain_mean_std():
    def __init__(self): #时间复杂度：O(1), 空间复杂度：O(1)
        self.nums = []
        self.mean = None
        self.std = None
        self.summ_sqe = 0
        self.summ = 0
    def insert_number(self, number):#时间复杂度：O(1), 空间复杂度：O(1)
        self.nums.append(number)
        if self.mean == None:
            self.mean = number
            self.std = 0
            self.summ = number
            self.summ_sqe = number**2
        else:
            self.summ = self.summ + number
            self.summ_sqe = self.summ_sqe +number**2
            length = len(self.nums)
            self.mean = self.summ / length
            self.std = math.sqrt(self.summ_sqe - 2* self.mean**2*length + length*self.mean**2)

    def print_mean_std(self):#时间复杂度：O(1),空间复杂度：O(1)
        print(self.mean)
        print(self.std)
      
    
obj = maintain_mean_std()
obj.insert_number(10)
obj.insert_number(10)
obj.insert_number(10)
obj.insert_number(9)
obj.insert_number(10)
obj.print_mean_std()


# In[ ]:


'''
Given a string s and a string array dictionary, return the longest string in the dictionary that can be formed by deleting some of the given string characters. If there is more than one possible result, return the longest word with the smallest lexicographical order. If there is no possible result, return the empty string.'''

class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        dic = defaultdict(list)#创建一个字典，用于存放每个字母出现的位置
        dictionary.sort() #对目标字典数组按照字典序排序
        for i in range(len(s)): #遍历一遍数组，将每个字母出现的位置放入字典中
            dic[s[i]].append(i)
        maxx = -1
        def check(st): #定义一个check函数，用于检测每个字符串是否可以通过删减s中的某些元素形成
            mark = -1  #mark变量用于记录上一个字母在字符串中的位置
            for j in range(len(st)): #遍历该字符串
                if dictionary[i][j] in dic: #如果该字符出现在了字典中
                    if j == 0: #如果是第一个字母，前面就没有其他字母
                        if len(st) == 1: #该情况下，如果该字符串长度就是1，直接返回满足条件
                            return True
                        mark = dic[dictionary[i][j]][0] #我们记录下这一位的字母在s串中首次出现的位置
                        continue #继续下一轮遍历
                    exist = False  #定义一个exist变量用于判断在上一位字母出现的位置后面是否有该字母
                    for ele in dic[dictionary[i][j]]: #遍历当前字母对应的位置数组
                        if ele > mark: #如果找到了有一个位置在上一个字母出现的位置后面，我们更新mark，同时exist置为true，直接break
                            mark = ele
                            exist = True
                            break
                    if exist == False: #如果不存在该位置，该字符串不满足，返回false
                        return False 
                    elif exist == True and j == len(st) -1: #否则，如果存在并且已经到了字符串的最后一位，返回true
                        return True
                else:
                    return False       #如果该字符不存在字典中，返回false
        res= ""
        for i in range(len(dictionary)): #遍历字典数组
            if check(dictionary[i]): #如果该字符串满足条件，判断该字符串长度是否为目前最长，是的则更新结果，因为数组已经按照字典序排序，长度相等时我们不需要额外判断
                if len(dictionary[i]) > maxx:
                    res = dictionary[i]
                    maxx = len(dictionary[i])
        return res
'''len(dictionary)记为m，len(s)记为n，dictionary中最长字符串记为x， 时间复杂度：O(n*x)'''    
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        def issubstring(sb):
            j = 0
            for i in range(len(s)):
                if j >=len(sb):
                    break
                if s[i] == sb[j]:
                    j += 1 

            return j == len(sb)
        maxx = -1 
        res = ""
        for i in range(len(dictionary)):
            if issubstring(dictionary[i]):
                if len(dictionary[i]) > maxx:
                    maxx = len(dictionary[i])
                    res = dictionary[i]
                
                elif len(dictionary[i]) == maxx:
                    res = min(dictionary[i], res)
        return res
'''时间复杂度：O(m*n),空间复杂度：O(1)'''


# In[ ]:


'''A city's skyline is the outer contour of the silhouette formed by all the buildings in that city when viewed from a distance. Given the locations and heights of all the buildings, return the skyline formed by these buildings collectively.

The geometric information of each building is given in the array buildings where buildings[i] = [lefti, righti, heighti]:

lefti is the x coordinate of the left edge of the ith building.
righti is the x coordinate of the right edge of the ith building.
heighti is the height of the ith building.
You may assume all buildings are perfect rectangles grounded on an absolutely flat surface at height 0.

The skyline should be represented as a list of "key points" sorted by their x-coordinate in the form [[x1,y1],[x2,y2],...]. Each key point is the left endpoint of some horizontal segment in the skyline except the last point in the list, which always has a y-coordinate 0 and is used to mark the skyline's termination where the rightmost building ends. Any ground between the leftmost and rightmost buildings should be part of the skyline's contour.

Note: There must be no consecutive horizontal lines of equal height in the output skyline. For instance, [...,[2 3],[4 5],[7 5],[11 5],[12 7],...] is not acceptable; the three lines of height 5 should be merged into one in the final output as such: [...,[2 3],[4 5],[12 7],...]'''


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        comb = [] #再创建一个用于存储所有位置的数组
        dic = defaultdict(list) #创建一个字典用于存放一个tuple数组，key为开始位置，值为（-该楼的高度，该楼的结束位置）
        dic_end = defaultdict(list)#该字典用于存放所有的开始位置对应的楼的高度
        dic_stt = defaultdict(list)#该字典用存放所有结束位置对应的楼的高度
        for i in range(len(buildings)): #遍历一遍buildings数组，添加相应的元素
            comb.append(buildings[i][0])
            comb.append(buildings[i][1])
            dic_end[buildings[i][1]].append(buildings[i][2])
            dic_stt[buildings[i][0]].append(buildings[i][2])
            dic[buildings[i][0]].append((-buildings[i][2], buildings[i][1]))

        heap_end = [] #该堆用于判断目前最高的楼的高度
        res = [] #用于存放结果
        height = 0 #初始化楼的高度为1
        comb.sort() #将comb数组排序
        for i in  range(len(comb)): #遍历comb数组
            tmp_height_1 = 0 #初始化临时变量，一个是给是start点的
            tmp_height_2 = 0 #一个是给是end点的
            if comb[i] in dic_stt: #如果当前遍历的点是在dic_stt字典中，说明有楼在这里开始
                tmp_height_1 = max(dic_stt[comb[i]]) #我们找到对应的数组中楼最高的那个，作为这个点的为开始点的最高高度（不一定是这个高度）
                for x in dic[comb[i]]: #同时我们把dic中该开始位置对应的元组数组中的所有元素放入大顶堆中，更新目前最高楼高
                    heapq.heappush(heap_end, x)
            mark = 0 #临时变量用于标记是否该位置有楼end
            if comb[i] in dic_end: #如果有楼end
                mark = 1 #置为1
                while heap_end and heap_end[0][1] <= comb[i] : #如果高度较大的楼的结束点在目前位置或者在目前位置的的前面，我们直接pop掉不考虑
                    heapq.heappop(heap_end)
                if heap_end: #如果大顶堆中还有元素，开头的元素一定是，之前的大楼的延续中最高的那个
                    tmp_height_2 = -heap_end[0][0] #我们把该高度传给tmp_height_2
            if height == max(tmp_height_1, tmp_height_2): #如果两个临时最大高度中最大的那个和上一个高度是一样的，说明该位置高度不变，不用记录
                continue
            if mark == 0 and height > tmp_height_1: #如果该位置没有楼结束，并且上一个高度比以该位置为起点的所有楼中最大高度要大，该位置高度也不变，我们不记录
                continue
        #tmp_height_2一定小于等于height
            height = max(tmp_height_1, tmp_height_2)#因为如果有楼结束的话，tmp_height_2一定是除了有新楼出来的最大高度，所以如果max(tmp_height_1, tmp_height_2)不是height，该位置高度一定改变
            res.append([comb[i], height]) #添加一个结果到数组

        return res
    
'''时间复杂度：O(nlogn),空间复杂度：O(n)'''


class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        dic_pair = defaultdict(list)
        arr = []
        for i in range(len(buildings)):
            arr.append(buildings[i][0])
            arr.append(buildings[i][1])
            
        for i in range(len(buildings)): 
            dic_pair[buildings[i][0]].append((-buildings[i][2],buildings[i][1]))
           
        arr = set(arr)
        arr = list(arr)
        arr.sort()
        heap = []
        res = []
        maxx = 0
        for i in range(len(arr)):
            while heap and heap[0][1] <= arr[i]:
                heapq.heappop(heap)
            if arr[i] in dic_pair:
                for ele in dic_pair[arr[i]]:
                    heapq.heappush(heap, ele)
            if heap and -heap[0][0] != maxx:
                maxx = -heap[0][0]
                res.append([arr[i],maxx])
            elif heap == []:
                maxx= 0
                res.append([arr[i],maxx])
        return res
'''时间复杂度：O(nlogn), 空间复杂度：O(n)'''


# In[ ]:


'''You have a pointer at index 0 in an array of size arrLen. At each step, you can move 1 position to the left, 1 position to the right in the array, or stay in the same place (The pointer should not be placed outside the array at any time).

Given two integers steps and arrLen, return the number of ways such that your pointer still at index 0 after exactly steps steps. Since the answer may be too large, return it modulo 109 + 7.'''

class Solution:
    def numWays(self, steps: int, arrLen: int) -> int: 
        max_right = min(steps//2,arrLen-1) #判断可以最远走多远，不能超过总步数的一半并且不能超过数组长度，注意，steps//2结果是下标位置，arrlen也要减一去代表下标位置
        res_arr = [[0 for i in range(max_right+1)] for j in range(steps+1)] #创建一个二维数组
        res_arr[steps][0] = 1 #先将初始位置置为1
        for i in range(steps-1, -1 ,-1): #开始向[0][0]位置遍历
            for j in range(max_right+1):
                if j-1>=0 and j+1 <= max_right: #转移方程为：f(i)(j) = f(i+1)(j+1) +f(i+1)(j-1) +f(i+1)(j)即为前一步（步数加一）向左到当前状态和向右到当前状态和不动到当前状态的总可能数
                    res_arr[i][j] =  res_arr[i+1][j] + res_arr[i+1][j-1] + res_arr[i+1][j+1]

                elif j-1 >= 0 and j+1 > max_right:
                    res_arr[i][j] = res_arr[i+1][j] + res_arr[i+1][j-1]

                elif j-1 < 0 and j+1 <= max_right:
                    res_arr[i][j] = res_arr[i+1][j] + res_arr[i+1][j+1]
                else:
                    res_arr[i][j] = res_arr[i+1][j]
                    
        return res_arr[0][0]%(10**9+7) #最后结果取余
                
            
'''时间复杂度：O(n*m),空间复杂度：O(n*m)'''      

class Solution:
    def numWays(self, steps: int, arrLen: int) -> int: 
        max_right = min(steps//2,arrLen-1)
        res_arr = [[0 for i in range(max_right+1)] for j in range(2)]
        res_arr[1][0] = 1
        mark = 0
        for i in range(steps-1, -1 ,-1):
            for j in range(max_right+1):
                if j-1>=0 and j+1 <= max_right:
                    res_arr[mark][j] =  (res_arr[1-mark][j] + res_arr[1-mark][j-1] + res_arr[1-mark][j+1])%(10**9+7)

                elif j-1 >= 0 and j+1 > max_right:
                    res_arr[mark][j] = (res_arr[1-mark][j] + res_arr[1-mark][j-1])%(10**9+7)

                elif j-1 < 0 and j+1 <= max_right:
                    res_arr[mark][j] = (res_arr[1-mark][j] + res_arr[1-mark][j+1])%(10**9+7)
                else:
                    res_arr[mark][j] = res_arr[1-mark][j]
            mark = 1- mark 
        return res_arr[1-mark][0]%(10**9+7)
            


# In[ ]:





# In[ ]:


'''Given an integer n, return any array containing n unique integers such that they add up to 0.'''

class Solution:
    def sumZero(self, n: int) -> List[int]:
        res = []
        if n % 2 == 1:
            res.append(0)
            n -= 1 
        while n >0:
            res.append(-n)
            res.append(n)
            n -= 2 
        return res
'''时间复杂度：O(n/2),空间复杂度：O(n)'''


# In[ ]:


'''*Given two binary search trees root1 and root2.

Return a list containing all the integers from both trees sorted in ascending order.'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        def in_order(node, res):
            if node is None:
                return 
            if node.left:
                in_order(node.left, res)
            res.append(node.val)
            if node.right:
                in_order(node.right, res)
            return res

        res1 = in_order(root1,[])
        res2 = in_order(root2,[])
        result = []
        if not res1 :
            return res2
        if not res2:
            return res1
        i  = 0
        j = 0
        while i < len(res1) or j < len(res2):
            if i < len(res1) and j < len(res2):
                if res1[i]<res2[j]:
                    result.append(res1[i])
                    i += 1
                else:
                    result.append(res2[j])
                    j += 1 
            elif i < len(res1):
                result.append(res1[i])
                i += 1
            else:
                result.append(res2[j])
                j += 1 
        return result
'''时间复杂度：O(n),空间复杂度：O(n)'''  


# In[ ]:


'''Given a string s formed by digits ('0' - '9') and '#' . We want to map s to English lowercase characters as follows:

Characters ('a' to 'i') are represented by ('1' to '9') respectively.
Characters ('j' to 'z') are represented by ('10#' to '26#') respectively. 
Return the string formed after mapping.

It's guaranteed that a unique mapping will always exist.'''

class Solution:
    def freqAlphabets(self, s: str) -> str:
        dic = {'1':'a', '2':'b','3':'c', '4':'d', '5':'e','6':'f','7':'g','8':'h','9':'i','10#':'j','11#':'k','12#':'l','13#':'m','14#':'n','15#':'o','16#':'p','17#':'q','18#':'r','19#':'s', '20#':'t', '21#':'u','22#':'v','23#':'w','24#':'x','25#': 'y', '26#':'z'}
        res = ''
        i = 0
        while i <len(s):
            if i+2 < len(s) and s[i+2] =='#':
                res = res + dic[s[i:i+3]]
                i = i+3
                continue 

            res= res + dic[s[i]]
            i += 1 

        return res
    
'''时间复杂度：O(n),空间复杂度：O(n)'''  


# In[ ]:


'''Given the array arr of positive integers and the array queries where queries[i] = [Li, Ri], for each query i compute the XOR of elements from Li to Ri (that is, arr[Li] xor arr[Li+1] xor ... xor arr[Ri] ). Return an array containing the result for the given queries.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/xor-queries-of-a-subarray'''

class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        res = []
        dic = dict()
        query_new =copy.deepcopy(queries)
        heapq.heapify(query_new)
        while query_new:
            if not dic :
                tmp = 0
                for i in range(query_new[0][0], query_new[0][1]+1):
                    tmp = tmp ^ arr[i]
                dic[(query_new[0][0], query_new[0][1])] = tmp
                mark = heapq.heappop(query_new)

            else:
                if mark[1]>=query_new[0][0] >= mark[0] :
                    if query_new[0][1] <= mark[1]:
                        tmp = dic[(mark[0], mark[1])]
                        for i in range(mark[0],query_new[0][0]):
                            tmp = tmp ^ arr[i]
                        for i in range(query_new[0][1]+1, mark[1]+1):
                            tmp = tmp ^ arr[i]
                        dic[(query_new[0][0], query_new[0][1])] = tmp
                        mark = heapq.heappop(query_new)
                    else:
                        tmp = dic[(mark[0], mark[1])]
                        for i in range(mark[0],query_new[0][0]):
                            tmp = tmp ^ arr[i]
                        for i in range(mark[1]+1, query_new[0][1]+1):
                            tmp = tmp ^ arr[i]
                        dic[(query_new[0][0], query_new[0][1])] = tmp
                        mark = heapq.heappop(query_new)
                else:
                    tmp = 0
                    for i in range(query_new[0][0], query_new[0][1]+1):
                        tmp = tmp ^ arr[i]
                    dic[(query_new[0][0], query_new[0][1])] = tmp
                    mark = heapq.heappop(query_new)

        res = []
        for i in range(len(queries)):
            res.append(dic[(queries[i][0],queries[i][1])])

        return res
'''简单的方法'''    
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        s = [ 0 for i in range(len(arr))] #先创立一个数组，用于存放到每一位为止的累计异或计算的结果
        s[0] = arr[0] # 首先初始化第一位，因为第一位之前没有数字，我们就放入该数字本身
        for i in range(1, len(arr)):#遍历一遍数组， s中每一位存放的是arr数组中对应下标之前的所有数字的异或计算结果，我们利用的是异或计算的交换律与结合律与可逆性
            s[i] = s[i-1] ^ arr[i]
        res = [] #创建结果数组
        for j in range(len(queries)): #遍历queries数组
            if queries[j][0] == 0: #如果查询的区间开头是0，我们直接取s中以区间结尾作为下标的值
                res.append(s[queries[j][1]])
            else:
                res.append(s[queries[j][1]]^s[queries[j][0]-1]) #否则我们就取s中区间右下标的值与s中区间左下标的值做异或运算，就得到区间中的数字的异或运算的值

        return res
'''时间复杂度：O(n),空间复杂度：O(n)'''  


# In[ ]:


'''Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def height(node):
            if node is None: 
                return -1
            left = height(node.left)
            right = height(node.right)
            return 1+ max(left,right)
        def search(node, maxx):
            if node is None:
                return maxx
                
            maxx = max(maxx, 1+height(node.left)+1+height(node.right))
            maxx1 = search(node.left, maxx)
            maxx2 = search(node.right,maxx)

            return max(maxx1,maxx2)

        return search(root,0)
    
    
'''时间复杂度：O(n^2),空间复杂度：O(1)''' 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def dfs(node):
            if node is None:
                return 0, 0

            #该方法的目标是，计算一个节点左右子树的深度，同时用该深度计算该节点的最大diameter
            depth_l, md_l = dfs(node.left)
            depth_r, md_r = dfs(node.right)
            depth = max(depth_l, depth_r) +1
            if node.left is None and node.right is None:
                depth = 0
                diameter = 0
            elif node.left is None:
                diameter = depth_r + 1

            elif node.right is None:
                diameter = depth_l + 1 

            else:
                diameter = depth_l + depth_r + 2

            md = max(md_l, md_r, diameter)

            return depth, md

        _,md = dfs(root)

        return md 
'''时间复杂度：O(n),空间复杂度：O(1)'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.ans = 0 #创建一个全局变量
        def height(node):
            if node is None:#如果当前节点是空，我们返回该层的长度为0
                return 0
            left = height(node.left) #以左节点为根节点向左遍历左子树
            right = height(node.right)#以右节点为根节点向右遍历右子树
            self.ans = max(self.ans, left+right)#更新当前的diameter最大值，diameter为左侧高度+右侧高度
            return 1+max(left, right) #该函数返回左子树与右子树中深度较大的作为上一层的左子树或者右子树的左子树或者右子树高度

        height(root)
        return self.ans
'''时间复杂度：O(n),空间复杂度：O(1)'''  


# In[ ]:


'''Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where each path's sum equals targetSum.

A leaf is a node with no children.'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        if root is None:
            return []
        node = collections.deque([root])
        val = collections.deque([[root.val]])
        summ_val = collections.deque([root.val])
        res = []
        while node:
            node_tmp = node.popleft()
            val_tmp = val.popleft()
            summ_tmp = summ_val.popleft()
            if node_tmp.left is None and node_tmp.right is None:

                if summ_tmp == targetSum:
                    res.append(val_tmp)

            if node_tmp.left:

                node.append(node_tmp.left)
                k = node_tmp.left.val
                tmp = val_tmp.copy()
                tmp.append(k)
                val.append(tmp)
                summ_val.append(summ_tmp+node_tmp.left.val)

            if node_tmp.right:
                node.append(node_tmp.right)
                k = node_tmp.right.val
                tmp = val_tmp.copy()
                tmp.append(k)
                val.append(tmp)
                summ_val.append(summ_tmp+node_tmp.right.val)

        return res
    
'''疑问：当使用语句：val.append(val_tmp.copy().append(node_tmp.right.val))报错说是deque 栈 nonetype， 分开来写就没事'''

'''空间复杂度：O(n),时间复杂度：O(n)'''


# In[ ]:


'''Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of the values along the path equals targetSum.

The path does not need to start or end at the root 


or a leaf, but it must go downwards (i.e., traveling only from parent nodes to child nodes).'''

'''这道题用到的思想是贪心，假设一条路径上的数字是：[4,3,2,4,1], 我们的target是5.我们每走到一个节点，将该节点的数字累计到该节点之前的所有节点中去
走到 i= 0：[4], 走到i=1：[7,3],i=2: [9,5,2] 这时候出现了一个5，说明最后一个数字到出现5的位置中所有元素的和为5，我们将个数加一，
继续向后i = 3: [13,9,6,4], i =4: [14,10,7,5] 这里又出现了5，我们个数再加一，这样是保证不会出现算重复的情况的。
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        def dfs(node,sumlist):
            if node is None: #如果走到头了，结束该遍历，并且返回0
                return  0
            sumlist = [x+node.val for x in sumlist] # 每条路径都有个sumlist，每到下一个节点，我们将该节点值累积到前面所有节点
            sumlist.append(node.val) #同时将该节点放进数组
            count = 0
            for i in range(len(sumlist)): #统计有几个5在里面
                if sumlist[i] == targetSum:
                    count += 1
            return count + dfs(node.left, sumlist) +dfs(node.right, sumlist) #统计子路径的5的个数
        if not root:
            return 0
        return dfs(root, [])
'''时间复杂度：O(n^2),空间复杂度：O(n)''' 


# In[ ]:


'''If the depth of a tree is smaller than 5, then this tree can be represented by an array of three-digit integers. For each integer in this array:

The hundreds digit represents the depth d of this node where 1 <= d <= 4.
The tens digit represents the position p of this node in the level it belongs to where 1 <= p <= 8. The position is the same as that in a full binary tree.
The units digit represents the value v of this node where 0 <= v <= 9.
Given an array of ascending three-digit integers nums representing a binary tree with a depth smaller than 5, return the sum of all paths from the root towards the leaves.

It is guaranteed that the given array represents a valid connected binary tree.'''

#我们计算每个节点使用的次数
class Solution:
    def pathSum(self, nums: List[int]) -> int:
        dic = dict()
        for i in range(len(nums)): # 因为每个元素的前两位分别代表深度和在每个深度从左到右的位置，又因为只有4层最多，所以前两位数字是唯一的，不重复的，我们将所有元素的前两位数拿出来先存入字典，并且val 为0
            dic[nums[i]//10] = 0 #val 用于存储每个节点用了几次，经过每个节点有多少个路径其实就用了几次
        if len(nums) == 1: #如果数组长度就1，我们直接返回该元素个位数就行
            return nums[0]%10
        for i in range(len(nums)-1, 0, -1): #从后向前遍历数组，因为数组中元素是按照一层一层从左到右排序的，最后一排最后一个在数组尾部
            if dic[nums[i]//10] == 0: # 如果当前遍历到的元素前缀在字典中的val是0，因为这个节点本身就是一个值要加，我们把val置为1
                dic[nums[i]//10] = 1 
                tmp = math.ceil(nums[i]//10%10/2)+(nums[i]//100-1)*10 #同时在字典中找到它上一层的父亲节点，将父亲节点使用次数加1，代表经过该父亲节点的有了一条路径
                dic[tmp] = dic[tmp] + dic[nums[i]//10]
            else: #如果遍历到val不为0，说明该节点一定不是叶子节点，该val所代表的数字就是该节点下有多少路径，我们直接将该值传递给该节点的父亲节点
                tmp = math.ceil(nums[i]//10%10/2)+(nums[i]//100-1)*10
                dic[tmp] = dic[tmp] + dic[nums[i]//10]
        summ = 0 #当到根节点时，字典中根节点的val就是这棵树有多少个路径，并且字典中的每个节点key的val都代表该节点为父亲节点有多少个路径，也就是要被用多少次
        for i in range(len(nums)): #最后遍历一遍数组将字典中所有的val *每个节点的值加起来就是结果
            summ = summ + dic[nums[i]//10]*(nums[i]%10)
        return summ
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''
Given a non-empty array nums containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.'''
#深度优先搜索
#第一种情况，如果该数组长度为1，肯定不可能返回false
#第二种情况，如果该数组和为奇数，也一定不可能，返回false
#第三种情况：
#数组和为偶数，该情况我们可以进行分割：
#定义当前一个数组的和为sum_1,另一个我们不用定义，，因为另一个的和即是sum - sum_1 
#遍历大数组， 每次遍历到一个元素，我们选择加或者不加，更新当前sum_1的值
#剪枝操作：如果当前值加上sum_1已经超过一半，就直接停止该branch的遍历
#剪枝操作：如果我们遍历的时候，出现两种的和相同的话，我们只需要取其中一种就可,我们用一个字典记录，当前位置之前所有和的可能
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if len(nums) == 1 :
            return False 
        if sum(nums) % 2 == 1:
            return False 
        summ = sum(nums)
        dic_sum = dict()
        def dfs(sum_1, index):
            if (index, sum_1) in dic_sum:
                return dic_sum[(index, sum_1)]
            if sum_1 > summ/2 or (sum_1 != summ/2 and index >= len(nums)):
                dic_sum[(index, sum_1)] = False 
                return False
            if sum_1 == summ/2:
                dic_sum[(index, sum_1)] = True 
                return True 
            result = dfs(sum_1 + nums[index], index + 1 ) or dfs(sum_1, index + 1)
            dic_sum[(index, sum_1)] = result
            return result
        return dfs(0, 0)
''''''


#动态规划
#每个位置有两种状态，一个是我们用该位置的数字，一个是我们不用，如果我们知道上一个状态的是否可以找到相应位置的目标值，我们就可以推导出这一个位置是否可以找到相应的目标值
#定义状态：我们定义dp[i][j] 为： 当我们有i个数字可以选择时，是否可以选择出和为j，是则值为true否则为false
#初始状态：dp[0][0] 的意义是：当我们没有数字可以选择时，是否可以选择出和为0，当然可以，所以为True
#         所有的dp[i][0] i > 0都为false，因为如果我们没有数字选择，不可能选择出和大于0的情况
#         当j<nums[i]时，dp[i][j-nums[i]]为false，因为如果我们目标和小于nums[i]，我们没法执意选择用nums[i]这个值
#状态转移方程：
#  dp[i][j] = dp[i-1][j] or dp [i-1][j-nums[i]] 其中，dp[i-1][j]不用当前的nums[i] 这个元素，跳过他，所以就是从前i-1中选，并且，目标和为j， dp[i-1][j-nums[i]] 为用当前这个元素，用了的话，我们就要先把他从目标和中去掉->j-nums[i]，去掉之后呢，我们的新目标和要从前n-1个元素中选。 只要这两种情况中有一个是true，该状态就是true我们可以找到这样的数字
#目标状态：
# dp[len-1][target]
#目标状态：target in 这三个数组中

#看解析，用时1.5小时， 时间复杂度：O(target*n), 空间复杂度：O(target*n)
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target = int(sum(nums)/2)
        if sum(nums) % 2 != 0:
            return False 
        if len(nums) == 1 :
            return False
        dp = [[False for i in range(target+1)] for j in range(len(nums))]
        dp[0][0] =True
        for i in range(1, target + 1):
            if nums[0] == i:
                dp[0][i] = True 
        for i in range(1, len(nums)):
            for j in range(1, target + 1):
                if j < nums[i]:
                    tmp = False 
                elif j == nums[i]:
                    tmp = True 
                else:
                    tmp = dp[i-1][j- nums[i]]
                dp[i][j] = dp[i-1][j] or tmp
        return dp[len(nums)-1][target]
    
    

    
#动态规划
#每个位置有两种状态，一个是我们用该位置的数字，一个是我们不用，如果我们知道上一个状态的是否可以找到相应位置的目标值，我们就可以推导出这一个位置是否可以找到相应的目标值
#定义状态：我们定义dp[i][j] 为： 当我们有i个数字可以选择时，是否可以选择出和为j，是则值为true否则为false
#初始状态：dp[0][0] 的意义是：当我们没有数字可以选择时，是否可以选择出和为0，当然可以，所以为True
#         所有的dp[i][0] i > 0都为false，因为如果我们没有数字选择，不可能选择出和大于0的情况
#         当j<nums[i]时，dp[i][j-nums[i]]为false，因为如果我们目标和小于nums[i]，我们没法执意选择用nums[i]这个值
#状态转移方程：
#  dp[i][j] = dp[i-1][j] or dp [i-1][j-nums[i]] 其中，dp[i-1][j]不用当前的nums[i] 这个元素，跳过他，所以就是从前i-1中选，并且，目标和为j， dp[i-1][j-nums[i]] 为用当前这个元素，用了的话，我们就要先把他从目标和中去掉->j-nums[i]，去掉之后呢，我们的新目标和要从前n-1个元素中选。 只要这两种情况中有一个是true，该状态就是true我们可以找到这样的数字
#目标状态：
# dp[len-1][target]
#目标状态：target in 这三个数组中

#看解析，用时1.5小时， 时间复杂度：O(target*n), 空间复杂度：O(target*n)
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        target = int(sum(nums)/2)
        if sum(nums) % 2 != 0:
            return False 
        if len(nums) == 1 :
            return False
        dp = defaultdict(lambda: defaultdict(lambda: -1))
        for i in range(1, len(nums)):
            dp[i][0] = 0
        for j in range(1, target+1):
            if nums[0] == j:
                dp[0][j] = 1
            else:
                dp[0][j] = 0
        dp[0][0] = 1
        def dfs(i,j):
            if dp[i][j] != -1:
                return dp[i][j]
            elif j < 0: #避免越界，越界就返回0 false
                dp[i][j] = 0
                return 0
            else:
                if dfs(i-1, j - nums[i]) == 1 :
                    dp[i][j] = 1
                    return 1
                else:
                    dp[i][j] = dfs(i-1,j)
                    return dp[i][j]

        if dfs(len(nums)-1,target) == 1:
            return True
        else:
            return False    
        


# In[ ]:


'''You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?'''
#我们可以走n步，一次要么走1步要么走两步
#如果n是1的话，我们只有一种可能，就是走一步到达顶部
#如果n不是1
#每次我们走一步或者两步，用cur_sp代表目前已经走的步数，每次我们选择走一步，或者两步进行递归，递归的结束条件就是cur_sp == steps
#剪枝操作，如果总的步数大于n的话，我们排除
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1 

        def dfs(cur_sp,count):
            if cur_sp > n:
                return 0
            if cur_sp == n:
                return 1
            count = count + dfs(cur_sp+1, 0) + dfs(cur_sp + 2,0)

            return  count

        return dfs(0,0)
'''超时'''
'''动态规划'''
class Solution:
    def climbStairs(self, n: int) -> int:
        f_0 = 1
        f_1 = 1 
        i = 1
        while i <n:
            tmp = f_1
            f_1 = f_1 +f_0
            f_0 = tmp
            i += 1 

        return f_1
'''时间复杂度：O(n), 空间复杂度: O(1)'''


# In[ ]:


'''
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

 '''
#可以向下走可以向又走
#定义一个变量cur为当前的的位置，count用于统计一共有多少种可能
#
#剪枝：当越界的时候停止当前branch遍历
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m == 1 or n == 1:

            return 1 
        arr = [[0  for i in range(n)]for j in range(m)]
        arr[0][0] = 1
        for i in range(m):
            for j in range(n):
                if j-1>=0 and i -1 >=0:
                    arr[i][j] = arr[i-1][j] + arr[i][j-1]

                elif j-1 >=0:
                    arr[i][j] = arr[i][j-1]
                elif i -1 >= 0:
                    arr[i][j] = arr[i-1][j]  
        return arr[m-1][n-1]
'''时间复杂度：O(m*n), 空间复杂度：:O(m*n)'''


# In[ ]:


'''The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return the number of distinct solutions to the n-queens puzzle.'''


class Solution:
    def totalNQueens(self, n: int) -> int:
        def dfs(row, i, count):
            if row == n:
                return 1 
            for j in range(n):
                if i +j in dic_dia1 or i-j in dic_dia2 or j in dic_col:
                    continue 
                dic_col.add(j)
                dic_dia1.add(j+i)
                dic_dia2.add(i-j)
                count = count + dfs(row+1,  i+1,0)
                dic_col.remove(j)
                dic_dia1.remove(j+i)
                dic_dia2.remove(i-j)

            return count 
        dic_col = set()
        dic_dia1 = set()
        dic_dia2 = set()
        return dfs(0, 0, 0)
'''时间复杂度：O(m*n), 空间复杂度：:'''


# In[ ]:


'''Given an integer array nums and an integer k, return true if it is possible to divide this array into k non-empty subsets whose sums are all equal.'''

'''自己做的：用时5个小时'''

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        if sum(nums) %k !=0:
            return False 
        target = sum(nums)/k
        dic_out = Counter(nums)
        self.dicarr = []
        def dfs(index,dic,summ, arr):
            if index >= len(arr) or summ >target:
                return False 
            if summ == target:
                if len(self.dicarr) ==0:
                    self.dicarr.append(dic.copy())
                return True 
            if dic[arr[index]] == 1:
                dic.pop(arr[index])
            else:
                dic[arr[index]] -= 1 
            result_1= dfs(index+1, dic, summ+arr[index], arr)
            if arr[index] not in dic:
                dic[arr[index]] = 1 
            else:
                dic[arr[index]] += 1 
            result_2 = dfs(index+1, dic, summ, arr)
            if result_1 == True or result_2 == True:
                return True
            else:
                return False
        count = 0 
        result = True
        arr = nums.copy()
        arr.sort(reverse = True)
        while count < k:
            tmp_result = dfs(0, dic_out, 0, arr)
            result = result and tmp_result
            if result is True:
                dic_out = self.dicarr[0].copy()
                self.dicarr = []
            if result == False:
                return False 
            arr = []
            for key, val in dic_out.items():
                cot = 0
                while cot < val:
                    arr.append(key)
                    cot += 1 
            arr.sort(reverse = True)
            count += 1 
            if sum(arr) == target and count == k -1 :
                return True
        return True
'''时间复杂度：O(K*N),空间复杂度：O(N)'''


# In[ ]:


'''
Given an integer n. No-Zero integer is a positive integer which doesn't contain any 0 in its decimal representation.

Return a list of two integers [A, B] where:

A and B are No-Zero integers.
A + B = n
It's guarateed that there is at least one valid solution. If there are many valid solutions you can return any of them.'''

'''自己做的：30分钟'''
class Solution:
    def getNoZeroIntegers(self, n: int) -> List[int]:
        s = str(n)
        if 0 < n < 10:
            return [1, n-1]
        if int(s[0])-1 == 0:
            s1 = ""  + '9' *(len(s)-1)
        else:
            s1 = "" + str(int(s[0])-1) + '9' *(len(s) - 1)
        print(s1)
        x1 = int(s1)
        x2 = n -x1
        while "0" in str(x2) or "0" in str(x1):
            x1  -= 1
            x2 += 1 

        return [x1, x2]
'''时间复杂度：O(n),空间复杂度：O(1)'''


# In[15]:


'''Given 3 positives numbers a, b and c. Return the minimum flips required in some bits of a and b to make ( a OR b == c ). (bitwise OR operation).
Flip operation consists of change any single bit 1 to 0 or change the bit 0 to 1 in their binary representation.'''

class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        s_a = str(bin(a))[2:]
        s_b = str(bin(b))[2:]
        s_c = str(bin(c))[2:]
        n = max(len(s_a), len(s_b), len(s_c))
        s_a = s_a.zfill(n)
        s_b = s_b.zfill(n)
        s_c = s_c.zfill(n)
        i = len(s_c) - 1 
        count = 0
        while i >=0:
            if int(s_a[i]) | int(s_b[i]) != int(s_c[i]):
                if s_c[i] == '1':
                    count += 1
                else:
                    if s_a[i] == '1' and s_b[i] == '1':
                        count += 2
                    else:
                        count += 1 
            i -= 1 
        return count
'''时间复杂度：O(n), 空间复杂度：O(n)'''



# In[19]:


'''# 现在背包里有n个硬币coins, 和一个阈值上限m; 要求拿走的硬币面值之和最大, 而且不超过m.
# example: coins = [2, 3, 3, 5, 7] m = 12
# answer = 12 (5, 7)
# example: coins = [2, 3, 5, 12], m = 6
# answer = 5 (2, 3)'''
'''from collections import defaultdict 

def takeCoins(coins:list, m:int):
    coins.sort()
    maxx = 0
    dic = defaultdict(tuple)
    def dfs(summ, prev, index):
        if (summ, index) in dic:
            return dic[(summ,index)]
        if index >= len(coins) and summ <=m:
            dic[(summ,index)] = summ
            return summ
        if summ > m:
            dic[(summ, index)] = summ - prev
            return summ - prev
        maxx = max(dfs(summ, prev, index+1), dfs(summ+coins[index], coins[index], index + 1))
        return maxx

    maxx = dfs(0,0,0)


coins = [5,44,66,77,88,99,55]
m = 234
takeCoins(coins, m)
'''

# 定义状态：对于每个硬币，我们都可以选择拿或者不拿，如果我们知道上一个状态的最大面值，我们可以推导出当前状态的最大面值，定义dp[i][j]: 从[0,i]的区间范围内能否找到总价值为j的硬币结合
# 初始状态：dp[i][j] = false 因为每个硬币的面值一定大于0，我们要在有一个硬币面值大于0的硬币里面找到总价值为0的，不可能的，所以为false
#动态转移方程：
# 第一种情况：
# 我们使用当前硬币，想要从上一个状态推到过来，总价值为j，因为用的当前的硬币，上一个状态的价值就是j-coins[i]， 所以就是dp[i-1][j-coins[i]]
# 第二种情况：
# 我们不使用当前硬币，所以该位置的总价值就是上一个位置的总价值，即dp[i-1][j]
# 这个两个情况只要有一个为true，就是true
# dp[i][j] = dp[i-1][j] or dp[i-1][j-coins[i]]

#目标状态：我们遍历完整个二维数组后，需要找出符合条件的j，最大的 j<m 并且dp[i][j] = True

def takeCoins(coins:list,m:int):
    dp = [[False for i in range(m+1)] for j in range(len(coins))]
    if len(coins) == 0:
        return -1
    for i in range(1, m+1):
        if i == coins[0]:
            dp[0][i] = True

    for i in range(1 , len(coins)):
        for j in range(1, m+1):
            if j <coins[i]:
                tmp = False
            elif j == coins[i]:
                tmp = True
            else:
                tmp = dp[i-1][j-coins[i]]

            dp[i][j] = tmp or dp[i-1][j]
    maxx = 0
    for i in range(len(coins)):
        for j in range(m+1):
            if dp[i][j] == True:
                maxx = max(maxx,j)

    return maxx

print(takeCoins([2, 3,  5, 12], 6))


# In[ ]:


'''Knight problem:'''

def nKnights(m = 6, n = 6):
    moves = {(1,2), (2,1), (1,-2),(-1,2),(-1,-2), (2,-1),(-2,1),(-2,-1)}
    buildboard = [[0 for i in range(m)] for j in range(n)]
    dic = dict()
    def dfs(buildboard, count, maxx, dic, start_i, start_j):
          if len(dic) == m*n:
            return count
          for i in range(start_i, m):
            for j in range(n):
                if i == start_i and j < start_j:
                    continue 
                if (i,j) in dic:
                    continue
                buildboard[i][j] = 1 
                dic[(i,j)] = 1 
                for k, l in moves:
                    if (i+k,j+l) not in dic and 0<= i+k <m and 0<= j+l < n:
                        dic[(i+k,j+l)] = 1 
                    elif (i+k, j+l) in dic:
                        dic[(i+k, j+l)] += 1 
                maxx = max(maxx, dfs(buildboard, count+1,maxx,dic,i,j))
                buildboard[i][j] = 0
                dic.pop((i,j))
                for k, l in moves:
                    if (i+k, j+l) in dic and dic[(i+k,j+l)] == 1:
                        dic.pop((i+k,j+l))
                    elif (i+k, j+l) in dic and dic[(i+k,j+l)] != 1:
                        dic[(i+k,j+l)] -= 1 
            return maxx
    res = []
    maxx = -1 
    maxx = dfs(buildboard, 0, maxx, dic,0,0)
    print(maxx)
nKnights()


# In[ ]:


'''You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).'''

'''看答案思路，思考了2个多小时，以前做过这题，但是超时了'''
#定义状态：buy_1[i] 定义为选择第i天第一次买入的最大总体利润
#         sell_1[i] 定义为选择第i天第一次卖出的最大总体利润
#         buy_2[i] 定义为选择第i天第二次买入的最大总体利润
#         sell_2[i] 定义为选择第i天第二次卖出的最大总体利润
#初始状态：
# 选择在第一天第一次买入，当天利润即消耗的成本：buy_1[0] = -prices[0]   
# 选择在第一天第一次卖出， 当天卖买了又卖了， 总利润为0： sell_1[0] = 0
# 选择在第一天第二次买入， 当天买了卖卖了买，总利润为当天成本： buy_2[0] = -prices[0]
# 选择在第一天第二次卖出，当天第二次卖了，总利润又为0了：sell_2[0] = 0
#
# 定义动态转移方程：
# 第i天买入状态的总利润为，选择在第i-1天买入与第i天买入中的利润的最大值
# buy_1[i]= max(buy_1[i-1],- price[i])
# 第i天第一次卖出状态的总利润为，选择在第i-1天卖出的利润与第i天卖出后的总利润中的最大值，第i天第一次卖出的利润为，当天价格+之前买入时的利润
# sell_1[i] = max(sell_1[i-1], price[i] + buy_1[i-1])
# 第i天第二次买入的总利润为， 选择在第i-1天第二次买入的利润与第i天第二次买入的总利润中的最大值，第i天第二次买入的利润为，之前第一次卖出所获得的总利润-第i天的股价
# buy_2[i] = max(buy_2[i-1], sell_1[i-1] - price[i])
# 第i天第二次卖出的总利润为，选择在第i-1天第二次卖出的利润与第i天第二次卖出的总利润中的最大值，第i天第二次卖出的利润为，当天的股价——之前第二次买入时的总利润
# sell_2[i] = max(sell_2[i-1], price[i] + buy_2[i-1])


#目标状态：因为我们可以选择交易一次或两次，因此最大利润为这两者中的最大值
# max(sell_1[n], sell_2[n])

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy_1 = [0 for i in range(len(prices))]
        sell_1 = [0 for i in range(len(prices))]
        buy_2 = [0 for i in range(len(prices))]
        sell_2 = [0 for i in range(len(prices))]
        buy_1[0] = -prices[0]
        sell_1[0] = 0
        buy_2[0] = -prices[0]
        sell_2[0] = 0
        for i in range(1,len(prices)):
            buy_1[i] = max(buy_1[i-1], -prices[i])
            sell_1[i] = max(sell_1[i-1], prices[i] + buy_1[i-1])
            buy_2[i] = max(buy_2[i-1], sell_1[i-1] - prices[i])
            sell_2[i] = max(sell_2[i-1], prices[i] + buy_2[i-1])
        return max(sell_2[-1], sell_1[-1])

'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times) with the following restrictions:

After you sell your stock, you cannot buy stock on the next day (i.e., cooldown one day).
Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).'''

'''自己独立完成，根据上一题的思路'''
#每一天有：1. 第一次买 2. 卖股票，3. 第k次买股票 
#第i天第一次买股票的利润：buy_1[i] = max(buy_1[i-1], -prices[i])
#第i天第k次买股票的利润：buy_k[i] = max(buy_k[i-1], sell[i-2] - prices[i])
#第i天卖股票的利润：sell[i] = max(sell[i-1], prices[i] + buy_1[i-1], prices[i] + buy_k[i-1])
#我们返回sell[-1]
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        buy_1 = [0 for i in range(len(prices))]
        buy_k = [0 for i in range(len(prices))]
        sell = [0 for i in range(len(prices))]
        if len(prices) == 1:
            return 0
        buy_1[0] = -prices[0]
        buy_1[1] = max(buy_1[0], -prices[1])
        buy_k[0] = -prices[0]
        buy_k[1] = max(buy_1[0], -prices[1])
        sell[0] = 0
        sell[1] = max(sell[0], prices[1]+buy_1[0], prices[1]+buy_k[0])

        for i in range(2, len(prices)):
            buy_1[i] = max(buy_1[i-1], -prices[i])
            buy_k[i] = max(buy_k[i-1], sell[i-2] - prices[i])
            sell[i] = max(sell[i-1], prices[i] + buy_1[i-1], prices[i] + buy_k[i-1])
        return sell[-1]
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.
'''
'''自己思考的过程：'''
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dic_text1 = Counter(text1)
        dic_text2 = Counter(text2)
        for i in range(len(text1)):
            if text1[i] in dic_text1 and text1[i] not in dic_text2:
                dic_text1.pop(text1[i])
        for i in range(len(text2)):
            if text2[i] not in dic_text1 and text2[i]  in dic_text2:
                dic_text2.pop(text2[i])
        if dic_text1 is None or dic_text2 is None:
            return 0
        new_text1 =""
        for i in range(len(text1)):
            if text1[i] in dic_text1:
                new_text1 = new_text1 + text1[i]
        new_text2 = ""
        for i in range(len(text2)):
            if text2[i] in dic_text2:
                new_text2 = new_text2 + text2[i]
        print(new_text1)
        print(new_text2)
        new_text1 = text1
        new_text2 = text2
        dic_text1 = dict()
        dic_text2 = dict()

        if len(new_text1) < len(new_text2):
            s_1 = new_text1.ljust(len(new_text2), ".")
            s_2 = new_text2
        else:
            s_1 = new_text2.ljust(len(new_text1), ".")
            s_2 = new_text1
        i = 0
        j = 0
        count = 0
        count_same = 0
        mark = 0
        while i < len(s_1) and j <len(s_2):
            #print(s_2)
            if s_1[i] != s_2[j] or (s_1[i] == s_2[j] and (s_1[i] in dic_text2 or s_2[j] in dic_text1)):
                if s_1[i] not in dic_text2 and s_2[j] not in dic_text1:
                    dic_text1[s_1[i]] = i
                    dic_text2[s_2[j]] = j
                    i += 1
                    j += 1 
                elif s_1[i] in dic_text2 and s_2[j] not in dic_text1:
                    if dic_text2[s_1[i]] < j
                    count += 1
                    
                    j = dic_text2[s_1[i]] +1 
                    i += 1 
                    dic_text1.clear()
                    dic_text2.clear()
                elif s_1[i] not in dic_text2 and s_2[j]  in dic_text1:
                    count += 1
                    i = dic_text1[s_2[j]] +1 
                    j += 1 
                    dic_text1.clear()
                    dic_text2.clear()
                else:
                    print(s_1[i])
                    print(dic_text2)
                    print(s_2[j])
                    print(dic_text1)
                    count += 1 
                    tmp_i = i
                    i = dic_text1[s_2[j]] +1
                    j = dic_text2[s_1[tmp_i]] +1 
                    dic_text1.clear()
                    dic_text2.clear()
            else:
                count_same = count + 1 
                i += 1
                j += 1 
        return count 
# 动态规划      
# 如果我们知道，每个字符串到目前位置上一位置的最长子序列，我们可以推导出到当前位置的最长子数列
#定义状态：
# arr[i][j] 为第一个字符串text1的到第j个位置与第二个字符串text2到第i个位置的最长子序列长度
#动态转移方程：
# 第一种情况：当前两个字符串相应位置的字符不一样， 这种情况下，arr[i][j]位置的最长子序列长度为text1退一个字符text2不变的最长子序列长度arr[i][j-1]与text1不变text2退一个字符的最长子序列中的最大值arr[i-1][j]
# arr[i][j] = max(arr[i][j-1], arr[i-1][j])

# 第二种情况：当两个字符串相应位置的字符一样，arr[i][j] 就是两个字符各向前退一个字符的最长子序列长度 + 1 即当前位置这一个
# arr[i][j] = arr[i-1][j-1] + 1 
#目标状态：
#arr[-1][-1] 即到达两个字符结尾的子序列长度
'''用时：5个小时，没有做出来，动态规划，当一样的字符时取斜对角这一步没有想到'''        
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m = len(text1)
        n = len(text2)
        arr = [[0 for i in range(m+1)] for j in range(n+1)]
        #如果text1[j-1] = text2[i-1], arr[i][j] = arr[i-1][j-1]+1， 原因是当遇到两个当前位置字符一样时，那么当前位置之前的最大子序列长度一定是两个序列分别往前退一位的对应在二维数组中的值，分别退一位，在二维数组中就是对角线位置
        #如果text1[j-1] != text2[i-1], arr[i][j] = max(arr[i-1][j], arr[i][j-1])
        for i in range(1, n+1):
            for j in range(1, m+1):
                if text1[j-1] == text2[i-1]:
                    arr[i][j] = arr[i-1][j-1] + 1 
                else:
                    arr[i][j] = max(arr[i-1][j], arr[i][j-1])
        return arr[n][m]
'''时间复杂度：O(n*m), 空间复杂度：O(n*m)'''


# In[ ]:


'''
You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.

Find the maximum profit you can achieve. You may complete at most k transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).'''

#动态规划
#一共可以交易k次， 我们如果知道上一次交易后的利润，就可以推到当前交易的利润
#定义状态：
#buy_k[i][j] 为第j天在第i次买股票后的总利润
#sell_k[i][j] 为第j天在第i次买股票后的总利润

#初始状态：所有的buy_k[0][j] = 都为在j位置之前每天股价中的最大值，因为i代表的是第i次购买
#         sell_k[0][j] 为当前j天之前的最小的成本与第j天股价的差值和sell_k[0][j-1]中的最大值
#动态转移方程：
#第一种情况：在第j天第i次买股票：当天获得的最大利润为之前卖掉股票所赚利润减去当天的股价,可以选择买或者不买，取决于哪一种的利润最大
# buy_k[i][j] = max(buy_k[i][j-1], sell_k[i-1][j-1]-prices[j])
# 第二种情况：在第j天第i次卖股票：
#sell_k[i][j] = max(sell_k[i][j-1], prices[j] + buy_k[i][j-1])
#目标动态：
#max(sell_k)

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        buy_k = [[ 0 for i in range(len(prices))] for j in range(k)]
        sell_k = [[0 for i in range(len(prices))] for j in range(k)]
        if prices == [] or len(prices) ==1:
            return 0 
        if k == 0:
            return 0
        maxx1 =-prices[0]
        maxx = 0
        for j in range(len(prices)):
            maxx1= max(maxx1, -prices[j])
            buy_k[0][j] = maxx1
            if j >= 1 :
                sell_k[0][j] = max(sell_k[0][j-1], prices[j]+maxx1)
                maxx = max(maxx, sell_k[0][j])
        for i in range(k):
            buy_k[i][0] = -prices[0]
        sell_k[0][0] = 0
        buy_k[0][0] = -prices[0]
        for i in range(1, k):
            for j in range(1, len(prices)):
                buy_k[i][j] = max( buy_k[i][j-1], sell_k[i-1][j-1]-prices[j])
                sell_k[i][j] = max( sell_k[i][j-1], prices[j]+buy_k[i][j])
                maxx = max(maxx,sell_k[i][j])
        return maxx

'''时间复杂度：O(k*n), 空间复杂度：O(k*n)'''


# In[ ]:


'''Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

A palindrome string is a string that reads the same backward as forward.'''
'''两个小时独立完成'''
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def judge(subs):
            if len(subs) % 2 ==0:
                i = len(subs) //2
                j = i -1 
                while j >= 0 and i <len(subs):
                    if subs[i] != subs[j]:
                        return False 
                    j -= 1
                    i += 1 
                return True
            else:
                i = len(subs)//2
                j = i -1 
                i = i + 1 
                while j >= 0 and i <len(subs):
                    if subs[i]!= subs[j]:
                        return False
                    j -=1 
                    i += 1
                return True
        self.res = []
        def dfs(index_right,st):
            if judge(s[index_right:]): #如果右侧的子串本身就是回文序列，我们就将该结果直接添加到res，因为index_right指针一直向右走，会将最后右侧只剩一个自付的情况包含进去
                st.append(s[index_right:]) 
                self.res.append(st.copy())
                st.pop()#用完之后就pop
            
            index_left = index_right #如果右侧不全是回文序列，因为我们已经确保了从left 到right是回文序列，我们将left指针放到right
            if index_left == len(s) : #如果left指针已经到了最后，我们就结束该branch遍历
                return 
            for index_right in range(index_left +1, len(s)): #否则遍历从left往后的一个个字符
                if judge(s[index_left:index_right]):#我们要确保index_right左侧的所有子串都是回文序列，然后进行判断右侧的子串
                    st.append(s[index_left:index_right])
                    dfs(index_right,st)
                    st.pop()  
        dfs(0,[])
        return self.res


# In[ ]:


'''There is a row of n houses, where each house can be painted one of three colors: red, blue, or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by an n x 3 cost matrix costs.

For example, costs[0][0] is the cost of painting house 0 with the color red; costs[1][2] is the cost of painting house 1 with color green, and so on...
Return the minimum cost to paint all houses.'''


        self.minn = 1000000000

        def dfs(prev_col, house, cost):
            if  cost > self.minn:
                return 
            if house >= len(costs):
                self.minn = min(cost, self.minn)

                return  
            if prev_col != 0:
                dfs(0, house +1, cost + costs[house][0])
            if prev_col != 1:
                dfs(1, house +1, cost + costs[house][1])
            
            if prev_col != 2:
                dfs(2, house +1, cost + costs[house][2])
        dfs(-1,0,0)
        return self.minn
    
    
#动态规划

#状态分析：
#我们想要知道当前最小的paint成本，如果我们知道上一个状态的最小成本，我们可以推导出当前状态的最小成本
#状态定义：
# 我们用dp[i][j] 代表第i个房子paint 第j种颜料的总成本
#初始化状态：
# dp[0][j] = cost[0][j] 因为第一个房子没有前面房子的限制，可以涂任意颜色，所以就是分别的三个方案
# 动态转移方程：
# dp[i][j] = costs[i][j] + min(dp[i-1][!j]) 其中!j表示所有非一样颜色的颜色。该方程的意义就是当前房子的最小总体成本就是当前该方案的成本加上上一个房子非一样颜色方案中的最小成本

# 目标状态：min(dp[len(costs)-1])

'''独立完成，2小时，一开始想复杂了'''
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        dp = [[0 for i in range(len(costs[0]))] for j in range(len(costs))]
        dp[0][0] = costs[0][0]
        dp[0][1] = costs[0][1]
        dp[0][2] = costs[0][2]

        for i in range(1, len(costs)):
            for j in range(3):
                if j == 0:
                    dp[i][j] = costs[i][j] + min(dp[i-1][1], dp[i-1][2])
                    
                elif j ==1:
                    dp[i][j] =costs[i][j] + min(dp[i-1][0], dp[i-1][2])

                else:
                    dp[i][j] =costs[i][j] + min(dp[i-1][0], dp[i-1][1])
        return min(dp[len(costs)-1])
'''时间复杂度:O(3n), 空间复杂度：O(3n)'''


# In[ ]:


'''
There are a row of n houses, each house can be painted with one of the k colors. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by an n x k cost matrix costs.

For example, costs[0][0] is the cost of painting house 0 with color 0; costs[1][2] is the cost of painting house 1 with color 2, and so on...
Return the minimum cost to paint all houses.'''

class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        dp  = [[0 for i in range(len(costs[0]))] for j in range(len(costs))]
        minn = []
        hp = []
        for i in range(len(costs[0])):
            dp[0][i] = costs[0][i]
        for i in range(1, len( costs)):
            for j in range(len(costs[0])):
                if j == 0:
                    left = 10000000
                else:
                    left = min(dp[i-1][0:j])
                if j == len(costs[0])-1:
                    right = 100000000
                else: 
                    right = min(dp[i-1][j+1:])
                dp[i][j] = costs[i][j] + min(left,right)
        return min(dp[len(costs)-1])


# In[ ]:


'''We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I will tell you whether the number I picked is higher or lower than your guess.

You call a pre-defined API int guess(int num), which returns 3 possible results:

-1: The number I picked is lower than your guess (i.e. pick < num).
1: The number I picked is higher than your guess (i.e. pick > num).
0: The number I picked is equal to your guess (i.e. pick == num).
Return the number that I picked.'''

# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num: int) -> int:

class Solution:
    def guessNumber(self, n: int) -> int:
        left = 1
        right = n
        while left <= right:
            mid= (left +right)//2
            if guess(mid)  == -1:
                right = mid -1
            elif guess(mid) == 0:
                return mid
            else:
                left = mid + 1 
'''时间复杂度：O(logn), 空间复杂度：O(1)'''


# In[ ]:


#动态规划
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp = [ 0 for i in range(n+1)]
        if len(dp) ==1 or len(dp) ==2:
            return 0
        elif len(dp) ==3:
            return 1
        elif len(dp) == 4:
            return 2
            
        dp[0] = 0
        dp[1] = 0
        dp[2] = 1
        dp[3] = 2
        def judge(n):
            maxx = -1
            minn = 10000000
            i = n
            tmp_right = 0
            while i >= 0:
                tmp_right = tmp_right + 2*i - 4
                tmp_left = dp[i-4] + i-3
                tmp_summ = max(tmp_left  tmp_right)
                if tmp_summ > minn:
                    return minn
                else:
                    minn = tmp_summ
                i = i-4
                
        for i in range (4, n+1):



            max_right = i -1 + i-
            max_left = dp[i-4] + i-3
            print(max_left)
            print(max_right)
            dp[i] = max(max_left,max_right)
            print(dp)
            print("\n")
  
        return dp[n]
'''We are playing the Guessing Game. The game will work as follows:

I pick a number between 1 and n.
You guess a number.
If you guess the right number, you win the game.
If you guess the wrong number, then I will tell you whether the number I picked is higher or lower, and you will continue guessing.
Every time you guess a wrong number x, you will pay x dollars. If you run out of money, you lose the game.
Given a particular n, return the minimum amount of money you need to guarantee a win regardless of what number I pick.'''

#动态规划
#当我们随机猜一个数时，我们有可能猜大也有可能猜小了，再猜的时候我们考虑我们猜法中损失最大的猜法损失是多少，不断根据提示继续猜，继续考虑每个细分的猜法的最大损失。总共有n个数字可以猜，我们先计算其中的一小段[i,j]中的最大损失。再逐渐扩大数字范围至[1,n]。
# 定义状态：
# dp[i][j] 代表从 [i,j] 范围内猜数字，最大的成本是多少，计算过程：我们从数字i开始，一个一个作为我们猜的数字，计算相应的最大损失是多少，之后取所有可能损失中的最大值
# 初始状态：
# dp[i][i] = 0 如果我们只有一个数字可以选择，那我们成本一定是0
#状态转移方程：
# pivot 作为我们选择猜的数字
# dp[i][j] = min(pivot +max(dp[i][pivot -1], dp[pivot+1][j]))
#目标状态：
# dp[1][n]

'''用时1天，看解析，有点想法，但是还是想歪了'''
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp = [[0 for i in range(n+1)] for j in range(n+1)]
        for i in range(1, n):
            j = 1 +i
            tmp_i = 1
            while j < n+1:
                minn = 10000000
                for pivot in range(tmp_i, j+1):
                    if pivot +1 >=n+1:
                        maxx = max(dp[tmp_i][pivot -1], 0) + pivot 
                    else:
                        maxx = max(dp[tmp_i][pivot -1], dp[pivot+1][j]) + pivot
                    minn = min(maxx, minn)
                dp[tmp_i][j] = minn
                tmp_i += 1
                j += 1 
        return dp[1][n]
'''时间复杂度：O(n^2/2), 空间复杂度：O(n^2)'''
 


# In[ ]:


'''You are given an array of transactions transactions where transactions[i] = [fromi, toi, amounti] indicates that the person with ID = fromi gave amounti $ to the person with ID = toi.

Return the minimum number of transactions required to settle the debt.

Input: transactions = [[0,1,10],[2,0,5]]
Output: 2
Explanation:
Person #0 gave person #1 $10.
Person #2 gave person #0 $5.
Two transactions are needed. One way to settle the debt is person #1 pays person #0 and #2 $5 each.'''


class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        dic = defaultdict(list)
        name = set()
        for i in range(len(transactions)):
            name.add(transactions[i][0])
            name.add(transactions[i][1])
        n = len(name)
        for i in range(n):
            dic[name.pop()]= [0 for j in range(2)]
        for i in range(len(transactions)):
            dic[transactions[i][0]][0] += -transactions[i][2]
            dic[transactions[i][1]][1] += transactions[i][2]
        pos = []
        neg = []
        #到这的时间复杂度：O(n), 空间复杂度：O(n)
        for val in dic.values():
            if val[0] + val[1] > 0:
                pos.append(val[0]+val[1])
            elif val[0] +val[1] < 0:
                neg.append(val[0]+val[1])
        if len(pos) == 0:
            return len(neg)
        if len(neg) == 0:
            return len(pos)
        self.minn = 1000000
        dic_pos = Counter(pos)#先求出每个人的收支余额，负数放一个集合，正数放一个集合
        dic_neg = Counter(neg)#递归每一种组合的可能, 适当剪枝，如果有的组合没到最后count就超过了之前的最小count，我们就停止该branch回溯
        def helper(index_pos, index_neg, val, count, dic_p, dic_n):
            if count >self.minn:
                return 
            if not dic_p and not dic_n and val == 0:
                self.minn = min(self.minn, count)
                return 
            elif dic_p and not dic_n:
                if val != 0:
                    for val in dic_p.values():
                        count = count +val
                    self.minn = min(self.minn, count)
                    return 
            elif dic_n and not dic_p:
                for val in dic_n.values():
                    count = count +val
                self.minn = min(self.minn, count)
                return 
            else:
                if val == 0:
                    for i in range(index_pos, len (pos)):
                        if pos[i] not in dic_p:
                            continue
                        else:
                            dic_p[pos[i]] -= 1 
                            if dic_p[pos[i]] == 0:
                                dic_p.pop(pos[i])
                        for j in range(index_neg, len(neg)):
                            if neg[j] not in dic_n:
                                continue
                            else:
                                dic_n[neg[j]] -= 1 
                                if dic_n[neg[j]] == 0:
                                    dic_n.pop(neg[j])
                            val = pos[i] + neg[j]
                            helper(0, 0, val, count + 1, dic_p, dic_n)
                            if neg[j] not in dic_n:
                                dic_n[neg[j]] = 1
                            else:
                                dic_n[neg[j]] += 1 
                        if pos[i] not in dic_p:
                            dic_p[pos[i]] = 1
                        else:
                            dic_p[pos[i]] += 1       
                elif val < 0:
                    for i in range(index_pos+1, len(pos)):
                        if pos[i] not in dic_p:
                            continue
                        else:
                            dic_p[pos[i]] -= 1 
                            if dic_p[pos[i]] == 0:
                                dic_p.pop(pos[i])
                        val = val + pos[i]
                        helper(0, 0, val, count + 1, dic_p, dic_n)
                        if pos[i] not in dic_p:
                            dic_p[pos[i]] = 1
                        else:
                            dic_p[pos[i]] += 1 
                elif val > 0:
                    for i in range(index_neg+1, len(neg)):
                        if neg[i] not in dic_n:
                                continue
                        else:
                            dic_n[neg[i]] -= 1 
                            if dic_n[neg[i]] == 0:
                                dic_n.pop(neg[i])
                        val = val + neg[i]
                        helper(0,0, val, count +1,dic_p,dic_n)
                        if neg[i] not in dic_n:
                            dic_n[neg[i]] = 1
                        else:
                            dic_n[neg[i]] += 1 
        #时间复杂度：O(count1+ count2+...+countk), 空间复杂度：O(n)
        helper(0,0,0,0, dic_pos, dic_neg)
        return self.minn


# In[ ]:


'''
You are given an array of binary strings strs and two integers m and n.

Return the size of the largest subset of strs such that there are at most m 0's and n 1's in the subset.

A set x is a subset of a set y if all elements of x are also elements of y.

 '''
'''自己思考的过程：在left[0]==down[0]的条件下如何操作这里出现问题，尝试修复失败'''
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:   
        dic_count = dict()
        for i in range(len(strs)):
            count_1 = 0
            count_0 = 0
            for j in range(len(strs[i])):
                if strs[i][j] == "0":
                    count_0 += 1
                else:
                    count_1 += 1 
                
                dic_count[i] = (count_0, count_1)

        board = [[[0 for i in range(3)] for j in range(len(strs))] for k in range(len(strs))]
        for i in range(len(strs)):
            if dic_count[i][0] <= m and dic_count[i][1] <= n:
                board[i][i][0] = 1 
                board[i][i][1] = m - dic_count[i][0]
                board[i][i][2] = n - dic_count[i][1]
            else:
                board[i][i][0] = 0
                board[i][i][1] = m
                board[i][i][2] = n

        print(board)
        dic = defaultdict(list)
        for i in range(1, len(strs)):
            j = i
            k = 0
            while j <len(strs):
               
                if board[k][j-1][1] - dic_count[j][0] >=0 and board[k][j-1][2] - dic_count[j][1] >= 0:
                    left = [board[k][j-1][0]+1, board[k][j-1][1] - dic_count[j][0],board[k][j-1][2] - dic_count[j][1]]

                else:
                    left = board[k][j-1].copy()  

                if board[k+1][j][1] - dic_count[k][0] >= 0 and board[k+1][j][2] - dic_count[k][1] >=0:
                    down = [board[k+1][j][0]+1, board[k+1][j][1] - dic_count[k][0],board[k+1][j][2] - dic_count[k][1]]
                else:
                    down = board[k+1][j].copy()

                if left == down:
                    board[k][j] = left
                elif left[0] > down[0]: 
                    #print("left[0]:", left[0])
                    #print("down[0]:", down[0])
                    #print("\n")
                    board[k][j] = left
                elif left[0] == down[0]:  # 当遇到最大的set长度相同时，如何取两个中的值
                    if left[1] >=0 or left[2] >=0:
                        board[k][j] = left
                    elif down[1] >= 0 or down[2] >= 0:
                        board[k][j] = down
                
                elif down[0] > left[0]:
                    board[k][j] = down
                j += 1
                k += 1 
        print(board)

        return board[0][len(strs)-1][0]

    
'''用时：6个小时，看解析，0-1背包问题升级版
时间复杂度：O(len(strs)*m*n), 空间复杂度:O(len(strs)*n*m)'''    
#board[i][j][k] = board[i-1][j][k] if j < count(0) or k < count(1)
#               = max(board[i-1][j-count(0)][k-count(1)]+1, board[i-1][j][k]) if j >= count(0) and k >= count(1)
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        count_0 = 0
        count_1 = 0
        board = [[[0 for i in range(n+1)] for j in range(m+1)] for k in range(len(strs))]
        for i in range(len(strs[0])):
            if strs[0][i] == "0":
                count_0 += 1
            else:
                count_1 += 1 
        for j in range(0, m+1):
            for k in range(0,n+1):
                if j < count_0 or k < count_1:
                    board[0][j][k] = 0 
                else:
                    board[0][j][k] = 1 
        for i in range(1, len(strs)): #和常规的0-1背包问题一样，我们从头开始逐步增加字符串的数量，并从确定的范围里选取j 个0， k个1， 求出最大的每种情况最多可以拿有几个字符串
            count_0 = 0
            count_1 = 0
            for l in range(0,len(strs[i])): #每次增加一个字符我们首先统计该字符中0,1的个数
                if strs[i][l] == "0":
                    count_0 += 1
                else:
                    count_1 += 1 
            for j in range(0, m+1): #我们从给定的范围内拿j个0，k个1
                for k in range(0, n+1): 
                    if j < count_0 or k < count_1: #我们可以选择拿比上次多出来的那个字符串，也可以不拿，但是如果j，k都小于该字符串中的0,1个数，我们不能拿，因为拿了的话，就会成为负数，我们不能总共拿4个0却只有3个0给我们拿
                        board[i][j][k] = board[i-1][j][k] #所以该值就是上一位的值
                    else:
                        board[i][j][k] = max(board[i-1][j][k], 1+board[i-1][j-count_0][k-count_1]) #否则，我们可以拿或者不拿，拿的话就是上一位的去掉我们拿了的这一位里面的所有0,1的最大长度，不拿就是上一位的最大长度

        return board[len(strs)-1][m][n]


# In[ ]:


'''
Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.

 '''



#定义状态：我们遍历整个二维数组， 遍历时将每个元素作为假象的一个正方形的右下角，从边长为1的正方形逐渐扩大，直到该范围内出现0为止，记录每个元素作为右下角时的正方形个数，最后将每个点的正方形个数累计起来

#初始状态：
# 定义dp[i][j] 为(i,j)点为正方形右下角的正方形总个数，如过matrix[i][j] = 0 dp[i][j] 就是0

#状态转移：
# 我们要求dp[i][j] 的正方形个数，我们可以查询该点左边，斜对角 和上边的点分别为左下角的正方形个数，该点的正方形个数就是这三个点中的最小值+1：
# dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) +1 

# 目标状态： sum(dp[i][j] for i < len(matrix) and j <len(matrix[0]))

class Solution:
    def countSquares(self, matrix: List[List[int]]) -> int:
        dp = [[0 for i in range(len(matrix[0])+1)] for j in range(len(matrix)+1)]

        for i in range(1, len(matrix)+1):
            for j in range(1, len(matrix[0])+1):
                if matrix[i-1][j-1] == 0:
                    dp[i][j] = 0
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 
        summ = 0
        for i in range(len(dp)):
            for j in range(len(dp[0])):
                summ = summ + dp[i][j]
        return summ


# In[ ]:


'''There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique'''


class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        def dfs(start, index, gass):
            if gass < 0:
                return -1
            elif index == start:
                return start   
            else:
                gass = gass +gas[index]
                if index+ 1 >= len(cost):
                    gass = gass - cost[index] 
                    index = 0
                else:
                    gass = gass - cost[index] 
                    index += 1 
                return dfs(start, index, gass)
        result = -1
        if len(cost) == 1:
            if cost[0] > gas[0]:
                return -1 
            else:
                return 0
        for i in range(len(gas)):  
            if gas[i] - cost[i] <= 0:
                continue
            gass = gas[i] - cost[i] 
            if i  == len(gas) -1 :
                index = 0
            else:
                index = i + 1 
            result = dfs(i,index, gass)
            if result != -1:
                return result

        return -1 

'''自己独立思考，用时5小时
#当我们进入遍历arr数组时就已经确定一定存在一个加油站可以走一圈，遍历arr数组，判断哪一个加油站的思路是，从一个非负的位置开始，（该数字代表的是每次经过一个加油站到下一个加油站的油的盈亏），
不断累加盈亏和，如果没有到最后并且和变成负数，该位置不会能回到原点也说明在这一段中都不会是起始位置，
因为该段中下一个正数向后累加和一定是负数因为少了前面的那个正数。所以就从出现负数的那一位置开始找下一个大于0的盈亏再往后累加遍历，直到出现
到数组结尾累计盈亏和一直为正数的加油站。'''
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:

        arr = [0 for i in range(len(gas))]
        neg = 0
        pos = 0
        cot_neg = 0
        cot_pos = 0
        for i in range(len(gas)): # 将每个加油站可以加的油-该加油站到下一个加油站所需要的成本，放入到arr数组中， 并统计正负数的各自的和
            arr[i] = gas[i] - cost[i]
            if arr[i] <0:
                neg += arr[i]
                cot_neg += 1
            else:
                pos += arr[i]
                cot_pos += 1 
        if -neg > pos: #因为如果要回到同一个加油站，不管是哪个加油站，都需要走过全程，所以如果消耗的和大于加油的总和，是没法走完全程的，直接返回-1
            return -1 
        i = 0
        while i <len(arr):
            if arr[i] >= 0:
                j = i + 1 
                summ = arr[i] 
                while j <len(arr):
                    summ += arr[j]
                    if summ < 0: # 如果和为负数停止遍历
                        break 
                    j += 1
                if j == len(arr) and summ >=0: #如果是已经到最后一位并且和一直为正数，找到了该加油站并返回i
                    return i 
                i = j #否则从j的位置开始向后遍历
            else: #跳过所有小于0的位置
                i += 1 
        return -1
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''Given a positive integer num consisting only of digits 6 and 9.

Return the maximum number you can get by changing at most one digit (6 becomes 9, and 9 becomes 6).'''

class Solution:
    def maximum69Number (self, num: int) -> int:
        s = str(num)
        for i in range(len(s)):
            if s[i] == '6':
                s = s[:i] + '9' + s[i+1:]
                break

        return int(s)
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''Given a string s. Return all the words vertically in the same order in which they appear in s.
Words are returned as a list of strings, complete with spaces when is necessary. (Trailing spaces are not allowed).
Each word would be put on only one column and that in one column there will be only one word.'''

class Solution:
    def printVertically(self, s: str) -> List[str]:
        maxx = 0
        mark = 0
        arr_s = s.split()
        for i in range(len(arr_s)):
            if len(arr_s[i]) >= maxx:
                maxx = len(arr_s[i])
                mark = i
        res = [[" " for i in range(len(arr_s))] for j in range(maxx)]
        for i in range(len(arr_s)):
            for j in range(len(arr_s[i])):
                res[j][i] = arr_s[i][j]
        for i in range(len(res)):
            res[i] = ''.join(res[i])
            res[i] = res[i].rstrip()
        return res
'''时间复杂度：O(n), 空间复杂度：O(n+k)'''


# In[ ]:


'''Given a binary tree root and an integer target, delete all the leaf nodes with value target.

Note that once you delete a leaf node with value target, if it's parent node becomes a leaf node and has the value target, it should also be deleted (you need to continue doing that until you can't).'''

'''用时2小时，独立完成'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        def dfs(node, parent, left, mark):
            if node is None :
                return 
            if node.left is None  and node.right is None and node.val == target: #一次一次的判断是否叶子节点为target，每次dfs只会删除最后的叶子节点为target的节点
                if mark != -1:
                    if left:
                        parent.left  = None
                    else:
                        parent.right = None 
                else:
                    node = None
                return 
            dfs(node.left, node, True, 0) #如果该节点不是叶子节点，继续遍历该节点的左右节点
            dfs(node.right, node, False, 0)
        def havetarget(node): #该函数用于判断是否树中还有为target的叶子节点
            if node is None:
                return False 
            if node.val == target and node.left is None and node.right is None:
                return True
            result = False
            result = result or havetarget(node.left) or havetarget(node.right)
            return result
        while havetarget(root): #如果树中有target的叶子节点，一直遍历
            dfs(root, root, True, -1)
            if root.left is None and root.right is None and root.val == target:
                return None
        return root
            
'''时间复杂度：O（2n）, 空间复杂度：O(1)'''


# In[ ]:


class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        index_left = 0
        index_right = 0
        i = 0 
        count = 0
        record = []
        while i < n+1:
            tmp_left = i - ranges[i] 
            tmp_right = i + ranges[i]
            if tmp_left == tmp_right:
                i += 1 
                continue 
            if tmp_left < 0:
                tmp_left = 0 
            if tmp_right > n:
                tmp_right = n
            if tmp_left >index_right:
                continue
            
            if tmp_right >= index_right:
                if index_left < tmp_left:
                    if tmp_right == index_right:
                        i += 1 
                        

                    else:
                        if record:
                            print("record:", record)
                            print("i:", i)
                            tmp = -heapq.heappop(record)
                            num_pop = 1
                            while  record and tmp_left < tmp:
                                tmp = -heapq.heappop(record)
                                num_pop += 1 
                            if tmp_left >= tmp:
                                heapq.heappush(record, -tmp)
                                num_pop -=1 
                            if num_pop == 0:
                                count += 1
                            else:
                                count -= num_pop-1
                            index_right = tmp_right
                            i += 1 
                        
                        else:
                            heapq.heappush(record, -index_right)
                            count += 1 
                            index_right = tmp_right
                            i += 1 
                elif tmp_left <= index_left :
                    index_left = tmp_left
                    index_right = tmp_right
                    count = 1 
                    i += 1 
                
                                
            elif tmp_left <= index_left and index_right >=tmp_right:
                count += 1 
                i += 1  
            elif tmp_left >= index_left and index_right >= tmp_right:
                i += 1 
            
            print("index_left:", index_left)
            print("index_right:", index_right)
            print("count: ", count)
            print("i:", i)
            print("\n")

        if index_left == 0 and index_right >= n:
            return count

        else:
            return -1
        
        
'''
There is a one-dimensional garden on the x-axis. The garden starts at the point 0 and ends at the point n. (i.e The length of the garden is n).

There are n + 1 taps located at points [0, 1, ..., n] in the garden.

Given an integer n and an integer array ranges of length n + 1 where ranges[i] (0-indexed) means the i-th tap can water the area [i - ranges[i], i + ranges[i]] if it was open.

Return the minimum number of taps that should be open to water the whole garden, If the garden cannot be watered return -1.'''


class Solution:
    def minTaps(self, n: int, ranges: List[int]) -> int:
        dic = []
        for i in range(len(ranges)): #先将所有的水龙头洒水范围放入一个数组，跳过所有的0， 因为0 只能浇其中一个点
            if i - ranges[i] == i + ranges[i]:
                continue
            dic.append([max(0, i - ranges[i]), min(n, i + ranges[i])])#如果洒水范围超过garden，那就取garden边界
        dic.sort() #将数组排序
        if dic == []:#如果该数组是空，我们没法浇garden
            return -1 
        if dic[0][0] >0:#如果该数组第一个元素数组开头大于0，我们也不可以浇garden，因为最左侧的tap都无法浇到garden最左侧
            return -1 
        i = 0 #初始化i=0
        mark = 0 #mark用于记录上一个以一个选定的开头为开头的所有tap中的最大结尾
        count = 0#用于记录需要开几个水龙头
        maxx = 0 #maxx用于记录当前的选定开头为开头的所有tap中的最大结尾
        if len(dic) == 1 :#如果dic长度是1，直接返回1，因为garden长度就是0，只要开1个水龙头就行
            return 1 
        for i in range(len(dic)):#先初始化下mark，遍历所有左侧范围为0的taps，得到其中的最大右侧范围
            if dic[i][0] == 0:
                mark = max(mark, dic[i][1])
            else:
                break
        count += 1  #count + 1 因为以0开头的我们已经算过了，就需要1个tap来覆盖该最大范围
        while i <len(dic): #接着往后遍历
            if i+1 <len(dic) and dic[i+1][0] > mark: # 如果 下一个元素的左侧范围比上一个最大右侧范围大的话，说明该garden没法全部覆盖，返回-1
                return -1 
            while i< len(dic) and dic[i][0] <= mark: # 如果当前的左侧范围一直小于上一个最大的右侧范围，一直遍历
                if dic[i][1] > maxx: #不断更新当前的最大的右侧范围
                    maxx = dic[i][1]  
                i += 1 
            if mark < maxx # 如果 当前最大右侧范围 大于上一个最大右侧范围，就说明要多开一个水龙头，否则则不需要
                count  += 1 
                mark = maxx
        return count  #最后返回count
'''时间复杂度：O(n) ,空间复杂度：O(n)'''
    


# In[ ]:


'''You are given a string s consisting only of letters 'a' and 'b'. In a single step you can remove one palindromic subsequence from s.

Return the minimum number of steps to make the given string empty.

A string is a subsequence of a given string if it is generated by deleting some characters of a given string without changing its order. Note that a subsequence does not necessarily need to be contiguous.

A string is called palindrome if is one that reads the same backward as well as forward.'''

class Solution:
    def removePalindromeSub(self, s: str) -> int:
        if len(s) == 0 :
            return 0
        if s == s[::-1]:
            return 1 
        else:
            return 2
        
'''时间复杂度：O(n),空间复杂度：O(1)'''        


# In[ ]:


'''Given the array restaurants where  restaurants[i] = [idi, ratingi, veganFriendlyi, pricei, distancei]. You have to filter the restaurants using three filters.

The veganFriendly filter will be either true (meaning you should only include restaurants with veganFriendlyi set to true) or false (meaning you can include any restaurant). In addition, you have the filters maxPrice and maxDistance which are the maximum value for price and distance of restaurants you should consider respectively.

Return the array of restaurant IDs after filtering, ordered by rating from highest to lowest. For restaurants with the same rating, order them by id from highest to lowest. For simplicity veganFriendlyi and veganFriendly take value 1 when it is true, and 0 when it is false.'''


class Solution:
    def filterRestaurants(self, restaurants: List[List[int]], veganFriendly: int, maxPrice: int, maxDistance: int) -> List[int]:
        res = []
        for i in range(len(restaurants)):
            if restaurants[i][2] == veganFriendly == 1 and restaurants[i][3] <= maxPrice and restaurants[i][4] <= maxDistance:
                res.append(restaurants[i])
            elif veganFriendly == 0 and restaurants[i][3] <= maxPrice and restaurants[i][4] <= maxDistance:
                res.append(restaurants[i])

        res.sort(reverse = True)
        res.sort(key = lambda x : x[1], reverse = True)
        final = []
        for i in range(len(res)):
            final.append(res[i][0])
        return final
    
    
'''时间复杂度：O(nlogn), 空间复杂度：O(n)'''  


# In[ ]:


'''You want to schedule a list of jobs in d days. Jobs are dependent (i.e To work on the i-th job, you have to finish all the jobs j where 0 <= j < i).

You have to finish at least one task every day. The difficulty of a job schedule is the sum of difficulties of each day of the d days. The difficulty of a day is the maximum difficulty of a job done in that day.

Given an array of integers jobDifficulty and an integer d. The difficulty of the i-th job is jobDifficulty[i].

Return the minimum difficulty of a job schedule. If you cannot find a schedule for the jobs return -1.'''


class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        if len(jobDifficulty) < d:
            return -1 
        if len(jobDifficulty) == d:
            return sum(jobDifficulty)
        if d == 1:
            return max(jobDifficulty)
        self.minn = 10000000
        dic = dict()
        def dfs(index, cur_d, summ):
            if len(jobDifficulty[index+1:]) < cur_d:
                return 
            if summ >self.minn:
                return 
            if cur_d == len(jobDifficulty[index+1:]):
                #print("here")
                self.minn = min(self.minn, summ+ sum(jobDifficulty[index+1:]))
                return 
            if cur_d == 1:
                self.minn = min(self.minn, summ + max(jobDifficulty[index+1:]))
                return 
            else:
                for i in range(index+1, len(jobDifficulty)):
                    if cur_d - 1 > len(jobDifficulty[i+1:]):
                        break
                    summ = summ + max(jobDifficulty[index+1:i+1])
                    #dic[(i, summ, cur_d-1)] = 1 
                    dfs(i, cur_d - 1, summ)
                    #dic.pop((i,summ, cur_d-1))
                    summ = summ - max(jobDifficulty[index+1:i+1])
        dfs(-1, d, 0)
        return self.minn
    
    
    
    
    
# dp[i][j] 代表当前执行的任务数 i， 且当前天数j时最小的工作难度
# dp[i][j] = min(dp[i-1][j-1] + max(jobDifficulty[i:i+1]), dp[i-2][j-1]+max(jobDifficulty[i-1:i+1])... dp[j-2][j-1] + max(jobDifficulty[j-1])
# 初始状态：dp[0][0] = 0, dp[i][j] = -1 for i < j
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        dp = [[0 for i in range(d+1)] for j in range(len(jobDifficulty))]
        for i in range(len(jobDifficulty)):
            for j in range(d+1):
                if i+1 < j :
                    dp[i][j] = -1 
                if i +1  >=j and j == 1 :
                    dp[i][j] = max(jobDifficulty[:i+1])
        for i in range(len(jobDifficulty)):
            for j in range(2, d+1):
                if j > i+1: # 如果需要分成的的份数大于总共有的个数，我们没法分配任务，跳过
                    continue
                if j == i+1: #如果分成的份数正好等于总共的个数，说明每一份里面只有一个，当前任务数的总体难度为每个任务难度之和
                    dp[i][j] = sum(jobDifficulty[:i+1]) 
                    continue #并且跳过下面的循环
                minn = 1000000 # 设置一个最小val
                for k in range(j-1, i+1): # 如果任务数大于要分的天数，那么用j天时间完成前i个任务的总体难度就是用j-1天时间完成j-1或j，..或 i-1个任务并且分别加上每一个剩余任务数中的难度最大值中的最小值。 
                    tmp = dp[k-1][j-1] 
                    minn_lat = max(jobDifficulty[k:i+1])
                    minn = min(tmp + minn_lat, minn)    
                dp[i][j] = minn 
                
        return dp[len(jobDifficulty) -1][d] #返回二维数组最后一个元素
    
'''时间复杂度：O(d*n), 空间复杂度：O(d*n)'''


# In[ ]:


'''Given a string s, partition s such that every substring of the partition is a palindrome.

Return the minimum cuts needed for a palindrome partitioning of s.'''


# f[i] = min(f(j)) + 1  for j in range(0, i-1)， f(i) 的意义就是长度为i+1的字符串最小需要的cut数量，s[j:i] 为回文串
#      = 0， 如果[0,i] 字符串本身就是一个回文子串

class Solution:
    def minCut(self, s: str) -> int:
        dp = [[ False for i in range(len(s))] for j in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = True 
        for j in range(len(s)):
            for i in range(j+1):
                if i+1 <= j-1:
                    if s[i] == s[j] and dp[i+1][j-1]:
                        dp[i][j] = True
                elif i+1 == j:
                    if s[i] == s[j]:
                        dp[i][j] = True 
        fc = [0 for i in range(len(s))]
        fc[0] = 0
        for i in range(1, len(s)):
            minn = 100000
            if dp[0][i] == True:
                fc[i] = 0
                continue
            for j in range(0, i):  
                if dp[j+1][i] == True:
                    minn = min(fc[j], minn)
            if minn != 100000:
                fc[i] = minn + 1 
        return fc[-1]


# In[ ]:


'''There is an integer array nums that consists of n unique elements, but you have forgotten it. However, you do remember every pair of adjacent elements in nums.

You are given a 2D integer array adjacentPairs of size n - 1 where each adjacentPairs[i] = [ui, vi] indicates that the elements ui and vi are adjacent in nums.

It is guaranteed that every adjacent pair of elements nums[i] and nums[i+1] will exist in adjacentPairs, either as [nums[i], nums[i+1]] or [nums[i+1], nums[i]]. The pairs can appear in any order.

Return the original array nums. If there are multiple solutions, return any of them.'''


class Solution:
    def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
        dic = defaultdict(set)
        for i in range(len(adjacentPairs)):
            dic[adjacentPairs[i][0]].add(adjacentPairs[i][1])
            dic[adjacentPairs[i][1]].add(adjacentPairs[i][0])

        res = [0 for i in range(len(adjacentPairs)+1)]
        num_l = []
        for key in dic.keys():
            if len(dic[key]) == 1:
                num_l.append(key)  
        res[0] = num_l[0]
        res[-1] = num_l[1]
        res[1] = dic[num_l[0]].pop()
        res[-2] = dic[num_l[1]].pop()
        if dic[res[1]]:
            dic[res[1]].remove(num_l[0])
        if dic[res[-2]]:
            dic[res[-2]].remove(num_l[1])
        dic.pop(num_l[0])
        dic.pop(num_l[1])
        i = 1
        j = len(res)-2
        while i < j:
            if dic:
                tmp_l = dic[res[i]].pop()
                tmp_r = dic[res[j]].pop()
                res[i+1] = tmp_l
                res[j-1] = tmp_r
                if dic[tmp_l]:
                    dic[tmp_l].remove(res[i])
                if dic[tmp_r]:
                    dic[tmp_r].remove(res[j])
                dic.pop(res[i])
                dic.pop(res[j])                
            else:
                break
            i += 1 
            j -= 1 
        return res
'''时间复杂度：O(n),空间复杂度：O(2n)'''


# In[ ]:


'''You are given a string s containing lowercase letters and an integer k. You need to :

First, change some characters of s to other lowercase English letters.
Then divide s into k non-empty disjoint substrings such that each substring is a palindrome.
Return the minimal number of characters that you need to change to divide the string.'''


# dp[i][j]: 长度为j的字符串分成i份回文子串要修改的最少字符数
# dp[i][j] = min(dp[i-1][k=j-1], dp[i-1][k=j-2],...,dp[i-1][k=i-1]) + stp[k+1][j]


class Solution:
    def palindromePartition(self, s: str, k: int) -> int:
        stp = [[0 for i in range(len(s))] for j in range(len(s))]
        for i in range(len(s)):
            for j in range(0,i+1):
                if i == j:
                    stp[j][i] = 0
                else:
                    if s[j] == s[i]:
                        if j + 1 == i:
                            stp[j][i] = 0
                        else:
                            stp[j][i] = stp[j+1][i-1]
                    else:
                        stp[j][i] = stp[j+1][i-1] + 1 
        dp = [[ 0 for i in range(len(s))] for j in range(k)]
        for i in range(len(dp[0])):
            dp[0][i] = stp[0][i]
        for i in range(1, len(dp)): #找到分成i-1份时，使用前i-1, i, ..., j 的长度的字符串所需要更改的最小字符数 + 每一种情况中剩余字符串最小需要更改的字符个数，即dp[i][j]的值
            for j in range(i, len(dp[0])):
                if i == j: #如果i, j相等，则该值为0，因为一个字符本身就是回文
                    dp[i][j] = 0
                    continue
                minn = 10000 
                for k in range(i,j+1): # 要计算用j个字符的情况，我们可以将计算分成i-1份时的每个情况，这里的i-1是最后一个分割的位置，这个位置可以是 [分割数，目标使用的字符串长度]中的一个
                    tmp = stp[k][j] #查询最后一个分割位置到目标字符串长度需要改几个字符
                    minn = min(dp[i-1][k-1]+tmp, minn) # 更新当前的最小值
                dp[i][j] = minn #录入最小值
            
        return dp[-1][len(s)-1]


'''时间复杂度：O(k*n^3), 空间复杂度：O(k*n)'''                
                


# In[ ]:


'''You are given an array target that consists of distinct integers and another integer array arr that can have duplicates.

In one operation, you can insert any integer at any position in arr. For example, if arr = [1,4,1,2], you can add 3 in the middle and make it [1,4,3,1,2]. Note that you can insert the integer at the very beginning or end of the array.

Return the minimum number of operations needed to make target a subsequence of arr.

A subsequence of an array is a new array generated from the original array by deleting some elements (possibly none) without changing the remaining elements' relative order. For example, [2,7,4] is a subsequence of [4,2,3,7,2,1,4] (the underlined elements), while [2,4,2] is not.'''

class Solution:
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        def binarysearch(left, right, pivot):
            mark = -1
            while left <= right:
                mid = (left+right)//2
                if res[mid] < pivot:
                    mark = mid + 1
                    left = mid + 1
                elif res[mid] > pivot:
                    mark = mid 
                    right = mid - 1 
                else:
                    return -1 
            return mark 

        dic_target = dict()
        dic_arr = defaultdict(list)
        for i in range(len(target)):
            dic_target[target[i]] = i
        new_arr = []
        for i in range(len(arr)):
            if arr[i] in dic_target:
                new_arr.append(dic_target[arr[i]])
        if new_arr == []:
            return (len(target))
        '''dp = [[0 for i in range(len(target)+1)] for j in range(len(new_arr)+1)]

        for j in range(1, len(target)+1):
            for i in range(1, len(new_arr)+1):
                if target[j-1] == new_arr[i-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return len(target) -dp[-1][-1] '''
        res = []
        res.append(new_arr[0])
        pos = -1 
        record = set()
        record.add(new_arr[0])
        for i in range(1, len(new_arr)):
            if res[-1] < new_arr[i]:
                res.append(new_arr[i])
                record.add(new_arr[i])
            else:
                if new_arr[i] in record:
                    continue 
                pos = binarysearch(0, len(res)-1, new_arr[i])
                record.remove(res[pos])
                res[pos] = new_arr[i]
                record.add(res[pos])
        return len(target) - len(res)


# In[ ]:


'''Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.'''

'''从开头开始，如果bar一直连续在降低，我们就把这些bar放入一个由大到小排列的栈中，同时dp[i] = dp[i-1], 因为如果是bar连续一直变小，我们没法蓄水。

如果开始遇到两个相同高度的bar，我们只取其中一个。

如果一旦遇到高度比栈中最小bar高的bar 但是小于最左侧的bar高度，我们遍历判断我们当前的bar比之前连续递减的bar高多少。
比如，如果当前bar是3，stack = [6, 2, 0], 我们去用 3 -0 + 3 -2 得到了这一组中可以有多少水。
同时与单调栈pop不同，在最左侧bar的高度大于当前bar 3 的高度的情况下，我们将其中所有比3小的bar的高度全部改为3。
该行为的用意是因为最左侧的bar还要继续用，但是比3小的所有bar我们都填满了，所以bar就被我们“填”成了高度为3的bar，在接下来计算中，我们把这些bar按照高度为3计算, stack = [6,3,3,3]。


当出现当前bar高度要大于最左侧bar高度时，每一次我们多算了一部分水，比如，我们现在在上一步的基础上有：bar = 7, stack = [6,3,3,3], 
我们按照上一步计算， 为7-3+7-3+7-3+7-6. 该结果为13, 真实结果为9. 原因是我们左侧高度没法到7， 所以在该情况下，我们需要减去左侧和右侧高度的高度差*这两个bar之间的距离。
即： 7-3+7-3+7-3-(7-6)*3 = 9.

每一次得出的结果需要加上上一步的可以蓄水数， 即当前步可以蓄多少水。'''

#单调栈+dp
#dp[i] = dp[i-1]+  ele popped out of stack * 每一个pop出来的bar与pivot的高度差, 并且最终由最左侧的最高高度juedin

class Solution:
    def trap(self, height: List[int]) -> int:
        dp = [0 for i in range(len(height))] #创建一个dp数组
        if dp == []:
            return 0
        dp[0] = 0 #初始化dp，如果只有一个bar，没法存，所以是0
        stack = []
        stack.append(height[0]) # 创建栈，将第一个元素入栈
        for i in range(1, len(dp)): 
            if len(stack) == 1 and height[i] == stack[0]: #如果栈中只有一个bar，并且当前bar的高度与与这个bar一样，我们跳过
                dp[i] = dp[i-1]
                continue
            if height[i] <= stack[-1]: # 如果当前bar高度小于栈中最后一个bar高度，我们一直入栈操作，并且dp就是上一个位置的值
                stack.append(height[i])         
                dp[i] = dp[i-1]
            else:
                tmp = 0 
                count = 0
                while stack != [] and height[i] > stack[-1]: #否则，我们计算新的bar可以存多少水
                    if len(stack) == 1: #当bar长度只有一个时并且当前bar高大于这个bar，我们需要减去我们多算的水的数量
                        tmp = tmp - (height[i]-stack[-1])* count
                    else:
                        tmp = tmp + (height[i]- stack[-1]) #我们一直累加高度差
                    stack.pop(-1) #将比当前bar小的bar pop掉
                    count += 1 #统计pop了多少bar（左侧与右侧的距离）
                if stack != []: #如果stack不是空，我们将所有之前的bar填满到最大高度，即当前height[i]高度，入栈
                    while count > 0:
                        stack.append(height[i])
                        count -= 1 
                stack.append(height[i])
                dp[i] = dp[i-1] + tmp #最后计算该位置最多储水量
        return dp[-1] 


# In[ ]:


'''Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has exactly two or zero sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes. More formally, the property root.val = min(root.left.val, root.right.val) always holds.

Given such a binary tree, you need to output the second minimum value in the set made of all the nodes' value in the whole tree.

If no such second minimum value exists, output -1 instead.'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        self.val = 10000000000
        self.mark = 0
        def dfs(node, val):
            if node is None :
                return 
            if node.val > val:
                self.val = min(self.val, node.val)
                return 
            else:
                dfs(node.left, val)
                dfs(node.right, val)


        dfs(root, root.val)
        if self.val == 10000000000:
            return -1 
        return self.val


# In[ ]:


'''Given an array of non-negative integers nums, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.'''

'''贪心'''
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        how_far = 0 + nums[0]
        if how_far >= len(nums) - 1 :
            return True 
        i = 0
        while i <len(nums):
            if nums[i] == 0 and i <len(nums)-1:
                return False
            maxx = 0
            for j in range(i+1, how_far+1):
                if nums[j] +j > maxx:
                    i = j
                    maxx = nums[j] +j
                maxx = max(maxx, nums[j] + j)
            if maxx >= len(nums) - 1 :
                return True 
            how_far = nums[i] + i
'''动态规划'''

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        dp = [0 for i in range(len(nums))]

        dp[0] = nums[0] + 0

        for i in range(1, len(nums)):
            if dp[i-1] < i:
                return False 
            dp[i] = max(dp[i-1], nums[i] + i)


        return dp[-1] >= len(nums)-1


# In[ ]:


'''A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and space is marked as 1 and 0 respectively in the grid.'''


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        dp = [[0 for i in range(len(obstacleGrid[0]))] for j in range(len(obstacleGrid))]
        if obstacleGrid[0][0] == 1:
            return 0
        else:
            dp[0][0] = 1 
        for i in range(1, len(obstacleGrid[0])):
            if obstacleGrid[0][i] ==1:
                dp[0][i] = 0
            else:
                dp[0][i] = dp[0][i-1]
            

        for j in range(1,len(obstacleGrid)):
            if obstacleGrid[j][0] == 1:
                dp[j][0] = 0
            else:
                dp[j][0] = dp[j-1][0]

        for i in range(1, len(obstacleGrid)):
            for j in range(1, len(obstacleGrid[0])):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] += dp[i-1][j] +dp[i][j-1] 
        return dp[-1][-1]


# In[ ]:


'''Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

Insert a character
Delete a character
Replace a character'''

# 动态规划：
# 定义状态: 我们可以选择删除一个字符，添加一个字符，替换一个字符。删除字符串A中的一个字符，目的是为了与字符换B保持一致，如果我们选择在字符串B中添加一个字符与A中要删除的字符一样的字符，得到的效果是一样的。 因此我们有如下想法：
# 如果我们给定了A 的字符长度 为i， B的字符长度为j，我们也知道了，当A长度为i-1， B 长度为j， A 长度为i，B长度为j-1， A 长度为 i-1，B 长度为j -1 时的最小需要操作的步骤使A,B 一致， 我们可以推导出i，j的情况
# 我们定义 dp[i][j] 为 长度为A 的字符串与长度为B的字符串最少需要几步能互相相等。
# 第一种一致的方式， B 不动， 让A与B一致：
#i 长度的A 与 j长度的B，我们要与B的最后一个字符保持一致的话，相当于原先B只有j-1个字符，现在多了一个字符在B的末尾，我们要让A末尾与该字符保持一致，我们就在A的结尾添加一个相同字符，操作一次。 所以为A字符长度为i， B字符长度为j-1的情况下，我们添加一个与B中新增字符一样的字符到A的结尾->dp[i][j] = dp[i][j-1] + 1
#第二种方式， A不动， 让B 与A 一致：
#方法同理： dp[i][j] = dp[i-1][j] + 1 

# 第三种方式，A 和B 中选一个不动，将A的第i个字符和B的第j个字符都当做新增字符，我们要让这两个保持一致·，我们要么改A 要么B, 改一个就好了，所以操作次数加一，如果这两个新增字符正巧是一个字符，我们什么都不用做了，他们已经一致了，所以就是等于原先两个字符的最小步骤：dp[i][j] = dp[i-1][j-1] if words[i] = words[j], else: dp[i][j] = dp[i-1][j-1] + 1 

# 我们的 dp[i][j] 就是这三种情况中的最小值：dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1, dp[i-1][j-1]+1) 如果结尾字符不相等，相同的话：dp[i][j] = dp[i-1][j-1]

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        len_1 = len(word1)
        len_2 = len(word2)
        if len_1 == 0:
            return len_2
        if len_2 ==0:
            return len_1
        dp = [[0 for i in range(len_1)] for j in range(len_2)]
        
        if word1[0] == word2[0]:
            dp[0][0] = 0
        else:
            dp[0][0] = 1 
        mark = 0
        for i in range(1, len_1):
            if word1[i] == word2[0] and mark == 0:
                dp[0][i] = dp[0][i-1]
                mark = 1 
            else:
                dp[0][i] = dp[0][i-1] + 1 
        mark = 0 
        for j in range(1, len_2):
            if word2[j] == word1[0] and mark == 0:
                dp[j][0] = dp[j-1][0]
                mark = 1 
            else:
                dp[j][0] =  dp[j-1][0] + 1 
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if word1[j] == word2[i]:
                    dp[i][j] = dp[i-1][j-1] 
                else:

                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1] + 1)  
        return dp[-1][-1]
    
    
                


# In[ ]:


'''Given the root of a binary tree, the value of a target node target, and an integer k, return an array of the values of all nodes that have a distance k from the target node.

You can return the answer in any order.'''

'''利用哈希表，先存储每个节点的父亲节点地址， 这样在深度优先遍历时可以向前遍历，比普通深度优先遍历多了一个向前遍历的语句，同时要注意避免重复搜索，要多设置一个来源节点参数'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        self.dic = dict()
        self.target = None 
        def dfs(node):
            if node is None:
                return 
            if node == target:
                self.target = node
            if node.left:
                self.dic[node.left] = node

            if node.right:
                self.dic[node.right] = node

            dfs(node.left)
            dfs(node.right)
        dfs(root)
        self.res = []
        def dfs_dis(node, count, prev):
            if node is None:
                return 
            if count == k :
                self.res.append(node.val)
                return 
            if prev != node.left:
                dfs_dis(node.left, count + 1, node)
            if prev != node.right:
                dfs_dis(node.right, count + 1, node)
            if node in self.dic and prev != self.dic[node]:
                dfs_dis(self.dic[node], count + 1, node)
        dfs_dis(self.target, 0, None)
        return self.res
    
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.'''



#我们每次都要添加一个节点，这个节点的数字一定是所有之前数字中最大的，所以
# 1. 一个最简单的情况就是，这个最大节点作为根节点，那么所有剩下的节点都不可能在该接节点上面，所有节点在该节点下面的，那就是dp数组上一个状态的所有可能。
# 2. 下面讨论如果我们想要把其中一个节点拿到最大的数字上面一层，我们任意哪一个都行，因为就一个节点拿上去了，如果就看新增节点以上的部分，就只有一种情况，因为就一个节点，不管该数字是多少，就是dp[1] =1， 新增节点下方，还剩下k-1个节点（假设上一个状态有k个节点, 那我们就找dp[k-1]是多少，因为下方的节点也一定是有序的，只不过可能其中有两个数字相差2，所以数字的大小不影响结果

# 3. 我们推广一下，如果拿3 个 节点上去，那么上方排列方法就是dp[3], 下方还剩k-3个，那排列方法就是dp[k-3]，上方的每一种排列会产生dp[k-3]个下方的结果，所以以3为例的所有排列方法就是dp[3]*dp[k-3]

# 4. generally， dp[i] = sum(dp[j]*dp[k-j] for j in range(k))
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [0 for i in range(n+1)]
        dp[0] = 1
        dp[1] = 1 
        summ = 0
        for i in range(2,len(dp)):
            num = i - 1 
            tmp = 0
            for j in range(0, num+1):
                tmp = dp[j] * dp[num-j]
                summ = summ + tmp

            dp[i] = summ
            summ = 0
        return dp[-1]
'''时间复杂度：O(n^2), 空间复杂度：O(n)'''


# In[ ]:


'''Given an integer n, return all the structurally unique BST's (binary search trees), which has exactly n nodes of unique values from 1 to n. Return the answer in any order.'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        dp = [[] for i in range(n+1)]
        dp[0] = [TreeNode(None)]
        dp[1] = [TreeNode(1)]
        def copy_tree(node):
            if node is None:
                return 
            l = copy_tree(node.left)
            r = copy_tree(node.right)
            root = TreeNode(node.val, l, r)
            return root
        def dfs(node, target):
            if node is None:
                return 
            if node.val == target:
                return node
            r2 = dfs(node.right, target)
            if r2 :
                return r2
        for i in range(2, n+1):
            tmp = []
            tmp_summ = []
            num = i - 1 
            for j in range(num+1):
                if j == 0:
                    tmp = []
                    for k in range(len(dp[i-1])):
                        tmp.append(copy_tree(dp[i-1][k]))

                    for k in range(len(tmp)):
                        new_root = TreeNode(i)
                        new_root.left = tmp[k]
                        tmp_summ.append(new_root)
                else:
                    tmp = []
                    for k in range(len(dp[i-1])):
                        tmp.append(copy_tree(dp[i-1][k]))
                    for k in range(len(tmp)):
                        pos = dfs(tmp[k], j)
                        if not pos:
                            continue 
                        tmp_nxt = copy_tree(pos.right)  
                        pos.right = TreeNode(i)
                        pos.right.left = tmp_nxt
                        tmp_summ.append(tmp[k])
            dp[i]=tmp_summ  
        return dp[-1]


# In[ ]:


'''Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.
链接：https://leetcode-cn.com/problems/largest-rectangle-in-histogram
'''

'''思想就是我们从左向右遍历，如果是一直大于等于上一个数字，就放入stack，一旦遇到第一个严格小于上一个数字的，我们在stack数组里向前遍历，

直到遍历到stack为空或者某一元素严格小于该数字。在这过程中我们同时计算每个长方形的最大面积，计算公式是， 1.(第一个小于该长条的坐标-前一个长条的坐标)*该长条的高度，如果该长条之前还有元素在stack

2. 如果stack中只有一个元素，说明这元素之前的所有元素长度都大于该元素，要不然这个元素早该被pop掉了，(第一个小于该长条的坐标 - 0)*该元素高度'''

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        if len(heights) == 1:
            return heights[0]
        stack.append((heights[0], 0))
        heights.append(0)
        i = 1 
        maxx = 0
        while i <len(heights):
            while i < len(heights) and heights[i] >= stack[-1][0]:
                stack.append((heights[i], i))
                i += 1  
            j = i -1 
            if i >= len(heights):
                right = 0
            else:
                right = heights[i]
            while stack and stack[-1][0] >= right:
                if len(stack) != 1:
                    tmp = (i-stack[-2][1]-1)*stack[-1][0]
                else:
                    tmp = (i)*stack[-1][0]
                maxx = max(tmp, maxx)
                stack.pop(-1)
            stack.append((right,i))
            i += 1 
        return maxx

            
'''时间复杂度：O(n^2), 空间复杂度：O(n)'''


# In[ ]:


'''Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

链接：https://leetcode-cn.com/problems/maximal-rectangle
'''


class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if matrix == []:
            return 0
        height = [[0 for i in range(len(matrix[0])+1)] for j in range(len(matrix))]  
        for i in range(len(height[0])-1):
            height[0][i] = int(matrix[0][i])
        for i in range(1, len(height)):
            for j in range(len(height[0])-1):
                if int(matrix[i][j]) == 0:
                    height[i][j] = 0
                else:
                    height[i][j] = height[i-1][j] + 1 
        maxx = 0
        for each in height:
            stack = []
            stack.append((each[0], 0))
            j = 1 
            while j < len(each):
                while j <len(each) and stack[-1][0] <= each[j]:
                    stack.append((each[j],j))
                    j += 1         
                if j >= len(each):
                    right = 0
                else:
                    right = each[j]
                while stack and stack[-1][0] >= right:
                    if len(stack) != 1:
                        tmp = (j- stack[-2][1] -1) *stack[-1][0]
                    else:
                        tmp = j*stack[-1][0]
                    maxx = max(tmp, maxx)
                    stack.pop(-1)
                stack.append((right,j))
                j += 1      
        return maxx
    
'''时间复杂度：O((nm)^2), 空间复杂度：O(n*m)'''


# In[ ]:


'''
Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

An interleaving of two strings s and t is a configuration where they are divided into non-empty substrings such that:

s = s1 + s2 + ... + sn
t = t1 + t2 + ... + tm
|n - m| <= 1
The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
Note: a + b is the concatenation of strings a and b.
https://leetcode-cn.com/problems/interleaving-string/
 '''

# 动态规划：
# 定义状态：dp[i][j] 为s1 前i个字符与s2前j个字符是否可以组成s3的i+j个字符.我们每次添加一个字符，该字符可以来自s1页可以来自s2. 

#假设来自s1， 我们就查询dp[i-1][j] 是否为True， 如果是true，我们就判断新增的字符与s3新增字符是否一致，一致的dp[i][j]就是true，

#不一致的话，我们再同样判断是否来自s2的情况，即dp[i][j-1] 是否为True，为true的话我们判断是否新增字符一致，如果还不一致dp[i][j] = false

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        dp = [[False for i in range(len(s2)+1)] for j in range(len(s1)+1)]
        dp[0][0] = True    
        for i in range(len(dp)):
            for j in range(len(dp[0])):
                if i == 0 and j == 0:
                    continue
                if i-1 >=0 and dp[i-1][j] == True:
                    if s1[i-1] == s3[i+j-1]:
                        dp[i][j] = True 
                if j - 1>=0 and dp[i][j-1] == True:
                    if s2[j-1] == s3[i+j-1]:
                        dp[i][j] = True 
        return dp[-1][-1]


# In[ ]:


'''Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.'''



class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if len(s) == 1 :
            if s in wordDict:
                return True 
            else:
                return False 
        dp = [False for i in range(len(s)+1)]
        dic = set(wordDict)
        dic_True = []
        dp[0] = False 
        mark = -1 
        for i in range(1, len(dp)): 
            if dp[i-1] is False: 
                if s[0:i] in dic:
                    dp[i] = True
                    dic_True.append(i-1)
                    continue
                if mark != -1:
                    if s[mark:i] in dic:
                        dp[i] = True 
                        dic_True.append(i-1)
                    else:
                        j = len(dic_True) -1 
                        while j >= 0:
                            if dp[dic_True[j]+1] is True:
                                if s[dic_True[j]+1:i] in dic:
                                    dp[i] = True 
                                    dic_True.append(i-1)                   
                            j -= 1 
                else:
                    mark = i-1
            else:
                if s[i-1] in dic:
                    dp[i] = True
                    dic_True.append(i-1)
                    continue
                if s[0:i] in dic:
                    dp[i] = True
                    dic_True.append(i-1)
                    continue
                if s[i-1] not in dic:
                    if len(dic_True) <=1:
                        dp[i] = False 
                        mark = i-1
                    else:
                        j = len(dic_True) -1 
                        while j >= 0:
                            if dp[dic_True[j]+1] is True:
                                if s[dic_True[j]+1:i] in dic:
                                    dp[i] = True 
                                    dic_True.append(i-1)
                                    break
                            j -= 1 
                        if dp[i] == False:
                            mark = i-1
                else:
                    dp[i] = True 
        return dp[-1]


# In[ ]:


'''
Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.

It is guaranteed that the answer will fit in a 32-bit integer.

A subarray is a contiguous subsequence of the array.'''


#动态规划：
#定义状态：因为是要求乘积最大的子数组，二数相乘，有3种情况，正*正= 正，正*负 = 负，负*负= 正。 最简单的情况，如果全都是正数，那该数组中最大的乘积子数组，我们可以不断地累积每个元素，一旦出现0，我们将当前最大乘积清0，再继续向后累积，每次比较累积的最大值，dp[i] = max(dp[i-1]*nums[i], nums[i]). 但是真实情况要比这个复杂，数组中可以有负数，如果上一个状态的成绩为正数，但是是由负数得到的正数，当前元素为负数的话，最大值就是错误的。为此我们可以创立一个dp_max 和 dp_min用于记录以每一个元素为结尾的最大乘积和最小乘积。
# dp_max[i]： 以i为结尾数组子串的最大乘积
# dp_min[i]: 以i为结尾数组子串的最小乘积

#每一个状态有三种可能结果，1. 上一个状态的最小乘积*当前元素 2. 上一个状态的最大乘积 * 当前元素 3. 当前元素。 我们需要比较这三者的最大值，放入dp_max[i]中， 最小值放入dp_min[i] 中。 

#目标状态：我们要求的是dp_max中的最大值

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        dp_max = [0 for i in range(len(nums))]
        dp_min = [0 for i in range(len(nums))]

        dp_min[0] = nums[0]
        dp_max[0]= nums[0]

        for i in range(1, len(nums)):
            dp_max[i] = max(dp_max[i-1]*nums[i], dp_min[i-1]*nums[i], nums[i])
            dp_min[i] = min(dp_min[i-1]*nums[i], dp_max[i-1]*nums[i], nums[i])
        return max(dp_max)


# In[ ]:


'''The demons had captured the princess and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of m x n rooms laid out in a 2D grid. Our valiant knight was initially positioned in the top-left room and must fight his way through dungeon to rescue the princess.

The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.

Some of the rooms are guarded by demons (represented by negative integers), so the knight loses health upon entering these rooms; other rooms are either empty (represented as 0) or contain magic orbs that increase the knight's health (represented by positive integers).

To reach the princess as quickly as possible, the knight decides to move only rightward or downward in each step.

Return the knight's minimum initial health so that he can rescue the princess.

Note that any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.
链接：https://leetcode-cn.com/problems/dungeon-game
'''

'''定义dp[i][j]为勇士要能够走到最后的话，需要在第i行第j列至少需要多少血， 我们采取逆向的动态规划
dp[i][j] 有两种可能的状态，勇士在 i， j可以选择向右也可以下走，因为是逆向推，我们已经分别知道了向下的那条路走到最后需要多少血，向右的那条路走到最后需要多少滴血，再看看当前位置需要多少滴血，
当前位置可能是补血，也可能是失血，如果是补血的话，在当前位置我们一开始就不需要下一个位置那么多血，因为在当前位置可以补充。 如果是失血，我们在当前位置就需要额外的这失血的部分去补充。

动态转移方程：
dp[i][j] = max(-(dungeon[i][j]-dp[i+1][j]), -(dungeon[i][j] - dp[i][j+1])) 如果dungeon[i][j]-dp[i+1][j] < 0 并且 dungeon[i][j] - dp[i][j+1] <0
           max(1,-(dungeon[i][j] - dp[i][j+1]))如果dungeon[i][j]-dp[i+1][j] >= 0 并且 dungeon[i][j] - dp[i][j+1] <0
           max(-(dungeon[i][j]-dp[i+1][j]), 1) 如果dungeon[i][j]-dp[i+1][j] < 0 并且 dungeon[i][j] - dp[i][j+1] >= 0
           1, 如果dungeon[i][j]-dp[i+1][j] >= 0 并且 dungeon[i][j] - dp[i][j+1] >= 0
           
目标状态：dp[0][0]
'''
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        dp = [[0 for i in range(len(dungeon[0]))] for j in range(len(dungeon))]
        if dungeon[-1][-1] >=0:
            dp[-1][-1] = 1
        else:
            dp[-1][-1] = -dungeon[-1][-1] + 1 
        for i in range( len(dp[0])-2, -1,-1):
            if dungeon[-1][i] - dp[-1][i+1] >=0:
                dp[-1][i] = 1 
            else:
                dp[-1][i] = -(dungeon[-1][i] - dp[-1][i+1]) 
        for i in range(len(dp)-2, -1, -1):
            if dungeon[i][-1] -dp[i+1][-1] >= 0:
                dp[i][-1] = 1
            else:
                dp[i][-1] = -(dungeon[i][-1] - dp[i+1][-1])  
        for i in range(len(dp)-2, -1,-1):
            for j in range(len(dp[0])-2,-1, -1):
                if dungeon[i][j] -dp[i+1][j] >= 0:
                    tmp_1 = 1
                else:
                    tmp_1 = -(dungeon[i][j] - dp[i+1][j])  
                if dungeon[i][j] - dp[i][j+1] >=0:
                    tmp_2 = 1 
                else:
                    tmp_2 = -(dungeon[i][j] - dp[i][j+1]) 
                dp[i][j] = min(tmp_2, tmp_1)
        return dp[0][0]


# In[ ]:


'''1011. Capacity To Ship Packages Within D Days
A conveyor belt has packages that must be shipped from one port to another within days days.

The ith package on the conveyor belt has a weight of weights[i]. Each day, we load the ship with packages on the conveyor belt (in the order given by weights). We may not load more weight than the maximum weight capacity of the ship.

Return the least weight capacity of the ship that will result in all the packages on the conveyor belt being shipped within days days.


https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/'''



class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        def possible(capacity):
            i = 0
            cur_load = 0 
            count = 0
            while i < len(weights):
                while i < len(weights) and cur_load <= capacity:
                    cur_load = cur_load + weights[i]
                    i += 1 
                count += 1 
                if count > days:
                    return False 
                if i >= len(weights) and cur_load <= capacity:
                    if count > days:
                        return False 
                    else:
                        return True               
                i -= 1    
                cur_load = 0
            return False 
        if days ==1:
            return sum(weights)
        left = max(weights)
        right = sum(weights)
        res = -1 
        while left <= right:
            mid = (left + right) //2
            if not possible(mid):
                left = mid + 1 
            else:
                res = mid
                right = mid - 1 
        return res

        

'''时间复杂度：O(nlogn), 空间复杂度：O(1)'''


# In[ ]:


'''213. House Robber II
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

https://leetcode-cn.com/problems/house-robber-ii/

'''


class Solution:
    def rob(self, nums: List[int]) -> int:
        dp_first = [0 for i in range(len(nums))]

        dp_sec = [0 for i in range(len(nums))]
        if len(nums) < 3:
            return max(nums)

        dp_first[0] = nums[0]
        dp_first[1] = nums[0]

        dp_sec[0] = 0
        dp_sec[1] = nums[1]

        for i in range(2, len(dp_first)):
            dp_first[i] = max(dp_first[i-2] + nums[i], dp_first[i-1])

        dp_first[-1] = dp_first[-2]

        for i in range(2, len(dp_sec)):
            dp_sec[i] = max(dp_sec[i-2] + nums[i], dp_sec[i-1] )


        return max(dp_first[-1], dp_sec[-1])
    
'''时间复杂度：O(n), 空间复杂度：O(n)'''


# In[ ]:


'''221. Maximal Square
Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
https://leetcode-cn.com/problems/maximal-square/

'''


class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        dp = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]
        maxx = 0
        if matrix[0][0] == "1":
            dp[0][0] = 1 
            maxx = 1
        for i in range(1, len(dp)):
            if matrix[i][0] == "1":
                dp[i][0] = 1 
                maxx = dp[i][0]
        
        for i in range(1, len(dp[0])):
            if matrix[0][i] == "1":
                dp[0][i] = 1 
                maxx = dp[0][i]
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if matrix[i][j] == "1":
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1 
                    maxx = max(dp[i][j], maxx) 
        return maxx**2


# In[ ]:


'''518. Coin Change 2
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.

You may assume that you have an infinite number of each kind of coin.

The answer is guaranteed to fit into a signed 32-bit integer.

https://leetcode-cn.com/problems/coin-change-2/

 '''
#动态规划

#定义dp[i][j] 为长度为i的数组有多少种方法可以组成和为j

# 对于第i个数字，我们可以选择拿也可以选择不拿，如果我们拿了之后和为j的话，我们就要找dp[i-1][j-coins[i]], 即我们上一个状态还没有拿到coins[i] 的时候的总拿法数，如果j - coins[i] < 0,意味着我们没法拿第i个数，那我们只能找dp[i-1][j]

# 我们取 dp[i-1][j-coins[i]] + dp[i-1][j] 如果j - coins[i] > 0 , 我们在每个状态不断地拿新增的硬币，直到j - coins[i] <0 为止

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [[ 0 for i in range(amount+1)] for j in range(len(coins)+1)]

        dp[0][0] = 1 
        for i in range(1, len(dp)):
            dp[i][0] = 1 

        for i in range(1, len(dp[0])):
            dp[0][i] = 0

        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                tmp = j 
                while tmp >= 0:
            
                    dp[i][j] = dp[i-1][tmp] +dp[i][j]
                    tmp = tmp - coins[i-1]

           
        return dp[-1][-1]
'''时间复杂度：O(amount*n), 空间复杂度：O(amount*n)'''


# In[ ]:


'''115. Distinct Subsequences
Given two strings s and t, return the number of distinct subsequences of s which equals t.

A string's subsequence is a new string formed from the original string by deleting some (can be none) of the characters without disturbing the remaining characters' relative positions. (i.e., "ACE" is a subsequence of "ABCDE" while "AEC" is not).

It is guaranteed the answer fits on a 32-bit signed integer.

https://leetcode-cn.com/problems/distinct-subsequences/'''
'''动态规划
    定义状态：我们定义dp[i][j] 为长度j的字符串有几个长度为i的字符子串。如何定义动态转移方程，最简单的入手，如果目标字符串的长度要比当前字符串的长度
    要长，那该状态一定是0，什么时候会有第一个1出现呢？那就是两个字符串第一次出现一个公共字符串时，那个状态就是1.
    下面，我们需要知道，其实每个状态是由前面两种状态转移过来的，因为我们知道了只要遇到一样的字符，我们就有可能有额外的子序列
    
    对于这两个一样的字符我们需要考虑站在什么角度去看。
    
    1. 因为这两个字符相等，我们可以考虑把这两个字符都去掉，看看这个状态下有多少个子序列，因为加上一对相同的字符，是不会影响子序列的个数的，
    所以这种情况下，我们要看dp[i-1][j-1]
    
    2. 第一种角度我们漏考虑了一种情况，如果目标子串的尾字符原来就在那，我们不是后加的，也就是上个状态的目标字符串和当前的目标字符串一样的，就算
    
    在非目标子串结尾添加的字符不一样，下个状态也是要继承上个状态的子序列的个数的，最简单地一个例子就是如果target = [r], s = [r,a,b,b,b,i,t]
    也就是第一行，我们知道这一行的结果是dp[0] = [1,1,1,1,1,1,1], 第一个1我们知道是来自r = r 后面的1都是继承了第一个，就算后面的字符都不相同。
    
    所以同理，如果目标字符串不变，我们也需要继承相同目标字符串的子序列的个数， 所以我们要加上dp[i][j-1]
    
    dp[i][j] = dp[i][j-1] + dp[i-1][j-1] if s[j] = t[i], dp[i][j] = dp[i][j-1] if s[j] != t[i]
    
    目标状态：dp[-1][-1]
    '''

class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        dp = [[0 for i in range(len(s))] for j in range(len(t))]
        if s[0]==t[0]:
            dp[0][0] = 1 
        for i in range(1, len(s)):
            if s[i] == t[0]:
                dp[0][i]= dp[0][i-1] + 1
            else:
                dp[0][i] = dp[0][i-1]      
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if s[j] == t[i]:
                    dp[i][j] = dp[i-1][j-1] + dp[i][j-1]
                else:
                    dp[i][j] = dp[i][j-1]
        return dp[-1][-1]
    
'''空间复杂度：O(s*t), 时间复杂度：O(s*t)'''


# In[ ]:


'''887. Super Egg Drop
You are given k identical eggs and you have access to a building with n floors labeled from 1 to n.

You know that there exists a floor f where 0 <= f <= n such that any egg dropped at a floor higher than f will break, and any egg dropped at or below floor f will not break.

Each move, you may take an unbroken egg and drop it from any floor x (where 1 <= x <= n). If the egg breaks, you can no longer use it. However, if the egg does not break, you may reuse it in future moves.

Return the minimum number of moves that you need to determine with certainty what the value of f is.


'''

class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        dp = [[0 for i in range(k+1)] for j in range(n+1)]
        dp[0][0] = 0
        for i in range(1, len(dp[0])):
            dp[1][i] = 1
        for i in range(1, len(dp)):
            dp[i][1] = i 
        def binarysearch(bound):
            left = 1
            right = bound
            res = 100000000000
            while left <= right:
                mid = (left + right)//2
                if dp[mid-1][j-1] < dp[i-mid][j] :
                    res = min(res, dp[i-mid][j]+1)
                    left = mid +1
                elif dp[mid-1][j-1] == dp[i-mid][j]:
                    return min(res, dp[mid-1][j-1]+1)
                else:
                    right = mid -1
                    res= min(res, dp[mid-1][j-1]+1)
            return res
        for i in range(2, len(dp)):
            for j in range(2, len(dp[0])):
                
                res = binarysearch(i)
                dp[i][j] = res

'''https://leetcode-cn.com/problems/super-egg-drop/

时间复杂度：O(k*n*logn), 空间复杂度：O(k*n)
'''



# In[ ]:


'''179. Largest Number
Given a list of non-negative integers nums, arrange them such that they form the largest number.

Note: The result may be very large, so you need to return a string instead of an integer.'''


class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        str_nums = []
        for i in range(len(nums)):
            str_nums.append(str(nums[i]))
        print("3" <"30")
        def quicksort(arr, left, right):
            if left < right:
                index = partition(arr, left, right)
                quicksort(arr, left, index - 1 )
                quicksort(arr, index + 1, right)
            return arr

        def partition(arr, left, right):
            pivot = random.randint(left, right)
            arr[pivot], arr[right] = arr[right], arr[pivot]
            pivot = right 
            i = left - 1 
            for j in range(left, right):
                if arr[j] + arr[pivot]> arr[pivot] + arr[j]:
                    i += 1
                    arr[i], arr[j] = arr[j] , arr[i]
            arr[i+1], arr[pivot] = arr[pivot], arr[i+1]

            return i+1
        str_nums = quicksort(str_nums, 0, len(str_nums)-1)
        i = 0
        while i < len(str_nums):
            if str_nums[i] != "0":
                return "".join(str_nums[i:])
            i += 1 
        return "0"
   


# In[ ]:


'''670. Maximum Swap
You are given an integer num. You can swap two digits at most once to get the maximum valued number.

Return the maximum valued number you can get.'''


class Solution:
    def maximumSwap(self, num: int) -> int:
        dic = defaultdict(list)
        str_num = str(num)
        arr = []
        arr_num = []
        for i in range(len(str_num)):
            arr.append(int(str_num[i]))
            arr_num.append(int(str_num[i]))
        arr.sort(reverse = -1)
        for i in range(len(str_num)):
            dic[str_num[i]].append(i)

        for i in range(len(arr)):
            if arr[i] != arr_num[i]:
                arr_num[dic[str(arr[i])][-1]] , arr_num[i] = arr_num[i], arr_num[dic[str(arr[i])][-1]]

                break
        res = ""
        for i in range(len(arr_num)):
            res += str(arr_num[i])
        res = int(res)      
        return res


# In[ ]:


'''1337. The K Weakest Rows in a Matrix
You are given an m x n binary matrix mat of 1's (representing soldiers) and 0's (representing civilians). The soldiers are positioned in front of the civilians. That is, all the 1's will appear to the left of all the 0's in each row.

A row i is weaker than a row j if one of the following is true:

The number of soldiers in row i is less than the number of soldiers in row j.
Both rows have the same number of soldiers and i < j.
Return the indices of the k weakest rows in the matrix ordered from weakest to strongest.

 '''


class Solution:
    def kWeakestRows(self, mat: List[List[int]], k: int) -> List[int]:
        arr = []
        dic = defaultdict(list)
        for i in range(len(mat)):
            count = 0
            for j in range(len(mat[i])):
                if mat[i][j] == 1 :
                    count += 1 
            arr.append(count)
            dic[count].append(i)
        arr.sort()
        res = []
        count = 0
        for i in range(len(arr)):
            if arr[i] in dic:
                for j in range(len(dic[arr[i]])):
                    res.append(dic[arr[i]][j])
                    count += 1 

                    if count == k:
                        break
                dic.pop(arr[i])
            if count == k:
                break 
        return res
    
    
'''时间复杂度：O(n*m), 空间复杂度：O(n*m)'''


# In[ ]:


'''1338. Reduce Array Size to The Half
You are given an integer array arr. You can choose a set of integers and remove all the occurrences of these integers in the array.

Return the minimum size of the set so that at least half of the integers of the array are removed.
https://leetcode-cn.com/problems/reduce-array-size-to-the-half/
 '''

class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        dic_cot = Counter(arr)
        nums = []
        for val in dic_cot.values():
            nums.append(val)
        nums.sort(reverse = 1)
        delete = 0
        count = 0
        length = math.ceil(len(arr)/2)
        for i in range(len(nums)):
            count += nums[i]
            if count >= length:
                return i + 1 
        return i+1


# In[ ]:


'''1339. Maximum Product of Splitted Binary Tree
Given the root of a binary tree, split the binary tree into two subtrees by removing one edge such that the product of the sums of the subtrees is maximized.

Return the maximum product of the sums of the two subtrees. Since the answer may be too large, return it modulo 109 + 7.

Note that you need to maximize the answer before taking the mod and not after taking it.

https://leetcode-cn.com/problems/maximum-product-of-splitted-binary-tree/
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxProduct(self, root: TreeNode) -> int:
        def tree_sum(node):
            if node is None:
                return 0
            return tree_sum(node.left) + tree_sum(node.right) + node.val

        self.total = tree_sum(root)
        #print(self.total)
        self.maxx = 0
        self.diff = 1000000000
        def finder(node):
            if node is None:
                return 0
            tmp = node.val + finder(node.left) + finder(node.right)
            tmp_diff = abs(tmp - (self.total - tmp))
            if tmp_diff < self.diff:
                self.maxx = tmp * (self.total - tmp)
                self.diff = tmp_diff
            
            return tmp

        finder(root)
    
        return self.maxx % (10**9 + 7)
'''时间复杂度：O(n)'''


# In[ ]:


'''1346. Check If N and Its Double Exist
Given an array arr of integers, check if there exists two integers N and M such that N is the double of M ( i.e. N = 2 * M).

More formally check if there exists two indices i and j such that :

i != j
0 <= i, j < arr.length
arr[i] == 2 * arr[j]'''


class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        dic = Counter(arr)
        for i in range(len(arr)):
            if arr[i] /2 in dic or arr[i] * 2 in dic:
                if arr[i] == 0:
                    if dic[0] > 1:
                        return True 
                    else:
                        continue 
                return True 


        return False 


# In[ ]:


'''1347. Minimum Number of Steps to Make Two Strings Anagram
Given two equal-size strings s and t. In one step you can choose any character of t and replace it with another character.

Return the minimum number of steps to make t an anagram of s.

An Anagram of a string is a string that contains the same characters with a different (or the same) ordering.

 '''

class Solution:
    def minSteps(self, s: str, t: str) -> int:
        dic_s = Counter(s)
        dic_t = Counter(t)
        count_t = 0
        count_s = 0
        for key, val in dic_t.items():
            if key in dic_s:
                if dic_s[key] > val:
                    count_s += dic_s[key] - val
                elif dic_s[key] < val:
                    count_t += val - dic_s[key]


            else:
                count_t += val
        return count_t


# In[ ]:


'''494. Target Sum
You are given an integer array nums and an integer target.

You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.

 '''

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        self.count = 0
        def dfs(index, summ):
            if index >= len(nums):
                if summ == target:
                    self.count += 1 
                    return 
                else:
                    return 
            dfs(index + 1, summ + nums[index])
            dfs(index + 1, summ - nums[index])
            return 

        dfs(0, 0)
        return self.count



    
    


# In[ ]:


'''494. Target Sum
You are given an integer array nums and an integer target.

You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.

 '''

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        summ = 0
        for i in range(len(nums)):
            summ += abs(nums[i])
        length = summ * 2  + 1 
        dp = [[0 for i in range(length)] for j in range(len(nums)+1)]
        dp[1][summ + nums[0]] += 1 
        dp[1][summ - nums[0]] += 1 
        dp[0][summ] = 1  
        for i in range(2, len(dp)):
            for j in range(-summ, summ+1): 
                if 0 <= j + nums[i-1] and 0 <= j - nums[i-1]:
                    if summ + j + nums[i-1] < length and summ + j - nums[i-1] <length:
                        dp[i][summ + j] = dp[i-1][summ + j +nums[i-1]] + dp[i-1][summ + j - nums[i-1]]

                    elif summ + j + nums[i-1]<length and summ + j - nums[i-1]>= length:
                        dp[i][summ + j] = dp[i-1][summ + j +nums[i-1]]

                    elif summ + j + nums[i-1] >= length and summ + j - nums[i-1] < length:
                        dp[i][summ + j] = dp[i-1][summ + j - nums[i-1]]
                    else:
                        dp[i][summ+j] = 0
                elif j + nums[i-1] < 0 and 0 <= j -nums[i-1]:
                    if summ + j + nums[i-1] >=0 and summ + j - nums[i-1] <length:
                        dp[i][summ+j] = dp[i-1][summ + j +nums[i-1]] + dp[i-1][summ + j - nums[i-1]]

                    elif summ + j + nums[i-1] >= 0 and summ + j - nums[i-1] >= length:
                        dp[i][summ+j] = dp[i-1][summ + j +nums[i-1]]

                    elif summ + j + nums[i-1] < 0 and summ + j - nums[i-1] < length:
                        dp[i][summ+j] = dp[i-1][summ + j - nums[i-1]]

                    elif summ + j + nums[i-1] < 0 and summ + j - nums[i-1] >= length:
                        dp[i][summ+j] = 0
                elif j + nums[i-1] >= 0 and j - nums[i-1] < 0:
                    if summ + j + nums[i-1] < length and summ + j - nums[i-1] >= 0:
                        dp[i][summ+j] = dp[i-1][summ + j +nums[i-1]] + dp[i-1][summ + j - nums[i-1]]

                    elif summ + j + nums[i-1] < length and summ + j - nums[i-1] < 0 :
                        dp[i][summ+j] = dp[i-1][summ + j +nums[i-1]]

                    elif summ + j + nums[i-1] >= length and summ + j - nums[i-1] >= 0:
                        dp[i][summ+j] = dp[i-1][summ + j - nums[i-1]]

                    elif summ + j + nums[i-1] >= length and summ + j - nums[i-1] < 0:
                        dp[i][summ+j] = 0

                elif j + nums[i-1] < 0 and j - nums[i-1] < 0:
                    if summ + j + nums[i-1] >= 0 and summ + j - nums[i-1] >= 0:
                        dp[i][summ+j] = dp[i-1][summ + j +nums[i-1]] + dp[i-1][summ + j - nums[i-1]]

                    elif summ + j + nums[i-1] >= 0 and summ + j - nums[i-1] < 0:
                        dp[i][summ+j] = dp[i-1][summ + j +nums[i-1]]

                    elif summ + j + nums[i-1] < 0  and summ + j - nums[i-1] >= 0:
                        dp[i][summ+j] = dp[i-1][summ + j - nums[i-1]]

                    elif summ + j + nums[i-1] < 0 and summ + j - nums[i-1] < 0:
                        dp[i][summ+j] = 0
        if summ < target:
            return 0
        return dp[-1][summ + target]
    
'''时间复杂度：O((2*sum + 1)*n), 空间复杂度:O((2*sum + 1)*n)'''


# In[ ]:


'''560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.

 '''
'''思路：因为我们要找的是有多少个连续的子数组满足和为k，也就是有多少段 k = summ[j]-summ[i]，我们改变下等式=>有多少段：summ[j] - k = summ[i]

如果我们知道了前j项和，我们就可以知道任意一段子数组的和， 我们可以先遍历一遍数组，把每一种和的可能用字典记录下来，字典的值为数组，记录该和值在数组

中出现的位置。接下来，我们遍历一遍summ数组，如果遇到当前和为k的话count + 1， 接着我们查询 summ[i] - k 的值的下标位置，如果该下标位置在当前i的左侧，

说明我们可以通过summ[i]- summ[j] 得到 k， 并且count + 1 
'''
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int: 
        summ = [0 for i in range(len(nums))]
        summ[0] = nums[0]
        dic = defaultdict(list)
        dic[summ[0]].append(0)
        for i in range(1, len(nums)):
            summ[i] = summ[i-1] + nums[i]
            dic[summ[i]].append(i)
        count = 0
        for i in range(len(summ)):
            tmp = summ[i]
            if tmp == k :
                count += 1     
            if tmp - k in dic:
                for j in range(len(dic[tmp - k])):
                    if dic[tmp-k][j] >= i :
                        break
                    count += 1 
        return count 
'''时间复杂度：O(n^2), 空间复杂度：O(n)'''


# In[ ]:


'''647. Palindromic Substrings
Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.

A substring is a contiguous sequence of characters within the string.'''

class Solution:
    def countSubstrings(self, s: str) -> int:
        substr = [[False for i in range(len(s))] for j in range(len(s))]
        for i in range(len(substr)):
            substr[i][i] = True 

        for i in range(len(substr)):
            for j in range(i+1):
                if s[i] == s[j]:
                    if j+1 >= i:
                        substr[j][i] = True 
                    else:
                        if substr[j+1][i-1] == True:
                            substr[j][i] = True 
        count = 0
        for i in range(len(substr)):
            for j in range(len(substr)):
                if substr[i][j] == True:
                    count += 1 
        return count 


# In[ ]:


'''849. Maximize Distance to Closest Person
You are given an array representing a row of seats where seats[i] = 1 represents a person sitting in the ith seat, and seats[i] = 0 represents that the ith seat is empty (0-indexed).

There is at least one empty seat, and at least one person sitting.

Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. 

Return that maximum distance to the closest person.'''


class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        index = 0
        i = 0
        arr = []
        while i <len(seats):
            if seats[i] == 1:
                if i == 0:
                    index == i
                    i += 1 
                    continue 
                else:
                    if index == 0 and seats[index] == 0:
                        tmp = i - index
                    else:
                        tmp = i - index - 1 
                    arr.append(tmp)
                    index = i
                    i += 1 

            else: 
                i += 1 
        if index != len(seats) - 1:
            arr.append(len(seats)-index - 1)
        maxx = 0
        if seats[0] != 1:
            maxx = arr[0]
        else:
            if arr[0] %2 == 1:
                maxx = (arr[0] + 1) /2
            else:
                maxx = arr[0] /2 
        if seats[-1] != 1:
            maxx = max(maxx, arr[-1])
        else:
            if arr[-1] % 2 == 1:
                maxx = max(maxx, (arr[-1] + 1)/2)
            else:
                maxx = max(maxx, arr[0] /2)
        tmp = -1
        if seats[0] == 1 and seats[-1] == 1:
            if arr:
                tmp = max(arr)
        elif seats[0] == 0 and seats[-1] == 1:
            if arr[1:]:
                tmp = max(arr[1:])
        elif seats[0] == 1 and seats[-1] == 0:
            if arr[:len(arr)-1]:
                tmp = max(arr[:len(arr)-1])
        else:
            if arr[1:len(arr)-1]:            
                tmp = max(arr[1:len(arr)-1])

        if tmp %2 == 1:
            maxx = max(maxx, (tmp+1)/2)
        else:
            maxx = max(maxx, (tmp)/2) 
        return int(maxx)


# In[ ]:


class Nodelist:
    def __init__(self, val = None):
        self.val = val
        self.next = None 
    def print_list(self):
        
        while self:
            print(self.val)
            self = self.next

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        dic = defaultdict(list)
        if numCourses == 1 or prerequisites == []:
            return True
        head = Nodelist(prerequisites[0][0])
        dic[prerequisites[0][0]].append(head)
        if prerequisites[0][1] != prerequisites[0][0]:
            head.next = Nodelist(prerequisites[0][1])
            dic[prerequisites[0][1]].append(head.next)
        else:
            head.next = head
        
        for i in range(1, len(prerequisites)):
            if prerequisites[i][0] not in dic:
                dic[prerequisites[i][0]].append(Nodelist(prerequisites[i][0]))
                if prerequisites[i][1] != prerequisites[i][0]:
                    if prerequisites[i][1] not in dic:
                        dic[prerequisites[i][0]][-1].next = Nodelist(prerequisites[i][1])
                        dic[prerequisites[i][1]].append(dic[prerequisites[i][0]].next)
                    else:
                        dic[prerequisites[i][0]].next = dic[prerequisites[i][1]]
                else:
                    dic[prerequisites[i][0]].next = dic[prerequisites[i][0]]
                
                
            else:
                if prerequi
                if prerequisites[i][1] in dic:
                
                    dic[prerequisites[i][0]].next = dic[prerequisites[i][1]]
                else:
                
                    dic[prerequisites[i][0]].next =Nodelist(prerequisites[i][1])
                    dic[prerequisites[i][1]] = dic[prerequisites[i][0]].next
        def iscircle(head):
            slow = head
            fast = head.next
            while True:
                if slow.next is None:
                    return False 
                if fast.next is None:
                    return False 
                if fast.next.next is None:
                    return False 
                if slow == fast:
                    return True 
                slow = slow.next
                fast = fast.next.next
        for key, val in dic.items():
            if key == 5:
                print(val)
        for val in dic.values():
            if iscircle(val):
                return False 


        return True 
            

            
'''207. Course Schedule
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.'''


from queue import Queue,LifoQueue,PriorityQueue
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        indegree_zero = deque()
        indegree = dict()
        for i in range(numCourses):
            indegree[i] = 0
        adj = [[] for i in range(numCourses)]
        for i in range(len(prerequisites)):
            adj[prerequisites[i][0]].append(prerequisites[i][1])
            indegree[prerequisites[i][1]] += 1 
        for i in range(len(indegree)):
            if indegree[i] == 0:
                indegree_zero.append(i)
        count = 0
        while indegree_zero:
            tmp = indegree_zero.popleft()
            for i in adj[tmp]:
                indegree[i] -= 1 
                if indegree[i] == 0:
                    indegree_zero.append(i)

            count += 1 
        if count != numCourses:
            return False
        else:
            return True 

'''时间复杂度：O(n), 空间复杂度:O(n)'''
            


# In[ ]:


'''210. Course Schedule II
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

 '''

from queue import Queue,LifoQueue,PriorityQueue

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        indegree_zero = deque()
        indegree = dict()

        adj = [[] for i in range(numCourses)]

        for i in range(numCourses):
            indegree[i] = 0

        for i in range(len(prerequisites)):
            adj[prerequisites[i][1]].append(prerequisites[i][0])

            indegree[prerequisites[i][0]] += 1 


        for i in range(numCourses):
            if indegree[i] == 0:
                indegree_zero.append(i)
        res = []
        count = 0
        while indegree_zero:
            tmp = indegree_zero.popleft()
            for i in adj[tmp]:
                indegree[i] -= 1 
                if indegree[i] == 0:
                    indegree_zero.append(i)

            count += 1 
            res.append(tmp)

        if count != numCourses:
            return []
        else:
            return res


# In[ ]:


'''116. Populating Next Right Pointers in Each Node
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.'''


"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        arr = [root]
        if root is None :
            return None 
        root.next = None 
        cur = root 
        count = 0
        while arr[-1].left or arr[-1].right:
            time = len(arr) - 2 **count
            tmp = len(arr)
            while time < tmp:
                arr.append(arr[time].left)
                arr.append(arr[time].right)
                time += 1 
            count += 1 
        i = 0 
        count = 0 
        while i <len(arr):
            length = 2**count -1 
            while length > 0:
                arr[i].next = arr[i+1]
                length -= 1 
                i += 1 
            i  += 1 
            count += 1 
        return root

            


# In[ ]:


'''117. Populating Next Right Pointers in Each Node II
Given a binary tree

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

 '''


"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        arr = []
        if root == None:
            return None 
        queue = collections.deque([(root,1)]) 
        while queue:
            tmp = queue.popleft()
            arr.append(tmp)
            if tmp[0].left:
                queue.append((tmp[0].left, tmp[1]+1))
            if tmp[0].right:
                queue.append((tmp[0].right, tmp[1] + 1))
        i = 0 
        while i+1 <len(arr):
            while i + 1 < len(arr) and arr[i][1] ==arr[i+1][1]:
                arr[i][0].next = arr[i+1][0]
                i += 1 
            i += 1 
        return root
    
    
    

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        def dfs(node):
            if node is None:
                return 
            if node.left and node.right:
                
                node.left.next = node.right
                if node.val == 7:
                    print(node.next.next)
                cur = node.next

                while cur:
                    if cur.left:
                        node.right.next = cur.left
                        break
                    if cur.right:
                        node.right.next = cur.right
                        break
                    cur = cur.next
                    #print(cur)
                #if node.val == 3:
                #    print(node.left.next.val)
            elif node.left and node.right is None:
                if node.val == 0:
                    print("here")
                    print(node.next.val)
                cur = node.next
                while cur:
                    if cur.left:
                        node.left.next = cur.left
                        break
                    if cur.right:
                        node.left.next = cur.right
                        break
                    cur = cur.next
          

            elif node.right and node.left is None:
                cur = node.next
                while cur:
                    if cur.left:
                        node.right.next = cur.left
                        break
                    if cur.right:
                        node.right.next = cur.right
                        break
                    cur = cur.next
            if node.left:
                dfs(node.left)
            if node.right:
                dfs(node.right)
            return 

        dfs(root)   
        return root

    
    
    
    
    
    

"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        
        def connect(head):
            mark = -1 
            while head:
                if head.left and head.right:
                    if mark == -1:
                        mark = head.left
                    head.left.next = head.right
                    cur = head.right
                elif head.left is None and head.right:
                    if mark == -1:
                        mark = head.right
                    cur = head.right
                elif head.left and head.right is None:
                    if mark == -1:
                        mark = head.left
                    cur = head.left
                
                if head.left or head.right:
                    tmp = head.next
                    while tmp :
                        if tmp.left:
                            cur.next = tmp.left
                            break
                        elif tmp.right:
                            cur.next = tmp.right 
                            break

                        tmp = tmp.next

                head = head.next
            return mark 
        head = root
        while True:
            tmp = connect(head)
            if tmp == -1:
                break
            head = tmp
        return root


# In[ ]:


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        self.dic = defaultdict(list)
        def dfs(count_left, count_right,level, node):
            if node is None:
                return
            self.dic[count_left + count_right].append([level, node.val])

            dfs(count_left -1, count_right, level+1, node.left)
            dfs(count_left, count_right+1, level+1, node.right)


        
        dfs(0,0,1, root)
        print(self.dic)
        length = len(self.dic)
        for val in self.dic.values():
            print(val)
            val = val.sort()
        print(self.dic)
        res = [[] for i in range(length)]
        minn = 10000000
        for key in self.dic.keys():
            if key < minn:
                minn = key

        minn = - minn
        for key ,val in self.dic.items():
            for el in val:
                res[key+minn].append(el[1])

        return res 


# In[ ]:


'''314. Binary Tree Vertical Order Traversal
Given the root of a binary tree, return the vertical order traversal of its nodes' values. (i.e., from top to bottom, column by column).

If two nodes are in the same row and column, the order should be from left to right.'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        queue = collections.deque([(root,0)])
        res  = []
        minn = 0
        dic = set()
        dic.add(0)
        if root is None:
            return []
        while queue:
            tmp = queue.popleft()
            res.append(tmp)
            if tmp[0].left:
                queue.append((tmp[0].left, tmp[1]-1))
                minn = min(minn, tmp[1]-1)
                dic.add(tmp[1]-1)
            if tmp[0].right:
                queue.append((tmp[0].right, tmp[1]+1))
                minn = min(minn, tmp[1]+1)
                dic.add(tmp[1]+1)
        arr = [[]for i in range(len(dic))]
        for ele in res:
            arr[ele[1]-minn].append(ele[0].val)
        return arr


# In[ ]:


'''322. Coin Change
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.'''


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [[-1 for j in range(amount+1)] for i in range(len(coins)+1)]
        dp[0][0] = 0 
        for i in range(len(dp)):
            dp[i][0] = 0
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                tmp = j-coins[i-1]
                minn = 100000
                count = 1
                while tmp >=0:
                    if dp[i-1][tmp] >=0:
                        minn = min(minn, dp[i-1][tmp] + count)
                    tmp = tmp - coins[i-1]
                    count += 1 
                if minn == 100000:
                    if dp[i-1][j] == -1:
                        minn = -1
                    else:
                        minn = dp[i-1][j]
                else:
                    if dp[i-1][j] != -1:
                        minn = min(minn, dp[i-1][j])
                dp[i][j] = minn
        print(dp)
        return dp[-1][-1]
    
    

    
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [0 for i in range(amount+1)]
        dp[0] = 0
        coins.sort()
        for i in range(1, len(dp)):
            j = 0
            minn = 100000
            while j < len(coins) and i - coins[j] >= 0:
                if dp[i-coins[j]] != -1:
                    minn = min(minn, dp[i-coins[j]])
                j += 1 
            if minn == 100000:
                dp[i] = -1
            else:
                dp[i] = minn + 1 
        return dp[-1]    


# In[ ]:


'''10. Regular Expression Matching
Given an input string s and a pattern p, implement regular expression matching with support for '.' and '*' where:

'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).

 '''
'''动态规划：

正则表达式匹配字符串

定义状态：“*” 表示0 个或者多个前一个字符，“.”可以代表任意一个字符，如果我们当前不是这两个特殊字符而是普通的字符，我们只要判断当前两个字符串尾部

字符是否一样，一样的话，我两个字符各退一步，判断该状态下是否是一样的，一样的则true， 否则false。 当然，如果p字符最后一位现在是"."， 该位置可以表示任意一个
字母， 那我们也是一样的退位判断操作。

如果p字符串最后一位是"*"说明我们可以有0个或者多个上一位字符， 我们就判断该可重复字符是否与s串的最后一位相等， 如果相等，我们就把s最后一位去掉，

看看剩下的s和当前的p是否匹配，因为我们p串的长度没变，s串少一位后，判断的就是少一位的s串和包含可重复的末尾字符的p串是否可以匹配，如果可以匹配，就是true,
如果不匹配，我们就去掉p串结尾的正则表达式符号和前置字符，来判断s和新的p串是否可以匹配，可以即True， 否则就是false

dp[i][j] 定义为长度为i的s串，和长度为j的p串是否匹配
根据以上分析:dp[i][j] = if p[j] != "*": dp[i-1][j-1] if s[i] == p[j] 
                                       false  if s[i] != p[j]
                                       
                        if p[j] == "*": dp[i-1][j] or dp[i][j-2] if s[i] == p[j]
                                        dp[i][j-2] if s[i] != p[j]


'''

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False for i in range(len(p)+1)] for j in range(len(s)+1)]
        dp[0][0] = True 
        for i in range(1, len(dp[0])):
            if p[i-1] == "*":   
                dp[0][i] = dp[0][i-2] 
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if p[j-1] == "*":
                    if s[i-1] == p[j-2] or p[j-2] == ".":
                        if i - 1 >=0 and j -2 >=0:
                            dp[i][j] = dp[i-1][j] or dp[i][j-2]
                        elif j -2 >=0:
                            dp[i][j] = dp[i][j-2]
                        else:
                            dp[i][j] = False 
                    else:
                        if j -2 >= 0:
                            dp[i][j] = dp[i][j-2]
                else:
                    if p[j-1] == "." or p[j-1] == s[i-1]:
                        if i - 1 >= 0 and j -1 >= 0:
                            dp[i][j] = dp[i-1][j-1]
        return dp[-1][-1]
    
'''时间复杂度：O(mn),空间复杂度：O(mn)'''


# In[ ]:


'''44. Wildcard Matching
Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*' where:

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).'''


# if p[j] = "*", dp[i][j] = dp[i-1][j] or dp[i][j-1]
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        dp = [[False for i in range(len(p)+1)] for j in range(len(s)+1)]
        dp[0][0] = True 
        for i in range(1, len(dp[0])):
            if p[i-1] == "*":
                dp[0][i] = dp[0][i-1]
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if p[j-1] == "*":
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]

                elif p[j-1] == "?" or p[j-1] == s[i-1]:
                    dp[i][j] = dp[i-1][j-1]
        return dp[-1][-1]

'''时间复杂度：O(nm), 空间复杂度：O(nm)'''


# In[ ]:


'''363. Max Sum of Rectangle No Larger Than K
Given an m x n matrix matrix and an integer k, return the max sum of a rectangle in the matrix such that its sum is no larger than k.

It is guaranteed that there will be a rectangle with a sum no larger than k.'''


class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        maxx = -1000000
        for i in range(len(matrix)):
            for j in range(1,len(matrix[0])):
                matrix[i][j] = matrix[i][j-1] + matrix[i][j]
                
        print(matrix)
        def binarysearch(left, right,target, arr):
            res = 12345
            while left <= right:
                mid = (left+right)//2
                if arr[mid] > target:
                    res = arr[mid]
                    right = mid - 1
                elif arr[mid] == target:
                    return arr[mid]
                else:
                    left  = mid + 1 
            return res 
        for i in range(len(matrix[0])):
            for j in range(i, len(matrix[0])):
                ele = 0
                res = [0]
                for x in range(len(matrix)):
                    if i -1 >=0:
                        ele = ele + matrix[x][j]-matrix[x][i-1]
                    else:
                        ele = ele + matrix[x][j]
                    target = ele - k
                    tmp = binarysearch(0, len(res)-1, target, res)
                    if tmp != 12345:
                        maxx = max(maxx, ele-tmp)
                    res.append(ele)
                    res.sort()
        return maxx
    


# In[ ]:


'''120. Triangle
Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

 '''


class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = []
        for i in range(len(triangle)):
            dp.append([])
            for j in range(len(triangle[i])):
                dp[-1].append(0)
        dp[0][0] = triangle[0][0]
        minn = 10000000
        for i in range(1, len(dp)):
            for j in range(len(dp[i])):
                if j -1 >=0 and j < len(dp[i-1]):
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1]) + triangle[i][j]
                elif j -1 >= 0 and j >= len(dp[i-1]):
                    dp[i][j] = dp[i-1][j-1] + triangle[i][j]

                elif j - 1<0 and j < len(dp[i-1]):
                    dp[i][j] = dp[i-1][j] + triangle[i][j]    
        return min(dp[-1])


# In[ ]:


'''题目描述
评论 (414)
题解 (493)
提交记录
331. Verify Preorder Serialization of a Binary Tree
One way to serialize a binary tree is to use preorder traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as '#'.


For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where '#' represents a null node.

Given a string of comma-separated values preorder, return true if it is a correct preorder traversal serialization of a binary tree.

It is guaranteed that each comma-separated value in the string must be either an integer or a character '#' representing null pointer.

You may assume that the input format is always valid.

For example, it could never contain two consecutive commas, such as "1,,3".
Note: You are not allowed to reconstruct the tree.'''

class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        preorder = preorder.split(",")
        i = 1
        if len(preorder) == 1 and preorder[0] == "#":
            return True 
        if len(preorder) >=2 and preorder[0] =="#":
            return False
        stack = [2]
        while stack and i < len(preorder):
            if stack == [] or stack[-1] != 0:
                if preorder[i] == "#" :
                    
                    stack[-1] -= 1 
                    i += 1 
                else:
                    stack.append(2)
                    i += 1 
            elif stack and stack[-1] == 0:
                if len(stack) >=2:
                    stack[-2] -= 1 
                stack.pop(-1)
        if i <len(preorder):
            return False 
        while stack and stack[-1]==0:
            if len(stack) >=2:
                    stack[-2] -= 1 
            stack.pop(-1)

        if stack:
            return False
        else:
            return True 


# In[12]:


def solution(A, K):
    n = len(A)
    best = 0
    count = 0
    for i in range(n - K - 1):
        if (A[i] == A[i + 1]):
            count = count + 1
            best = max(best, count)
        else:
            best = max(best, count+1)
            count = 0
    result = best +  K
    
    return result

result = solution([1,1,3,3,3,4,5,5,5,5],2)
print(result)


# In[24]:



def solution(A, F, M):
    arr = A
    forget = F
    meann = M
    summ = meann*(len(arr)+forget)
    subsum = summ - sum(arr)
    if subsum> forget *6:
        return []
    
    dic = set()
    def dfs(remain_sum, res, count):
        if count * 6 < remain_sum:
            return 
        if remain_sum < 0 :
            return 
        if remain_sum == 0 and count > 0:
            return 
        if remain_sum == 0 and count == 0:
            if tuple(sorted(res)) not in dic:
                result.append(res.copy())
                print(result)
                dic.add(tuple(sorted(res)))
            return 
        for  i in range(1, 7):
            res.append(i)
            dfs(remain_sum - i, res, count - 1)
            res.pop(-1)
    result = []
    dfs(subsum,[],forget)
    print(result)
    print(dic)
    return result

result = solution([3,2,4,3],2,4)
print(result)


# In[ ]:


'''105. Construct Binary Tree from Preorder and Inorder Traversal
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        stack = collections.deque()
        dic = dict()
        for i in range(len(inorder)):
            dic[inorder[i]] = i

        for i in range(len(preorder)):
            stack.append(preorder[i])
   
        def recursive(left, right):
            if left > right:
                return 
            if stack == []:
                return 
            root = TreeNode(stack.popleft())
            index = dic[root.val]

            root.left = recursive(left, index -1)
            root.right = recursive(index+1,right)
            return root 

        root = recursive(0,len(inorder)-1)

        return root 

            


# In[4]:


import heapq

locker = "310"
num = 0
if locker[0] == "-":
	num = int(locker[1:])
else:
	num = int(locker)

arr = []
if num == 0:
	print( 0)
mark = 0
if num < 0:
	mark = 1 
arr_zero = 0
while num != 0:
	tmp = num % 10
	num = num //10
	if tmp == 0:
		arr_zero += 1 
		continue
	
	heapq.heappush(arr, tmp)
print(arr)
	

result = []
first = heapq.heappop(arr)
result.append(first)
while arr_zero > 0:
	result.append(0)
	arr_zero -= 1 
while arr:
	result.append(heapq.heappop(arr))
	
res = 0
for i in range(len(result)):
	res = res*10 + result[i]
if mark == 1:
	res = -res
print(res)


# In[ ]:


'''324. Wiggle Sort II
Given an integer array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....

You may assume the input array always has a valid answer.

 '''


class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        def quickselect(left, right, target, arr):
            if left <= right:
                index = partition(left, right,arr)
                if index < target:
                    quickselect(index+1, right, target,arr)
                elif index == target:
                    print(arr[index])
                    tmp = arr[index]
                    return tmp

                else:
                    quickselect(left, index -1, target, arr)
        def partition(left, right, arr):
            i = left -1  
            pivot = random.randint(left, right)
            arr[pivot], arr[right] = arr[right], arr[pivot]
            pivot = right
            j = left

            for j in range(left, right):
                if arr[j] < arr[pivot]:
                    i += 1 
                    arr[i], arr[j] = arr[j], arr[i]
            arr[pivot], arr[i+1] = arr[i+1], arr[pivot]
            return i + 1 
        if len(nums) == 1:
            return 
        if len(nums) == 2:
            nums.sort()
            return 
        length = len(nums)
        if length % 2 == 0:
            small_count = int(length / 2)
        else:
            small_count = length//2 + 1 
        quickselect(0, len(nums) -1, small_count-1, nums)
        mid = nums[small_count-1]
        def three_way(mid, nums):
            j = 0
            i = -1 
            n = len(nums) - 1 
            while j <= n:
                
                if nums[j] < mid:
                    i+= 1 
                    nums[j], nums[i] = nums[i], nums[j]
                    j += 1 
                elif nums[j] > mid:
                    nums[j], nums[n] = nums[n], nums[j]
                    n -= 1 
                else:
                    j += 1 
        three_way(mid, nums)
        nums_left = nums[:small_count].copy()
        nums_left.sort(reverse = True)
        nums_right = nums[small_count:].copy()
        nums_right.sort(reverse  = True)
        n = len(nums)
        j = 0
        k = 0
        for i in range(len(nums)):
            if i % 2 == 0 :
                nums[i] = nums_left[j]
                j += 1
            else:
                nums[i] = nums_right[k]
                k += 1 
        return 


# In[ ]:


'''162. Find Peak Element
A peak element is an element that is strictly greater than its neighbors.

Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞.

You must write an algorithm that runs in O(log n) time.'''
'''二分查找：思路是取中间开始搜索，记为mid, 如果mid不满足nums[mid-1] < nums[mid] > nums[mid+1], 我们就将left 或者right指针移动到两侧中大于nums[mid]
的位置，如果两侧的值都大于nums[mid] 我们任意取一个方向寻找，不管哪个方向都可以找到，这样搜索的依据是如果一直寻找较大的那个数字，除非连续的数字是一直非递减的

一旦有一个数字突然减小，我们就可以找到符合条件的数字，如果是一直递增走到了最后或者最前面，左右两端都是 负无穷，也就满足了条件。
'''

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        i = 0
        left = 0
        right =  len(nums)-1
        while left <= right:
            mid = (left + right)//2
            if mid == len(nums) - 1 :
                if mid - 1 >= 0:
                    if nums[mid-1] < nums[mid] :
                        return mid
                    else:
                        right = mid -1 
                else:
                    return mid
            elif mid == 0:
                if mid + 1 <len(nums):
                    if nums[mid+1] <nums[mid]:
                        return mid
                    else:
                        left = mid + 1 
                else:
                    return mid 
            else:
                if nums[mid] <= nums[mid -1] and nums[mid] > nums[mid+1]:
                    right = mid - 1 
                elif nums[mid] > nums[mid-1] and nums[mid] <= nums[mid +1]:
                    left = mid +1 
                elif nums[mid-1] < nums[mid ] > nums[mid+1]:
                    return mid 
                else:
                    left = mid +1 
'''时间复杂度: O(logn), 空间复杂度：O(1)'''


# In[ ]:


'''34. Find First and Last Position of Element in Sorted Array
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

 '''
'''binary search：因为数组已经排过序，我们先二分找到目标的其中一个位置，再二分找目标的左右两端坐标，排序过的target都是在一起的'''
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = 0
        right = len(nums) - 1 
        id_left = -1
        id_right = -1
        if nums == []:
            return [-1,-1]
        while left <= right:
            mid = (left + right) // 2 
            if nums[mid] < target:
                left = mid + 1 
            elif nums[mid] == target:
                id_left = mid
                id_right = mid
                
                break   
            else:
                right = mid - 1

        if nums[mid] != target:
            return [-1,-1]
        copy_mid = mid 
        left = 0 
        right = copy_mid
        while left <= right:
            mid = (left +right) //2 
            if nums[mid] < target:
                left = mid +1
            elif nums[mid] == target:
                id_left = mid
                right = mid - 1 

        left = copy_mid
        right = len(nums) - 1 
        while left <= right:
            mid = (left+right)//2
            if nums[mid] > target:
                right = mid - 1 
            else:
                id_right = mid 
                left = mid + 1 
           
        return [id_left, id_right] 
'''时间复杂度：O(logn), 空间复杂度：O(1)'''


# In[ ]:


'''267. Palindrome Permutation II
Given a string s, return all the palindromic permutations (without duplicates) of it.

You may return the answer in any order. If s has no palindromic permutation, return an empty list.

 '''


class Solution:
    def generatePalindromes(self, s: str) -> List[str]:
        dic_total = Counter(s)
        odd = dict()
        for key, val in dic_total.items():
            if val % 2 != 0:
                odd[key] = val
                odd_val = key
        if len(odd) > 1 :
            return []
        mark = 0
        new_s = []
        if len(odd) == 1 :
            mark = 1
            i = 0 
            while i < (odd[odd_val] -1)/2:
                new_s.append(odd_val)
                i += 1 
        for key, val in dic_total.items():
            if val % 2 != 0:
                continue
            i = 0
            while i < val/2:
                new_s.append(key)
                i += 1 
        st = []
        res = []
        record = set()
        def recursive(mark, count, st, used):    
            if count == len(new_s) and mark == 1:
                tmp = st.copy()
                st.append(odd_val)       
                st += tmp[::-1]
                if "".join(st) not in record:
                    record.add("".join(st))
                    res.append("".join(st))
                return 
            elif count == len(new_s) and mark == 0:
                st += st.copy()[::-1]
                if "".join(st) not in record:
                    record.add("".join(st))
                    res.append("".join(st))
                return 
            tmp = st.copy()
            for i in range(len(new_s)):
                if new_s[i] in used and used[new_s[i]] == dic_total[new_s[i]] //2:
                    continue 
                else:
                    if new_s[i] not in used:
                        used[new_s[i]] = 1 
                    else:
                        used[new_s[i]] += 1 
                    st.append(new_s[i])
                    recursive(mark, count+1, st, used)
                    st = tmp.copy()
                    if used[new_s[i]] == 0:
                        used.pop(new_s[i])
                    else:
                        used[new_s[i]] -= 1   
        used = dict()
        recursive(mark, 0, st, used)
        return res


# In[ ]:


'''32. Longest Valid Parentheses
Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.'''


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack  = [-1]
        maxx = 0
        tmp = 0
        maxx = 0
        for i in range( len(s)):     
            if stack == [] and s[i] == ")":
                stack.append(i)
            elif len(stack) == 1 and s[i] == ")":
                stack.pop(-1)
                stack.append(i)   
            elif s[i] == "(":
                if stack == []:
                    stack.append(i-1)
                stack.append(i)
            elif len(stack) > 1 and s[i] == ")":
                maxx = max(maxx, i - stack[-2])
                stack.pop(-1)
        return maxx

'''时间复杂度：O(n),空间复杂度：O(n)'''


# In[ ]:


'''347. Top K Frequent Elements
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 '''

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        dic = Counter(nums)
        arr = []
        for key, val in dic.items():
            arr.append((val, key))
        def quickselect(k,left, right,arr):
            if left <= right:
                index = partition(k, left, right, arr)
                if index < k-1:
                    return quickselect(k, index+1, right, arr)
                elif index == k-1:
                    return arr[index]
                else:
                    return quickselect(k,left, index-1,arr)
        def partition(k, left, right, arr):
            pivot = random.randint(left, right)
            arr[pivot], arr[right] = arr[right], arr[pivot]
            i = left - 1 
            j = left
            pivot = right
            while j < pivot:
                if arr[j][0] > arr[pivot][0]:
                    i += 1 
                    arr[i], arr[j] = arr[j], arr[i]
                j += 1 
            arr[pivot], arr[i+1]= arr[i+1], arr[pivot]
            return i+1

        index, val = quickselect(k, 0, len(arr)-1, arr)
        res = []
        i = 0 
        while i < len(arr):
            if arr[i][0] >= index:
                res.append(arr[i][1])
            i += 1 
        return res
'''时间复杂度: O(n),空间复杂度：O(n)'''


# In[ ]:


'''332. Reconstruct Itinerary
You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and the arrival airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin with "JFK". If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string.

For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.

 '''

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        dic = dict()
        record = defaultdict(list)
        for i in range(len(tickets)):
            record[tickets[i][0]].append(tickets[i][1])
            if (tickets[i][0], tickets[i][1]) not in dic:
                dic[(tickets[i][0], tickets[i][1])] = 1 
            else:
                dic[(tickets[i][0], tickets[i][1])] += 1 
        res = ['JFK']
        def dfs(record, res,prev):
            if len(res) == len(tickets)+1:
                return True 
            record[prev].sort()
            for des in record[prev]:
                if dic[(prev,des)] != 0:
                    dic[(prev,des)] -= 1 
                    res.append(des)
                    if dfs(record,res,des):
                        return True
                    res.pop()
                    dic[(prev, des)] += 1 
        dfs(record,res, 'JFK')

        return res
    
'''时间复杂度：O(m^2logm),空间复杂度：O(n)'''


# In[ ]:


'''341. Flatten Nested List Iterator
You are given a nested list of integers nestedList. Each element is either an integer or a list whose elements may also be integers or other lists. Implement an iterator to flatten it.

Implement the NestedIterator class:

NestedIterator(List<NestedInteger> nestedList) Initializes the iterator with the nested list nestedList.
int next() Returns the next integer in the nested list.
boolean hasNext() Returns true if there are still some integers in the nested list and false otherwise.
Your code will be tested with the following pseudocode:'''

# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """
#
#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """
#
#    def getList(self) -> [NestedInteger]:
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """

class NestedIterator:
    def dfs(self, arr):
        for i in arr:
            if i.isInteger():
                self.store.append(i.getInteger())
            else:
                l = i.getList()
                self.dfs(l)
            
    def __init__(self, nestedList: [NestedInteger]):
      
        self.store = collections.deque()
        self.dfs(nestedList)
    
    def next(self) -> int:         
        return self.store.popleft()
        
    
    def hasNext(self) -> bool:
        if not self.store:
            return False 
        else:
            return True 
         

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())


# In[ ]:


'''388. Longest Absolute File Path
Suppose we have a file system that stores both files and directories. An example of one system is represented in the following picture:



Here, we have dir as the only directory in the root. dir contains two subdirectories, subdir1 and subdir2. subdir1 contains a file file1.ext and subdirectory subsubdir1. subdir2 contains a subdirectory subsubdir2, which contains a file file2.ext.

In text form, it looks like this (with ⟶ representing the tab character):

dir
⟶ subdir1
⟶ ⟶ file1.ext
⟶ ⟶ subsubdir1
⟶ subdir2
⟶ ⟶ subsubdir2
⟶ ⟶ ⟶ file2.ext
If we were to write this representation in code, it will look like this: "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext". Note that the '\n' and '\t' are the new-line and tab characters.

Every file and directory has a unique absolute path in the file system, which is the order of directories that must be opened to reach the file/directory itself, all concatenated by '/'s. Using the above example, the absolute path to file2.ext is "dir/subdir2/subsubdir2/file2.ext". Each directory name consists of letters, digits, and/or spaces. Each file name is of the form name.extension, where name and extension consist of letters, digits, and/or spaces.

Given a string input representing the file system in the explained format, return the length of the longest absolute path to a file in the abstracted file system. If there is no file in the system, return 0.
'''

class Solution:
    def lengthLongestPath(self, input: str) -> int:
        maxx = 0
        s = ""
        i  = 0
        count_t = 0
        while i < len(input):
            if input[i] == ".":
                while i < len(input) and input[i] != "\n":
                    s += input[i]
                    i += 1 
                maxx = max(maxx, len(s))
            elif input[i] == "\n":
                if i + 1 <len(input):
                    if input[i+1] != "\t":
                        s= ""
                        count_t = 0
                    else: s += "!"
                else:
                    s += "!"
                i += 1 
            elif input[i] == "\t":
                tmp_t = 0
                while i <len(input) and input[i] == "\t":
                    tmp_t += 1 
                    i += 1 
                if tmp_t <= count_t:
                    back = count_t - tmp_t + 1 
                    count = 0
                    j = len(s) -1 
                    while s:
                        if s[-1] == "!":
                            count += 1 
                            if count == back + 1 :
                                break
                            s = s[:-1]
                            
                        else:
                            s= s[:-1]

                    count_t = tmp_t
                else:
                    count_t = tmp_t
            else:
                s += input[i]
                i += 1      
                        
        return maxx



# In[ ]:


'''1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
Given a m x n matrix mat and an integer threshold. Return the maximum side-length of a square with a sum less than or equal to threshold or return 0 if there is no such square.

 '''

class Solution:
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if j - 1 >= 0:
                    mat[i][j] = mat[i][j-1]+mat[i][j]
        maxx= 0 
        for i in range(len(mat[0])):
            for j in range(i, min(len(mat),len(mat[0]))):
                k = 0
                diff = 0
                while k <= j-i:
                    if i -1 >= 0:
                        diff = diff + mat[k][j] - mat[k][i-1]
                    else:
                        diff = diff + mat[k][j]
                    k += 1 
                first = 0
                last = j-i
                while last < len(mat):
                    if diff <= threshold:
                        maxx = max(maxx, j-i+1)
                        break 
                    else:
                        if i - 1>=0:
                            diff -= mat[first][j] - mat[first][i-1]
                            diff += mat[last][j] - mat[last][i-1]
                            first += 1 
                            last += 1 
                        else:
                            diff -= mat[first][j] 
                            diff += mat[last][j]
                            first += 1 
                            last += 1 
        return maxx


# In[ ]:


'''227. Basic Calculator II
Given a string s which represents an expression, evaluate this expression and return its value. 

The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of [-231, 231 - 1].

Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

 '''

class Solution:
    def calculate(self, s: str) -> int:
        arr = []
        i = 0 
        while i < len(s):
            if i < len(s) and s[i] != "*" and s[i] != "/" and s[i] != "+" and s[i] != "-":
                tmp = ""
                while i < len(s) and s[i] != "*" and s[i] != "/" and s[i] != "+" and s[i] != "-":
                    tmp += s[i]
                    i += 1 
                arr.append(tmp)
            else:
                
                arr.append(s[i])
                i += 1 
        i = 0
        stack = []
        while i <= len(arr)-1:
            if arr[i] == "/":
                tmp = int(stack[-1]) //int(arr[i+1])
                stack.pop(-1)
                stack.append(tmp)
                i = i + 2 
            elif arr[i] == "*":
                tmp = int(stack[-1]) * int(arr[i+1])
                stack.pop(-1)
                stack.append(tmp)
                i = i + 2 
            else:
                stack.append(arr[i])
                i += 1

        if len(stack) == 1 :
            return int(stack[0])
        res = int(stack[0])
        i = 1
        while i < len(stack):
            if stack[i] == "-" :
                res = res - int(stack[i+1])
                i = i + 2
            elif stack[i] == "+":
                res = res + int(stack[i+1])
                i = i + 2 
        return res 


# In[ ]:


'''373. Find K Pairs with Smallest Sums
You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u, v) which consists of one element from the first array and one element from the second array.

Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.'''
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        res = []
        size = min(k, len(nums1)* len(nums2))
        #print(size)
        for i in range(min(len(nums1),k)):
            for j in range(min(len(nums2),k)):
                if len(res) < size:
                    heapq.heappush(res, (-(nums1[i]+nums2[j]), nums1[i] , nums2[j]))
                else:
                    tmp = nums1[i] + nums2[j]
                    #print("here")
                    if -tmp >= res[0][0]:
                        heapq.heappop(res)
                        heapq.heappush(res, (-tmp, nums1[i], nums2[j]))
                #print(res)
        arr = []
        while res:
            tmp = heapq.heappop(res)
            arr.append([tmp[1], tmp[2]])

        return arr
                


# In[ ]:


'''286. Walls and Gates
You are given an m x n grid rooms initialized with these three possible values.

-1 A wall or an obstacle.
0 A gate.
INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.
'''

class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        """
        Do not return anything, modify rooms in-place instead.
        """
        direction = {(0,1), (1,0),(-1,0),(0,-1)}
        record = set()
        def dfs(index_1, index_2):
            for ele in direction:
                tmp_1, tmp_2  = index_1 + ele[0], index_2 + ele[1]
                if tmp_1 < 0 or tmp_1 >= len(rooms) or tmp_2 < 0 or tmp_2 >= len(rooms[0]):
                    continue 
                if rooms[tmp_1][tmp_2] == -1:
                    continue 
                if rooms[tmp_1][tmp_2] == 0:
                    if (tmp_1,tmp_2) not in record:
                        record.add((tmp_1,tmp_2))
                        dfs(tmp_1,tmp_2)
                    else:
                        continue 
                if rooms[index_1][index_2] + 1 >= rooms[tmp_1][tmp_2]:
                    continue 
                else:
                    rooms[tmp_1][tmp_2] = rooms[index_1][index_2] + 1 
                    dfs(tmp_1,tmp_2)
            return 
        i = 0
        j = 0
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if rooms[i][j] == 0:
                    if (i,j) in record:
                        continue 
                    else:
                        record.add((i,j))
                        dfs(i,j)        
        return 


# In[ ]:


'''337. House Robber III
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root.

Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

 '''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

'''动态规划： 不能偷相邻两个节点，每一个节点有两个选择，偷或者不偷，如果偷了，该节点的子节点一定不能偷，如果不偷，我们可以选择偷左右节点也可以选择不偷

选择是否偷左右节点，取决于，偷左节点或者不偷左节点的中的最大收入和偷右节点和不偷右节点中的最大收入的和。由此，我们意识到一个节点会有两个状态：一个是被偷的最大收益，

一个是不偷的最大收益。

我们用两个字典分别记录每个节点被偷或者不偷的最大收益值，并且利用后序遍历的方法遍历二叉树， 记录这些值，最后返回根节点这两个值中的最大值。

'''
class Solution:
    def rob(self, root: TreeNode) -> int:
        dic_c = dict()
        dic_p = dict()

        def dfs(node):
            if node is None:
                dic_c[node] = 0
                dic_p[node] = 0
                return 
            dfs(node.left)
            dfs(node.right)

            dic_c[node] = node.val + dic_p[node.left] + dic_p[node.right]
            dic_p[node] = max(dic_c[node.left], dic_p[node.left]) + max(dic_c[node.right], dic_p[node.right])

        dfs(root)
        return max(dic_c[root], dic_p[root])
    


# In[ ]:


'''300. Longest Increasing Subsequence
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].'''


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        arr = [nums[0]]
        record = set()
        record.add(nums[0])
        for i in range(1, len(nums)):
            if nums[i] <arr[-1]:
                if nums[i] in record:
                    continue 
                else:
                    record.add(nums[i])
                    cur = len(arr)-1
                    while cur >=0 and nums[i] < arr[cur]:
                        cur -= 1 
                    if cur >= 0:
                        record.remove(arr[cur+1])
                        arr[cur+1] = nums[i]
                        
                    else:
                        record.remove(arr[0])
                        arr[0] = nums[i]

            elif nums[i] > arr[-1]:
                record.add(nums[i])
                arr.append(nums[i])
        return len(arr)

