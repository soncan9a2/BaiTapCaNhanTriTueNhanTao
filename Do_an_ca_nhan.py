from collections import deque
import tkinter as tk
import time
import random 
import networkx as nx
from abc import ABC, abstractmethod
from tkinter import messagebox
import uuid
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import gc
from collections import deque
import itertools
import heapq



class PuzzleState:
    def __init__(self, state_string):
        self.state = state_string
    
    def get_blank_position(self):
        index = self.state.index('_')
        return divmod(index, 3)  
    
    def get_next_states(self):
        index = self.state.index('_')
        row, col = divmod(index, 3)
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_index = new_row * 3 + new_col
                new_state = list(self.state)
                new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
                moves.append(''.join(new_state))
        return moves
    
    def __eq__(self, other):
        if isinstance(other, PuzzleState):
            return self.state == other.state
        return self.state == other
    
    def __hash__(self):
        return hash(self.state)
    
    def __str__(self):
        return self.state


class SearchAlgorithm(ABC):
    def __init__(self, initial_state, target_state):
        self.initial_state = initial_state
        self.target_state = target_state
    
    @abstractmethod
    def solve(self):
        """Find path from initial state to target state"""
        pass
    

class BreadthFirstSearch(SearchAlgorithm):
    def solve(self):
        """Breadth-First Search su dung hang doi de luu"""
        initial = self.initial_state
        target = self.target_state
        
        queue = deque([initial]) #trang thai can duyet
        visited = set([initial]) #da duyet
        parent = {initial: None} #trang thai cha
        
        while queue:
            current = queue.popleft() #FIFO
            if current == target: 
                return self._reconstruct_path(current, parent)
                
            current_state = PuzzleState(current)
            for next_state in current_state.get_next_states():
                if next_state not in visited:
                    visited.add(next_state)
                    parent[next_state] = current
                    queue.append(next_state)
        
        return None
    
    def _reconstruct_path(self, current, parent):
        path = []
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse() #truy vet lai duong di
        return path

class UniformCostSearch(SearchAlgorithm):
    def solve(self):
        """
        Uniform Cost Search su dung heapq cho hang doi uu tien
        
        """
        initial = self.initial_state
        target = self.target_state

        queue = [(0, id(initial), initial, [])] #chi phi, id (neu trung cost), trang thai, duong di
        visited = {initial} #da tham

        while queue:
            cost, _, current, path = heapq.heappop(queue) #xep theo cost thap nhat truoc
            current_path = path + [current]

            if current == target:
                return current_path

            current_state = PuzzleState(current)
            next_states = current_state.get_next_states()

            for next_state in next_states:
                if next_state not in visited:
                    visited.add(next_state)
                    new_cost = cost + 1
                    heapq.heappush(queue, (new_cost, id(next_state), next_state, current_path)) #them trang thai moi 

        return None

class DepthFirstSearch(SearchAlgorithm):
    def solve(self):
        """Depth-First Search su dung stack de luu"""
        initial = self.initial_state
        target = self.target_state
        
        stack = [(initial, 0)] 
        visited = {initial: 0}  
        parent = {initial: None}
        max_depth = 31  #Gioi han do sau tranh lap trong bai lon
        
        while stack:
            current, depth = stack.pop()  #Lay trang thai moi nhat LIFO
            
            if current == target:
                return self._reconstruct_path(current, parent)
            
            if depth >= max_depth: #khong di sau nua
                continue
                
            current_state = PuzzleState(current)
            next_states = current_state.get_next_states()
            for next_state in reversed(next_states):
                if next_state not in visited or visited[next_state] > depth + 1:
                    visited[next_state] = depth + 1
                    parent[next_state] = current
                    stack.append((next_state, depth + 1))
                    
        return None  
    
    def _reconstruct_path(self, current, parent):
        path = []
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

class IterativeDeepeningDFS(SearchAlgorithm):
    def solve(self):
        """
        Iterative Deepening Depth-First Search su dung stack de luu
        """
        initial = self.initial_state
        target = self.target_state
        max_depth = 50  #Gioi han do sau
        
        for depth_limit in range(1, max_depth + 1):
            result = self._depth_limited_search(initial, target, depth_limit)
            if result is not None:
                return result
        
        return None  
    
    def _depth_limited_search(self, initial, target, depth_limit):
        stack = [(initial, 0)]
        visited = {initial: 0}
        parent = {initial: None}
        
        while stack:
            current, depth = stack.pop() #LIFO
            
            if current == target:
                return self._reconstruct_path(current, parent)
            
            if depth >= depth_limit:
                continue
            
            current_state = PuzzleState(current)
            next_states = current_state.get_next_states()
            for next_state in reversed(next_states):
                if next_state not in visited or visited[next_state] > depth + 1:
                    visited[next_state] = depth + 1
                    parent[next_state] = current
                    stack.append((next_state, depth + 1))
        
        return None  
    
    def _reconstruct_path(self, current, parent):
        path = []
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

class HeuristicSearch(SearchAlgorithm):
    def _calculate_manhattan_distance(self, state, target):
        """Khoang cach Manhattan"""
        distance = 0
        for i in range(len(state)):
            if state[i] != '_' and state[i] != target[i]:
                target_idx = target.index(state[i])
                curr_row, curr_col = divmod(i, 3)
                target_row, target_col = divmod(target_idx, 3)
                distance += abs(curr_row - target_row) + abs(curr_col - target_col)
        return distance
    
    def _reconstruct_path(self, current, parent):
        path = []
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

class GreedySearch(HeuristicSearch):
    def solve(self):
        """
        Greedy Best-First Search implementation.
        Uses only the heuristic (Manhattan distance) to guide the search.
        Does not consider path cost.
        """
        initial = self.initial_state
        target = self.target_state
        
        open_set = [(self._calculate_manhattan_distance(initial, target), initial)] #trang thai dang xet
        closed_set = set() #da xet 
        parent = {initial: None}
        
        while open_set:
            open_set.sort()  #sap xep heuristic tu thap den cao
            _, current = open_set.pop(0) #Lay thap nhat
            
            if current == target:
                return self._reconstruct_path(current, parent)
            
            if current in closed_set:
                continue
                
            closed_set.add(current) 

            current_state = PuzzleState(current)
            for next_state in current_state.get_next_states():
                #neu chua xet thi va khong co trong open_set thi tinh heuristic va them vao open_set
                if next_state not in closed_set and next_state not in [state for _, state in open_set]:
                    heuristic = self._calculate_manhattan_distance(next_state, target)
                    parent[next_state] = current
                    open_set.append((heuristic, next_state))
        
        return None  

class AStarSearch(HeuristicSearch):
    def solve(self):
        """A* search algorithm"""
        initial = self.initial_state
        target = self.target_state
        
        open_set = [(self._calculate_manhattan_distance(initial, target), 0, initial)]  #(f, g, state) 
        closed_set = set() #da xet
        g_score = {initial: 0}  #g(n) chi phi tu start den current
        parent = {initial: None}
        
        while open_set:
            open_set.sort()  #sap xep theo (f = g + h) nho nhat 
            _, g, current = open_set.pop(0)
            
            if current == target:
                return self._reconstruct_path(current, parent)
            
            closed_set.add(current)
            
            current_state = PuzzleState(current)
            for next_state in current_state.get_next_states():
                if next_state in closed_set:
                    continue
                    
                tentative_g = g_score[current] + 1 #moi buoc di chuyen
                
                if next_state not in [state for _, _, state in open_set] or tentative_g < g_score.get(next_state, float('inf')):
                    parent[next_state] = current
                    g_score[next_state] = tentative_g
                    f_score = tentative_g + self._calculate_manhattan_distance(next_state, target) #uoc luong chi phi hien tai den dich
                    
                    existing = [(i, item) for i, item in enumerate(open_set) if item[2] == next_state] #neu da co trong open_set
                    #chi phi nho hon thi cap nhat
                    if existing:
                        idx, _ = existing[0]
                        open_set[idx] = (f_score, tentative_g, next_state)
                    else:
                        open_set.append((f_score, tentative_g, next_state))
        
        return None  

class IDAStarSearch(HeuristicSearch):
    def solve(self):
        """
        IDA* (Iterative Deepening A*) search algorithm.
        Uses iterative deepening with A* heuristic to find optimal solution
        with lower memory usage than standard A*.
        """
        initial = self.initial_state
        target = self.target_state
        
        bound = self._calculate_manhattan_distance(initial, target) #bat dau tim kiem
        path = [initial]
        visited_states = {}  
        
        while True:
            visited_states = {initial: None}
            
            t, solution_path = self._search(path, 0, bound, target, visited_states)
            
            print(f"Completed search with bound {bound}, result: ", end="")
            if t == float('inf'):
                print("No solution found")
                return None  
            if isinstance(t, list):
                print(f"Solution found with {len(t) - 1} steps")
                return t  
            
            print(f"Increasing bound to {t}")
            bound = t
    
    def _search(self, path, g, bound, target, visited_states):
        """
        Recursive helper function for IDA*.
        
        Args:
            path: Current path being explored
            g: Cost so far
            bound: Current cost bound
            target: Target state
            visited_states: Dictionary to track visited states and their parents
        
        Returns:
            - float('inf') if no solution found within bound
            - solution path list if solution found
            - new bound if search needs to continue with higher bound
        """
        current = path[-1]
        f = g + self._calculate_manhattan_distance(current, target)
        if f > bound: #f lon hon bound thi tang boun
            return f, None
        
        if current == target:
            return path, path
        
        min_bound = float('inf')
        current_state = PuzzleState(current)
        
        for next_state in current_state.get_next_states():
            if next_state in path:
                continue
            
            path.append(next_state)
            visited_states[next_state] = current
            
            t, solution_path = self._search(path, g + 1, bound, target, visited_states) #de qui
            
            if isinstance(t, list):
                return t, solution_path  
            
            if t < min_bound:
                min_bound = t
            
            path.pop() #quay lai trang thai truoc
        
        return min_bound, None
    
    def _print_state_grid(self, state):
        """Print the puzzle state in a grid format for console output"""
        for i in range(0, 9, 3):
            print(" ".join(state[i:i+3]).replace("_", " "))

class HillClimbingSearch(HeuristicSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.max_iterations = 10000  # Giới hạn số lần lặp
        self.visited_states = {}  
        
    def _calculate_misplaced_tiles(self, state, target):
        """Đếm số ô không đúng vị trí"""
        count = 0
        for i in range(len(state)):
            if state[i] != '_' and state[i] != target[i]:
                count += 1
        return count
    
    def _calculate_linear_conflict(self, state, target):
        """Tính khoảng cách Manhattan + xung đột tuyến tính"""
        manhattan = self._calculate_manhattan_distance(state, target)
        conflicts = 0
        
        # Kiểm tra xung đột trong hàng
        for row in range(3):
            tiles_in_row = []
            for col in range(3):
                idx = row * 3 + col
                if state[idx] != '_':
                    tile = state[idx]
                    target_idx = target.index(tile)
                    target_row = target_idx // 3
                    if target_row == row:  # Nếu ô thuộc đúng hàng trong trạng thái đích
                        tiles_in_row.append((tile, target_idx % 3))  # (giá trị, cột đích)
            
            # Kiểm tra xung đột giữa các ô trong cùng hàng
            for i in range(len(tiles_in_row)):
                for j in range(i+1, len(tiles_in_row)):
                    tile_i, col_i = tiles_in_row[i]
                    tile_j, col_j = tiles_in_row[j]
                    if col_i > col_j: 
                        conflicts += 2  # Mỗi xung đột tính 2 bước di chuyển
        
        # Kiểm tra xung đột trong cột
        for col in range(3):
            tiles_in_col = []
            for row in range(3):
                idx = row * 3 + col
                if state[idx] != '_':
                    tile = state[idx]
                    target_idx = target.index(tile)
                    target_col = target_idx % 3
                    if target_col == col:  # Nếu ô thuộc đúng cột trong trạng thái đích
                        tiles_in_col.append((tile, target_idx // 3))  # (giá trị, hàng đích)
            
            for i in range(len(tiles_in_col)):
                for j in range(i+1, len(tiles_in_col)):
                    tile_i, row_i = tiles_in_col[i]
                    tile_j, row_j = tiles_in_col[j]
                    if row_i > row_j:  
                        conflicts += 2  # Mỗi xung đột tính 2 bước di chuyển
        
        return manhattan + conflicts
    
    def _calculate_pattern_database(self, state, target):
        """Heuristic dựa trên cơ sở dữ liệu mẫu đơn giản"""
        corner_positions = [0, 2, 6, 8]  
        
        corner_cost = 0
        for pos in corner_positions:
            if state[pos] != '_' and state[pos] != target[pos]:
                tile = state[pos]
                target_idx = target.index(tile)
                # Tính khoảng cách Manhattan
                corner_cost += abs(pos // 3 - target_idx // 3) + abs(pos % 3 - target_idx % 3)
        
        # Tính chi phí cho các ô còn lại
        other_cost = 0
        for pos in [1, 3, 4, 5, 7]:  
            if state[pos] != '_' and state[pos] != target[pos]:
                tile = state[pos]
                target_idx = target.index(tile)
                other_cost += abs(pos // 3 - target_idx // 3) + abs(pos % 3 - target_idx % 3)
        
        return corner_cost + other_cost
    
    def _calculate_combined_heuristic(self, state, target):
        """Kết hợp nhiều hàm heuristic để có đánh giá tốt hơn"""
        h1 = self._calculate_manhattan_distance(state, target)
        h2 = self._calculate_misplaced_tiles(state, target)
        h3 = self._calculate_linear_conflict(state, target)
        h4 = self._calculate_pattern_database(state, target)
        
        # Chọn heuristic mạnh nhất
        return max(h1, h2, h3, h4)
    
    def _is_solvable(self, state, target):
        """Kiểm tra xem bài toán có thể giải được không"""
        state_list = [0 if x == '_' else int(x) for x in state]
        target_list = [0 if x == '_' else int(x) for x in target]
        
        # Tính số nghịch thế trong state
        state_inversions = 0
        for i in range(len(state_list)):
            if state_list[i] == 0:  
                continue
            for j in range(i+1, len(state_list)):
                if state_list[j] != 0 and state_list[i] > state_list[j]:
                    state_inversions += 1
        
        # Tính số nghịch thế trong target
        target_inversions = 0
        for i in range(len(target_list)):
            if target_list[i] == 0:  
                continue
            for j in range(i+1, len(target_list)):
                if target_list[j] != 0 and target_list[i] > target_list[j]:
                    target_inversions += 1
        
        blank_row_state = state.index('_') // 3
        blank_row_target = target.index('_') // 3
        
        return (state_inversions % 2 == target_inversions % 2) == ((blank_row_state - blank_row_target) % 2 == 0)

class SimpleHC(HillClimbingSearch):
    def solve(self):
        """Thuật toán Leo đồi đơn giản (Simple Hill Climbing)"""
        import random
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        # BƯỚC 1: Chọn trạng thái hiện tại
        current = self.initial_state
        parent = {current: None}
        path = [current]
        iterations = 0
        max_iterations = self.max_iterations
        
        while current != self.target_state and iterations < max_iterations:
            iterations += 1
            
            if iterations % 1000 == 0:
                print(f"Đã thực hiện {iterations} bước...")
            
            # BƯỚC 2: Tạo các hàng xóm của trạng thái hiện tại
            current_state = PuzzleState(current)
            next_states = current_state.get_next_states()
            
            if not next_states:
                break
                
            # BƯỚC 3: Đánh giá hàm mục tiêu tại trạng thái hiện tại
            current_h = self._calculate_combined_heuristic(current, self.target_state)
            
            # Chọn ngẫu nhiên một hàng xóm để xem xét
            random_neighbor = random.choice(next_states)
            next_h = self._calculate_combined_heuristic(random_neighbor, self.target_state)
            
            # BƯỚC 4: Nếu hàng xóm tốt hơn trạng thái hiện tại, chuyển đến hàng xóm đó
            if next_h < current_h:
                parent[random_neighbor] = current
                current = random_neighbor
                path.append(current)
            
            # BƯỚC 5: Lặp lại BƯỚC 2-BƯỚC 4 cho đến khi không còn cải thiện
        
        # BƯỚC 6: Trả về trạng thái hiện tại và giá trị hàm mục tiêu của nó
        if current == self.target_state:
            print(f"Tìm thấy giải pháp sau {iterations} bước!")
            return self._reconstruct_path(current, parent)
        else:
            print(f"Không tìm thấy giải pháp sau {iterations} bước.")
            print(f"Giá trị hàm mục tiêu cuối cùng: {self._calculate_combined_heuristic(current, self.target_state)}")
            return self._reconstruct_path(current, parent)

class SteepestHC(HillClimbingSearch):
    def solve(self):
        """Thuật toán Leo đồi dốc nhất (Steepest-Ascent Hill Climbing)"""
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        # BƯỚC 1: Chọn trạng thái hiện tại
        current = self.initial_state
        parent = {current: None}
        path = [current]
        iterations = 0
        max_iterations = self.max_iterations
        
        while iterations < max_iterations:
            iterations += 1
            
            if iterations % 1000 == 0:
                print(f"Đã thực hiện {iterations} bước...")
            
            # BƯỚC 2: Tạo tất cả các trạng thái lân cận của trạng thái hiện tại
            current_state = PuzzleState(current)
            next_states = current_state.get_next_states()
            
            if not next_states:
                break
            
            # BƯỚC 3: Đánh giá hàm mục tiêu tại tất cả các lân cận
            current_h = self._calculate_combined_heuristic(current, self.target_state)
            neighbors_h = []
            
            for next_state in next_states:
                next_h = self._calculate_combined_heuristic(next_state, self.target_state)
                neighbors_h.append((next_h, next_state))
            
            # Sắp xếp theo giá trị heuristic tăng dần 
            neighbors_h.sort()
            
            # BƯỚC 4: Kiểm tra nếu trạng thái hiện tại tốt hơn tất cả các lân cận
            if not neighbors_h or neighbors_h[0][0] >= current_h:
                break
            
            # Lân cận có cải thiện cao nhất của hàm mục tiêu sẽ trở thành trạng thái hiện tại
            _, best_next_state = neighbors_h[0]
            parent[best_next_state] = current
            current = best_next_state
            path.append(current)
            
            if current == self.target_state:
                break
            
            # BƯỚC 5: Lặp lại BƯỚC 2-BƯỚC 4 (thông qua vòng lặp while)
        
        # BƯỚC 6: Trả về trạng thái hiện tại và giá trị hàm mục tiêu của nó
        if current == self.target_state:
            print(f"Tìm thấy giải pháp sau {iterations} bước!")
        else:
            print(f"Không tìm thấy giải pháp sau {iterations} bước.")
            print(f"Giá trị hàm mục tiêu cuối cùng: {self._calculate_combined_heuristic(current, self.target_state)}")
        
        return self._reconstruct_path(current, parent)

class BeamSteepestHC(SteepestHC):
    def __init__(self, initial_state, target_state, beam_width=10):
        super().__init__(initial_state, target_state)
        self.beam_width = beam_width  # Số lượng trạng thái giữ lại mỗi lần mở rộng
        self.max_iterations = 15000   # Tăng số lần lặp tối đa
        self.restart_threshold = 500  # Số bước không cải thiện trước khi khởi động lại
        self.dynamic_beam_adjustment = True  
        self.min_beam_width = 5       # Giới hạn dưới của beam width
        self.max_beam_width = 30      # Giới hạn trên của beam width
        
    def solve(self):
        """Beam Search kết hợp với Steepest Hill Climbing"""
        import random
        import math
        import heapq
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        # Khởi tạo tham số
        best_h_overall = float('inf')
        best_state_overall = None
        iterations = 0
        restarts = 0
        max_restarts = 5  # Số lần khởi động lại tối đa
        
        global_parent = {}
        
        current_beam = [(self._calculate_combined_heuristic(self.initial_state, self.target_state), 
                        0, self.initial_state)]  # (heuristic, tie_breaker, state)
        visited_states = {self.initial_state: 0}  # state -> cost (heuristic)
        parent = {self.initial_state: None}
        
        stagnation_counter = 0
        progress_counter = 0
        current_beam_width = self.beam_width
        
        initial_temperature = 1.0
        temperature = initial_temperature
        cooling_rate = 0.995
        min_temperature = 0.001
        
        while iterations < self.max_iterations and restarts < max_restarts:
            iterations += 1
            
            if iterations % 1000 == 0:
                print(f"Đã thực hiện {iterations} bước, restarts: {restarts}, beam width: {current_beam_width}, temperature: {temperature:.6f}")
            
            for _, _, state in current_beam:
                if state == self.target_state:
                    print(f"Tìm thấy giải pháp sau {iterations} bước và {restarts} lần khởi động lại!")
                    global_parent.update(parent)
                    return self._reconstruct_path(state, global_parent)
            
            next_beam_candidates = []
            current_best_h = float('inf')
            
            for h_current, _, current in current_beam:
                current_state = PuzzleState(current)
                next_states = current_state.get_next_states()
                
                for next_state in next_states:
                    next_h = self._calculate_combined_heuristic(next_state, self.target_state)
                    
                    if next_h < best_h_overall:
                        best_h_overall = next_h
                        best_state_overall = next_state
                        progress_counter += 1
                        stagnation_counter = 0
                    
                    if next_h < current_best_h:
                        current_best_h = next_h
                    
                    delta_h = next_h - h_current
                    
                    if delta_h <= 0: 
                        acceptance_probability = 1.0
                    else:  
                        acceptance_probability = math.exp(-delta_h / temperature)
                    
                    if next_state in visited_states and next_h > visited_states[next_state] + 2:
                        if random.random() > acceptance_probability: 
                            continue
                    
                    random_tiebreaker = random.random() * 0.01
                    
                    if random.random() < acceptance_probability:
                        heapq.heappush(next_beam_candidates, (next_h, random_tiebreaker, next_state))
                        
                        
                        parent[next_state] = current
                        visited_states[next_state] = next_h
            
            if not next_beam_candidates:
                stagnation_counter += 1
                print(f"Không tìm thấy trạng thái mới sau {iterations} bước. Khởi động lại tìm kiếm.")
                self._perform_restart(parent, visited_states, global_parent)
                current_beam = [(self._calculate_combined_heuristic(self.initial_state, self.target_state),
                                0, self.initial_state)]
                restarts += 1
                temperature = initial_temperature 
                continue
            
            current_beam = []
            best_candidates = []
            medium_candidates = []
            
            while next_beam_candidates and len(best_candidates) < current_beam_width:
                h, tie, state = heapq.heappop(next_beam_candidates)
                if h <= current_best_h + 2:
                    best_candidates.append((h, tie, state))
                elif len(medium_candidates) < current_beam_width // 2:
                    medium_candidates.append((h, tie, state))
            
            current_beam.extend(best_candidates)
            
            if len(current_beam) < current_beam_width and stagnation_counter > 10:
                num_to_add = min(len(medium_candidates), current_beam_width - len(current_beam))
                current_beam.extend(medium_candidates[:num_to_add])
            
            if len(current_beam) > current_beam_width:
                current_beam = current_beam[:current_beam_width]
            
            if not current_beam:
                stagnation_counter += 1
            else:
                best_in_beam_h = min(h for h, _, _ in current_beam)
                if best_in_beam_h < current_best_h:
                    progress_counter += 1
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            
            if self.dynamic_beam_adjustment and iterations % 50 == 0:
                current_beam_width = self._adjust_beam_width(stagnation_counter, progress_counter)
                progress_counter = 0
            
            temperature *= cooling_rate
            
            if stagnation_counter >= 100 and temperature < initial_temperature * 0.5:
                print(f"Tăng nhiệt độ do bị kẹt ({stagnation_counter} bước không cải thiện)")
                temperature = initial_temperature * 0.5
                stagnation_counter = max(0, stagnation_counter - 50)
            
            if stagnation_counter >= self.restart_threshold:
                print(f"Stagnation detected after {stagnation_counter} iterations. Restarting search.")
                self._perform_restart(parent, visited_states, global_parent)
                current_beam = [(self._calculate_combined_heuristic(self.initial_state, self.target_state),
                                0, self.initial_state)]
                restarts += 1
                temperature = initial_temperature  
                stagnation_counter = 0
        
        if best_state_overall is not None and best_state_overall != self.initial_state:
            print(f"Không tìm thấy giải pháp tối ưu sau {iterations} bước.")
            print(f"Trả về đường đi tới trạng thái tốt nhất với h = {best_h_overall}")
            global_parent.update(parent)
            return self._reconstruct_path(best_state_overall, global_parent)
        
        print(f"Không tìm thấy giải pháp sau {iterations} bước và {restarts} lần khởi động lại.")
        return None
    
    def _adjust_beam_width(self, stagnation, progress):
        """Điều chỉnh beam width dựa trên tiến độ tìm kiếm"""
        current_width = self.beam_width
        
        if stagnation > 50:  
            new_width = min(current_width + 5, self.max_beam_width)
            if new_width != current_width:
                print(f"Tăng beam width từ {current_width} lên {new_width} do mắc kẹt.")
                self.beam_width = new_width
        elif progress > 10:  
            new_width = max(current_width - 2, self.min_beam_width)
            if new_width != current_width:
                print(f"Giảm beam width từ {current_width} xuống {new_width} do tiến độ tốt.")
                self.beam_width = new_width
                
        return self.beam_width
    
    def _perform_restart(self, parent, visited, global_parent):
        """Thực hiện khởi động lại tìm kiếm từ trạng thái ban đầu hoặc trạng thái ngẫu nhiên"""
        import random
        
        global_parent.update(parent)
        
        parent.clear()
        visited.clear()
        
        if random.random() < 0.3 and len(global_parent) > 10:
            good_states = sorted([(self._calculate_combined_heuristic(state, self.target_state), state) 
                                 for state in global_parent.keys()])
            num_candidates = max(1, len(good_states) // 5)
            _, random_state = random.choice(good_states[:num_candidates])
            
            self.initial_state = random_state
            print(f"Khởi động lại từ trạng thái ngẫu nhiên với h = {self._calculate_combined_heuristic(random_state, self.target_state)}")
        else:
            print("Khởi động lại từ trạng thái ban đầu.")
        
        parent[self.initial_state] = None
        visited[self.initial_state] = self._calculate_combined_heuristic(self.initial_state, self.target_state)

class StochasticHC(HillClimbingSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
    
    def solve(self):
        """Thuật toán Leo đồi ngẫu nhiên (Stochastic Hill Climbing)"""
        import random
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        current = self.initial_state
        parent = {current: None}
        path = [current]
        iterations = 0
        max_iterations = self.max_iterations
        
        while current != self.target_state and iterations < max_iterations:
            iterations += 1
            
            if iterations % 1000 == 0:
                print(f"Đã thực hiện {iterations} bước...")
            
            # Tạo tất cả các trạng thái hàng xóm
            current_state = PuzzleState(current)
            next_states = current_state.get_next_states()
            
            if not next_states:
                break
            
            current_h = self._calculate_combined_heuristic(current, self.target_state)
            
            # Lọc ra các hàng xóm có heuristic tốt hơn hiện tại
            better_neighbors = []
            
            for next_state in next_states:
                next_h = self._calculate_combined_heuristic(next_state, self.target_state)
                if next_h < current_h:  
                    better_neighbors.append(next_state)
            
            if not better_neighbors:
                break
            
            # Chọn NGẪU NHIÊN một hàng xóm trong số các hàng xóm tốt hơn
            chosen_neighbor = random.choice(better_neighbors)
            
            parent[chosen_neighbor] = current
            current = chosen_neighbor
            path.append(current)
        
        if current == self.target_state:
            print(f"Tìm thấy giải pháp sau {iterations} bước!")
        else:
            print(f"Không tìm thấy giải pháp sau {iterations} bước.")
            print(f"Giá trị hàm mục tiêu cuối cùng: {self._calculate_combined_heuristic(current, self.target_state)}")
        
        return self._reconstruct_path(current, parent)

class SimulatedAnnealingHC(HillClimbingSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.initial_temperature = 1000.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.01
    
    def solve(self):
        """Thuật toán Mô phỏng luyện kim (Simulated Annealing)"""
        import random
        import math
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        # 1. Khởi tạo
        current = self.initial_state
        parent = {current: None}
        path = [current]
        
        best_state = current
        best_h = self._calculate_combined_heuristic(current, self.target_state)
        best_parent = {current: None}
        
        temperature = self.initial_temperature
        iterations = 0
        
        # 2. Lặp lại đến khi nhiệt độ gần 0
        while temperature > self.min_temperature and iterations < self.max_iterations:
            iterations += 1
            
            if iterations % 1000 == 0:
                print(f"Đã thực hiện {iterations} bước, nhiệt độ: {temperature:.2f}")
            
            # a. Sinh hàng xóm - Tạo một trạng thái hàng xóm ngẫu nhiên
            current_state = PuzzleState(current)
            next_states = current_state.get_next_states()
            
            if not next_states:
                break
            
            # Chọn ngẫu nhiên một trạng thái hàng xóm
            next_state = random.choice(next_states)
            
            # b. Tính độ thay đổi hàm mục tiêu
            current_h = self._calculate_combined_heuristic(current, self.target_state)
            next_h = self._calculate_combined_heuristic(next_state, self.target_state)
            
            # ΔE = heuristic(current_state) - heuristic(next_state)
            delta_E = current_h - next_h
            
            accept_new_state = False
            
            # Nếu ΔE > 0  → Chấp nhận luôn
            if delta_E > 0:
                accept_new_state = True
            # Nếu ΔE ≤ 0  → Chấp nhận với xác suất P = exp(ΔE / T)
            else:
                acceptance_probability = math.exp(delta_E / temperature)
                if random.random() < acceptance_probability:
                    accept_new_state = True
                    if iterations % 500 == 0:
                        print(f"Chấp nhận trạng thái xấu hơn với xác suất {acceptance_probability:.4f} ở nhiệt độ {temperature:.2f}")
            
            if accept_new_state:
                parent[next_state] = current
                current = next_state
                path.append(current)
                
                current_h = self._calculate_combined_heuristic(current, self.target_state)
                if current_h < best_h:
                    best_h = current_h
                    best_state = current
                    best_parent = parent.copy()
            
            if current == self.target_state:
                print(f"Tìm thấy giải pháp sau {iterations} bước!")
                return self._reconstruct_path(current, parent)
            
            # 3. Làm nguội - Cập nhật nhiệt độ
            temperature = temperature * self.cooling_rate
        
        # 4. Kết thúc - Trả về trạng thái hiện tại hoặc trạng thái tốt nhất đã tìm thấy
        print(f"Kết thúc tìm kiếm sau {iterations} bước, nhiệt độ cuối: {temperature:.2f}")
        
        if best_state == self.target_state:
            return self._reconstruct_path(best_state, best_parent)
        else:
            print(f"Không tìm thấy giải pháp sau {iterations} bước.")
            print(f"Giá trị hàm mục tiêu cuối cùng: {self._calculate_combined_heuristic(best_state, self.target_state)}")
            return self._reconstruct_path(current, parent)
    
    def set_annealing_params(self, initial_temp=1000.0, cooling_rate=0.95, min_temp=0.01):
        """Thiết lập các tham số cho Simulated Annealing"""
        self.initial_temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temp
        return self

class GeneticAlgorithm(SearchAlgorithm):
    def __init__(self, initial_state, target_state, population_size=20, max_generations=1000, 
                 mutation_rate=0.1, elite_size=10, crossover_rate=0.8):
        super().__init__(initial_state, target_state)
        self.population_size = population_size        # Kích thước quần thể
        self.max_generations = max_generations        # Số thế hệ tối đa
        self.mutation_rate = mutation_rate            # Tỷ lệ đột biến
        self.elite_size = elite_size                  # Số cá thể tốt nhấtnhất
        self.crossover_rate = crossover_rate          # Tỷ lệ lai ghép
        self.best_solution = None                     
        
    def _calculate_fitness(self, state):
        """Tính độ thích nghi (fitness) của một cá thể
        Trong trường hợp 8-puzzle, hàm fitness tỷ lệ nghịch với khoảng cách Manhattan"""
        distance = 0
        for i in range(len(state)):
            if state[i] != '_' and state[i] != self.target_state[i]:
                target_idx = self.target_state.index(state[i])
                curr_row, curr_col = divmod(i, 3)
                target_row, target_col = divmod(target_idx, 3)
                distance += abs(curr_row - target_row) + abs(curr_col - target_col)
        
        return 1 / (distance + 1)

    def _generate_initial_population(self):
        """Khởi tạo dân số ban đầu (Initial Population)
        Tạo một tập hợp các giải pháp ngẫu nhiên cho bài toán"""
        import random
        population = [self.initial_state]
        
        while len(population) < self.population_size:
            current = random.choice(population)  
            state_obj = PuzzleState(current)
            
            for _ in range(random.randint(3, 15)):
                next_states = state_obj.get_next_states()
                if next_states:
                    current = random.choice(next_states)
                    state_obj = PuzzleState(current)
            
            if current not in population and self._is_solvable(current):
                population.append(current)
        
        print(f"Đã khởi tạo quần thể ban đầu với {len(population)} cá thể")
        return population

    def _selection(self, population, fitness_scores):
        """Lựa chọn các cá thể tốt nhất (Selection)
        Chọn lọc cá thể dựa trên fitness để tạo ra thế hệ tiếp theo"""
        import random
        
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.sample(population, self.population_size)
        
        selection_probs = [f / total_fitness for f in fitness_scores]
        
        # Chọn elite_size cá thể tốt nhất 
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], 
                             reverse=True)[:self.elite_size]
        elite = [population[i] for i in elite_indices]
        
        # Chọn phần còn lại bằng phương pháp roulette wheel 
        selected = elite.copy()
        while len(selected) < self.population_size:
            pick = random.random()
            current = 0
            for i, prob in enumerate(selection_probs):
                current += prob
                if pick <= current:
                    if population[i] not in selected:
                        selected.append(population[i])
                    break
        
        while len(selected) < self.population_size:
            selected.append(random.choice(elite))
        
        return selected

    def _crossover(self, parent1, parent2):
        """Lai ghép (Crossover)
        Tạo cá thể con bằng cách kết hợp thông tin từ hai cá thể cha mẹ"""
        import random
        
        if random.random() > self.crossover_rate:
            return random.choice([parent1, parent2])
        
        cut_point = random.randint(1, 7)
        
        child = list(parent1[:cut_point])
        remaining = [c for c in parent2 if c not in child and c != '_']
        
        child.extend(remaining)
        
        while len(child) < 8:
            for char in "12345678":
                if char not in child:
                    child.append(char)
        
        blank_pos = random.randint(0, 8)
        child.insert(blank_pos, '_')
        
        result = ''.join(child)
        
        if len(result) != 9 or '_' not in result or not self._is_solvable(result):
            return random.choice([parent1, parent2])  
        
        return result

    def _mutate(self, state):
        """Đột biến (Mutation)
        Thay đổi ngẫu nhiên một phần nhỏ của cá thể để tạo sự đa dạng gen"""
        import random
        
        if random.random() > self.mutation_rate:
            return state
        
        state_obj = PuzzleState(state)
        next_states = state_obj.get_next_states()
        
        if next_states:
            mutated_state = random.choice(next_states)
            return mutated_state
        
        return state

    def _is_solvable(self, state):
        """Kiểm tra xem trạng thái có thể giải được không"""
        state_list = [0 if x == '_' else int(x) for x in state]
        target_list = [0 if x == '_' else int(x) for x in self.target_state]
        
        # Tính số nghịch thế trong state
        state_inversions = 0
        for i in range(len(state_list)):
            if state_list[i] == 0:  
                continue
            for j in range(i+1, len(state_list)):
                if state_list[j] != 0 and state_list[i] > state_list[j]:
                    state_inversions += 1
        
        # Tính số nghịch thế trong target
        target_inversions = 0
        for i in range(len(target_list)):
            if target_list[i] == 0: 
                continue
            for j in range(i+1, len(target_list)):
                if target_list[j] != 0 and target_list[i] > target_list[j]:
                    target_inversions += 1
        
        blank_row_state = state.index('_') // 3
        blank_row_target = self.target_state.index('_') // 3
        
        return (state_inversions % 2 == target_inversions % 2) == ((blank_row_state - blank_row_target) % 2 == 0)

    def solve(self):
        """Giải bài toán bằng Genetic Algorithm theo các bước:
        1. Khởi tạo dân số ban đầu
        2. Đánh giá các giải pháp
        3. Lựa chọn các cá thể tốt nhất
        4. Tạo thế hệ mới thông qua lai ghép và đột biến
        5. Lặp lại quá trình cho đến khi tìm được giải pháp hoặc đạt số thế hệ tối đa
        6. Kết thúc và trả về giải pháp tốt nhất"""
        import random
        import time
        start_time = time.time()
        
        if not self._is_solvable(self.initial_state):
            print("Bài toán không thể giải được!")
            return None

        # 1. Khởi tạo dân số ban đầu
        population = self._generate_initial_population()
        generation = 0
        best_fitness_overall = 0
        best_state_overall = self.initial_state
        
        while generation < self.max_generations:
            # 2. Đánh giá các giải pháp - Tính fitness cho từng cá thể
            fitness_scores = [self._calculate_fitness(state) for state in population]
            
            # Tìm cá thể tốt nhất trong thế hệ hiện tại
            best_fitness_idx = fitness_scores.index(max(fitness_scores))
            best_fitness = fitness_scores[best_fitness_idx]
            best_state = population[best_fitness_idx]
            
            # Cập nhật giải pháp tốt nhất tổng thể
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_state_overall = best_state
                self.best_solution = best_state
            
            if best_state == self.target_state:
                elapsed_time = time.time() - start_time
                print(f"Tìm thấy giải pháp ở thế hệ {generation} sau {elapsed_time:.3f}s!")
                self.best_solution = best_state
                return self._reconstruct_path_from_best()

            if generation % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Thế hệ {generation}: Fitness tốt nhất = {best_fitness:.4f}, Thời gian: {elapsed_time:.3f}s")

            # 3. Lựa chọn các cá thể tốt nhất
            selected = self._selection(population, fitness_scores)

            # 4. Tạo thế hệ mới
            new_population = []
            
            elite_indices = sorted(range(len(fitness_scores)), 
                                key=lambda i: fitness_scores[i],
                                reverse=True)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # Lai ghép và đột biến để tạo phần còn lại của quần thể
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                
                # Lai ghép để tạo con
                child = self._crossover(parent1, parent2)
                
                # Đột biến
                child = self._mutate(child)
                
                if self._is_solvable(child):
                    new_population.append(child)
            
            population = new_population
            generation += 1
        
        # 6. Kết thúc và trả về giải pháp tốt nhất
        elapsed_time = time.time() - start_time
        print(f"Hoàn thành sau {generation} thế hệ và {elapsed_time:.3f}s")
        print(f"Không tìm thấy giải pháp chính xác. Trả về trạng thái tốt nhất với fitness {best_fitness_overall:.4f}")
        
        self.best_solution = best_state_overall
        return self._reconstruct_path_from_best()

    def _reconstruct_path_from_best(self):
        """Tạo đường đi từ trạng thái ban đầu đến trạng thái tốt nhất tìm được"""
        if self.best_solution == self.initial_state:
            return [self.initial_state]
        
        astar = AStarSearch(self.initial_state, self.best_solution)
        path = astar.solve()
        
        if path is None:
            bfs = BreadthFirstSearch(self.initial_state, self.best_solution)
            path = bfs.solve()
            
            if path is None:
                return [self.initial_state, self.best_solution]
        
        return path

class BeliefStateSearch(SearchAlgorithm):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        # Khởi tạo belief state ban đầu - có thể là một tập hợp các trạng thái khả dĩ
        self.initial_belief = self._create_initial_belief()

    def _create_initial_belief(self):
        """Tạo belief state ban đầu dựa trên thông tin không chắc chắn"""
        # Ví dụ: giả sử chúng ta không chắc chắn vị trí của ô trống
        belief_states = set()
        
        belief_states.add(self.initial_state)
        
        # Tạo các biến thể khả dĩ 
        state_list = list(self.initial_state)
        blank_idx = state_list.index('_')
        
        if blank_idx != 0:  # Nếu ô trống không ở góc trên trái
            variant = state_list.copy()
            # Đổi chỗ ô trống với ô ở góc trên trái
            variant[blank_idx], variant[0] = variant[0], variant[blank_idx]
            belief_states.add(''.join(variant))
            
        if blank_idx != 2:  
            variant = state_list.copy()
            variant[blank_idx], variant[2] = variant[2], variant[blank_idx]
            belief_states.add(''.join(variant))
            
        return belief_states

    def _apply_action(self, action, belief_state):
        """Áp dụng một hành động lên tất cả các trạng thái trong belief state"""
        new_belief = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
        dr, dc = directions[action]
        
        for state in belief_state:
            state_obj = PuzzleState(state)
            blank_row, blank_col = state_obj.get_blank_position()
            new_row, new_col = blank_row + dr, blank_col + dc
            
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                blank_idx = blank_row * 3 + blank_col
                new_idx = new_row * 3 + new_col
                
                new_state_list = list(state)
                new_state_list[blank_idx], new_state_list[new_idx] = new_state_list[new_idx], new_state_list[blank_idx]
                new_belief.add(''.join(new_state_list))
        
        return new_belief if new_belief else None

    def _is_goal_belief(self, belief_state):
        """Kiểm tra xem tất cả các trạng thái trong belief state có phải là trạng thái đích không"""
        return all(state == self.target_state for state in belief_state)

    def solve(self):
        """Tìm dãy hành động để đưa tất cả trạng thái trong belief state về trạng thái đích"""
        queue = deque([(self.initial_belief, [])])  
        visited = {frozenset(self.initial_belief)}
        
        while queue:
            current_belief, actions = queue.popleft()
            
            if self._is_goal_belief(current_belief):
                return self._reconstruct_path(actions)
            
            for action in range(4): 
                next_belief = self._apply_action(action, current_belief)
                
                if next_belief is not None:
                    belief_key = frozenset(next_belief)
                    if belief_key not in visited:
                        visited.add(belief_key)
                        queue.append((next_belief, actions + [action]))
        
        return None  

    def _reconstruct_path(self, actions):
        """Tái tạo đường đi từ chuỗi hành động"""
        path = [self.initial_state]
        current = self.initial_state
        
        for action in actions:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
            dr, dc = directions[action]
            
            state_obj = PuzzleState(current)
            blank_row, blank_col = state_obj.get_blank_position()
            new_row, new_col = blank_row + dr, blank_col + dc
            
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                blank_idx = blank_row * 3 + blank_col
                new_idx = new_row * 3 + new_col
                
                new_state_list = list(current)
                new_state_list[blank_idx], new_state_list[new_idx] = new_state_list[new_idx], new_state_list[blank_idx]
                current = ''.join(new_state_list)
                path.append(current)
        
        return path

class ANDORSearch(SearchAlgorithm):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.max_depth = 31  # Giới hạn độ sâu để tránh vòng lặp vô hạn

    def _get_nondeterministic_next_states(self, state, action_index):
        """Mô phỏng tính phi tất định: một hành động có thể dẫn đến nhiều trạng thái"""
        state_obj = PuzzleState(state)
        blank_row, blank_col = state_obj.get_blank_position()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
        if action_index >= len(directions):
            return []

        dr, dc = directions[action_index]
        new_row, new_col = blank_row + dr, blank_col + dc
        next_states = []

        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_index = new_row * 3 + new_col
            old_index = blank_row * 3 + blank_col
            new_state = list(state)
            new_state[old_index], new_state[new_index] = new_state[new_index], new_state[old_index]
            next_states.append(''.join(new_state))

        import random
        for other_action in range(len(directions)):
            if other_action != action_index and random.random() < 0.3:  # 30% cơ hội nhiễu
                dr, dc = directions[other_action]
                new_row, new_col = blank_row + dr, blank_col + dc
                if 0 <= new_row < 3 and 0 <= new_col < 3:
                    new_index = new_row * 3 + new_col
                    old_index = blank_row * 3 + blank_col
                    new_state = list(state)
                    new_state[old_index], new_state[new_index] = new_state[new_index], new_state[old_index]
                    next_states.append(''.join(new_state))

        return next_states

    def solve(self):
        """Tìm kiếm cây AND-OR để tìm cây con giải pháp"""
        # Stack chứa: (state, depth, action_sequence, parent_map)
        stack = [(self.initial_state, 0, [], {self.initial_state: None})]
        visited = set()  

        while stack:
            state, depth, action_sequence, parent = stack.pop()
            if state == self.target_state:
                return self._reconstruct_path(state, parent)

            if depth >= self.max_depth or state in visited:
                continue

            visited.add(state)

            # Nút OR: thử tất cả các hành động
            for action_index in range(4):
                next_states = self._get_nondeterministic_next_states(state, action_index)
                if not next_states:
                    continue

                new_parent = parent.copy()
                for next_state in next_states:
                    if next_state not in visited:
                        new_parent[next_state] = state
                        new_action_sequence = action_sequence + [action_index]
                        stack.append((next_state, depth + 1, new_action_sequence, new_parent))

        return None

    def _reconstruct_path(self, state, parent):
        """Tái tạo đường đi từ trạng thái đích về ban đầu"""
        path = []
        while state is not None:
            path.append(state)
            state = parent.get(state)
        path.reverse()
        return path

class PartiallyObservableSearch(SearchAlgorithm):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.max_iterations = 1000  # Số bước tìm kiếm tối đa
        self.max_belief_size = 1000  # Kích thước belief state tối đa
        self.observation_noise = 0.1  # Xác suất quan sát không chính xác
        self.hidden_tiles = 2  
        self.belief_sizes = []  
    
    def _is_solvable(self, state, target):
        """Kiểm tra xem bài toán có thể giải được không"""
        state_list = [0 if x == '_' else int(x) for x in state]
        target_list = [0 if x == '_' else int(x) for x in target]
        
        state_inversions = 0
        for i in range(len(state_list)):
            if state_list[i] == 0:  
                continue
            for j in range(i+1, len(state_list)):
                if state_list[j] != 0 and state_list[i] > state_list[j]:
                    state_inversions += 1
        
        # Tính số nghịch thế 
        target_inversions = 0
        for i in range(len(target_list)):
            if target_list[i] == 0: 
                continue
            for j in range(i+1, len(target_list)):
                if target_list[j] != 0 and target_list[i] > target_list[j]:
                    target_inversions += 1
        
        # Kiểm tra tính chẵn lẻ của số nghịch thế
        return (state_inversions % 2) == (target_inversions % 2)
    
    def _create_initial_belief(self):
        """Tạo belief state ban đầu dựa trên giả định không biết vị trí chính xác của một số ô"""
        import random
        
        belief_states = set()
        belief_states.add(self.initial_state)
        
        state_list = list(self.initial_state)
        blank_idx = state_list.index('_')
        
        tiles = [i for i in range(9) if i != blank_idx]
        hidden_tiles = random.sample(tiles, min(self.hidden_tiles, len(tiles)))
        
        for perm in itertools.permutations(hidden_tiles):
            new_state = state_list.copy()
            
            for i, j in zip(hidden_tiles, perm):
                new_state[i], new_state[j] = new_state[j], new_state[i]
            
            new_state_str = ''.join(new_state)
            if self._is_solvable(new_state_str, self.target_state):
                belief_states.add(new_state_str)
            
            if len(belief_states) >= self.max_belief_size:
                break
        
        print(f"Khởi tạo belief state với {len(belief_states)} trạng thái khả dĩ")
        return belief_states
    
    def _get_observation(self, action, next_state, current_state):
        """Mô phỏng quan sát sau khi thực hiện hành động
           Trả về thông tin về ô đã được đổi chỗ với ô trống"""
        import random
        
        blank_idx_current = current_state.index('_')
        blank_idx_next = next_state.index('_')
        
        moved_tile = next_state[blank_idx_current]
        
        if random.random() < self.observation_noise:
            other_tiles = [next_state[i] for i in range(9) if i != blank_idx_next and i != blank_idx_current]
            if other_tiles:
                moved_tile = random.choice(other_tiles)
        
        return moved_tile
    
    def _update_belief_state(self, belief_state, action, observation):
        """Cập nhật belief state dựa trên hành động và quan sát"""
        new_belief = set()
        
        for state in belief_state:
            state_obj = PuzzleState(state)
            next_states = state_obj.get_next_states()
            
            for next_state in next_states:
                expected_observation = self._get_observation(action, next_state, state)
                
                if expected_observation == observation:
                    new_belief.add(next_state)
        
        if len(new_belief) > self.max_belief_size:
            new_belief = set(random.sample(list(new_belief), self.max_belief_size))
        
        return new_belief
    
    def _calculate_belief_heuristic(self, belief_state):
        """Tính heuristic cho belief state (trung bình Manhattan distance)"""
        if not belief_state:
            return float('inf')
        
        total_distance = 0
        for state in belief_state:
            distance = 0
            for i in range(len(state)):
                if state[i] != '_' and state[i] != self.target_state[i]:
                    target_idx = self.target_state.index(state[i])
                    curr_row, curr_col = divmod(i, 3)
                    target_row, target_col = divmod(target_idx, 3)
                    distance += abs(curr_row - target_row) + abs(curr_col - target_col)
            total_distance += distance
        
        return total_distance / len(belief_state)
    
    def _is_goal_belief(self, belief_state):
        """Kiểm tra xem belief state có chứa trạng thái đích không"""
        return self.target_state in belief_state
    
    def solve(self):
        """Tìm đường đi từ belief state ban đầu đến belief state chứa trạng thái đích"""
        import random
        import time
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        start_time = time.time()
        
        initial_belief = self._create_initial_belief()
        self.belief_sizes.append(len(initial_belief))
        
        import heapq
        queue = [(self._calculate_belief_heuristic(initial_belief), id(initial_belief), 
                 initial_belief, [self.initial_state], {self.initial_state: None})]
        visited_beliefs = set([frozenset(initial_belief)])
        
        iterations = 0
        
        while queue and iterations < self.max_iterations:
            iterations += 1
            
            _, _, current_belief, path, parent = heapq.heappop(queue)
            
            if self._is_goal_belief(current_belief):
                for state in current_belief:
                    if state == self.target_state:
                        parent[state] = path[-1]
                        path.append(state)
                        print(f"Tìm thấy giải pháp sau {iterations} bước, {time.time() - start_time:.3f}s")
                        print(f"Kích thước belief state qua các bước: {self.belief_sizes}")
                        return path
            
            current_state = path[-1]
            current_state_obj = PuzzleState(current_state)
            
            next_states = current_state_obj.get_next_states()
            
            next_states_with_heuristic = []
            for next_state in next_states:
                h = 0
                for i in range(len(next_state)):
                    if next_state[i] != '_' and next_state[i] != self.target_state[i]:
                        target_idx = self.target_state.index(next_state[i])
                        curr_row, curr_col = divmod(i, 3)
                        target_row, target_col = divmod(target_idx, 3)
                        h += abs(curr_row - target_row) + abs(curr_col - target_col)
                next_states_with_heuristic.append((h, next_state))
            
            next_states_with_heuristic.sort()
            
            for _, next_state in next_states_with_heuristic:
                observation = self._get_observation(None, next_state, current_state)
                
                new_belief = self._update_belief_state(current_belief, None, observation)
                
                if not new_belief:
                    continue  
                
                belief_key = frozenset(new_belief)
                
                if belief_key not in visited_beliefs:
                    visited_beliefs.add(belief_key)
                    
                    new_parent = parent.copy()
                    new_parent[next_state] = current_state
                    new_path = path + [next_state]
                    
                    belief_h = self._calculate_belief_heuristic(new_belief)
                    
                    heapq.heappush(queue, (belief_h, id(new_belief), new_belief, new_path, new_parent))
                    
                    self.belief_sizes.append(len(new_belief))
        
        print(f"Không tìm thấy giải pháp sau {iterations} bước, {time.time() - start_time:.3f}s")
        return None 

class BacktrackingSearch(SearchAlgorithm):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.max_depth = 31  # Giới hạn độ sâu 
        self.visited = set()  
        self.path = []  
        self.best_path = None  
        self.best_path_length = float('inf')  
        self.num_backtracks = 0  # Số lần quay lui
    
    def _is_solvable(self, state, target):
        """Kiểm tra xem bài toán có thể giải được không"""
        # Chuyển '_' thành '0' 
        state_list = [0 if x == '_' else int(x) for x in state]
        target_list = [0 if x == '_' else int(x) for x in target]
        
        # Tính số nghịch thế 
        state_inversions = 0
        for i in range(len(state_list)):
            if state_list[i] == 0: 
                continue
            for j in range(i+1, len(state_list)):
                if state_list[j] != 0 and state_list[i] > state_list[j]:
                    state_inversions += 1
        
        # Tính số nghịch thế 
        target_inversions = 0
        for i in range(len(target_list)):
            if target_list[i] == 0: 
                continue
            for j in range(i+1, len(target_list)):
                if target_list[j] != 0 and target_list[i] > target_list[j]:
                    target_inversions += 1
        
        return (state_inversions % 2) == (target_inversions % 2)
    
    def _calculate_heuristic(self, state):
        """Tính heuristic (khoảng cách Manhattan) để sắp xếp các nước đi"""
        distance = 0
        for i in range(len(state)):
            if state[i] != '_' and state[i] != self.target_state[i]:
                target_idx = self.target_state.index(state[i])
                curr_row, curr_col = divmod(i, 3)
                target_row, target_col = divmod(target_idx, 3)
                distance += abs(curr_row - target_row) + abs(curr_col - target_col)
        return distance
    
    def _backtrack(self, current_state, depth):
        """Hàm đệ quy thực hiện backtracking"""
        if current_state == self.target_state:
            if len(self.path) < self.best_path_length:
                self.best_path = self.path.copy()
                self.best_path_length = len(self.path)
            return True
        
        if depth >= self.max_depth:
            return False
        
        state_obj = PuzzleState(current_state)
        next_states = state_obj.get_next_states()
        
        next_states_with_heuristic = [(self._calculate_heuristic(state), state) for state in next_states]
        next_states_with_heuristic.sort()
        
        found_solution = False
        
        for _, next_state in next_states_with_heuristic:
            if next_state in self.visited:
                continue
            
            self.path.append(next_state)
            self.visited.add(next_state)
            
            # Đệ quy
            if self._backtrack(next_state, depth + 1):
                found_solution = True
                break  
            
            # Quay lui
            self.path.pop()
            self.num_backtracks += 1
        
        return found_solution
    
    def solve(self):
        """Giải bài toán sử dụng backtracking"""
        # Kiểm tra tính khả thi
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        self.visited = set([self.initial_state])
        self.path = [self.initial_state]
        self.best_path = None
        self.best_path_length = float('inf')
        self.num_backtracks = 0
        
        self._backtrack(self.initial_state, 0)
        
        print(f"Số lần quay lui: {self.num_backtracks}")
        
        return self.best_path

class ForwardCheckingSearch(BacktrackingSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.num_forward_checks = 0  # Số lần kiểm tra forward
        self.max_lookahead = 3  # Số bước nhìn trước tối đa
    
    def _is_promising(self, state, depth):
        """Kiểm tra xem trạng thái có triển vọng dẫn đến đích không"""
        # Nếu độ sâu gần đạt giới hạn, kiểm tra xem có thể đạt đến đích không
        if depth > self.max_depth - self.max_lookahead:
            return self._calculate_heuristic(state) <= (self.max_depth - depth)
        
        return self._forward_check(state, 0)
    
    def _forward_check(self, state, lookahead_depth):
        """Kiểm tra forward để xác định xem có đường đi khả thi không"""
        self.num_forward_checks += 1
        
        if state == self.target_state:
            return True
        
        if lookahead_depth >= self.max_lookahead:
            return True
        
        state_obj = PuzzleState(state)
        next_states = state_obj.get_next_states()
        
        next_states_with_heuristic = [(self._calculate_heuristic(next_state), next_state) 
                                     for next_state in next_states 
                                     if next_state not in self.visited]
        next_states_with_heuristic.sort()
        
        if not next_states_with_heuristic:
            return False
        
        for _, next_state in next_states_with_heuristic[:3]:  # Chỉ xem xét 3 trạng thái tốt nhất
            if self._forward_check(next_state, lookahead_depth + 1):
                return True
        
        return False
    
    def _backtrack(self, current_state, depth):
        """Hàm đệ quy thực hiện backtracking với forward checking"""
        if current_state == self.target_state:
            if len(self.path) < self.best_path_length:
                self.best_path = self.path.copy()
                self.best_path_length = len(self.path)
            return True
        
        if depth >= self.max_depth:
            return False
        
        state_obj = PuzzleState(current_state)
        next_states = state_obj.get_next_states()
        
        next_states_with_heuristic = [(self._calculate_heuristic(state), state) for state in next_states]
        next_states_with_heuristic.sort()
        
        found_solution = False
        
        for _, next_state in next_states_with_heuristic:
            # Bỏ qua nếu đã thăm
            if next_state in self.visited:
                continue
            
            # Forward checking 
            if not self._is_promising(next_state, depth + 1):
                continue
            
            self.path.append(next_state)
            self.visited.add(next_state)
            
            # Đệ quy
            if self._backtrack(next_state, depth + 1):
                found_solution = True
                break  
            
            # Quay lui
            self.path.pop()
            self.num_backtracks += 1
        
        return found_solution
    
    def solve(self):
        """Giải bài toán sử dụng forward checking"""
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        self.visited = set([self.initial_state])
        self.path = [self.initial_state]
        self.best_path = None
        self.best_path_length = float('inf')
        self.num_backtracks = 0
        self.num_forward_checks = 0
        
        self._backtrack(self.initial_state, 0)
        
        print(f"Số lần quay lui: {self.num_backtracks}")
        print(f"Số lần kiểm tra forward: {self.num_forward_checks}")
        
        return self.best_path

class MinConflictsSearch(HeuristicSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.max_iterations = 10000  # Số lần lặp tối đa
        self.max_plateau = 100  # Số bước tối đa không cải thiện
        self.restart_threshold = 500  # Số bước không cải thiện trước khi khởi động lại
        self.random_move_prob = 0.1  # Xác suất di chuyển ngẫu nhiên
        self.num_iterations = 0  
        self.num_restarts = 0  
        self.conflicts_history = []  
    
    def _calculate_conflicts(self, state):
        """Tính số xung đột (số ô không đúng vị trí so với trạng thái đích)"""
        conflicts = 0
        for i in range(9):
            if state[i] != '_' and state[i] != self.target_state[i]:
                conflicts += 1
        return conflicts
    
    def _get_conflicting_tiles(self, state):
        """Trả về danh sách các vị trí có xung đột"""
        positions = []
        for i in range(9):
            if state[i] != '_' and state[i] != self.target_state[i]:
                positions.append(i)
        return positions
    
    def _get_possible_moves(self, state):
        """Trả về danh sách các bước di chuyển có thể từ trạng thái hiện tại"""
        state_obj = PuzzleState(state)
        return state_obj.get_next_states()
    
    def _min_conflict_move(self, state, moves):
        """Chọn bước di chuyển giảm xung đột nhiều nhất"""
        import random
        
        current_conflicts = self._calculate_conflicts(state)
        min_conflicts = current_conflicts
        best_moves = []
        
        for move in moves:
            move_conflicts = self._calculate_conflicts(move)
            
            if move_conflicts < min_conflicts:
                min_conflicts = move_conflicts
                best_moves = [move]
            elif move_conflicts == min_conflicts:
                best_moves.append(move)
        
        if not best_moves or min_conflicts >= current_conflicts:
            if random.random() < self.random_move_prob:
                return random.choice(moves)
            
            same_conflict_moves = [move for move in moves if self._calculate_conflicts(move) == current_conflicts]
            if same_conflict_moves:
                return random.choice(same_conflict_moves)
            
            moves_by_conflicts = [(self._calculate_conflicts(move), move) for move in moves]
            moves_by_conflicts.sort()
            return moves_by_conflicts[0][1]
        
        return random.choice(best_moves)
    
    def _is_solvable(self, state, target):
        """Kiểm tra xem bài toán có thể giải được không"""
        state_list = [0 if x == '_' else int(x) for x in state]
        target_list = [0 if x == '_' else int(x) for x in target]
        
        # Tính số nghịch thế trong state
        state_inversions = 0
        for i in range(len(state_list)):
            if state_list[i] == 0:  
                continue
            for j in range(i+1, len(state_list)):
                if state_list[j] != 0 and state_list[i] > state_list[j]:
                    state_inversions += 1
        
        # Tính số nghịch thế trong target
        target_inversions = 0
        for i in range(len(target_list)):
            if target_list[i] == 0:  
                continue
            for j in range(i+1, len(target_list)):
                if target_list[j] != 0 and target_list[i] > target_list[j]:
                    target_inversions += 1
        
        # Kiểm tra tính chẵn lẻ của số nghịch thế
        return (state_inversions % 2) == (target_inversions % 2)
    
    def solve(self):
        """Giải bài toán sử dụng Min-Conflicts"""
        import random
        import time
        
        # Kiểm tra tính khả thi
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        start_time = time.time()
        
        # Khởi tạo
        current = self.initial_state
        best_state = current
        best_conflicts = self._calculate_conflicts(current)
        
        path = [current]
        parent = {current: None}
        
        self.num_iterations = 0
        self.num_restarts = 0
        plateau_counter = 0
        
        # Lưu số xung đột ban đầu
        self.conflicts_history.append(best_conflicts)
        
        while self.num_iterations < self.max_iterations:
            self.num_iterations += 1
            
            if current == self.target_state:
                print(f"Tìm thấy giải pháp sau {self.num_iterations} lần lặp, {self.num_restarts} lần khởi động lại, {time.time() - start_time:.3f}s")
                return self._reconstruct_path(current, parent)
            
            # In tiến trình mỗi 1000 bước
            if self.num_iterations % 1000 == 0:
                print(f"Đã thực hiện {self.num_iterations} bước, {self.num_restarts} lần khởi động lại, xung đột hiện tại: {self._calculate_conflicts(current)}")
            
            possible_moves = self._get_possible_moves(current)
            
            next_state = self._min_conflict_move(current, possible_moves)
            
            if next_state not in parent:
                parent[next_state] = current
            
            current = next_state
            path.append(current)
            
            current_conflicts = self._calculate_conflicts(current)
            self.conflicts_history.append(current_conflicts)
            
            if current_conflicts < best_conflicts:
                best_state = current
                best_conflicts = current_conflicts
                plateau_counter = 0
            else:
                plateau_counter += 1
            
            if plateau_counter >= self.restart_threshold:
                self.num_restarts += 1
                print(f"Khởi động lại lần {self.num_restarts} sau {plateau_counter} bước không cải thiện")
                
                if best_state != self.initial_state and best_state not in path:
                    parent[best_state] = path[-1]
                
                current = self.initial_state
                for _ in range(random.randint(5, 20)):
                    moves = self._get_possible_moves(current)
                    if moves:
                        next_move = random.choice(moves)
                        if next_move not in parent:
                            parent[next_move] = current
                        current = next_move
                
                path.append(current)
                plateau_counter = 0
        
        print(f"Không tìm thấy giải pháp chính xác sau {self.max_iterations} lần lặp, {self.num_restarts} lần khởi động lại")
        print(f"Trả về trạng thái tốt nhất với {best_conflicts} xung đột")
        
        if best_state != self.initial_state:
            return self._reconstruct_path(best_state, parent)
        
        return None

class QLearningPuzzleSolver(SearchAlgorithm):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.3
        self.max_episodes = 5000
        self.max_steps_per_episode = 100
        self.q_table = {}
        self.best_solution = None
        self.best_solution_length = float('inf')
        
    def _calculate_reward(self, state):
        reward = 0
        for i in range(len(state)):
            if state[i] != '_' and state[i] == self.target_state[i]:
                reward += 1
        return reward
    
    def _calculate_manhattan_distance(self, state):
        distance = 0
        for i in range(len(state)):
            if state[i] != '_' and state[i] != self.target_state[i]:
                target_idx = self.target_state.index(state[i])
                curr_row, curr_col = divmod(i, 3)
                target_row, target_col = divmod(target_idx, 3)
                distance += abs(curr_row - target_row) + abs(curr_col - target_col)
        return distance
    
    def _get_state_actions(self, state):
        state_obj = PuzzleState(state)
        next_states = state_obj.get_next_states()
        
        actions = []
        for next_state in next_states:
            blank_idx_current = state.index('_')
            blank_idx_next = next_state.index('_')
            
            row_current, col_current = divmod(blank_idx_current, 3)
            row_next, col_next = divmod(blank_idx_next, 3)
            
            if row_next < row_current:
                actions.append(("UP", next_state))
            elif row_next > row_current:
                actions.append(("DOWN", next_state))
            elif col_next < col_current:
                actions.append(("LEFT", next_state))
            elif col_next > col_current:
                actions.append(("RIGHT", next_state))
                
        return actions
    
    def _get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)
    
    def _choose_action(self, state, epsilon):
        import random
        
        actions = self._get_state_actions(state)
        if not actions:
            return None, None
            
        if random.random() < epsilon:
            action, next_state = random.choice(actions)
            return action, next_state
            
        q_values = [(self._get_q_value(state, action), action, next_state) for action, next_state in actions]
        max_q_value = max(q_values, key=lambda x: x[0])
        
        best_actions = [x for x in q_values if x[0] == max_q_value[0]]
        _, action, next_state = random.choice(best_actions)
        
        return action, next_state
    
    def _update_q_value(self, state, action, reward, next_state):
        current_q = self._get_q_value(state, action)
        
        next_actions = self._get_state_actions(next_state)
        max_next_q = max([self._get_q_value(next_state, next_action) for next_action, _ in next_actions], default=0)
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def _is_solvable(self, state, target):
        state_list = [0 if x == '_' else int(x) for x in state]
        target_list = [0 if x == '_' else int(x) for x in target]
        
        state_inversions = 0
        for i in range(len(state_list)):
            if state_list[i] == 0:
                continue
            for j in range(i+1, len(state_list)):
                if state_list[j] != 0 and state_list[i] > state_list[j]:
                    state_inversions += 1
        
        target_inversions = 0
        for i in range(len(target_list)):
            if target_list[i] == 0:
                continue
            for j in range(i+1, len(target_list)):
                if target_list[j] != 0 and target_list[i] > target_list[j]:
                    target_inversions += 1
        
        return (state_inversions % 2) == (target_inversions % 2)
    
    def save_q_table(self, filename="q_table.pkl"):
        import pickle
        
        print(f"Đang lưu bảng Q với {len(self.q_table)} cặp (state, action)...")
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Đã lưu bảng Q vào {filename}")

    def load_q_table(self, filename="q_table.pkl"):
        import pickle
        import os
        
        if not os.path.exists(filename):
            print(f"Không tìm thấy tệp {filename}. Sử dụng bảng Q trống.")
            return False
        
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Đã tải bảng Q từ {filename} với {len(self.q_table)} cặp (state, action)")
            return True
        except Exception as e:
            print(f"Lỗi khi tải bảng Q: {e}")
            return False
    
    def _find_path_with_q_table(self):
        current_state = self.initial_state
        path = [current_state]
        visited = {current_state}
        max_steps = 100
        
        for _ in range(max_steps):
            if current_state == self.target_state:
                return path
            
            actions = self._get_state_actions(current_state)
            if not actions:
                break
                
            best_action = None
            best_q_value = float('-inf')
            best_next_state = None
            
            for action, next_state in actions:
                q_value = self._get_q_value(current_state, action)
                if q_value > best_q_value and next_state not in visited:
                    best_q_value = q_value
                    best_action = action
                    best_next_state = next_state
            
            if best_next_state is None:
                break
                
            current_state = best_next_state
            path.append(current_state)
            visited.add(current_state)
        
        return None if current_state != self.target_state else path
    
    def solve(self):
        import random
        import time
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
            
        start_time = time.time()
        
        q_table_loaded = self.load_q_table()
        
        if q_table_loaded and len(self.q_table) > 1000:
            print("Thử giải bằng bảng Q đã lưu...")
            direct_solution = self._find_path_with_q_table()
            if direct_solution:
                print(f"Tìm thấy giải pháp trực tiếp từ bảng Q đã lưu!")
                return direct_solution
            print("Không tìm thấy giải pháp trực tiếp, tiếp tục huấn luyện...")
        
        best_path = None
        parent = {}
        
        successful_episodes = 0
        
        for episode in range(self.max_episodes):
            current_state = self.initial_state
            
            episode_path = [current_state]
            episode_parent = {current_state: None}
            
            current_epsilon = max(0.1, self.epsilon * (1 - episode / self.max_episodes))
            
            for step in range(self.max_steps_per_episode):
                if current_state == self.target_state:
                    successful_episodes += 1
                    
                    if len(episode_path) < self.best_solution_length:
                        self.best_solution_length = len(episode_path)
                        self.best_solution = episode_path.copy()
                        parent = episode_parent.copy()
                        
                        print(f"Tập {episode}: Tìm thấy đường đi ngắn hơn với {len(episode_path)} bước")
                    
                    break
                
                action, next_state = self._choose_action(current_state, current_epsilon)
                
                if action is None:
                    break
                
                reward = -0.1
                
                current_distance = self._calculate_manhattan_distance(current_state)
                next_distance = self._calculate_manhattan_distance(next_state)
                
                if next_distance < current_distance:
                    reward += 1
                elif next_distance > current_distance:
                    reward -= 1
                
                if next_state == self.target_state:
                    reward += 100
                
                self._update_q_value(current_state, action, reward, next_state)
                
                current_state = next_state
                
                episode_path.append(current_state)
                episode_parent[current_state] = episode_path[-2]
            
            if episode % 100 == 0:
                print(f"Tập {episode}: Epsilon = {current_epsilon:.3f}, Thành công = {successful_episodes}, "
                      f"Độ dài tốt nhất = {self.best_solution_length if self.best_solution else 'N/A'}")
        
        self.save_q_table()
        
        end_time = time.time()
        print(f"Hoàn thành sau {self.max_episodes} tập, {end_time - start_time:.3f}s")
        print(f"Tổng số tập thành công: {successful_episodes}")
        print(f"Kích thước bảng Q: {len(self.q_table)}")
        
        if self.best_solution:
            return self.best_solution
        
        print("Không tìm thấy giải pháp tối ưu bằng Q-learning, chuyển sang A*...")
        astar = AStarSearch(self.initial_state, self.target_state)
        return astar.solve()
    
    def _reconstruct_path(self, current, parent):
        path = []
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        return path


class PuzzleRenderer:
    def __init__(self, canvas):
        self.canvas = canvas
    
    def draw_state(self, state, prev_state, target):
        self.canvas.delete("all")
        
        for i in range(1, 3):
            self.canvas.create_line(0, i*100, 300, i*100, fill="black")
            self.canvas.create_line(i*100, 0, i*100, 300, fill="black")
        
        for i in range(9):
            row, col = divmod(i, 3)
            x0, y0 = col * 100, row * 100
            x1, y1 = x0 + 100, y0 + 100
            tile = state[i]
            
            if tile == '_':
                continue  
            
            color = "green" if state[i] == target[i] else "red"
            
            if prev_state is not None:
                p1 = prev_state.index('_')
                p2 = state.index('_')
                if i == p1:  
                    color = "blue"
            
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
            self.canvas.create_text(x0 + 50, y0 + 50, text=tile, font=("Arial", 24))

class AlgorithmFactory:
    @staticmethod
    def create_algorithm(algorithm_name, initial_state, target_state):
        algorithms = {
            "A*": AStarSearch,
            "BFS": BreadthFirstSearch,
            "DFS": DepthFirstSearch,
            "IDDFS": IterativeDeepeningDFS,
            "UCS": UniformCostSearch,
            "Greedy": GreedySearch,
            "Greedy Search": GreedySearch,
            "IDA*": IDAStarSearch,
            "SimpleHC": SimpleHC,
            "Simple Hill Climbing": SimpleHC,
            "SteepestHC": SteepestHC,
            "Steepest Hill Climbing": SteepestHC,
            "BeamSteepestHC": BeamSteepestHC,
            "Beam Steepest Hill Climbing": BeamSteepestHC,
            "StochasticHC": StochasticHC,
            "Stochastic Hill Climbing": StochasticHC,
            "SimulatedAnnealingHC": SimulatedAnnealingHC,
            "Simulated Annealing": SimulatedAnnealingHC,
            "GeneticAlgorithm": GeneticAlgorithm,
            "Genetic Algorithm": GeneticAlgorithm,
            "BeliefState": BeliefStateSearch,
            "Belief State": BeliefStateSearch,
            "ANDOR": ANDORSearch,
            "AND-OR Search": ANDORSearch,
            "PartiallyObservable": PartiallyObservableSearch,
            "Partially Observable": PartiallyObservableSearch,
            "Backtracking": BacktrackingSearch,
            "ForwardChecking": ForwardCheckingSearch,
            "Forward Checking": ForwardCheckingSearch,
            "MinConflicts": MinConflictsSearch,
            "Min Conflicts": MinConflictsSearch,
            "QLearning": QLearningPuzzleSolver,
            "Q-Learning": QLearningPuzzleSolver,
            "Uniform Cost Search": UniformCostSearch,
        }
        
        algorithm_class = algorithms.get(algorithm_name, BreadthFirstSearch)
        return algorithm_class(initial_state, target_state)

class PuzzleSolver:
    def __init__(self, root, initial, target):
        self.root = root
        self.initial = initial
        self.target = target
        self.path = None
        self.current_index = 0
        self.animation_speed = 1000  
        self.animation_running = False
        self.pause_animation = False
        
        self.create_ui()
        
        self.renderer = PuzzleRenderer(self.canvas)
        
        self.calculate_solution()
        
    def create_ui(self):
            top_frame = tk.Frame(self.root)
            top_frame.pack(pady=10)
    
            canvas_frame = tk.Frame(self.root)
            canvas_frame.pack(pady=10)
    
            control_frame = tk.Frame(self.root)
            control_frame.pack(pady=10)
    
            comparison_frame = tk.Frame(self.root)
            comparison_frame.pack(pady=5)
    
            self.algorithm_var = tk.StringVar(value="BFS")
            algorithm_label = tk.Label(top_frame, text="Thuật toán:")
            algorithm_label.pack(side=tk.LEFT, padx=5)
    
            algorithm_menu = tk.Menubutton(top_frame, textvariable=self.algorithm_var, relief=tk.RAISED)
            algorithm_menu.pack(side=tk.LEFT, padx=5)
    
            menu = tk.Menu(algorithm_menu, tearoff=0)
            algorithm_menu["menu"] = menu
    
            self.algorithm_groups = {
                "Informed Search": ["A*", "IDA*", "Greedy Search"],
                "Uninformed Search": ["BFS", "DFS", "Uniform Cost Search", "IDDFS"],
                "Local Search": ["Simple Hill Climbing", "Steepest Hill Climbing", 
                               "Beam Steepest Hill Climbing", "Stochastic Hill Climbing", 
                               "Simulated Annealing", "Genetic Algorithm"],
                "Complex Environments": ["Belief State", "AND-OR Search", "Partially Observable"],
                "CSPs": ["Backtracking", "Forward Checking", "Min Conflicts"],
                "Reinforcement Learning": ["Q-Learning"]
            }
    
            informed_menu = tk.Menu(menu, tearoff=0)
            menu.add_cascade(label="Informed Search", menu=informed_menu)
            for algo in self.algorithm_groups["Informed Search"]:
                informed_menu.add_command(label=algo, command=lambda a=algo: self.set_algorithm(a))
    
            uninformed_menu = tk.Menu(menu, tearoff=0)
            menu.add_cascade(label="Uninformed Search", menu=uninformed_menu)
            for algo in self.algorithm_groups["Uninformed Search"]:
                uninformed_menu.add_command(label=algo, command=lambda a=algo: self.set_algorithm(a))
    
            local_menu = tk.Menu(menu, tearoff=0)
            menu.add_cascade(label="Local Search", menu=local_menu)
            for algo in self.algorithm_groups["Local Search"]:
                local_menu.add_command(label=algo, command=lambda a=algo: self.set_algorithm(a))
    
            complex_menu = tk.Menu(menu, tearoff=0)
            menu.add_cascade(label="Complex Environments", menu=complex_menu)
            for algo in self.algorithm_groups["Complex Environments"]:
                complex_menu.add_command(label=algo, command=lambda a=algo: self.set_algorithm(a))
    
            csp_menu = tk.Menu(menu, tearoff=0)
            menu.add_cascade(label="CSPs", menu=csp_menu)
            for algo in self.algorithm_groups["CSPs"]:
                csp_menu.add_command(label=algo, command=lambda a=algo: self.set_algorithm(a))
    
            rl_menu = tk.Menu(menu, tearoff=0)
            menu.add_cascade(label="Reinforcement Learning", menu=rl_menu)
            for algo in self.algorithm_groups["Reinforcement Learning"]:
                rl_menu.add_command(label=algo, command=lambda a=algo: self.set_algorithm(a))
    
            solve_button = tk.Button(top_frame, text="Giải Mới", command=self.calculate_solution)
            solve_button.pack(side=tk.LEFT, padx=20)
    
            self.canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="white")
            self.canvas.pack()
    
            reset_button = tk.Button(control_frame, text="Reset", command=self.reset_animation)
            reset_button.pack(side=tk.LEFT, padx=5)
    
            self.play_pause_button = tk.Button(control_frame, text="Play", command=self.toggle_animation)
            self.play_pause_button.pack(side=tk.LEFT, padx=5)
    
            next_button = tk.Button(control_frame, text="Next", command=self.next_step)
            next_button.pack(side=tk.LEFT, padx=5)
    
            prev_button = tk.Button(control_frame, text="Prev", command=self.prev_step)
            prev_button.pack(side=tk.LEFT, padx=5)
    
            speed_label = tk.Label(control_frame, text="Tốc độ:")
            speed_label.pack(side=tk.LEFT, padx=(20, 5))
    
            self.speed_var = tk.IntVar(value=1)
            speed_scale = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, 
                                  variable=self.speed_var, command=self.update_speed)
            speed_scale.pack(side=tk.LEFT)
    
            compare_label = tk.Label(comparison_frame, text="So sánh thuật toán:")
            compare_label.pack(side=tk.LEFT, padx=5)
    
            self.compare_group_var = tk.StringVar(value="Informed Search")
            compare_group_menu = tk.OptionMenu(comparison_frame, self.compare_group_var, 
                                             *self.algorithm_groups.keys())
            compare_group_menu.pack(side=tk.LEFT, padx=5)
    
            compare_button = tk.Button(comparison_frame, text="So sánh", command=self.compare_algorithms)
            compare_button.pack(side=tk.LEFT, padx=5)
    
            solve_all_button = tk.Button(comparison_frame, text="Giải đồng thời", command=self.solve_all_in_group)
            solve_all_button.pack(side=tk.LEFT, padx=5)
    
            self.status_frame = tk.Frame(self.root)
            self.status_frame.pack(pady=10, fill=tk.X)
    
            self.status_label = tk.Label(self.status_frame, text="Trạng thái: Đang khởi tạo...")
            self.status_label.pack(side=tk.LEFT, padx=10)
    
            self.step_label = tk.Label(self.status_frame, text="Bước: 0/0")
            self.step_label.pack(side=tk.RIGHT, padx=10)
    
            export_button = tk.Button(control_frame, text="Export Steps", command=self.export_steps_to_console)
            export_button.pack(side=tk.LEFT, padx=5)

    def compare_algorithms(self):
        group_name = self.compare_group_var.get()
    
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title(f"So sánh thuật toán - {group_name}")
        comparison_window.geometry("800x600")
    
        result_frame = tk.Frame(comparison_window)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        status_label = tk.Label(result_frame, text=f"Đang so sánh các thuật toán trong nhóm {group_name}...", 
                              font=("Arial", 12))
        status_label.pack(pady=10)
        comparison_window.update()
    
        comparison = AlgorithmComparison(self.initial, self.target, group_name)
    
        def run_comparison():
            try:
                comparison.run_comparison()
                summary = comparison.get_summary()
            
                status_label.destroy()
            
                result_text = tk.Text(result_frame, wrap=tk.WORD, font=("Courier New", 12))
                result_text.pack(fill=tk.BOTH, expand=True)
                result_text.insert(tk.END, summary)
            
                self.create_comparison_charts(comparison, result_frame)
            
            except Exception as e:
                status_label.config(text=f"Lỗi khi so sánh: {str(e)}")
    
        threading.Thread(target=run_comparison, daemon=True).start()

    def create_comparison_charts(self, comparison, parent_frame):
        charts_frame = tk.Frame(parent_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        algorithms = []
        times = []
        path_lengths = []
    
        for algo, result in comparison.results.items():
            algorithms.append(algo)
            times.append(result['execution_time'])
            path_lengths.append(result['path_length'] if result['path_length'] is not None else 0)
    
        ax1.bar(algorithms, times, color='skyblue')
        ax1.set_title('Thời gian thực thi (giây)')
        ax1.set_ylabel('Thời gian (s)')
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    
        ax2.bar(algorithms, path_lengths, color='lightgreen')
        ax2.set_title('Độ dài đường đi')
        ax2.set_ylabel('Số bước')
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    
        fig.tight_layout()
    
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def solve_all_in_group(self):
        group_name = self.compare_group_var.get()
        algorithms = self.algorithm_groups.get(group_name, [])

        if not algorithms:
            messagebox.showinfo("Thông báo", f"Không có thuật toán nào trong nhóm {group_name}")
            return

        results_window = tk.Toplevel(self.root)
        results_window.title(f"Giải đồng thời - {group_name}")
        results_window.geometry("1200x800")

        main_frame = tk.Frame(results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        control_frame = tk.Frame(scrollable_frame)
        control_frame.pack(fill=tk.X, pady=5)

        title_label = tk.Label(control_frame, text=f"Giải đồng thời - {group_name}", font=("Arial", 12, "bold"))
        title_label.pack(side=tk.LEFT, padx=5)

        step_label = tk.Label(control_frame, text="0/0")
        step_label.pack(side=tk.RIGHT, padx=10)

        reset_button = tk.Button(control_frame, text="Reset", width=8)
        reset_button.pack(side=tk.RIGHT, padx=5)

        play_pause_button = tk.Button(control_frame, text="Play", width=8)
        play_pause_button.pack(side=tk.RIGHT, padx=5)

        status_labels = {}
        algorithm_frames = {}
        renderers = {}
        animation_controls = {}
    
        algorithm_grid_frame = tk.Frame(scrollable_frame)
        algorithm_grid_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
        cols_per_row = 3
        rows_needed = (len(algorithms) + cols_per_row - 1) // cols_per_row

        for i, algorithm in enumerate(algorithms):
            row = i // cols_per_row
            col = i % cols_per_row
        
            algo_frame = tk.LabelFrame(algorithm_grid_frame, text=algorithm, padx=5, pady=5)
            algo_frame.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
        
            algorithm_grid_frame.columnconfigure(col, weight=1)
        
            algo_canvas = tk.Canvas(algo_frame, width=300, height=300, bg="white")
            algo_canvas.pack(pady=2)
        
            status_label = tk.Label(algo_frame, text="Đang giải...", anchor="w", font=("Arial", 8))
            status_label.pack(fill=tk.X, expand=True, pady=1)

            status_labels[algorithm] = status_label
            algorithm_frames[algorithm] = algo_frame
            renderers[algorithm] = PuzzleRenderer(algo_canvas)

            animation_controls[algorithm] = {
                "current_step": 0,
                "is_playing": False,
                "path": None
            }

        results_window.update()

        threads = []
        results = {}

        def solve_algorithm(algorithm_name):
            try:
                algorithm = AlgorithmFactory.create_algorithm(algorithm_name, self.initial, self.target)
        
                start_time = time.time()
                path = algorithm.solve()
                end_time = time.time()
        
                execution_time = end_time - start_time
                path_length = len(path) - 1 if path else None
        
                results[algorithm_name] = {
                    "path": path,
                    "execution_time": execution_time,
                    "path_length": path_length
                }
        
                if path:
                    status_labels[algorithm_name].config(
                        text=f"Đã giải. Thời gian: {execution_time:.3f}s | Bước: {path_length}"
                    )
            
                    animation_controls[algorithm_name]["path"] = path
                    animation_controls[algorithm_name]["current_step"] = 0
            
                    renderers[algorithm_name].draw_state(path[0], None, self.target)
                else:
                    status_labels[algorithm_name].config(text="Không tìm thấy lời giải!")
        
            except Exception as e:
                status_labels[algorithm_name].config(text=f"Lỗi: {str(e)}")

        for algorithm in algorithms:
            thread = threading.Thread(target=solve_algorithm, args=(algorithm,), daemon=True)
            threads.append(thread)
            thread.start()

        def check_threads():
            if any(thread.is_alive() for thread in threads):
                results_window.after(100, check_threads)
            else:
                setup_common_controls()
            
                if len(algorithms) <= 6:
                    self.create_comparison_charts_from_results(results, scrollable_frame)

        def setup_common_controls():
            max_path_length = 0
            for algo, controls in animation_controls.items():
                if controls["path"]:
                    max_path_length = max(max_path_length, len(controls["path"]) - 1)
        
            step_label.config(text=f"0/{max_path_length}")
        
            is_playing = [False]
            current_step = [0]
        
            def update_all_displays(step):
                for algo, controls in animation_controls.items():
                    path = controls["path"]
                    if path and step < len(path):
                        current_state = path[step]
                        prev_state = path[step-1] if step > 0 else None
                        renderers[algo].draw_state(current_state, prev_state, self.target)
                        controls["current_step"] = step
            
                current_step[0] = step
                step_label.config(text=f"{step}/{max_path_length}")
        
            def toggle_play_pause():
                is_playing[0] = not is_playing[0]
                play_pause_button.config(text="Pause" if is_playing[0] else "Play")
            
                if is_playing[0]:
                    animate_all()
        
            def reset_all():
                is_playing[0] = False
                play_pause_button.config(text="Play")
                update_all_displays(0)
        
            def animate_all():
                if not is_playing[0]:
                    return
            
                if current_step[0] < max_path_length:
                    current_step[0] += 1
                    update_all_displays(current_step[0])
                    results_window.after(200, animate_all)
                else:
                    is_playing[0] = False
                    play_pause_button.config(text="Play")
        
            play_pause_button.config(command=toggle_play_pause)
            reset_button.config(command=reset_all)
        
            update_all_displays(0)

        results_window.after(100, check_threads)

    def show_algorithm_details(self, algorithm_name, result):
        if not result["path"]:
            messagebox.showinfo("Thông báo", f"Không có đường đi cho thuật toán {algorithm_name}")
            return
    
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Chi tiết - {algorithm_name}")
        details_window.geometry("500x600")
    
        info_frame = tk.Frame(details_window)
        info_frame.pack(fill=tk.X, padx=10, pady=10)
    
        info_text = f"Thuật toán: {algorithm_name}\n"
        info_text += f"Thời gian thực thi: {result['execution_time']:.3f}s\n"
        info_text += f"Độ dài đường đi: {result['path_length']}\n"
    
        info_label = tk.Label(info_frame, text=info_text, justify=tk.LEFT, anchor="w")
        info_label.pack(fill=tk.X)
    
        canvas_frame = tk.Frame(details_window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="white")
        canvas.pack(pady=10)
    
        renderer = PuzzleRenderer(canvas)
    
        control_frame = tk.Frame(details_window)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
    
        current_step = tk.IntVar(value=0)
    
        def update_display():
            step = current_step.get()
            if 0 <= step < len(result["path"]):
                current_state = result["path"][step]
                prev_state = result["path"][step-1] if step > 0 else None
                renderer.draw_state(current_state, prev_state, self.target)
                step_label.config(text=f"Bước: {step}/{len(result['path']) - 1}")
    
        prev_button = tk.Button(control_frame, text="Prev", 
                              command=lambda: [current_step.set(max(0, current_step.get() - 1)), update_display()])
        prev_button.pack(side=tk.LEFT, padx=5)
    
        step_label = tk.Label(control_frame, text=f"Bước: 0/{len(result['path']) - 1}")
        step_label.pack(side=tk.LEFT, padx=10)
    
        next_button = tk.Button(control_frame, text="Next", 
                              command=lambda: [current_step.set(min(len(result['path']) - 1, current_step.get() + 1)), update_display()])
        next_button.pack(side=tk.LEFT, padx=5)
    
        update_display()

    def create_comparison_charts_from_results(self, results, parent_frame):
        charts_frame = tk.LabelFrame(parent_frame, text="So sánh", padx=10, pady=10)
        charts_frame.pack(fill=tk.X, expand=True, pady=10)
    
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
        algorithms = []
        times = []
        path_lengths = []
    
        for algo, result in results.items():
            algorithms.append(algo)
            times.append(result['execution_time'])
            path_lengths.append(result['path_length'] if result['path_length'] is not None else 0)
    
        ax1.bar(algorithms, times, color='skyblue')
        ax1.set_title('Thời gian thực thi (giây)')
        ax1.set_ylabel('Thời gian (s)')
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    
        ax2.bar(algorithms, path_lengths, color='lightgreen')
        ax2.set_title('Độ dài đường đi')
        ax2.set_ylabel('Số bước')
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    
        fig.tight_layout()
    
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def set_algorithm(self, algorithm_name):
        self.algorithm_var.set(algorithm_name)
        
    def export_steps_to_console(self):
        if not self.path:
            print("No solution path available to export.")
            return
    
        print("\n===== SOLUTION PATH =====")
        print(f"Algorithm: {self.algorithm_var.get()}")
        print(f"Total steps: {len(self.path) - 1}")
    
        algorithm_name = self.algorithm_var.get()
        if algorithm_name == "PartiallyObservable":
            belief_sizes = getattr(self.algorithm, 'belief_sizes', [])
            if belief_sizes:
                print(f"Initial belief state size: {belief_sizes[0]}")
                print(f"Final belief state size: {belief_sizes[-1]}")
                print(f"Average belief state size: {sum(belief_sizes) / len(belief_sizes):.2f}")
                print(f"Max belief state size: {max(belief_sizes)}")
        elif algorithm_name == "MinConflicts":
            num_iterations = getattr(self.algorithm, 'num_iterations', 0)
            num_restarts = getattr(self.algorithm, 'num_restarts', 0)
            conflicts_history = getattr(self.algorithm, 'conflicts_history', [])
            print(f"Number of iterations: {num_iterations}")
            print(f"Number of restarts: {num_restarts}")
            if conflicts_history:
                print(f"Initial conflicts: {conflicts_history[0]}")
                print(f"Final conflicts: {conflicts_history[-1]}")
    
        print("=======================\n")

        for i, state in enumerate(self.path):
            print(f"Step {i}:")
            for row in range(3):
                print(" ".join(state[row*3:row*3+3]).replace("_", " "))
        
            if algorithm_name == "PartiallyObservable" and i < len(belief_sizes) - 1:
                print(f"Belief state size: {belief_sizes[i]}")
            elif algorithm_name == "MinConflicts" and i < len(conflicts_history):
                print(f"Conflicts: {conflicts_history[i]}")
        
            print()
    
    def calculate_solution(self):
        self.status_label.config(text="Trạng thái: Đang tính toán...")
        self.root.update()
    
        algorithm_name = self.algorithm_var.get()
        self.algorithm = AlgorithmFactory.create_algorithm(algorithm_name, self.initial, self.target)
    
        start_time = time.time()
        self.path = self.algorithm.solve()
        end_time = time.time()
    
        if self.path:
            if algorithm_name == "PartiallyObservable":
                belief_sizes = getattr(self.algorithm, 'belief_sizes', [])
                avg_belief_size = sum(belief_sizes) / len(belief_sizes) if belief_sizes else 0
                self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s. Kích thước belief trung bình: {avg_belief_size:.1f}")
            elif algorithm_name == "MinConflicts":
                num_iterations = getattr(self.algorithm, 'num_iterations', 0)
                num_restarts = getattr(self.algorithm, 'num_restarts', 0)
                self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s. Số lần lặp: {num_iterations}, Khởi động lại: {num_restarts}")
            else:
                self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s")
        
            self.step_label.config(text=f"Bước: 0/{len(self.path) - 1}")
        else:
            self.status_label.config(text="Trạng thái: Không tìm thấy lời giải!")
            self.step_label.config(text="Bước: 0/0")
    
        self.current_index = 0
        self.animation_running = False
        self.pause_animation = False
        self.play_pause_button.config(text="Play")
    
        self.update_display()
    
    def update_display(self):
        if not self.path:
            return
            
        current_state = self.path[self.current_index]
        prev_state = None
        
        if self.current_index > 0:
            prev_state = self.path[self.current_index - 1]
            
        self.renderer.draw_state(current_state, prev_state, self.target)
        self.step_label.config(text=f"Bước: {self.current_index}/{len(self.path) - 1}")
    
    def toggle_animation(self):
        if not self.path:
            return
            
        if self.animation_running:
            self.pause_animation = True
            self.animation_running = False
            self.play_pause_button.config(text="Play")
        else:
            self.animation_running = True
            self.pause_animation = False
            self.play_pause_button.config(text="Pause")
            self.animate_solution()
    
    def animate_solution(self):
        if not self.path or not self.animation_running or self.pause_animation:
            return
            
        if self.current_index < len(self.path) - 1:
            self.current_index += 1
            self.update_display()
            
            delay = int(1000 / self.speed_var.get())
            self.root.after(delay, self.animate_solution)
        else:
            self.animation_running = False
            self.play_pause_button.config(text="Play")
    
    def next_step(self):
        if not self.path:
            return
            
        if self.current_index < len(self.path) - 1:
            self.current_index += 1
            self.update_display()
    
    def prev_step(self):
        if not self.path:
            return
            
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def reset_animation(self):
        if not self.path:
            return
            
        self.current_index = 0
        self.animation_running = False
        self.pause_animation = False
        self.play_pause_button.config(text="Play")
        self.update_display()
    
    def update_speed(self, value):
        pass

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver")
        
        self.setup_frame = tk.Frame(root)
        self.setup_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.create_setup_ui()
    
    def create_setup_ui(self):
        """Create puzzle setup UI"""
        title_label = tk.Label(self.setup_frame, text="8-Puzzle Solver", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        input_frame = tk.Frame(self.setup_frame)
        input_frame.pack(pady=10)
        
        initial_frame = tk.LabelFrame(input_frame, text="Trạng thái ban đầu")
        initial_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.initial_entries = []
        self.create_grid_entries(initial_frame, self.initial_entries)
        
        target_frame = tk.LabelFrame(input_frame, text="Trạng thái đích")
        target_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.target_entries = []
        self.create_grid_entries(target_frame, self.target_entries)
        
        default_initial = "265_87431"
        default_target = "12345678_"
        
        self.set_entries_values(self.initial_entries, default_initial)
        self.set_entries_values(self.target_entries, default_target)
        
        button_frame = tk.Frame(self.setup_frame)
        button_frame.pack(pady=20)
        
        solve_button = tk.Button(button_frame, text="Giải", command=self.solve_puzzle)
        solve_button.pack(side=tk.LEFT, padx=10)
        
        random_button = tk.Button(button_frame, text="Random", command=self.randomize_initial)
        random_button.pack(side=tk.LEFT, padx=10)
        
        clear_button = tk.Button(button_frame, text="Xóa", command=self.clear_entries)
        clear_button.pack(side=tk.LEFT, padx=10)
    
    def create_grid_entries(self, parent, entries_list):
        """Create a 3x3 grid of entry widgets"""
        for i in range(3):
            row_frame = tk.Frame(parent)
            row_frame.pack(pady=5)
            
            for j in range(3):
                entry = tk.Entry(row_frame, width=3, font=("Arial", 18), justify="center")
                entry.pack(side=tk.LEFT, padx=5)
                entries_list.append(entry)
    
    def set_entries_values(self, entries, values_str):
        """Set entry values from a string"""
        for i, entry in enumerate(entries):
            entry.delete(0, tk.END)
            entry.insert(0, values_str[i])
    
    def get_entries_values(self, entries):
        """Get values from entries as a string"""
        values = []
        for entry in entries:
            value = entry.get().strip()
            if value == "":
                value = "_"
            values.append(value[0])  
        return ''.join(values)
    
    def clear_entries(self):
        """Clear all entries"""
        for entry in self.initial_entries + self.target_entries:
            entry.delete(0, tk.END)
    
    def randomize_initial(self):
        """Randomize initial state (ensure it's solvable)"""
        import random
        
        target = self.get_entries_values(self.target_entries)
        if not self._is_valid_state(target):
            target = "12345678_"
            self.set_entries_values(self.target_entries, target)
        
        current = target
        for _ in range(100):  
            state = PuzzleState(current)
            next_states = state.get_next_states()
            if next_states:
                current = random.choice(next_states)
        
        self.set_entries_values(self.initial_entries, current)
    
    def _is_valid_state(self, state):
        """Check if state is valid (contains 0-8 with one blank)"""
        if len(state) != 9:
            return False
            
        if state.count('_') != 1:
            return False
            
        for i in range(1, 9):
            if str(i) not in state:
                return False
                
        return True
    
    def _is_solvable(self, initial, target):
        """Check if the puzzle is solvable from initial to target state"""
        
        initial = initial.replace('_', '0')
        target = target.replace('_', '0')
        
        initial_inversions = 0
        for i in range(9):
            if initial[i] == '0':
                continue
            for j in range(i + 1, 9):
                if initial[j] == '0':
                    continue
                if initial[i] > initial[j]:
                    initial_inversions += 1
        
        target_inversions = 0
        for i in range(9):
            if target[i] == '0':
                continue
            for j in range(i + 1, 9):
                if target[j] == '0':
                    continue
                if target[i] > target[j]:
                    target_inversions += 1
        
        return initial_inversions % 2 == target_inversions % 2
    
    def solve_puzzle(self):
        """Start solving the puzzle"""
        initial = self.get_entries_values(self.initial_entries)
        target = self.get_entries_values(self.target_entries)
        
        if not self._is_valid_state(initial) or not self._is_valid_state(target):
            tk.messagebox.showerror("Lỗi", "Trạng thái không hợp lệ. Mỗi trạng thái phải có các số từ 1-8 và một ô trống (_).")
            return
        
        if not self._is_solvable(initial, target):
            tk.messagebox.showerror("Lỗi", "Không thể giải bài toán với trạng thái ban đầu và đích này.")
            return
        
        self.setup_frame.pack_forget()
        
        self.solver = PuzzleSolver(self.root, initial, target)
        
        back_button = tk.Button(self.root, text="Trở lại", command=self.back_to_setup)
        back_button.pack(side=tk.BOTTOM, pady=10)
    
    def back_to_setup(self):
        """Return to puzzle setup screen"""
        for widget in self.root.winfo_children():
            if widget != self.setup_frame:
                widget.destroy()
        
        self.setup_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

class CSPVisualizer:
    def __init__(self, root, algorithm, initial_state, target_state):
        self.root = root
        self.algorithm = algorithm
        self.initial_state = initial_state
        self.target_state = target_state
        
        self.visual_window = tk.Toplevel(root)
        self.visual_window.title(f"CSP Visualization - {algorithm.__class__.__name__}")
        self.visual_window.geometry("900x700")
        
        state_frame = tk.Frame(self.visual_window)
        state_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.state_canvas = tk.Canvas(state_frame, width=300, height=300, bg="white")
        self.state_canvas.pack(pady=10)
        
        tree_frame = tk.Frame(self.visual_window)
        tree_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tree_canvas = tk.Frame(tree_frame, bg="white")
        self.tree_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tree_canvas)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        info_frame = tk.Frame(self.visual_window)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.status_label = tk.Label(info_frame, text="Initializing...", anchor="w", justify="left")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        control_frame = tk.Frame(self.visual_window)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.pause_button = tk.Button(control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.IntVar(value=5)
        self.speed_scale = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                    variable=self.speed_var, command=self.update_speed)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.renderer = PuzzleRenderer(self.state_canvas)
        
        self.paused = False
        self.delay = 200  # ms
        
        self.search_tree = nx.DiGraph()
        self.node_positions = {}
        self.node_colors = {}
        self.current_node = None
        
        self._add_node_to_tree(self.initial_state, None, 0)
        
    def _add_node_to_tree(self, state, parent_state, depth, status="normal"):
        """Thêm nút vào cây tìm kiếm"""
        node_id = str(uuid.uuid4())
        
        self.search_tree.add_node(node_id, state=state, depth=depth)
        
        x = depth
        y = len([n for n, d in self.search_tree.nodes(data=True) if d.get('depth') == depth])
        self.node_positions[node_id] = (x, -y)
        
        if status == "current":
            self.node_colors[node_id] = "yellow"
        elif status == "backtrack":
            self.node_colors[node_id] = "red"
        elif status == "promising":
            self.node_colors[node_id] = "green"
        elif status == "goal":
            self.node_colors[node_id] = "lime"
        else:
            self.node_colors[node_id] = "lightblue"
        
        if parent_state:
            parent_id = None
            for n, d in self.search_tree.nodes(data=True):
                if d.get('state') == parent_state:
                    parent_id = n
                    break
            
            if parent_id:
                self.search_tree.add_edge(parent_id, node_id)
        
        self.current_node = node_id
        return node_id
    
    def update_tree(self, state, parent_state, depth, status="normal"):
        """Cập nhật cây tìm kiếm với trạng thái mới"""
        node_id = self._add_node_to_tree(state, parent_state, depth, status)
        self.draw_tree()
        return node_id
    
    def draw_tree(self):
        """Vẽ cây tìm kiếm"""
        self.ax.clear()
        
        if len(self.search_tree) > 100:
            current_depth = self.search_tree.nodes[self.current_node]['depth']
            visible_nodes = [n for n, d in self.search_tree.nodes(data=True) 
                            if current_depth - 2 <= d['depth'] <= current_depth + 2]
            subgraph = self.search_tree.subgraph(visible_nodes)
        else:
            subgraph = self.search_tree
        
        # Vị trí các nút
        pos = {n: self.node_positions[n] for n in subgraph.nodes()}
        
        # Màu các nút
        node_colors = [self.node_colors.get(n, "lightblue") for n in subgraph.nodes()]
        
        # Vẽ đồ thị
        nx.draw(subgraph, pos, ax=self.ax, with_labels=False, 
                node_size=300, node_color=node_colors, 
                edge_color='gray', arrows=True)
        
        labels = {}
        for node in subgraph.nodes():
            state = self.search_tree.nodes[node]['state']
            label = state[:3] + '\n' + state[3:6] + '\n' + state[6:]
            labels[node] = label
        
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=7)
        
        self.canvas.draw()
    
    def update_status(self, message):
        """Cập nhật thông báo trạng thái"""
        self.status_label.config(text=message)
    
    def toggle_pause(self):
        """Tạm dừng/tiếp tục hiển thị"""
        self.paused = not self.paused
        self.pause_button.config(text="Continue" if self.paused else "Pause")
    
    def update_speed(self, value):
        """Cập nhật tốc độ hiển thị"""
        self.delay = int(1000 / int(value))
    
    def wait(self):
        """Đợi nếu đang tạm dừng"""
        while self.paused:
            self.visual_window.update()
            time.sleep(0.1)
        
        self.visual_window.update()
        time.sleep(self.delay / 1000)

class BacktrackingSearchVisual(BacktrackingSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.visualizer = None
        self.console_path = []
    
    def solve(self, renderer=None):
        """Giải bài toán với hiển thị trực quan"""
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        print("\n===== BẮT ĐẦU BACKTRACKING SEARCH =====")
        print(f"Trạng thái ban đầu: {self.initial_state}")
        print(f"Trạng thái đích: {self.target_state}")
        print("========================================\n")
        
        self.visualizer = renderer
        self.visited = set([self.initial_state])
        self.path = [self.initial_state]
        self.best_path = None
        self.best_path_length = float('inf')
        self.num_backtracks = 0
        self.console_path = []
        
        self.console_path.append(("start", self.initial_state, None))
        print("0. START:", self.initial_state)
        self._print_state_grid(self.initial_state)
        print()
        
        self._backtrack(self.initial_state, 0)
        
        print(f"\nSố lần quay lui: {self.num_backtracks}")
        if self.best_path:
            print(f"Độ dài đường đi: {len(self.best_path) - 1}")
        else:
            print("Không tìm thấy đường đi")
        print("===== KẾT THÚC BACKTRACKING SEARCH =====\n")
        
        return self.best_path
    
    def _backtrack(self, current_state, depth):
        """Hàm đệ quy thực hiện backtracking với hiển thị trực quan"""
        if self.visualizer:
            self.visualizer.renderer.draw_state(current_state, None, self.target_state)
            self.visualizer.update_tree(current_state, self.path[-2] if len(self.path) > 1 else None, depth, "current")
            self.visualizer.update_status(f"Depth: {depth}, Backtracks: {self.num_backtracks}")
            self.visualizer.wait()
        
        if current_state == self.target_state:
            if len(self.path) < self.best_path_length:
                self.best_path = self.path.copy()
                self.best_path_length = len(self.path)
                
                step_num = len(self.console_path)
                self.console_path.append(("goal", current_state, self.path[-2] if len(self.path) > 1 else None))
                print(f"{step_num}. GOAL: {current_state} (from {self.path[-2] if len(self.path) > 1 else None})")
                self._print_state_grid(current_state)
                print()
                
                if self.visualizer:
                    self.visualizer.update_tree(current_state, self.path[-2] if len(self.path) > 1 else None, depth, "goal")
                    self.visualizer.update_status(f"Found solution with {len(self.path)} steps!")
                    self.visualizer.wait()
            return True
        
        # Nếu đạt đến độ sâu tối đa
        if depth >= self.max_depth:
            return False
        
        # Lấy các trạng thái kế tiếp
        state_obj = PuzzleState(current_state)
        next_states = state_obj.get_next_states()
        
        # Sắp xếp các trạng thái kế tiếp 
        next_states_with_heuristic = [(self._calculate_heuristic(state), state) for state in next_states]
        next_states_with_heuristic.sort()
        
        found_solution = False
        
        # Thử từng trạng thái kế tiếp
        for _, next_state in next_states_with_heuristic:
            if next_state in self.visited:
                continue
            
            self.path.append(next_state)
            self.visited.add(next_state)
            
            step_num = len(self.console_path)
            self.console_path.append(("explore", next_state, current_state))
            print(f"{step_num}. EXPLORE: {next_state} (from {current_state})")
            self._print_state_grid(next_state)
            print()
            
            if self.visualizer:
                self.visualizer.update_tree(next_state, current_state, depth + 1, "promising")
                self.visualizer.update_status(f"Exploring depth {depth + 1}")
                self.visualizer.wait()
            
            # Đệ quy
            if self._backtrack(next_state, depth + 1):
                found_solution = True
                break
            
            # Quay lui
            self.path.pop()
            self.num_backtracks += 1
            
            # Thêm vào đường đi 
            step_num = len(self.console_path)
            self.console_path.append(("backtrack", next_state, current_state))
            print(f"{step_num}. BACKTRACK: {next_state} (from {current_state})")
            self._print_state_grid(next_state)
            print()
            
            if self.visualizer:
                self.visualizer.update_tree(next_state, current_state, depth + 1, "backtrack")
                self.visualizer.update_status(f"Backtracking: {self.num_backtracks}")
                self.visualizer.wait()
        
        return found_solution
    
    def _print_state_grid(self, state):
        """In trạng thái dưới dạng lưới 3x3"""
        for row in range(3):
            print("   " + " ".join(state[row*3:row*3+3]).replace("_", " "))

class ForwardCheckingSearchVisual(ForwardCheckingSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.visualizer = None
        self.console_path = []
    
    def solve(self, renderer=None):
        """Giải bài toán với hiển thị trực quan"""
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        print("\n===== BẮT ĐẦU FORWARD CHECKING SEARCH =====")
        print(f"Trạng thái ban đầu: {self.initial_state}")
        print(f"Trạng thái đích: {self.target_state}")
        print("============================================\n")
        
        self.visualizer = renderer
        self.visited = set([self.initial_state])
        self.path = [self.initial_state]
        self.best_path = None
        self.best_path_length = float('inf')
        self.num_backtracks = 0
        self.num_forward_checks = 0
        self.console_path = []
        
        # Thêm trạng thái ban đầu 
        self.console_path.append(("start", self.initial_state, None))
        print("0. START:", self.initial_state)
        self._print_state_grid(self.initial_state)
        print()
        
        # Bắt đầu backtracking với forward checking
        self._backtrack(self.initial_state, 0)
        
        print(f"\nSố lần quay lui: {self.num_backtracks}")
        print(f"Số lần kiểm tra forward: {self.num_forward_checks}")
        if self.best_path:
            print(f"Độ dài đường đi: {len(self.best_path) - 1}")
        else:
            print("Không tìm thấy đường đi")
        print("===== KẾT THÚC FORWARD CHECKING SEARCH =====\n")
        
        return self.best_path
    
    def _forward_check(self, state, lookahead_depth):
        """Kiểm tra forward để xác định xem có đường đi khả thi không"""
        self.num_forward_checks += 1
        
        step_num = len(self.console_path)
        self.console_path.append(("forward_check", state, None))
        print(f"{step_num}. FORWARD CHECK: {state} (depth {lookahead_depth})")
        self._print_state_grid(state)
        print()
        
        if self.visualizer:
            self.visualizer.update_status(f"Forward checking at depth {lookahead_depth}")
            self.visualizer.wait()
        
        if state == self.target_state:
            return True
        
        if lookahead_depth >= self.max_lookahead:
            return True
        
        state_obj = PuzzleState(state)
        next_states = state_obj.get_next_states()
        
        next_states_with_heuristic = [(self._calculate_heuristic(next_state), next_state) 
                                     for next_state in next_states 
                                     if next_state not in self.visited]
        next_states_with_heuristic.sort()
        
        if not next_states_with_heuristic:
            return False
        
        for _, next_state in next_states_with_heuristic[:3]:  # Chỉ xem xét 3 trạng thái tốt nhất
            if self._forward_check(next_state, lookahead_depth + 1):
                return True
        
        return False
    
    def _backtrack(self, current_state, depth):
        """Hàm đệ quy thực hiện backtracking với forward checking và hiển thị trực quan"""
        if self.visualizer:
            self.visualizer.renderer.draw_state(current_state, None, self.target_state)
            self.visualizer.update_tree(current_state, self.path[-2] if len(self.path) > 1 else None, depth, "current")
            self.visualizer.update_status(f"Depth: {depth}, Backtracks: {self.num_backtracks}, Forward Checks: {self.num_forward_checks}")
            self.visualizer.wait()
        
        if current_state == self.target_state:
            if len(self.path) < self.best_path_length:
                self.best_path = self.path.copy()
                self.best_path_length = len(self.path)
                
                step_num = len(self.console_path)
                self.console_path.append(("goal", current_state, self.path[-2] if len(self.path) > 1 else None))
                print(f"{step_num}. GOAL: {current_state} (from {self.path[-2] if len(self.path) > 1 else None})")
                self._print_state_grid(current_state)
                print()
                
                if self.visualizer:
                    self.visualizer.update_tree(current_state, self.path[-2] if len(self.path) > 1 else None, depth, "goal")
                    self.visualizer.update_status(f"Found solution with {len(self.path)} steps!")
                    self.visualizer.wait()
            return True
        
        if depth >= self.max_depth:
            return False
        
        state_obj = PuzzleState(current_state)
        next_states = state_obj.get_next_states()
        
        next_states_with_heuristic = [(self._calculate_heuristic(state), state) for state in next_states]
        next_states_with_heuristic.sort()
        
        found_solution = False
        
        for _, next_state in next_states_with_heuristic:
            if next_state in self.visited:
                continue
            
            if not self._is_promising(next_state, depth + 1):
                step_num = len(self.console_path)
                self.console_path.append(("prune", next_state, current_state))
                print(f"{step_num}. PRUNE: {next_state} (from {current_state})")
                self._print_state_grid(next_state)
                print()
                
                if self.visualizer:
                    self.visualizer.update_tree(next_state, current_state, depth + 1, "backtrack")
                    self.visualizer.update_status(f"Pruning: not promising")
                    self.visualizer.wait()
                continue
            
            self.path.append(next_state)
            self.visited.add(next_state)
            
            step_num = len(self.console_path)
            self.console_path.append(("explore", next_state, current_state))
            print(f"{step_num}. EXPLORE: {next_state} (from {current_state})")
            self._print_state_grid(next_state)
            print()
            
            if self.visualizer:
                self.visualizer.update_tree(next_state, current_state, depth + 1, "promising")
                self.visualizer.update_status(f"Exploring depth {depth + 1}")
                self.visualizer.wait()
            
            if self._backtrack(next_state, depth + 1):
                found_solution = True
                break
            
            self.path.pop()
            self.num_backtracks += 1
            
            step_num = len(self.console_path)
            self.console_path.append(("backtrack", next_state, current_state))
            print(f"{step_num}. BACKTRACK: {next_state} (from {current_state})")
            self._print_state_grid(next_state)
            print()
            
            if self.visualizer:
                self.visualizer.update_tree(next_state, current_state, depth + 1, "backtrack")
                self.visualizer.update_status(f"Backtracking: {self.num_backtracks}")
                self.visualizer.wait()
        
        return found_solution
    
    def _print_state_grid(self, state):
        """In trạng thái dưới dạng lưới 3x3"""
        for row in range(3):
            print("   " + " ".join(state[row*3:row*3+3]).replace("_", " "))

class MinConflictsSearchVisual(MinConflictsSearch):
    def __init__(self, initial_state, target_state):
        super().__init__(initial_state, target_state)
        self.visualizer = None
        self.console_path = []
    
    def solve(self, renderer=None):
        """Giải bài toán với hiển thị trực quan"""
        import random
        import time
        
        if not self._is_solvable(self.initial_state, self.target_state):
            print("Bài toán không thể giải được!")
            return None
        
        print("\n===== BẮT ĐẦU MIN CONFLICTS SEARCH =====")
        print(f"Trạng thái ban đầu: {self.initial_state}")
        print(f"Trạng thái đích: {self.target_state}")
        print("========================================\n")
        
        self.visualizer = renderer
        start_time = time.time()
        
        current = self.initial_state
        best_state = current
        best_conflicts = self._calculate_conflicts(current)
        
        path = [current]
        parent = {current: None}
        
        self.num_iterations = 0
        self.num_restarts = 0
        plateau_counter = 0
        self.console_path = []
        
        step_num = len(self.console_path)
        self.console_path.append(("start", current, None, best_conflicts))
        print(f"{step_num}. START: {current} (conflicts: {best_conflicts})")
        self._print_state_grid(current)
        print()
        
        self.conflicts_history.append(best_conflicts)
        
        if self.visualizer:
            self.visualizer.renderer.draw_state(current, None, self.target_state)
            self.visualizer.update_tree(current, None, 0, "current")
            self.visualizer.update_status(f"Initial state with {best_conflicts} conflicts")
            self.visualizer.wait()
        
        while self.num_iterations < self.max_iterations:
            self.num_iterations += 1
            
            if self.visualizer:
                self.visualizer.update_status(f"Iterations: {self.num_iterations}, Restarts: {self.num_restarts}, Conflicts: {self._calculate_conflicts(current)}")
                self.visualizer.wait()
            
            if current == self.target_state:
                step_num = len(self.console_path)
                self.console_path.append(("goal", current, path[-2] if len(path) > 1 else None, 0))
                print(f"{step_num}. GOAL: {current} (from {path[-2] if len(path) > 1 else None}, conflicts: 0)")
                self._print_state_grid(current)
                print()
                
                if self.visualizer:
                    self.visualizer.update_tree(current, path[-2] if len(path) > 1 else None, self.num_iterations, "goal")
                    self.visualizer.update_status(f"Found solution after {self.num_iterations} iterations, {self.num_restarts} restarts")
                    self.visualizer.wait()
                
                print(f"\nTìm thấy giải pháp sau {self.num_iterations} lần lặp, {self.num_restarts} lần khởi động lại, {time.time() - start_time:.3f}s")
                print("===== KẾT THÚC MIN CONFLICTS SEARCH =====\n")
                
                return self._reconstruct_path(current, parent)
            
            if self.num_iterations % 1000 == 0:
                print(f"Đã thực hiện {self.num_iterations} bước, {self.num_restarts} lần khởi động lại, xung đột hiện tại: {self._calculate_conflicts(current)}")
            
            possible_moves = self._get_possible_moves(current)
            
            next_state = self._min_conflict_move(current, possible_moves)
            next_conflicts = self._calculate_conflicts(next_state)
            
            step_num = len(self.console_path)
            self.console_path.append(("move", next_state, current, next_conflicts))
            print(f"{step_num}. MOVE: {next_state} (from {current}, conflicts: {next_conflicts})")
            self._print_state_grid(next_state)
            print()
            
            if next_state not in parent:
                parent[next_state] = current
            
            if self.visualizer:
                self.visualizer.renderer.draw_state(next_state, current, self.target_state)
                status = "promising" if next_conflicts < self._calculate_conflicts(current) else "current"
                self.visualizer.update_tree(next_state, current, self.num_iterations, status)
                self.visualizer.wait()
            
            current = next_state
            path.append(current)
            
            current_conflicts = self._calculate_conflicts(current)
            self.conflicts_history.append(current_conflicts)
            
            if current_conflicts < best_conflicts:
                best_state = current
                best_conflicts = current_conflicts
                plateau_counter = 0
                
                step_num = len(self.console_path)
                self.console_path.append(("improve", current, path[-2] if len(path) > 1 else None, best_conflicts))
                print(f"{step_num}. IMPROVE: {current} (from {path[-2] if len(path) > 1 else None}, conflicts: {best_conflicts})")
                self._print_state_grid(current)
                print()
                
                if self.visualizer:
                    self.visualizer.update_tree(current, path[-2] if len(path) > 1 else None, self.num_iterations, "promising")
                    self.visualizer.update_status(f"Improved to {best_conflicts} conflicts")
                    self.visualizer.wait()
            else:
                plateau_counter += 1
            
            if plateau_counter >= self.restart_threshold:
                self.num_restarts += 1
                
                step_num = len(self.console_path)
                self.console_path.append(("restart", self.initial_state, current, self._calculate_conflicts(self.initial_state)))
                print(f"{step_num}. RESTART: {self.initial_state} (from {current}, conflicts: {self._calculate_conflicts(self.initial_state)})")
                self._print_state_grid(self.initial_state)
                print()
                
                print(f"Khởi động lại lần {self.num_restarts} sau {plateau_counter} bước không cải thiện")
                
                if self.visualizer:
                    self.visualizer.update_status(f"Restarting search ({self.num_restarts})")
                    self.visualizer.wait()
                
                if best_state != self.initial_state and best_state not in path:
                    parent[best_state] = path[-1]
                
                current = self.initial_state
                for _ in range(random.randint(5, 20)):
                    moves = self._get_possible_moves(current)
                    if moves:
                        next_move = random.choice(moves)
                        if next_move not in parent:
                            parent[next_move] = current
                        current = next_move
                
                path.append(current)
                
                if self.visualizer:
                    self.visualizer.renderer.draw_state(current, None, self.target_state)
                    self.visualizer.update_tree(current, None, self.num_iterations, "current")
                    self.visualizer.update_status(f"Restarted with {self._calculate_conflicts(current)} conflicts")
                    self.visualizer.wait()
                
                plateau_counter = 0
        
        print(f"\nKhông tìm thấy giải pháp chính xác sau {self.max_iterations} lần lặp, {self.num_restarts} lần khởi động lại")
        print(f"Trả về trạng thái tốt nhất với {best_conflicts} xung đột")
        print("===== KẾT THÚC MIN CONFLICTS SEARCH =====\n")
        
        if best_state != self.initial_state:
            return self._reconstruct_path(best_state, parent)
        
        return None
    
    def _print_state_grid(self, state):
        """In trạng thái dưới dạng lưới 3x3"""
        for row in range(3):
            print("   " + " ".join(state[row*3:row*3+3]).replace("_", " "))

class AlgorithmFactoryVisual:
    @staticmethod
    def create_algorithm(algorithm_name, initial_state, target_state):
        algorithms = {
            "Backtracking": BacktrackingSearchVisual,
            "ForwardChecking": ForwardCheckingSearchVisual,
            "Forward Checking": ForwardCheckingSearchVisual,
            "MinConflicts": MinConflictsSearchVisual,
            "Min Conflicts": MinConflictsSearchVisual,
        }
        return algorithms.get(algorithm_name, BacktrackingSearchVisual)(initial_state, target_state)

class PuzzleSolverVisual(PuzzleSolver):
    def __init__(self, root, initial, target):
        self.root = root
        self.initial = initial
        self.target = target
        self.path = None
        self.current_index = 0
        self.animation_speed = 1000
        self.animation_running = False
        self.pause_animation = False
        self.visualization_active = False
        self.visualizer = None
        
        self.create_ui()
        
        self.renderer = PuzzleRenderer(self.canvas)
        
        self.calculate_solution()
    
    def create_ui(self):
        super().create_ui()
        
        visualize_button = tk.Button(self.status_frame, text="Visualize CSP", command=self.show_visualization)
        visualize_button.pack(side=tk.RIGHT, padx=10)
    
    def show_visualization(self):
        """Hiển thị trực quan cho thuật toán CSP"""
        if self.visualization_active:
            messagebox.showinfo("Thông báo", "Hiển thị trực quan đang hoạt động!")
            return
        
        algorithm_name = self.algorithm_var.get()
        
        if algorithm_name not in ["Backtracking", "ForwardChecking", "Forward Checking", "MinConflicts", "Min Conflicts"]:
            messagebox.showinfo("Thông báo", "Chỉ hỗ trợ hiển thị trực quan cho các thuật toán CSP!")
            return
        
        self.visualization_active = True
        
        self.algorithm = AlgorithmFactoryVisual.create_algorithm(algorithm_name, self.initial, self.target)
        
        self.visualizer = CSPVisualizer(self.root, self.algorithm, self.initial, self.target)
        
        threading.Thread(target=self.run_visualization, daemon=True).start()
    
    def run_visualization(self):
        """Chạy thuật toán với hiển thị trực quan trong luồng riêng biệt"""
        try:
            self.status_label.config(text="Trạng thái: Đang hiển thị trực quan...")
            self.root.update()
        
            print("\n===== BẮT ĐẦU HIỂN THỊ TRỰC QUAN =====")
            print(f"Thuật toán: {self.algorithm_var.get()}")
            print(f"Trạng thái ban đầu: {self.initial}")
            print(f"Trạng thái đích: {self.target}")
            print("=======================================\n")
        
            start_time = time.time()
            self.path = self.algorithm.solve(self.visualizer)
            end_time = time.time()
        
            print(f"\n===== KẾT QUẢ HIỂN THỊ TRỰC QUAN =====")
            print(f"Thời gian thực thi: {end_time - start_time:.3f}s")
            if self.path:
                print(f"Độ dài đường đi: {len(self.path) - 1}")
            else:
                print("Không tìm thấy đường đi")
            print("=======================================\n")
        
            if self.path:
                self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s")
                self.step_label.config(text=f"Bước: 0/{len(self.path) - 1}")
            else:
                self.status_label.config(text="Trạng thái: Không tìm thấy lời giải!")
                self.step_label.config(text="Bước: 0/0")
        
            self.current_index = 0
            self.animation_running = False
            self.pause_animation = False
            self.play_pause_button.config(text="Play")
        
            self.update_display()
        
        except Exception as e:
            print(f"Lỗi khi hiển thị trực quan: {e}")
        finally:
            self.visualization_active = False
    
    def calculate_solution(self):
        """Tính toán lời giải sử dụng thuật toán đã chọn"""
        if self.visualization_active:
            messagebox.showinfo("Thông báo", "Hiển thị trực quan đang hoạt động! Vui lòng đợi hoặc đóng cửa sổ hiển thị trực quan.")
            return
        
        self.status_label.config(text="Trạng thái: Đang tính toán...")
        self.root.update()
        
        algorithm_name = self.algorithm_var.get()
        self.algorithm = AlgorithmFactory.create_algorithm(algorithm_name, self.initial, self.target)
        
        start_time = time.time()
        self.path = self.algorithm.solve()
        end_time = time.time()
        
        if self.path:
            if algorithm_name in ["Backtracking", "ForwardChecking", "Forward Checking"]:
                num_backtracks = getattr(self.algorithm, 'num_backtracks', 0)
                self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s. Số lần quay lui: {num_backtracks}")
                
                if algorithm_name in ["ForwardChecking", "Forward Checking"]:
                    num_forward_checks = getattr(self.algorithm, 'num_forward_checks', 0)
                    self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s. Quay lui: {num_backtracks}, Forward checks: {num_forward_checks}")
            
            elif algorithm_name in ["MinConflicts", "Min Conflicts"]:
                num_iterations = getattr(self.algorithm, 'num_iterations', 0)
                num_restarts = getattr(self.algorithm, 'num_restarts', 0)
                self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s. Số lần lặp: {num_iterations}, Khởi động lại: {num_restarts}")
            else:
                self.status_label.config(text=f"Trạng thái: Đã giải. Thời gian: {end_time - start_time:.3f}s")
            
            self.step_label.config(text=f"Bước: 0/{len(self.path) - 1}")
        else:
            self.status_label.config(text="Trạng thái: Không tìm thấy lời giải!")
            self.step_label.config(text="Bước: 0/0")
        
        self.current_index = 0
        self.animation_running = False
        self.pause_animation = False
        self.play_pause_button.config(text="Play")
        
        self.update_display()
    
    def export_steps_to_console(self):
        """Export all steps from the solution path to the console"""
        if not self.path:
            print("No solution path available to export.")
            return
        
        print("\n===== SOLUTION PATH =====")
        print(f"Algorithm: {self.algorithm_var.get()}")
        print(f"Total steps: {len(self.path) - 1}")
        
        algorithm_name = self.algorithm_var.get()
        if algorithm_name in ["Backtracking", "ForwardChecking", "Forward Checking"]:
            num_backtracks = getattr(self.algorithm, 'num_backtracks', 0)
            print(f"Number of backtracks: {num_backtracks}")
            
            if algorithm_name in ["ForwardChecking", "Forward Checking"]:
                num_forward_checks = getattr(self.algorithm, 'num_forward_checks', 0)
                print(f"Number of forward checks: {num_forward_checks}")
        
        elif algorithm_name in ["MinConflicts", "Min Conflicts"]:
            num_iterations = getattr(self.algorithm, 'num_iterations', 0)
            num_restarts = getattr(self.algorithm, 'num_restarts', 0)
            conflicts_history = getattr(self.algorithm, 'conflicts_history', [])
            print(f"Number of iterations: {num_iterations}")
            print(f"Number of restarts: {num_restarts}")
            if conflicts_history:
                print(f"Initial conflicts: {conflicts_history[0]}")
                print(f"Final conflicts: {conflicts_history[-1]}")
        
        print("=======================\n")
        
        for i, state in enumerate(self.path):
            print(f"Step {i}:")
            for row in range(3):
                print(" ".join(state[row*3:row*3+3]).replace("_", " "))
            
            if algorithm_name in ["MinConflicts", "Min Conflicts"] and i < len(getattr(self.algorithm, 'conflicts_history', [])):
                conflicts = getattr(self.algorithm, 'conflicts_history', [])[i]
                print(f"Conflicts: {conflicts}")
            
            print()


class ApplicationVisual(Application):
    def solve_puzzle(self):
        """Start solving the puzzle"""
        initial = self.get_entries_values(self.initial_entries)
        target = self.get_entries_values(self.target_entries)
        
        if not self._is_valid_state(initial) or not self._is_valid_state(target):
            messagebox.showerror("Lỗi", "Trạng thái không hợp lệ. Mỗi trạng thái phải có các số từ 1-8 và một ô trống (_).")
            return
        
        if not self._is_solvable(initial, target):
            messagebox.showerror("Lỗi", "Không thể giải bài toán với trạng thái ban đầu và đích này.")
            return
        
        self.setup_frame.pack_forget()
        
        self.solver = PuzzleSolverVisual(self.root, initial, target)
        
        back_button = tk.Button(self.root, text="Trở lại", command=self.back_to_setup)
        back_button.pack(side=tk.BOTTOM, pady=10)

class AlgorithmComparison:
    def __init__(self, initial_state, target_state, algorithm_group):
        self.initial_state = initial_state
        self.target_state = target_state
        self.algorithm_group = algorithm_group
        self.results = {}
        self.algorithms = self._get_algorithms_in_group()
    
    def _get_algorithms_in_group(self):
        algorithm_groups = {
            "Informed Search": ["A*", "IDA*", "Greedy Search"],
            "Uninformed Search": ["BFS", "DFS", "Uniform Cost Search", "IDDFS"],
            "Local Search": ["Simple Hill Climbing", "Steepest Hill Climbing", 
                           "Beam Steepest Hill Climbing", "Stochastic Hill Climbing", 
                           "Simulated Annealing", "Genetic Algorithm"],
            "Complex Environments": ["Belief State", "AND-OR Search", "Partially Observable"],
            "CSPs": ["Backtracking", "Forward Checking", "Min Conflicts"],
            "Reinforcement Learning": ["Q-Learning"]
        }
        
        return algorithm_groups.get(self.algorithm_group, [])
    
    def run_comparison(self):
        self.results = {}
        
        for algorithm_name in self.algorithms:
            print(f"Đang chạy thuật toán: {algorithm_name}")
            algorithm = AlgorithmFactory.create_algorithm(algorithm_name, self.initial_state, self.target_state)
            
            start_time = time.time()
            path = algorithm.solve()
            end_time = time.time()
            
            execution_time = end_time - start_time
            path_length = len(path) - 1 if path else None
            
            additional_info = {}
            
            if algorithm_name in ["Backtracking", "Forward Checking"]:
                additional_info["backtracks"] = getattr(algorithm, 'num_backtracks', 0)
                if algorithm_name == "Forward Checking":
                    additional_info["forward_checks"] = getattr(algorithm, 'num_forward_checks', 0)
            
            elif algorithm_name in ["Min Conflicts"]:
                additional_info["iterations"] = getattr(algorithm, 'num_iterations', 0)
                additional_info["restarts"] = getattr(algorithm, 'num_restarts', 0)
            
            self.results[algorithm_name] = {
                "path": path,
                "execution_time": execution_time,
                "path_length": path_length,
                "additional_info": additional_info
            }
        
        return self.results
    
    def get_summary(self):
        if not self.results:
            return "Chưa có kết quả so sánh. Vui lòng chạy so sánh trước."
        
        summary = f"=== SO SÁNH CÁC THUẬT TOÁN TRONG NHÓM {self.algorithm_group} ===\n\n"
        
        headers = ["Thuật toán", "Thời gian (s)", "Số bước", "Thông tin thêm"]
        rows = []
        
        for algorithm_name, result in self.results.items():
            time_str = f"{result['execution_time']:.3f}" if result['execution_time'] is not None else "N/A"
            path_length = str(result['path_length']) if result['path_length'] is not None else "Không tìm thấy"
            
            additional_info = []
            for key, value in result['additional_info'].items():
                additional_info.append(f"{key}: {value}")
            
            additional_info_str = ", ".join(additional_info) if additional_info else ""
            
            rows.append([algorithm_name, time_str, path_length, additional_info_str])
        
        col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
        
        header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        summary += header_str + "\n"
        summary += "-" * len(header_str) + "\n"
        
        for row in rows:
            summary += " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)) + "\n"
        
        return summary

if __name__ == "__main__":
    root = tk.Tk()
    app = ApplicationVisual(root)
    
    root.mainloop()