#thu vien gui
import tkinter as tk
from tkinter import scrolledtext
from tkinter import simpledialog
from tkinter import messagebox
from tkinter import ttk
#thu vien bang 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
#thu vien do thi
import networkx as nx
#thu vien mang, ma tran
import numpy as np
#thu vien de chay thuat toan dijkstra
import heapq
from PIL import Image, ImageTk

'''
8
0 4 3 0 0 0 0 0
4 0 2 5 0 0 0 0
3 2 0 3 6 0 0 0
0 5 3 0 1 5 0 0
0 0 6 1 0 0 5 0 
0 0 0 5 0 0 2 7 
0 0 0 0 5 2 0 4
0 0 0 0 0 7 4 0

6
0 0 0 0 0 0
7 0 0 2 0 0
0 9 0 0 0 2
2 0 0 0 0 6
0 2 0 2 0 0
0 1 0 0 3 0

'''


# check ma tran ke co duong cheo chinh bnag 0 hay khong
def check_diagonal(matrix):
    for i in range(len(matrix)):
        if matrix[i][i] != 0:
            return False
    return True

# check trọng số âm
def check_negative_weights(matrix):
    for row in matrix:
        if any(weight < 0 for weight in row):
            return False
    return True

#check ma trận có hướng
def check_directed(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] != matrix[j][i]:
                return True
    return False





# thuật toán dijkstra
def dijkstra_algorithm(graph, start_vertex, target_vertex):
    #tao 1 tu dien voi cac dinh la vo cung
    distances = {vertex: float('infinity') for vertex in graph.nodes}
    #dinh bat dau se = 0
    distances[start_vertex] = 0
    #tu dien khoang cac cac dinh toi dinh ban dau gia trị la none
    predecessors = {vertex: None for vertex in graph.nodes}
    # tao hang doi uu tien luu tru cac dinh cung voi khoang cach hien tai tu dinh bat dau den di do
    priority_queue = [(0, start_vertex)]
    #lam cho den khi hang doi khong con gia tri
    while priority_queue:
        #goi heappop de lay gia tri dau tien trong hang doi ra su lu
        #gan current_distance cho gia tri khoang cach trong hang doi
        #gan current_verter cho gia tri dinh trong hang doi
        current_distance, current_vertex = heapq.heappop(priority_queue)
        #neu khoang cach tu dinh hien tai den dinh bat dau lon hon gia tri da biet thi tiep tuc
        if current_distance > distances[current_vertex]:
            continue
        #duyet qua tat cac cac dinh ke voi dinh hien tai
        for neighbor, edge_data in graph[current_vertex].items():
            #tinhh khoang cah tu dinh ban dau den di ke bang cach cong cho trong so cua dinh ke
            distance = distances[current_vertex] + edge_data['weight']
            #neu khoang cach do be hon khoang cac da biet cua canh thi gan cho khoang cach do bàn distance
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                #gan cho dinh nho nhat la dinh ke 
                predecessors[neighbor] = current_vertex
                #day vao khoang cach ngan nhat hien tai va dinh ngan nhat hien tai
                heapq.heappush(priority_queue, (distance, neighbor))

    #tao mot mang de luu cac dinh cua duong di ngan nhat 
    shortest_path = []
    #
    current_vertex = target_vertex
    #neu current_vertex khong none
    while current_vertex is not None:
        #thi insert current_vertex vao shortest_path
        shortest_path.insert(0, current_vertex)
        #cap nhat lai current_vertex
        current_vertex = predecessors[current_vertex]
    #return mang dinh cua do thi ngan nhat, khoang cach tu dinh nguon de dinh cuoi
    return shortest_path, distances[target_vertex]


def draw_shortest_path(G, pos, ax, canvas, shortest_path):
    # Vẽ xen kẽ đỉnh và cạnh
    for i in range(len(shortest_path) - 1):
        # Vẽ đỉnh
        nx.draw_networkx_nodes(G, pos, nodelist=[shortest_path[i]], node_size=200, node_color='red', ax=ax)
        canvas.draw()
        canvas.get_tk_widget().update()
        canvas.get_tk_widget().after(1000)  # Độ trễ 2 giây (2000 milliseconds)

        # Vẽ cạnh
        nx.draw_networkx_edges(G, pos, edgelist=[(shortest_path[i], shortest_path[i + 1])], connectionstyle="arc3,rad=0.2", edge_color='red', ax=ax)
        canvas.draw()
        canvas.get_tk_widget().update()
        canvas.get_tk_widget().after(1000)  # Độ trễ 2 giây (2000 milliseconds)

    # Vẽ đỉnh cuối cùng
    nx.draw_networkx_nodes(G, pos, nodelist=[shortest_path[-1]], node_size=200, node_color='red', ax=ax)
    canvas.draw()
    canvas.get_tk_widget().update()
    canvas.get_tk_widget().after(2000)  # Độ trễ 2 giây (2000 milliseconds)




# gui dijkstra
def dijkstra():
    try:
        num_vertices = int(entry_vertices.get())
        matrix_values = scrolledtext_matrix.get("1.0", tk.END).strip().split('\n')

        adjacency_matrix = []
        for row in matrix_values:
            values = list(map(int, row.split()))
            adjacency_matrix.append(values)

        if len(adjacency_matrix) != num_vertices or any(len(row) != num_vertices for row in adjacency_matrix):
            raise ValueError("Đâu vao không hợp lệ. Hãy đảm bảo ma trận là hình vuông và có số đỉnh chính xác.")
        
        if not check_diagonal(adjacency_matrix):
            raise ValueError("Đường chéo chính của ma trận phải được điền bằng số 0.")
        
        if not check_negative_weights(adjacency_matrix):
            raise ValueError("Trọng số trong ma trận không thể âm.")            


        
        if check_directed(adjacency_matrix):
            G = nx.from_numpy_array(np.array(adjacency_matrix), create_using=nx.DiGraph())

        else:
            G = nx.from_numpy_array(np.array(adjacency_matrix), create_using=nx.Graph())


        source_node = simpledialog.askinteger("Đỉnh bắt đầu", "Nhập đỉnh bắt đầu (0 đến {}):".format(num_vertices - 1),
                                                       minvalue=0, maxvalue=num_vertices - 1)

        target_node = simpledialog.askinteger("Đỉnh kết thúc", "Nhập đỉnh kết thúc(0 đến {}):".format(num_vertices - 1),
                                                       minvalue=0, maxvalue=num_vertices - 1)

        if source_node is None or target_node is None:
            return
        
        #lấy được đường đi ngắn nhất, tổng độ dài
        shortest_path, shortest_path_length = dijkstra_algorithm(G, source_node, target_node)
        graph_window = tk.Toplevel(root)
        graph_window.title("Đồ thị và đường đi ngắn nhất")

        figure, ax = plt.subplots(figsize=(10, 10))
        canvas = FigureCanvasTkAgg(figure, master=graph_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()


        #laays ddinh
        pos = nx.spring_layout(G, scale=5.0)
        #lay trong so
        labels = nx.get_edge_attributes(G, 'weight')
        # set title la dijkstra
        ax.set_title("Thuật toán Dijkstra")
        # xuat duong di ngan nhat
        # Vẽ các đỉnh
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue')
        # Vẽ các cạnh với kiểu kết nối là "arc3"
        nx.draw_networkx_edges(G, pos, edgelist=G.edges() ,connectionstyle="arc3,rad=0.2", edge_color='black')
        # thêm số cho đỉnh
        nx.draw_networkx_labels(G, pos, font_color='black', font_weight='bold')
        # them trong so
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)


        draw_shortest_path(G, pos, ax, canvas, shortest_path)
        plt.text(0.5, -0.1, f"Đường đi ngắn nhất từ đỉnh {source_node} tới đỉnh {target_node}: {shortest_path} = {shortest_path_length}", fontsize=10, color='red', ha='center', va='center', transform=plt.gca().transAxes)


        canvas.draw()
            

    except ValueError as e:
        tk.messagebox.showerror("Lỗi", str(e))


# thuật toán primm
def prim_algorithm(graph, start_vertex):
    #khoi tao mot do thi de chua cay
    min_spanning_tree = nx.Graph()
    #dung tap hop de theo doi cac dinh da di qua
    visited = set([start_vertex])

    # lam cho đến khi hết đỉnh
    while len(visited) < len(graph.nodes):
        possible_edges = []
        #duyet qua tat cac cac dinh da di qua
        for node in visited:
            #trả về tất cả các cạnh kết nối đỉnh node, bao gồm dữ liệu cạnh (nếu có).
            #possible_edges.extend(...) mở rộng danh sách 
            #possible_edges với các cạnh mới thu được từ đỉnh node.
            possible_edges.extend(graph.edges(node, data=True))
        #Loại bỏ các cạnh nối đến các đỉnh đã thăm:
        possible_edges = [edge for edge in possible_edges if edge[1] not in visited]
        #Tìm cạnh có trọng số nhỏ nhất trong possible_edges:
        min_edge = min(possible_edges, key=lambda x: x[2]['weight'])    
        #Thêm cạnh có trọng số nhỏ nhất vào cây bao trùm nhỏ nhất 
        min_spanning_tree.add_edge(min_edge[0], min_edge[1], weight=min_edge[2]['weight'])
        #Đánh dấu đỉnh đích của cạnh là đã thăm:
        visited.add(min_edge[1])

    return min_spanning_tree

def draw_tree_with_delay(min_spanning_tree, pos, ax, canvas, start_vertex):
    edges = list(min_spanning_tree.edges(data=True))
    nodes = list(min_spanning_tree.nodes())
    endpoints_of_last_edge = list(min_spanning_tree.edges())[-1]
    last_vertex = endpoints_of_last_edge[1]

    # Function to draw a single edge
    def draw_edge(edge):
        nx.draw_networkx_edges(min_spanning_tree, pos, edgelist=[(edge[0], edge[1])], edge_color='red', ax=ax)
        canvas.draw()
        canvas.get_tk_widget().update()
        canvas.get_tk_widget().after(1000)  # Delay: 2 seconds

    # Function to draw a single node
    def draw_node(node):
        nx.draw_networkx_nodes(min_spanning_tree, pos, nodelist=[node], node_size=200, node_color='red', ax=ax)
        canvas.draw()
        canvas.get_tk_widget().update()
        canvas.get_tk_widget().after(1000)  # Delay: 2 seconds

    # Draw the edges and nodes with delay
    for edge in edges:
        draw_node(edge[0])
        draw_edge(edge)
    draw_node(last_vertex)



# gui thuật toán prim
def prim():
    try:
        num_vertices = int(entry_vertices.get())
        matrix_values = scrolledtext_matrix.get("1.0", tk.END).strip().split('\n')

        adjacency_matrix = []
        for row in matrix_values:
            values = list(map(int, row.split()))
            adjacency_matrix.append(values)

        if len(adjacency_matrix) != num_vertices or any(len(row) != num_vertices for row in adjacency_matrix):
            raise ValueError("Đâu vao không hợp lệ. Hãy đảm bảo ma trận là hình vuông và có số đỉnh chính xác.")

        if not check_diagonal(adjacency_matrix):
            raise ValueError("Đường chéo chính của ma trận phải bằng 0.")

        if not check_negative_weights(adjacency_matrix):
            raise ValueError("Ma trận không thể chứa trọng số âm.")

        if check_directed(adjacency_matrix):
            raise ValueError("Đồ thị có hướng. Thuật toán Prim chỉ áp dụng được cho đồ thị vô hướng.")
        
        G = nx.from_numpy_array(np.array(adjacency_matrix))
        
        start_vertex = simpledialog.askinteger("Đỉnh bắt đầu", "Nhập đỉnh bắt đầu (0 đến {}):".format(num_vertices - 1),
                                               minvalue=0, maxvalue=num_vertices - 1)

        if start_vertex is None:
            return

        

        min_spanning_tree = prim_algorithm(G, start_vertex)
        nodes = list(min_spanning_tree.nodes())
        edges = list(min_spanning_tree.edges())
        total_weight = sum(data['weight'] for u, v, data in min_spanning_tree.edges(data=True))


        graph_window = tk.Toplevel(root)
        graph_window.title("Cây bao trùm tối thiểu")

        figure, ax = plt.subplots(figsize=(10, 10))
        canvas = FigureCanvasTkAgg(figure, master=graph_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        pos = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, 'weight')
        ax.set_title("Cây khung nhỏ nhất")
        nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue')
        # Vẽ các cạnh
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black')
        # thêm số cho đỉnh
        nx.draw_networkx_labels(G, pos, font_color='black', font_weight='bold')
        # them trong so
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)



        draw_tree_with_delay(min_spanning_tree, pos, ax, canvas, start_vertex)
        plt.text(0.5, -0.1, f"Cây khung nhỏ nhất từ đỉnh {start_vertex}: {edges} = {total_weight}", fontsize=10, color='red', ha='center', va='center', transform=plt.gca().transAxes)
        canvas.draw()

    except ValueError as e:
        tk.messagebox.showerror("Lỗi", str(e))


#gui menu
def perform_algorithm():
    selected_algorithm = combobox.get()
    if selected_algorithm == 'Prim':
        prim()
    elif selected_algorithm == 'Dijkstra':
        dijkstra()

root = tk.Tk()
root.title("Trực quan hóa thuật toán")
root.geometry('500x500')
root['bg'] = 'pink'
label_vertices = tk.Label(root, text="Nhập số đỉnh:", font=('Verdena', 18), bg='pink')
label_vertices.place(x=20, y=30)

entry_vertices = tk.Entry(root, width=15, font=('Verdena', 16))
entry_vertices.place(x=220, y=30)

label_matrix = tk.Label(root, text="Nhập ma trận kề:", font=('Verdena', 18), bg='pink')
label_matrix.place(x=20, y=80)

scrolledtext_matrix = scrolledtext.ScrolledText(root, width=40, height=15, font=('Verdena', 14))
scrolledtext_matrix.place(x=220, y=80)

label_choose = tk.Label(root, text="Thuật toán:", font=('Verdena', 18), bg='pink')
label_choose.place(x=20, y=500)

combobox = ttk.Combobox(root, font=('Verdena', 16), width=15)
combobox['value'] = ('Prim', 'Dijkstra')
combobox.set('Prim')
combobox.place(x=220, y=500)

button_execute = tk.Button(root, text="Thực hiện", font=('verdena', 16),command=perform_algorithm, bg="blue", fg="white")
button_execute.place(x=450, y=490)

root.mainloop()