import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq

# Generating map
map = np.zeros((500, 1200, 3))

# Generating rectangular clearance
cv2.rectangle(map, pt1=(100-5, 0), pt2=(175+5, 400+5), color=(0, 0, 255), thickness=-1)
cv2.rectangle(map, pt1=(275-5, 100-5), pt2=((275+75+5), 500), color=(0, 0, 255), thickness=-1)
# Generating rectangles obstacle
cv2.rectangle(map, pt1=(100, 0), pt2=(175, 400), color=(255, 0, 0), thickness=-1)
cv2.rectangle(map, pt1=(275, 100), pt2=((275+75), 500), color=(255, 0, 0), thickness=-1)
# Generating hexagonal clearance (second method)
vertices_hexagon = 6
center_hexagon = (650, 250)
radius_hexagon = 155
hexagon_points = cv2.ellipse2Poly(center_hexagon, (radius_hexagon, radius_hexagon), 0, 0, 360, 60)
cv2.fillPoly(map, [hexagon_points], (0, 0, 255))
# Generating hexagonal obstacle (second method)
vertices_hexagon = 6
center_hexagon = (650, 250)
radius_hexagon = 150
hexagon_points = cv2.ellipse2Poly(center_hexagon, (radius_hexagon, radius_hexagon), 0, 0, 360, 60)
cv2.fillPoly(map, [hexagon_points], (255, 0, 0))

#Generating concave clearance (inverted C)
cv2.rectangle(map,pt1=(900-5, 50-5), pt2 = (1100+5, 125+5), color = (0, 0, 255), thickness=-1)
cv2.rectangle(map,pt1=((1200-180-5), (125+5)), pt2 = ((1200-100+5), (50+400-75)), color = (0, 0, 255), thickness=-1)
cv2.rectangle(map,pt1=((1200-100-200-5), (50+400-75-5)), pt2 = ((1100+5),(50+400+5)), color = (0, 0, 255), thickness=-1)
#Generating concave obstacle (inverted C)
cv2.rectangle(map,pt1=(900, 50), pt2 = (1100, 125), color = (255, 0, 0), thickness=-1)
cv2.rectangle(map,pt1=((1200-180), (50+75)), pt2 = ((1200-100), (50+400-75)), color = (255, 0, 0), thickness=-1)
cv2.rectangle(map,pt1=((1200-100-200), (50+400-75)), pt2 = ((1100),(50+400)), color = (255, 0, 0), thickness=-1)

# Display the image
plt.imshow(map.astype(int))
plt.title("Initial Map (close this window to continue)")
plt.show()

# Define the actions set
actions_set = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
cost_straight = 1.0
cost_diagonal = 1.4
video_name = 'dijkstra_scan.mp4'
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (map.shape[1], map.shape[0]))

# Define the function to calculate the cost of moving from one node to another
def calculate_cost(current_cost, action):
    if action in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        return current_cost + cost_diagonal
    else:
        return current_cost + cost_straight

# Dijkstra's algorithm function
def dijkstra(start, goal, map):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, (0, start)) # 0 being the priority for start
    came_from = {} # parent nodes dictionary
    cost_so_far = {start: 0} # node: cost to reach

    while open_list:
        current_cost, current_node = heapq.heappop(open_list)

        if current_node == goal:
            # Reached the goal, backtrack to save the path
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start)
            path.reverse()
            return path

        closed_list.add(current_node)

        for action in actions_set:
            dx, dy = action
            next_node = (current_node[0] + dx, current_node[1] + dy)

            if (5 <= next_node[0] < (map.shape[0]-5) and 5 <= next_node[1] < (map.shape[1]-5) and
                    next_node not in closed_list and np.all(map[next_node[0], next_node[1]] == 0)):
                new_cost = calculate_cost(cost_so_far[current_node], action)

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current_node
                    # Update the color of the scanned pixel to green
                    map[next_node[0], next_node[1]] = [0, 255, 0]  # Green color
                    

    return None  # No path found

# Prompt user for the start coordinates
while True:
    start_x = int(input("\nEnter your start x-coordinate: "))
    start_y = int(input("Enter your start y-coordinate: "))

    # Access the pixel value using the index
    start_pixel_value = map[start_y, start_x]

    # Check if the pixel value corresponds to red or blue and if the start coordinates are on the boundary
    if (start_pixel_value == [255, 0, 0]).all() or (start_pixel_value == [0, 0, 255]).all() or (5 >= start_y >= (map.shape[0]-5)) or (5 >= start_y < (map.shape[1]-5)):
        print("Your start is on an obstacle. Please re-enter the goal coordinates.")

    else:
        print("Your start cordinates are correct, proceed ahead" )
        break   

# Prompt user for the goal coordinates
while True:

    goal_x = int(input("\nEnter your goal x-coordinate: "))
    goal_y = int(input("Enter your goal y-coordinate: "))

    # Access the pixel value using the index
    goal_pixel_value = map[goal_y, goal_x]

    # Check if the pixel value corresponds to red or blue and if the goal coordinates are on the boundary
    if (goal_pixel_value == [255, 0, 0]).all() or (goal_pixel_value == [0, 0, 255]).all() or (5 >= goal_x >= (map.shape[0]-5)) or (5 >= goal_x < (map.shape[1]-5)):
        print("Your goal is on an obstacle. Please re-enter the goal coordinates.")
        print(goal_pixel_value)
    else:
        print("Your goal cordinates are correct. Search in progress PLEASE WAIT..." )
        print("Goal coordinate has following pixel values: ", goal_pixel_value)
        break 

# Run Dijkstra's algorithm
start_node = (start_x, start_y)  # Assuming the robot starts from the bottom-left corner
goal_node = (goal_x, goal_y)
path = dijkstra(start_node, goal_node, map)

# Print the path
if path:
    print("Path found:", path)
    for node in path:
        # Change color to black for nodes in the path
        map[node[0], node[1]] = [0, 0, 0]
        frame = cv2.cvtColor(map.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(frame)
        
else:
    print("No path found.")

# Display the updated image after scanning
plt.imshow(map.astype(int))
plt.title("Map after scanning")
plt.show()