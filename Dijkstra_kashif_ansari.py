import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq

# Generating map
map = np.zeros((500, 1200, 3))
red = np.array([255, 0, 0])
blue = np.array([0, 0, 255])
black = np.array([0, 0, 0])

# Generating rectangular clearance
cv2.rectangle(map, pt1=(95, 0), pt2=(180, 405), color=(0, 0, 255), thickness=-1)
cv2.rectangle(map, pt1=(270, 95), pt2=((355), 500), color=(0, 0, 255), thickness=-1)
# Generating rectangles obstacle
cv2.rectangle(map, pt1=(100, 0), pt2=(175, 400), color=(255, 0, 0), thickness=-1)
cv2.rectangle(map, pt1=(275, 100), pt2=((350), 500), color=(255, 0, 0), thickness=-1)
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
cv2.rectangle(map,pt1=(895, 45), pt2 = (1105, 130), color = (0, 0, 255), thickness=-1)
cv2.rectangle(map,pt1=((1015), (130)), pt2 = ((1105), (425)), color = (0, 0, 255), thickness=-1)
cv2.rectangle(map,pt1=((895), (370)), pt2 = ((1105),(455)), color = (0, 0, 255), thickness=-1)
#Generating concave obstacle (inverted C)
cv2.rectangle(map,pt1=(900, 50), pt2 = (1100, 125), color = (255, 0, 0), thickness=-1)
cv2.rectangle(map,pt1=((1020), (125)), pt2 = ((1100), (50+375)), color = (255, 0, 0), thickness=-1)
cv2.rectangle(map,pt1=((900), (375)), pt2 = ((1100),(450)), color = (255, 0, 0), thickness=-1)

# Display the image
plt.imshow(map.astype(int))
plt.title("Initial Map (close this window to continue)")
plt.show()

# Define the actions set
actions_set = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
cost_straight = 1.0
cost_diagonal = 1.4
video_name = 'D:\Desktop\djikstra_scan.mp4'
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
    visited_list = []
    heapq.heappush(open_list, (0, start)) # 0 being the priority for start
    came_from = {} # parent nodes dictionary
    cost_so_far = {start: 0} # node: cost to reach
    path_iteration =    0
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
        visited_list.append(current_node)

        for action in actions_set:
            dx, dy = action
            next_node = (current_node[0] + dx, current_node[1] + dy)

            if (5 <= next_node[0] < (map.shape[0]-5)) and (5 <= next_node[1] < (map.shape[1]-5)) and (next_node not in closed_list) and (np.array_equal(map[next_node[0], next_node[1]], black)):
                new_cost = calculate_cost(cost_so_far[current_node], action)

                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current_node
                    # Update the color of the scanned pixel to green
                    map[next_node[0], next_node[1]] = [0, 255, 0]  # Green color
                    # frame1 = cv2.cvtColor(map.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    # out.write(frame1)
                    if path_iteration%10000==0:
                        frame = cv2.cvtColor(map.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        out.write(frame)
                    path_iteration+=1


    return None  # No path found

# Prompt user for the start coordinates
while True:
    start_x = int(input("\nEnter your start x-coordinate: "))
    start_y = int(input("Enter your start y-coordinate: "))

    # Access the pixel value using the index
    if (5 <= start_y <= (map.shape[0]-5)) and (5 <= start_x < (map.shape[1]-5)):
        start_pixel_value = [int(value) for value in map[start_y, start_x]]
    else:
        print("Your values are not within boundary or on clearances. Please re-enter the start coordinates.")
        continue

    # Check if the pixel value corresponds to red or blue and if the start coordinates are on the boundary
    if np.array_equal(start_pixel_value, red) or np.array_equal(start_pixel_value, blue) or (5 >= start_x >= (map.shape[0]-5)) or (5 >= start_y < (map.shape[1]-5)):
        print("Your start is on an obstacle. Please re-enter the start coordinates.")

    else:
        print("Your start cordinates are correct, proceed ahead >>" )
        break

# Prompt user for the goal coordinates
while True:

    goal_x = int(input("\nEnter your goal x-coordinate: "))
    goal_y = int(input("Enter your goal y-coordinate: "))

    # Access the pixel value using the index
    if (5 <= goal_y <= (map.shape[0]-5)) and (5 <= goal_x <= (map.shape[1]-5)):
        goal_pixel_value = [int(value) for value in map[goal_y, goal_x]]
    else:
        print("Your values are not within boundary or on clearances. Please re-enter the goal coordinates.")
        continue

    # Check if the pixel value corresponds to red or blue and if the goal coordinates are on the boundary
    if np.array_equal(goal_pixel_value, red) or np.array_equal(goal_pixel_value, blue):

        print("Your goal is on an obstacle. Please re-enter the goal coordinates.")
        print(goal_pixel_value)
    else:
        print("Your goal cordinates are correct. Search in progress PLEASE WAIT..." )
        print("Goal coordinate has following pixel values: ", goal_pixel_value)
        break

# Run Dijkstra's algorithm
start_node = (start_y, start_x)  # Assuming the robot starts from the bottom-left corner
goal_node = (goal_y, goal_x)
path = dijkstra(start_node, goal_node, map)

path_iteration = 0
# Print the path
if path :
    print("Path found:", path)
    for node in path:
        # Change color to black for nodes in the path
        map[node[0], node[1]] = [0, 0, 0]
        if path_iteration%10==0:
            frame = cv2.cvtColor(map.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(frame)
        path_iteration+=1

else:
    print("No path found.")

# Display the updated image after scanning
plt.imshow(map.astype(int))
plt.title("Map after scanning")
plt.show()

out.release()
cv2.destroyAllWindows()