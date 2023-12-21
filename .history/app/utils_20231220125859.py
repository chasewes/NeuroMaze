from PIL import Image
import random
import torch
from torchvision import transforms
from PIL import ImageDraw
from torchvision.transforms import ToTensor


def generate_maze(width, height, block_size):

    #seed random number generator
    random.seed()
    
    # Initialize maze with walls (1) everywhere
    maze = [[1 for _ in range(width // block_size)] for _ in range(height // block_size)]

    # Randomly choose a starting position for the player
    start_x, start_y = random.randint(1, (width // block_size) - 2), random.randint(1, (height // block_size) - 2)
    maze[start_y][start_x] = 3  # Player position marked as 3

    # Depth-first search to carve out a path
    stack = [(start_x, start_y)]
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx * 2, y + dy * 2
            if 1 <= nx < (width // block_size) - 1 and 1 <= ny < (height // block_size) - 1 and maze[ny][nx] == 1:
                neighbors.append((nx, ny))
        if neighbors:
            next_cell = random.choice(neighbors)
            wall_x, wall_y = (x + next_cell[0]) // 2, (y + next_cell[1]) // 2
            maze[wall_y][wall_x] = 0  # Remove wall between cells
            maze[next_cell[1]][next_cell[0]] = 0  # Carve out new cell
            stack.append(next_cell)
        else:
            stack.pop()

    # Randomly choose a goal position
    end_x, end_y = random.randint(1, (width // block_size) - 2), random.randint(1, (height // block_size) - 2)
    while maze[end_y][end_x] == 3:  # Ensure goal is not at player's position
        end_x, end_y = random.randint(1, (width // block_size) - 2), random.randint(1, (height // block_size) - 2)

    maze[end_y][end_x] = 2  # Goal position marked as 2
    print(maze)

    return maze_to_image(maze)

def maze_to_image(maze):
    width, height = len(maze[0]), len(maze)
    image = Image.new("RGB", (width, height), (0, 0, 255))  # Blue background
    draw = ImageDraw.Draw(image)

    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == 1:  # Wall
                draw.point((x, y), fill=(0, 0, 0))  # Black pixel for walls
            elif cell == 2:  # Goal
                draw.point((x, y), fill=(0, 255, 0))  # Green pixel for the goal
            elif cell == 3:  # Player
                draw.point((x, y), fill=(255, 0, 0))  # Red pixel for the player

    return image


def preprocess_image(image, action_one_hot):
    """ Preprocess the image and action for model input """
    image_tensor = ToTensor()(image)  # Convert PIL Image to tensor
    action_channel = action_one_hot.view(-1, 1, 1).expand(-1, image_tensor.shape[1], image_tensor.shape[2])
    combined_tensor = torch.cat((image_tensor, action_channel), dim=0).unsqueeze(0)  # Add batch dimension
    return combined_tensor


def postprocess_image(image_tensor):
    """ Postprocess the model output to viewable image """
    return transforms.functional.to_pil_image(image_tensor.squeeze(0))

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint['epoch']


def scale_image(maze_image, scale_factor=10):
    """ Scale up an image by a certain factor. """
    width, height = maze_image.size
    new_width, new_height = width * scale_factor, height * scale_factor

    # Create a new image with the scaled dimensions
    scaled_image = Image.new("RGB", (new_width, new_height))

    # Get pixel data from the original image
    pixels = maze_image.load()

    draw = ImageDraw.Draw(scaled_image)

    # Draw each pixel from the original image as a larger block
    for y in range(height):
        for x in range(width):
            color = pixels[x, y]
            draw.rectangle([
                x * scale_factor, 
                y * scale_factor, 
                (x + 1) * scale_factor - 1, 
                (y + 1) * scale_factor - 1
            ], fill=color)

    return scaled_image