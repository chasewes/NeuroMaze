from PIL import Image
import random
import torch
from torchvision import transforms
from PIL import ImageDraw


def generate_maze(width, height, block_size):
    maze = [[1 for _ in range(width // block_size)] for _ in range(height // block_size)]
    start_x, start_y = random.randint(1, (width // block_size) - 2), random.randint(1, (height // block_size) - 2)
    stack = [(start_x, start_y)]
    maze[start_y][start_x] = 0

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
            maze[wall_y][wall_x] = 0
            maze[next_cell[1]][next_cell[0]] = 0
            stack.append(next_cell)
        else:
            stack.pop()

    end_x, end_y = random.randint(1, (width // block_size) - 2), random.randint(1, (height // block_size) - 2)
    while maze[end_y][end_x] != 0:
        end_x, end_y = random.randint(1, (width // block_size) - 2), random.randint(1, (height // block_size) - 2)
    maze[end_y][end_x] = 2
    
    image_maze = maze_to_image(maze,block_size)

    return image_maze

def maze_to_image(maze, block_size):
    width, height = len(maze[0]) * block_size, len(maze) * block_size
    image = Image.new("RGB", (width, height), (0, 0, 255))  # Blue background
    draw = ImageDraw.Draw(image)

    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            if cell == 1:  # Wall
                draw.rectangle([x*block_size, y*block_size, (x+1)*block_size, (y+1)*block_size], fill=(0, 0, 0))
            elif cell == 2:  # Goal
                draw.rectangle([x*block_size, y*block_size, (x+1)*block_size, (y+1)*block_size], fill=(0, 255, 0))
            elif cell == :  # Player
                draw.rectangle([x*block_size, y*block_size, (x+1)*block_size, (y+1)*block_size], fill=(255, 0, 0))

    return image

def preprocess_image(image_tensor, action_one_hot):
    """ Preprocess the image and action for model input """
    action_channel = action_one_hot.view(-1, 1, 1).expand(-1, image_tensor.shape[1], image_tensor.shape[2])
    return torch.cat((image_tensor, action_channel), dim=0).unsqueeze(0)  # Add batch dimension

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
