import streamlit as st
from PIL import Image
import torch
from seq2seq import Seq2Seq
from utils import generate_maze, preprocess_image, postprocess_image, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = Seq2Seq(num_channels=3, num_actions=4, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(32, 32), num_layers=5)
model.to(device)
load_checkpoint('checkpoints/checkpoint_lr_0.0001_bs_1_epoch_5.pth.tar', model, device=device)  # Load your saved model
model.eval()

st.title("NeuroMaze")

# Generate a random maze
maze_image = generate_maze(32,32,1)
st.write(maze_image.size)
rescaled_maze_image = maze_image.resize((320,320))
st.image(rescaled_maze_image, caption='Initial Maze', use_column_width=True)

# Display buttons for actions
action = st.radio("Choose your action:", ('Up', 'Down', 'Left', 'Right'))

if st.button('Make Move'):
    # Process the action and get the next maze state
    action_one_hot = ...  # Convert the action to one-hot encoding
    input_tensor = preprocess_image(maze_image, action_one_hot)
    with torch.no_grad():
        predicted_frame = model(input_tensor)
    new_maze_image = postprocess_image(predicted_frame)

    st.image(new_maze_image, caption='Next Maze State', use_column_width=True)
