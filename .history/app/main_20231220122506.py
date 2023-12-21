import streamlit as st
from PIL import Image
import torch
from seq2seq import Seq2Seq
from utils import generate_maze, preprocess_image, postprocess_image, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache(allow_output_mutation=True)
def load_model():
    model = Seq2Seq(num_channels=3, num_actions=4, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(32, 32), num_layers=5)
    model.to(device)
    load_checkpoint('checkpoints/checkpoint_lr_0.0001_bs_1_epoch_5.pth.tar', model, device=device)  # Load your saved model
    model.eval()
    return model

@st.cache
def generate_initial_maze():
    return generate_maze(32, 32, 1)

model = load_model()
maze_image_small = generate_initial_maze()

st.title("NeuroMaze")
st.image(maze_image_big, caption='Initial Maze', use_column_width=False)

action_dict = {'Up': [1, 0, 0, 0], 'Down': [0, 1, 0, 0], 'Left': [0, 0, 1, 0], 'Right': [0, 0, 0, 1]}
action_buttons = st.columns(3)

with action_buttons[1]:
    if st.button('Up'):
        action = 'Up'
with action_buttons[0]:
    if st.button('Left'):
        action = 'Left'
with action_buttons[2]:
    if st.button('Right'):
        action = 'Right'
with action_buttons[1]:
    if st.button('Down'):
        action = 'Down'

if 'action' in locals():
    # Convert action to one-hot encoding
    action_one_hot = torch.tensor(action_dict[action]).unsqueeze(0)
    input_tensor = preprocess_image(maze_image_small, action_one_hot)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        predicted_frame = model(input_tensor)
        st.write("I think the next frame will look like this:")
    new_maze_image = postprocess_image(predicted_frame.cpu())

    st.image(new_maze_image, caption='Next Maze State', use_column_width=True)
