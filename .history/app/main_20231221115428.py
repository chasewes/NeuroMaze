import streamlit as st
import torch
from seq2seq import Seq2Seq
from utils import generate_maze, preprocess_image, postprocess_image, load_checkpoint, scale_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache(allow_output_mutation=True)
def load_model():
    model = Seq2Seq(num_channels=3, num_actions=4, num_kernels=64, kernel_size=(3, 3), padding=(1, 1), activation="relu", frame_size=(32, 32), num_layers=5)
    model.to(device)
    load_checkpoint('checkpoints/checkpoint_lr_0.0001_bs_1_epoch_5.pth.tar', model, device=device)
    model.eval()
    return model

def generate_initial_maze():
    return generate_maze(32, 32, 1)

model = load_model()

if 'current_maze_image' not in st.session_state:
    st.session_state.current_maze_image = generate_initial_maze()

st.markdown("<h1 style='text-align: center;'>NeuroMaze</h1>", unsafe_allow_html=True)
# Centering the maze image
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write("")  # Empty column for spacing

with col2:
    st.image(scale_image(st.session_state.current_maze_image), caption='Maze State', use_column_width=True)

with col3:
    st.write("")  # Empty column for spacing
action_dict = {'Up': [1, 0, 0, 0], 'Down': [0, 1, 0, 0], 'Left': [0, 0, 1, 0], 'Right': [0, 0, 0, 1]}

# Centering and organizing the action buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    left_action = st.button("⬅️")

with col2:
    up_action = st.button("⬆️")
    down_action = st.button("⬇️")

with col3:
    right_action = st.button("➡️")

action = None
if up_action:
    action = 'Up'
elif down_action:
    action = 'Down'
elif left_action:
    action = 'Left'
elif right_action:
    action = 'Right'

if action:
    action_one_hot = torch.tensor(action_dict[action]).unsqueeze(0)
    input_tensor = preprocess_image(st.session_state.current_maze_image, action_one_hot)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        predicted_frame = model(input_tensor)
        st.session_state.current_maze_image = postprocess_image(predicted_frame.cpu())
    st.experimental_rerun()


    st.markdown("<div style='display: flex; justify-content: center;'><button>Generate New Maze</button></div>", unsafe_allow_html=True)
    if st.button('Generate New Maze'):
        st.session_state.current_maze_image = generate_initial_maze()
        st.experimental_rerun()
