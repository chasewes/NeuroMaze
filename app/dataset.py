from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



class MazeGameDataset(Dataset):
    # Adding reverse mapping
    def __init__(self, image_dir, action_file):
        self.image_dir = image_dir
        self.actions = self._load_actions(action_file)
        self.num_actions = 4
        self.action_index = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.index_action = {v: k for k, v in self.action_index.items()}  # Reverse mapping
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # why these values? 
        ])

    def action_to_one_hot(self, action):
        one_hot = torch.zeros(self.num_actions)
        one_hot[self.action_index[action]] = 1
        return one_hot

    def one_hot_to_action(self, one_hot):
        index = one_hot.argmax().item()
        return self.index_action[index]

    def _load_actions(self, action_file):
        with open(action_file, 'r') as f:
            actions = json.load(f)
        return actions

    def __len__(self):
        return len(self.actions) - 1

    def __getitem__(self, idx):
        action = self.actions[str(idx+1)]
        
        if action is None: 
            idx += 1
            action = self.actions[str(idx+1)]
            if action is None: 
                idx += 1

        frame_file = os.path.join(self.image_dir, f"{idx}.png")
        next_frame_file = os.path.join(self.image_dir, f"{idx + 1}.png")

        frame = Image.open(frame_file).convert('RGB')
        next_frame = Image.open(next_frame_file).convert('RGB')

        frame = self.transform(frame)
        next_frame = self.transform(next_frame)
        
        action = self.actions[str(idx+1)]

        # print(action)
        action_one_hot = self.action_to_one_hot(action)

        action_one_hot = action_one_hot.view(self.num_actions, 1, 1)
        action_channel = action_one_hot.expand(-1, frame.shape[1], frame.shape[2])
        frame_with_action = torch.cat((frame, action_channel), dim=0)

        return frame_with_action, next_frame