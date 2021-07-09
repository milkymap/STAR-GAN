import torch as th 

from glob import glob 

from torch.utils.data import Dataset 
from torchvision import transforms as T 
from PIL import Image 

from libraries.strategies import * 

class Data(Dataset):
    def __init__(self, path_to_images, path_to_attributes, attributes=["Black_Hair", "Blond_Hair", "Brown_Hair"]):
        self.transform = T.Compose([
        		T.Resize((128, 128)),
    			T.RandomHorizontalFlip(),
    			T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    		])
        
        self.selected_attrs = attributes
        self.files = sorted(glob(f'{path_to_images}/*.jpg'))[:1000]
        self.label_path = path_to_attributes
        self.annotations = self.get_annotations()

    def get_annotations(self):
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, "r")]
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in self.selected_attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == "1"))
            annotations[filename] = labels
        return annotations

    def get_random_label(self, source_label):
    	shift = np.random.randint(len(source_label) - 1)
    	return np.roll(source_label, shift)

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split("/")[-1]
        image = read_image(filepath, by='th')
        image = image / th.max(image)  # between 0 ~ 1
        image = self.transform(image)

        source_label = np.array(self.annotations[filename])
        target_label = self.get_random_label(source_label)


        source_label = th.as_tensor(source_label).float()
        target_label = th.as_tensor(target_label).float()

        return image, source_label, target_label

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
	D = Data('dump/images', 'dump/attributes.txt')
	print(D[0])