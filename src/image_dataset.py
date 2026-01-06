import os
from PIL import Image
from torch.utils.data import Dataset


class SatelliteImageDataset(Dataset):
    """
    PyTorch Dataset for satellite images aligned with housing tabular data.

    Each item returns:
        image_tensor, house_id
    """

    def __init__(self, dataframe, image_dir, transform=None):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            Must contain an 'id' column.
        image_dir : str
            Directory containing satellite images named as <id>.jpg
        transform : callable, optional
            torchvision transforms to apply to images
        """
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        house_id = row["id"]

        image_path = os.path.join(self.image_dir, f"{house_id}.jpg")

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found for house ID {house_id} at {image_path}"
            )

        # Safe image loading
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, house_id
