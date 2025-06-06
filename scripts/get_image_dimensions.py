from PIL import Image
from pathlib import Path
from typing import Tuple, Union

def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Helper to get image dimensions (width, height) without loading full image data.

    Parameters
    ----------
    image_path : Union[str, Path]
        The path to the image file.

    Returns
    -------
    Tuple[int, int]
        A tuple containing (width, height) of the image.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    Exception
        For other errors during image opening.
    """
    image_path = Path(image_path) # Ensure it's a Path object
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    try:
        with Image.open(image_path) as img:
            return img.size # Returns (width, height)
    except Exception as e:
        raise Exception(f"Error opening image {image_path}: {e}")

# Example Usage:
if __name__ == "__main__":
    # Assuming your images are in a folder named 'images' relative to where this script runs
    # You can adjust this path as needed
    image_folder = Path(__file__).resolve().parent.parent / "images" 
    
    shepp_logan_path = image_folder / "Shepp_Logan_phantom.png"
    spiral_path = image_folder / "spiral.png"
    spiral_zigzag_path = image_folder / "spiral_and_zigzag.png"

    try:
        width, height = get_image_dimensions(shepp_logan_path)
        print(f"Shepp_Logan_phantom.png dimensions: Width = {width}, Height = {height}")

        width, height = get_image_dimensions(spiral_path)
        print(f"spiral.png dimensions: Width = {width}, Height = {height}")

        width, height = get_image_dimensions(spiral_zigzag_path)
        print(f"spiral_and_zigzag.png dimensions: Width = {width}, Height = {height}")

    except FileNotFoundError as e:
        print(e)
        print("Please ensure your image files are in the correct 'images' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")