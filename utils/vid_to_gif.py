import glob

import cv2
from PIL import Image

import utils

source_images = [
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI01_Pixar_Trump.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI02_Pixar_WonderWoman.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI03_Realistic_JohnWick.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI04_Realistic_DarthVader.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI05_Realistic_ScoobyDoo.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI06_Pixel_Penguins.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI07_Anime_Countryside.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI08_Anime_WhatDreams.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI09_Anime_Gundam.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI10_VanGogh_Table.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI11_VanGogh_Cat.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI12_Pixel_Mario.mp4",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI13_Realistic_Dinosaur.mp4",
]

target_images = [
    "/home/liam/Downloads/compressed/DI01_Pixar_Trump_thumb.gif",
    "/home/liam/Downloads/compressed/DI02_Pixar_WonderWoman_thumb.gif",
    "/home/liam/Downloads/compressed/DI03_Realistic_JohnWick_thumb.gif",
    "/home/liam/Downloads/compressed/DI04_Realistic_DarthVader_thumb.gif",
    "/home/liam/Downloads/compressed/DI05_Realistic_ScoobyDoo_thumb.gif",
    "/home/liam/Downloads/compressed/DI06_Pixel_Penguins_thumb.gif",
    "/home/liam/Downloads/compressed/DI07_Anime_Countryside_thumb.gif",
    "/home/liam/Downloads/compressed/DI08_Anime_WhatDreams_thumb.gif",
    "/home/liam/Downloads/compressed/DI09_Anime_Gundam_thumb.gif",
    "/home/liam/Downloads/compressed/DI10_VanGogh_Table_thumb.gif",
    "/home/liam/Downloads/compressed/DI11_VanGogh_Cat_thumb.gif",
    "/home/liam/Downloads/compressed/DI12_Pixel_Mario_thumb.gif",
    "/home/liam/Downloads/compressed/DI13_Realistic_Dinosaur_thumb.gif",
]

frame_folder = "/home/liam/Downloads/compressed/frames"


def convert_mp4_to_jpgs(path):
    video_capture = cv2.VideoCapture(path)
    still_reading, image = video_capture.read()
    frame_count = 0
    while still_reading:
        cv2.imwrite(f"{frame_folder}/frame_{frame_count:03d}.jpg", image)

        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1


def make_gif(path):
    images = glob.glob(f"{frame_folder}/*.jpg")
    images.sort()
    frames = [Image.open(image) for image in images]
    utils.export_frames_to_gif(frames, path)


if __name__ == "__main__":
    for source, target in zip(source_images, target_images):
        convert_mp4_to_jpgs(source)
        make_gif(target)
