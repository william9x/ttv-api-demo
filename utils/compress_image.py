from PIL import Image

source_images = [
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI01_Pixar_Trump_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI02_Pixar_WonderWoman_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI03_Realistic_JohnWick_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI04_Realistic_DarthVader_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI05_Realistic_ScoobyDoo_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI06_Pixel_Penguins_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI07_Anime_Countryside_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI08_Anime_WhatDreams_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI09_Anime_Gundam_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI10_VanGogh_Table_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI11_VanGogh_Cat_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI12_Pixel_Mario_thumb.jpg",
    "/home/liam/Downloads/03-01-2024-09-42-38_files_list/DI13_Realistic_Dinosaur_thumb.jpg",
]

target_images = [
    "/home/liam/Downloads/compressed/DI01_Pixar_Trump_thumb.jpg",
    "/home/liam/Downloads/compressed/DI02_Pixar_WonderWoman_thumb.jpg",
    "/home/liam/Downloads/compressed/DI03_Realistic_JohnWick_thumb.jpg",
    "/home/liam/Downloads/compressed/DI04_Realistic_DarthVader_thumb.jpg",
    "/home/liam/Downloads/compressed/DI05_Realistic_ScoobyDoo_thumb.jpg",
    "/home/liam/Downloads/compressed/DI06_Pixel_Penguins_thumb.jpg",
    "/home/liam/Downloads/compressed/DI07_Anime_Countryside_thumb.jpg",
    "/home/liam/Downloads/compressed/DI08_Anime_WhatDreams_thumb.jpg",
    "/home/liam/Downloads/compressed/DI09_Anime_Gundam_thumb.jpg",
    "/home/liam/Downloads/compressed/DI10_VanGogh_Table_thumb.jpg",
    "/home/liam/Downloads/compressed/DI11_VanGogh_Cat_thumb.jpg",
    "/home/liam/Downloads/compressed/DI12_Pixel_Mario_thumb.jpg",
    "/home/liam/Downloads/compressed/DI13_Realistic_Dinosaur_thumb.jpg",
]

if __name__ == "__main__":
    for source, target in zip(source_images, target_images):
        foo = Image.open(source)
        foo.save(target, optimize=True, quality=85)
