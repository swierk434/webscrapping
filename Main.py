import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def get_png_linksnnames(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        return [urljoin(url, link.get('href')) for link in soup.find_all('a') if link.get('href').endswith('.png')], [link.get('href').split('/')[-1] for link in soup.find_all('a') if link.get('href').endswith('.png')]
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def get_image_from_link(link):
    with requests.get(link, stream=True) as response:
        image_data = BytesIO(response.content)
        buffered_image = Image.open(image_data)
    return buffered_image       

def save_image_from_link(link, name):
    with requests.get(link, stream=True) as response:
        with open(name, 'wb') as f:
            f.write(response.content)

def save_image_from_image(image, name):
    image.save(name)

def create_gaussian_filter(size=3, sigma=1):
    space = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gaussian_filter_1d = np.exp(-space**2 / (2. * sigma**2))
    gaussian_filter_2d = np.outer(gaussian_filter_1d, gaussian_filter_1d)
    return gaussian_filter_2d / gaussian_filter_2d.sum()

def job1(element_pair):
    element1, element2 = element_pair
    return save_image_from_link(element1, element2)

def filter_image(image):
    image_array = np.array(image)

    gaussian_filter = create_gaussian_filter()

    red_array = convolve2d(image_array[:,:,0], gaussian_filter, mode='same', boundary='symm')
    green_array = convolve2d(image_array[:,:,1], gaussian_filter, mode='same', boundary='symm')
    blue_array = convolve2d(image_array[:,:,2], gaussian_filter, mode='same', boundary='symm')

    rgb_array = np.stack([red_array, green_array, blue_array], axis=-1).astype(np.uint8)

    image = Image.fromarray(rgb_array)
    return image

def func_stack(link, name):
    image = get_image_from_link(link)
    image = filter_image(image)
    save_image_from_image(image, name) 

def main1():
    start_time = time.time()
    links, names = get_png_linksnnames("https://www.if.pw.edu.pl/~mrow/dyd/wdprir/")
    for link, name in zip(links, names):
        save_image_from_link(link, name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main1: {elapsed_time} seconds")

def main2():
    start_time = time.time()
    links, names = get_png_linksnnames("https://www.if.pw.edu.pl/~mrow/dyd/wdprir/")

    num_threads = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for link, name in zip(links, names):
            executor.submit(save_image_from_link, link, name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main2: {elapsed_time} seconds")

def main3():
    start_time = time.time()
    links, names = get_png_linksnnames("https://www.if.pw.edu.pl/~mrow/dyd/wdprir/")
    for link, name in zip(links, names):
        image = get_image_from_link(link)
        image = filter_image(image)
        save_image_from_image(image, name)      
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main3: {elapsed_time} seconds")

def main4():
    start_time = time.time()
    links, names = get_png_linksnnames("https://www.if.pw.edu.pl/~mrow/dyd/wdprir/")
    
    num_threads = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for link, name in zip(links, names):
            executor.submit(func_stack, link, name)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main4: {elapsed_time} seconds")

if __name__ == '__main__':
    main1()
    main2()
    main3()
    main4()