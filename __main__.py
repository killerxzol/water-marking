import numpy as np
from Picture import Picture
from PIL import Image


def get_text(path):
    text = ''
    file = open(path, mode='r', encoding='utf-8-sig')
    for line in file:
        text = text + line.strip('\n')
    text = text.replace(' ﻿', '')
    return text


def main():

    print('\n   - - - - - - - - - - - -\n')
    mode = input('Select mode: { Embed, Decode, Exit }: ')

    while mode != 'Exit':

        if mode == 'Embed':
            image_path = input('\nEnter the path to the image: ')
            image = np.array(Image.open(image_path))
            sub_mode = input('\nText from keyboard or file { Keyboard, File }: ')
            if sub_mode == 'Keyboard':
                text = input('\nEnter text: ')
            else:
                text_path = input('\nEnter the path to the text file: ')
                text = get_text(text_path)
            columns = int(input('\nProtected columns: '))
            block_size = int(input('\nBlock size: '))
            redundancy = int(input('\nRedundancy: '))
            iterations = int(input('\nIterations: '))
            noise = float(input('\nNoise: '))
            picture = Picture(mode, image, text, columns, block_size, redundancy, iterations, noise)
            picture.execution()

        if mode == 'Decode':
            image_path = input('\nEnter the path to the image: ')
            image = np.array(Image.open(image_path))
            columns = int(input('\nProtected columns: '))
            block_size = int(input('\nBlock size: '))
            redundancy = int(input('\nRedundancy: '))
            text = Picture(mode, image, '', columns, block_size, redundancy)
            text.execution()

        mode = input('\nSelect mode: { Embed, Decode, Exit }: ')


if __name__ == '__main__':
    main()