from PIL import Image
import numpy as np
import time
import math


class Picture:

    def __init__(self, method, image, message, column_protected, block_size, redundancy, iterations=None, noise=None):
        self.method = method
        self.image = image
        self.original_image = np.copy(image)
        self.message = message
        self.column_protected = column_protected
        self.block_size = block_size
        self.redundancy = redundancy
        self.iterations = iterations
        self.noise = noise
        self.binary_message = None

    def control(self):
        difference = np.array(self.image.tolist()) - self.original_image
        fro = np.linalg.norm(difference)
        print('\nНорма Фробениуса: ', fro)
        print('\nНорма на пиксель: ', fro / (self.image.shape[0] * self.image.shape[1]))

    def encode_test(self):
        original, binary_original = self.message, self.binary_message
        self.decode_picture()
        recover, binary_recover = self.message[:len(original)], self.binary_message[:len(binary_original)]
        binary_original = [0 if ch == -1 else 1 for ch in binary_original]
        binary_recover = [0 if ch == '0' else 1 for ch in binary_recover]
        print('   - - - - Results - - - -\n')
        print('Error [Char]     :', 1 - np.mean(np.array(list(original)) == np.array(list(recover))))
        print('Error [Bit]      :', 1 - np.mean(np.array(binary_original) == np.array(binary_recover)))
        print('Original  [text] :', original)
        print('Recovered [text] :', recover)
        print('Original  [bits] :', binary_original)
        print('Recovered [bits] :', binary_recover)
        print('\n   - - - - - - - - - - - -\n')

    def make_noise(self):
        noise = np.random.rand(self.image.shape[0], self.image.shape[1]) * self.noise
        self.image = self.image + noise

    def orthogonalization_process(self, block, column):
        coefficient = np.zeros((column, column))
        solution = np.zeros(column)
        for i in range(column):
            for j in range(column):
                coefficient[i, j] = block[j + self.block_size - column, i]
            solution[i] = -np.dot(block[0:self.block_size - column, column], block[0:self.block_size - column, i])
        try:
            result = np.linalg.solve(coefficient, solution)
            block[self.block_size - column: self.block_size, column] = result
        except:
            pass
        return block

    def decode_message(self, raw_message):
        result = ''
        message = []
        for i in range(0, len(raw_message) % self.redundancy):
            raw_message.append(0)
        for i in range(0, len(raw_message), self.redundancy):
            sign = sum(raw_message[i:i + self.redundancy])
            if sign < 0:
                message.append('0')
            else:
                message.append('1')
        self.binary_message = message
        for i in range(0, len(message) % 7):
            message.append('0')
        for i in range(0, len(message), 7):
            temp_data = ''.join(message[i:i + 7])
            string = int(temp_data, 2)
            result = result + chr(string)
        return result

    def encode_message(self):
        bin_string = list(''.join(format(ord(i), '07b') for i in self.message))
        self.binary_message = [-1 if bit == '0' else 1 for bit in bin_string]

    def decode_block(self, block):
        binary_message = []
        u = np.linalg.svd(block, full_matrices=True)[0]
        for column_ in range(self.block_size):
            if u[0, column_] < 0:
                for row_ in range(self.block_size):
                    u[row_, column_] = (-1) * u[row_, column_]
        for column_ in range(self.column_protected, self.block_size):
            for row_ in range(1, self.block_size - column_):
                if u[row_, column_] < 0:
                    binary_message.append(-1)
                else:
                    binary_message.append(1)
        return binary_message

    def encode_block(self, block, sigma, message):
        average = (sigma[1] + sigma[-1]) / (self.block_size - 2)
        for index in range(2, self.block_size - 1):
            sigma[index] = sigma[1] - (index - 1) * average
        index = 0
        for column in range(self.column_protected, self.block_size):
            for row in range(1, self.block_size - column):
                block[row, column] = message[index] * abs(block[row, column])
                index = index + 1
            block = self.orthogonalization_process(block, column)
            norm = np.sqrt(np.dot(block[:, column], block[:, column]))
            if norm != 0:
                block[:, column] = block[:, column] / norm
        return block, sigma

    def decode_picture(self):
        message = []
        for column_ in range(math.floor(self.image.shape[1] / self.block_size)):
            for row_ in range(math.floor(self.image.shape[0] / self.block_size)):
                block = self.image[self.block_size * row_:self.block_size * (row_ + 1),
                                   self.block_size * column_:self.block_size * (column_ + 1)]
                message.extend(self.decode_block(block))
        text = self.decode_message(message)
        self.message = text

    def encode_picture(self):
        row_limit = math.floor(self.image.shape[0] / self.block_size)
        col_limit = math.floor(self.image.shape[1] / self.block_size)
        bit_limit = int(((self.block_size - self.column_protected - 1) * (self.block_size - self.column_protected)) / 2)
        bit_number = math.floor(row_limit * col_limit * bit_limit / self.redundancy)
        char_number = int(bit_number / 7)

        if char_number >= len(self.message):
            self.encode_message()
        else:
            mode = input('Too long message. Char limit: {0}. Bit limit: {1} Cut? (Y/N) '.format(char_number, bit_number))
            print('\n   - - - - - - - - - - - -\n')
            if mode == 'Y':
                self.message = self.message[:char_number]
                self.encode_message()
            else:
                exit()

        binary_message_ = [bit for bit in self.binary_message for _ in range(self.redundancy)]

        print('Embedding the message...')
        print('\n   - - - - - - - - - - - -\n')

        break_point = False
        for iteration in range(self.iterations):
            binary_message = binary_message_
            for column_ in range(col_limit):
                if break_point:
                    break_point = False
                    break
                for row_ in range(row_limit):
                    if not binary_message:
                        break_point = True
                        break
                    else:
                        if len(binary_message) <= bit_limit:
                            block_message = binary_message
                            binary_message = ''
                            while len(block_message) < bit_limit:
                                block_message.append(1)
                        else:
                            block_message = binary_message[:bit_limit]
                            binary_message = binary_message[bit_limit:]
                    block = self.image[self.block_size * row_:self.block_size * (row_ + 1),
                                       self.block_size * column_:self.block_size * (column_ + 1)]
                    u, s, vt = np.linalg.svd(block, full_matrices=True)
                    for column in range(self.block_size):
                        if u[0, column] < 0:
                            u[:, column] = -u[:, column]
                            vt[column, ] = -vt[column, ]
                    u, s = self.encode_block(u, s, block_message)
                    block = np.round(np.dot(u * s, vt))
                    block = block.clip(0, 255)
                    self.image[self.block_size * row_:self.block_size * (row_ + 1),
                               self.block_size * column_:self.block_size * (column_ + 1)] = block
        print('The picture is ready!\n')

    def image_saver(self):
        path = './output/images/' + time.asctime().replace(':', '-').replace(' ', '_') + '.bmp'
        image = Image.fromarray(self.image)
        image.save(path)

    def text_saver(self):
        path = './output/text/' + time.asctime().replace(':', '-').replace(' ', '_') + '.txt'
        file = open(path, 'w')
        file.write(self.message + '\n\n' + ''.join(self.binary_message))

    def encode_information(self):
        print('\n   - - - Information - - -\n')
        print('Columns protected     :', self.column_protected)
        print('Block size            :', self.block_size)
        print('Iterations            :', self.iterations)
        print('Redundancy            :', self.redundancy)
        print('Noise                 :', self.noise)
        print('\nImage size            : {0}x{1}'.format(self.image.shape[0], self.image.shape[1]))
        print('Message length [char] :', len(self.message))
        print('Message length [bit]  :', len(self.message) * 7)
        print('\n   - - - - - - - - - - - -\n')

    def decode_information(self):
        print('\n   - - - Information - - -\n')
        print('Columns protected     :', self.column_protected)
        print('Block size            :', self.block_size)
        print('Redundancy            :', self.redundancy)
        print('\n   - - - - - - - - - - - -\n')

    def execution(self):

        if self.method == 'Embed':
            print('\nSelected method: Embed')
            self.encode_information()
            self.encode_picture()
            self.image_saver()
            if self.noise:
                self.make_noise()
            self.encode_test()
            self.control()

        if self.method == 'Decode':
            print('\nSelected method: Decode')
            self.decode_information()
            self.decode_picture()
            self.text_saver()
