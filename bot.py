from telebot import *
import urllib.request
import torch
import torchvision
from PIL import Image

from src.model.dataset import process_img_to_show, ImageDataset
from src.model.generator import Generator

def generate_image(images_path: str):
    data = ImageDataset(images_path)
    model = Generator()
    model.load_state_dict(torch.load(
        'model_weights/gen_XY',
        map_location=torch.device('cpu')
    ))
    model.eval()

    output = model(data[0]) * 0.5 + 0.5  # denormalize
    torchvision.utils.save_image(output, 'data/inference/output_image.jpg')

class GenBot:
    def __init__(self, token: str):
        self.bot = TeleBot(token)

        @self.bot.message_handler(commands=["start"])
        def send_greetings(message):
            self.bot.send_message(
                message.chat.id,
                'Hi! Send me your selfies',
            )

        @self.bot.message_handler(content_types=['photo'])
        def send_generation(message):
            photo_id = message.photo[-1].file_id
            file_info = self.bot.get_file(photo_id)
            file_url = (
                'https://api.telegram.org/file/'
                f'bot{token}/{file_info.file_path}'
            )
            urllib.request.urlretrieve(
                file_url, 'data/inference/input_image.jpg'
            )
            generate_image('inference')
            img = (
                Image.open('data/inference/output_image.jpg')
                .resize((256, 256)).convert('RGB')
            )
            self.bot.send_photo(message.chat.id, img)

    def start(self):
        self.bot.polling()
