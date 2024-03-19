from selfie2anime.bot import GenBot
import os

if __name__ == '__main__':
    token = os.getenv('BOT_TOKEN')
    bot = GenBot(token)
    bot.start()
