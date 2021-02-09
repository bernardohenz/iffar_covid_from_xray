import os
from google_drive_downloader import GoogleDriveDownloader as gdd
# Fazendo download do dataset utilizado
gdd.download_file_from_google_drive(file_id='1Ln3JVkZEQfGKzie5x39hPNCEcmNlp3eG',
                                    dest_path='./dataset/colab_covid.zip',
                                    unzip=True)

os.remove('./dataset/colab_covid.zip')