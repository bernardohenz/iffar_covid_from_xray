import os
from google_drive_downloader import GoogleDriveDownloader as gdd
# Fazendo download do dataset utilizado
gdd.download_file_from_google_drive(file_id='1x2S9h7ob6XXZxW8rays1dndrtL0G2tyJ',
                                    dest_path='./trained_model/trained_model.h5')

