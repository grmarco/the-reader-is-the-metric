import os
import random
import string
import csv
import shutil

def random_string(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

source_folder = 'D:/Dropbox/18 Creativity/AI-Art/StoriesAndChatbotting/ptero-project/raw-corpus'  # Replace with the path to your folder with text files
destination_folder = 'D:/Dropbox/18 Creativity/AI-Art/StoriesAndChatbotting/ptero-project/anonymized-corpus-grouped'  # Replace with the path to the folder where you want the anonymized files

os.makedirs(destination_folder, exist_ok=True)

mapping_file = 'mapping-grouped.csv'

with open(mapping_file, mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Original', 'Anonymized']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            group_name = 'group'+filename[-5]
            group_path = os.path.join(destination_folder, group_name)
            new_filename = f'{random_string(4)}.txt'  # You can change the length of the random string to your preference
            if not ( os.path.exists(group_path) ): 
                os.makedirs(group_path)
            shutil.copy2(os.path.join(source_folder, filename), os.path.join(group_path, new_filename))
            writer.writerow({'Original': filename, 'Anonymized': new_filename})