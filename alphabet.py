import os
import glob

datas = glob.glob(r'/home/yangna/deepblue/OCR/EAST2/ICDAR_2015/ch4_training_localization_transcription_gt/*.txt')
alphabet = []

for txtname in datas:

    with open(txtname) as f:
        perdata = f.readlines()

        for pd in perdata:
            pd = pd.rstrip().split(',')
            if pd[-1] != '###':
                alphabet += pd[-1]

temp = ''.join(set(alphabet))

with open('./ICDAR_2015/icdar2015_alphabet.txt', 'w') as outf:
    outf.write(temp)