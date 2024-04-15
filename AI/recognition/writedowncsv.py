import csv
import os
import numpy as np

landmarks = ['class']
# 33, 502, 544 pose = 33, hand = 42, face = 478
for val in range(1, 76):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
with open('coords.csv', mode='a', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)