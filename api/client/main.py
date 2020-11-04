import csv
import requests
from time import sleep, time
from datetime import timedelta, datetime

from tqdm import tqdm

count = 0
t1 = time()
if __name__ == "__main__":
    dataset_ = open("../../dataset/custom_dataset/since_january.csv")

    reader = csv.reader(dataset_, delimiter=',', quotechar='"')

    queries = []
    for idx, line in enumerate(reader):
        if len(line) < 3 or line[1] == "" or line[0] == "":
            continue

        try:
            queries.append((line[0], line[1], datetime.strptime(line[1], "%Y-%m-%d %H:%M:%S")))
        except Exception:
            pass

    queries = sorted(queries, key=lambda x: x[2])
    for idx, (query, date, _) in tqdm(enumerate(queries)):
        url = 'http://127.0.0.1:5000/clustering'
        body = {
            'title': query,
            'timestamp': date,
        }
        count += 1
        x = requests.post(url, json=body)

# print(count)
elapsed = time() - t1
print('Elapsed time is %f seconds.' % elapsed)
