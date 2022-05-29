import io
import itertools
import csv
import datetime

EMBEDDINGS_FILE = 'wiki-news-300d-1M.vec'
N_LIMIT = 2 ** 20

EMOJI_DATASET_INPUT = 'archive/full_emoji.csv'
DATASET_OUTPUT = 'dataset.csv'


def description_to_words_list(s):
    s = s.replace(',', '').replace(':', '').replace('?', '').replace('!', '')
    s = s.replace('“', '').replace('”', '').replace('(', '').replace(')', '')
    s = s.replace('’s', '').replace('⊛', '').replace('o’clock', 'oclock').replace('-', ' ').strip()
    return list(filter(lambda s: s.strip() != '', s.split(' ')))


def get_average_embedding(data, description_words):
    def add_vectors(a, b):
        return list(map(lambda pair: pair[0] + pair[1], zip(a, b)))

    embeddings = list(map(lambda word: data[word], description_words))
    res = embeddings[0]
    for embedding in embeddings[1:]:
        res = add_vectors(res, embedding)
    assert len(res) == 300
    return res


# building set of interesting words
print(f'building set of important words ...')
start_time = datetime.datetime.now()
interesting_words = set()
with open(EMOJI_DATASET_INPUT) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader) # skip title
    for row in csv_reader:
        interesting_words.update(description_to_words_list(row[3]))
print(f'important words set was built in {datetime.datetime.now() - start_time}')


# loading embedding dataset
print(f'loading embedding dataset (this can take a while) .....')
start_time = datetime.datetime.now()

fin = io.open(EMBEDDINGS_FILE, 'r', encoding='utf-8', newline='\n', errors='ignore')
n, d = map(int, fin.readline().split())
data = {}
for line in itertools.islice(fin, N_LIMIT):
    tokens = line.rstrip().split(' ')
    if tokens[0] in interesting_words:
        data[tokens[0]] = list(map(float, tokens[1:]))

print(f'embedding dataset loaded in {datetime.datetime.now() - start_time}')


id_to_embeddings = []
skipped = []


# converting emoji descriptions to embeddings 

with open(EMOJI_DATASET_INPUT) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader) # skip title
    for row in csv_reader:
        skip = False
        index = row[0]
        description_words = description_to_words_list(row[3])
        for word in description_words:
            if word not in data:
                print(f'! |{word}| not found in data. Skipping {index} {description_words}')
                skip = True
                break
        if skip:
            skipped.append((index, description_words))
        else:
            id_to_embeddings.append((index, get_average_embedding(data, description_words)))

print(f'total skipped: {len(skipped)} {skipped}')

# printing to DATASET_OUTPUT csv file
with open(DATASET_OUTPUT, 'w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for id_to_embedding in id_to_embeddings:
        csv_writer.writerow(list(id_to_embedding))
