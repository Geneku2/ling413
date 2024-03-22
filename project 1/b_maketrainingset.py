import a_datacleaning as a
import pandas as pd

windows_size = 2

file_path = "en_gum-ud-train.conllu"
all_words, no_det = a.create_data_structures(file_path)

# word, pos, word @ loc - n, pos @ loc - n, word @ loc - n + 1, pos @ loc n + 1...
heading = ["word", "word_tag"]
for i in range(-windows_size,windows_size+1):
    if i != 0:
        heading.append(f"word_at_pos{i}")
        heading.append(f"word_tag_at_pos{i}")

all_words_data = []
all_words_data.append(heading)
no_det_data = []
no_det_data.append(heading)


for i in range(len(all_words)):
    word_window = [all_words[i][0],all_words[i][1]]
    for j in range(-windows_size,windows_size+1):
        if (i + j) < 0 or (i + j) >= len(all_words):
            word_window.append("NaN")
            word_window.append("NaN")
        elif j != 0:
            word_window.append(all_words[i+j][0])
            word_window.append(all_words[i+j][1])

    all_words_data.append(word_window)

for i in range(len(no_det)):
    word_window = [no_det[i][0],no_det[i][1]]
    for j in range(-windows_size,windows_size+1):
        if (i + j) < 0 or (i + j) >= len(no_det):
            word_window.append("NaN")
            word_window.append("NaN")
        elif j != 0:
            word_window.append(no_det[i+j][0])
            word_window.append(no_det[i+j][1])

    no_det_data.append(word_window)

all_words_df = pd.DataFrame(all_words_data, columns = all_words_data[0], )
no_det_df = pd.DataFrame(no_det_data, columns = no_det_data[0])

print(all_words_df)
print(no_det_df)