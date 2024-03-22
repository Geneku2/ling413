from conllu import parse
import nltk
from nltk.tag import hmm

# Function to read the .conllu file and create the data structures
def create_data_structures(file_path):
    words = []  # Data structure to store all words
    words_no_det = []  # Data structure to store words without determiners

    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Parse the .conllu file
    sentences = parse(data)

    # Iterate through each sentence to extract words and determiners
    for sentence in sentences:
        words_in_sentence = []  # Words in the current sentence
        words_in_sentence_no_det = []  # Words in the current sentence without determiners
        for token in sentence:
            if token['upos'] != 'PUNCT':  # Exclude punctuation
                # Check if the token is a determiner
                if token['upos'] != 'DET':
                    words_in_sentence_no_det.append((token["form"], token["upos"]))
                words_in_sentence.append((token["form"], token["upos"]))
            elif token['form'] == '.':
                # Add '.' punctuation if it ends the sentence
                words_in_sentence.append((token["form"], token["upos"]))
                words_in_sentence_no_det.append((token["form"], token["upos"]))

        # Append the words of the current sentence to the data structures
        if words_in_sentence:
            words.extend(words_in_sentence)
        if words_in_sentence_no_det:
            words_no_det.extend(words_in_sentence_no_det)

    return words, words_no_det

if __name__ == "__main__":

    # Main function to demonstrate the usage
    file_path = "en_gum-ud-train.conllu"
    all_words, words_no_det = create_data_structures(file_path)

    print("All words:")
    print(all_words[100:130])  # Print the some words
    print("----------------------------------------------------")
    print("Words without determiners:")
    print(words_no_det[100:130])  # Print the first some words without determiners
