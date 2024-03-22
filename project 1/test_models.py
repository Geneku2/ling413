import nltk
import conllu
import pickle
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

# Load the POS tagger model
with open('det_tagger_model.pkl', 'rb') as f:
    det_pos_tagger = pickle.load(f)

with open('ndet_tagger_model.pkl', 'rb') as f:
    ndet_pos_tagger = pickle.load(f)

# Read the .conllu file and extract sentences with POS tags
def read_conllu_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
        for sent in conllu.parse(data):
            tokens = []
            for token in sent:
                tokens.append((token['form'], token['upostag']))
            sentences.append(tokens)
    return sentences

# Perform POS tagging on the testing data
def test_pos_tagger(test_sentences, tagger):
    predicted_tags = []
    true_tags = []
    for sent in test_sentences:
        tokens, tags = zip(*sent)
        predicted_tags.extend(tagger.tag(tokens))
        true_tags.extend(tags)
    return true_tags, predicted_tags

def test_model(data, model, filename, name):
    f = open(filename, "a")
    true_tags, predicted_tags = test_pos_tagger(data, model)
    
    predicted_tags = [tag for _, tag in predicted_tags]
    
    # Calculate precision, recall, and accuracy
    precision = precision_score(true_tags, predicted_tags, average='weighted', zero_division=1)
    recall = recall_score(true_tags, predicted_tags, average='weighted', zero_division=1)
    precisionu = precision_score(true_tags, predicted_tags, average='macro', zero_division=1)
    recallu = recall_score(true_tags, predicted_tags, average='macro', zero_division=1)
    accuracy = accuracy_score(true_tags, predicted_tags)
    
    
    # Print precision, recall, and accuracy
    f.write(f"{name} Precision (Weighted): {precision}\n")
    f.write(f"{name} Recall (Weighted): {recall}\n")
    f.write(f"{name} Precision (Unweighted): {precisionu}\n")
    f.write(f"{name} Recall (Unweighted): {recallu}\n")
    f.write(f"{name} Accuracy: {accuracy}\n")
    print(name)
    # Print confusion matrix
    conf_matrix = confusion_matrix(true_tags, predicted_tags)
    unique_tags = sorted(set(true_tags + predicted_tags))
    f.write(f"{name} Confusion Matrix:\n")
    header = " " * 5 + " ".join("{:<4}".format(tag[:4]) for tag in unique_tags)
    f.write(f"{header}\n")
    for i, row in enumerate(conf_matrix):
        label = unique_tags[i][:4]
        matrix_row = " ".join("{:<4}".format(item) for item in row)
        f.write(f"{label:<5}{matrix_row}\n")

    f.write("\n-----------------------------------------------------------------\n\n")

if __name__ == "__main__":
    # Path to the testing .conllu file
    test_conllu_file = "en_gum-ud-test.conllu"
    
    # Read testing .conllu file and extract sentences with POS tags
    test_sentences = read_conllu_file(test_conllu_file)
    
    no_det_test_sentences = []
    for sentence in test_sentences:
        sen = []
        for elem in sentence:
            if elem[1] != "DET":
                sen.append(elem)
            no_det_test_sentences.append(sen)

    # Test the POS tagger on the testing data
    test_model(test_sentences, det_pos_tagger, "results.txt", "Det On Det")
    test_model(test_sentences, ndet_pos_tagger, "results.txt", "Non-Det On Det")
    test_model(no_det_test_sentences, det_pos_tagger, "results.txt", "Det On Non-Det")
    test_model(no_det_test_sentences, ndet_pos_tagger, "results.txt", "Non-Det On Non-Det")