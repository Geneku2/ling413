import nltk
import conllu
import pickle

# Download the necessary resources for the POS tagger
nltk.download('universal_tagset')

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

# Train a POS tagger using the given sentences
def train_pos_tagger(sentences):
    # Default tagger
    default_tagger = nltk.DefaultTagger('NOUN')
    # Unigram tagger
    unigram_tagger = nltk.UnigramTagger(sentences, backoff=default_tagger)
    # Bigram tagger
    bigram_tagger = nltk.BigramTagger(sentences, backoff=unigram_tagger)
    return bigram_tagger

# Perform POS tagging on a given sentence
def pos_tag_sentence(sentence, tagger):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = tagger.tag(tokens)
    return pos_tags

# Main function
if __name__ == "__main__":
    # Path to the .conllu file
    conllu_file = "en_gum-ud-train.conllu"
    
    # Read .conllu file and extract sentences with POS tags
    training_sentences = read_conllu_file(conllu_file)
    no_det_sentences = []
    for sentence in training_sentences:
        sen = []
        for elem in sentence:
            if elem[1] != "DET":
                sen.append(elem)
            no_det_sentences.append(sen)
    
    # Train a POS tagger
    det_pos_tagger = train_pos_tagger(training_sentences)
    ndet_pos_tagger = train_pos_tagger(no_det_sentences)

    with open('det_tagger_model.pkl', 'wb') as f:
        pickle.dump(det_pos_tagger, f)

    with open('ndet_tagger_model.pkl', 'wb') as f:
        pickle.dump(ndet_pos_tagger, f)