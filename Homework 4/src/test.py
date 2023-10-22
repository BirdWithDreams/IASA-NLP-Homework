import pandas as pd
import spacy
import torch
from nltk.tokenize import sent_tokenize

from models import UniversalRNN
from data import TextTokenDataset
from post_processing import predict_text_spans
from text import get_locations, preprocess_text

tokenizer = spacy.load("xx_ent_wiki_sm", disable=["tagger", "parser", "ner", "textcat"])


def process_document(model_class, state_dict, series, vocab, tokenize):
    locations = []
    for doc in series:
        sentences = sent_tokenize(doc)
        input_df = pd.DataFrame(
            {
                'text': sentences,
                'loc_markers': [[]] * len(sentences)
            }
        )

        # dataset = TextTokenDataset(
        #     input_df['text'].tolist(),
        #     input_df['loc_markers'].tolist(),
        #     vocab,
        #     tokenize
        # )
        token_spans = predict_text_spans(model_class, state_dict, input_df, vocab, tokenizer)

        current_locations = []
        for sentence, spans in zip(sentences, token_spans):
            # if spans:
            #     for span in spans:
            #         current_locations.append(sentence[span[0]: span[1]])
            current_locations.extend(get_locations((sentence, spans)))
        locations.append(current_locations)
    return locations


unseen_df = pd.read_csv('../datasets/test.csv')
unseen_df.head()

unseen_df['clean_text'] = unseen_df['text'].apply(preprocess_text)

if __name__ == '__main__':
    path = '../checkpoints/2023-10-22 10.25.34/'
    vocab = torch.load(path + 'vocab.voc')
    state_dict = torch.load(path + 'Model 0.9527.pth')

    locs = process_document(UniversalRNN, state_dict, unseen_df['clean_text'], vocab, tokenizer)

    unseen_df['locations'] = pd.Series(locs)

    df_to_save = unseen_df.drop(columns=['clean_text', 'text'])
    df_to_save.to_csv('submission.csv', index=False)
