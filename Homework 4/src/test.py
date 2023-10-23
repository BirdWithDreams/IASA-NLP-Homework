import os
from datetime import datetime

import pandas as pd
import spacy
import torch

import models
from post_processing import predict_text_spans
from text import get_locations, preprocess_text, hard_processing

tokenizer = spacy.load("xx_ent_wiki_sm", disable=["tagger", "parser", "ner", "textcat"])


def predict_locations(model_class, state_dict, series, vocab, tokenize, threshold):
    locations = []
    input_df = pd.DataFrame(
        {
            'text': series.tolist(),
            'loc_markers': [[]] * len(series)
        }
    )

    token_spans = predict_text_spans(model_class, state_dict, input_df, vocab, tokenizer, threshold)

    for sentence, spans in zip(series, token_spans):
        locations.append(get_locations((sentence, spans)))
    return locations


unseen_df = pd.read_csv('../datasets/test.csv')
unseen_df.head()

unseen_df['clean_text'] = unseen_df['text'].apply(hard_processing)

# if __name__ == '__main__':
#     path = '../checkpoints/2023-10-22 18.54.33/'
#
#     vocab = torch.load(path + 'vocab.voc')
#     state_dict = torch.load(path + 'Model 0.9092 0.55.pth')
#
#     model = state_dict['model']
#
#     locs = predict_locations(getattr(models, model['name']), model, unseen_df['clean_text'], vocab, tokenizer, 0.55)
#
#     unseen_df['locations'] = pd.Series(locs)
#
#     df_to_save = unseen_df.drop(columns=['clean_text', 'text'])
#     df_to_save.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    lover_date_str = "2023-10-22 19.30.00"
    lover_date = datetime.strptime(lover_date_str, "%Y-%m-%d %H.%M.%S")

    upper_date_str = "2023-10-22 22.00.00"
    upper_date = datetime.strptime(upper_date_str, "%Y-%m-%d %H.%M.%S")

    for folder_name in os.listdir('../checkpoints'):
        try:
            folder_date = datetime.strptime(folder_name, "%Y-%m-%d %H.%M.%S")
            if upper_date < folder_date:
                path = '../checkpoints/' + folder_name
                vocab = torch.load(path + '/vocab.voc')
                for model_name in os.listdir(path):
                    if model_name.endswith('.pth'):
                        state_dict = torch.load(path + '/' + model_name)
                        break

                model = state_dict['model']

                locs = predict_locations(
                    getattr(models, model['name']), model, unseen_df['clean_text'], vocab, tokenizer,
                    float(model_name.removesuffix('.pth').split(' ')[-1])
                    )

                unseen_df['locations'] = pd.Series(locs)

                df_to_save = unseen_df.drop(columns=['clean_text', 'text'])
                df_to_save.to_csv(
                    f'{model["name"]} {model["model_params"]["num_layers"]}x{model["model_params"]["rnn_channels"]} sche-er2 submission.csv',
                    index=False
                    )

        except ValueError:
            pass
