from pprint import pprint

import pandas as pd

from topic_modeling import TopicModeling

if __name__ == '__main__':
    df = pd.read_csv('../data/processed_dataset.csv', usecols=['channelname', 'token_text', 'content'])
    modeler = TopicModeling()

    results = {}
    for channel in df['channelname'].unique():
        cluster = df[df['channelname'] == channel]['token_text']
        try:
            pred = modeler.fit_model(cluster)
            results[channel] = dict(pred.items())
        except Exception as e:
            pass
    pprint(results)
