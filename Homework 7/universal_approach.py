from collections import defaultdict, Counter
from pprint import pprint

import pandas as pd
import spacy
import torch
import transformers
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from langid import langid
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from torch import bfloat16
from transformers import TextGenerationPipeline
from umap import UMAP

torch.manual_seed(42)
torch.backends.cuda.deterministic = True


class TopicModeling:
    _base_prompt = """
[INST] <<SYS>>
You are a helpful, respectful and honest guide to finding the general topic of given documents and keywords. You return only the topic stamp and nothing else. You don't give your opinion.
<</SYS>>
I have a topic that contains the following documents:
- Traditional diets in most cultures consisted mostly of plant foods with a small amount of meat, but with the development of industrialized meat production and factory farming, meat became a staple food.
- Meat, but especially beef, is the best food in terms of emissions.
- Eating meat does not make you a bad person, not having meat does not make you a good person.

The topic is described by the following keywords: 'meat, beef, eat, eat, emissions, steak, food, health, processed, chicken'.

Based on the topic information above, create a short tag for that topic. Make sure you only return the label and nothing else.

[/INST] The impact of eating meat on the environment
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the topic information above, create a short tag for that topic. Make sure you only return the label and nothing else.
[/INST]
"""

    def __init__(
            self,
            tokenizers: dict[str, spacy.language.Language] = None,

    ):
        self.tokenizers = tokenizers or {
            'uk': spacy.load("uk_core_news_sm"),
            'ru': spacy.load("ru_core_news_sm")
        }
        self.topic_model = None
        self.cluster_size = None

    def clean_cluster(self, cluster):
        def get_most_common_words(series, num_common_words=None):
            combined_text = series.str.cat(sep=' ')

            words = combined_text.split()
            word_counts = Counter(words)
            if num_common_words is None:
                num_common_words = int(len(word_counts) * 0.001)

            # Get the most common words
            most_common_words = word_counts.most_common(num_common_words)

            return [word[0] for word in most_common_words]

        new_stop_words = set(get_most_common_words(cluster))
        lang, _ = langid.classify(cluster.sample(1).item())

        if lang not in ['uk', 'ru']:
            raise Exception()

        clean_texts = cluster.apply(
            lambda x: ' '.join(
                [token.text for token in self.tokenizers[lang](x) if
                 token.text not in new_stop_words and not token.is_stop]
                )
        )
        return clean_texts

    def fit_model(self, cluster):
        cluster = self.clean_cluster(cluster)
        cluster = cluster.to_list()# list(cluster)
        self.cluster_size = len(cluster)

        if self.topic_model is None:
            self.init_model()

        topics, probs = self.topic_model.fit_transform(cluster)
        df = DataFrame(
            {
                'topic': topics,
                'text': cluster
            }
        )
        self.grouped_docs = df.groupby('topic').aggregate(lambda x: list(x))

        topics_df = self.topic_model.get_topic_info()
        topics = self.get_topics_and_subtopics(topics_df, representation_column='MMR')

        return topics, topics_df

    def init_model(
            self,
            embedding_model=None,
            umap_model=None,
            hdbscan_model=None,
            ctfidf_model=None,
            representation_model=None,
    ):
        cluster_size = self.cluster_size or 750
        cluster_size = max(cluster_size, 750)
        print(cluster_size)

        if embedding_model is None:
            embedding_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
        elif isinstance(embedding_model, str):
            embedding_model = SentenceTransformer(embedding_model)
        elif not isinstance(embedding_model, SentenceTransformer):
            raise Exception()

        if umap_model is None:
            umap_model = UMAP(
                n_neighbors=int(0.02 * cluster_size),
                n_components=16,
                min_dist=0.0,
                metric='cosine',
                random_state=42,
            )
        elif not isinstance(umap_model, UMAP):
            raise Exception()

        if hdbscan_model is None:
            hdbscan_model = HDBSCAN(
                min_cluster_size=int(0.01 * cluster_size),
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
            )
        elif not isinstance(hdbscan_model, HDBSCAN):
            raise Exception()

        if ctfidf_model is None:
            ctfidf_model = ClassTfidfTransformer(
                bm25_weighting=True
                # reduce_frequent_words=True
            )
        elif not isinstance(ctfidf_model, ClassTfidfTransformer):
            raise Exception()

        if representation_model is None:
            keybert = KeyBERTInspired()
            mmr = MaximalMarginalRelevance(diversity=0.3)

            generator = self.init_llama()
            prompt = self._base_prompt

            llama2 = TextGeneration(generator, prompt=prompt)
            representation_model = {
                "KeyBERT": keybert,
                "Llama2": [mmr, llama2],
                "MMR": mmr,
            }
        elif not isinstance(representation_model, ClassTfidfTransformer):
            raise Exception()

        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,

            verbose=True,
        )

    @staticmethod
    def get_topic(gen: TextGenerationPipeline, keywords: list[str]):
        topic_prompt = f'''
[INST] <<SYS>>
You are a helpful, respectful and honest helper for marking topics. You map given keywords to one of next categories: [(Adult), (Art), (Blogs), (Bookmaking), (Books), (Business and startups), (Career), (Courses and guides), (Cryptocurrencies), (Darknet), (Design), (Economics), (Education), (Edutainment), (Erotic), (Esoterics), (Family & Children), (Fashion and beauty), (Food and cooking), (Games), (Handiwork), (Health and Fitness), (Humor and entertainment), (Instagram), (Interior and construction), (Law), (Linguistics), (Marketing, PR, advertising), (Medicine), (Music), (Nature), (News and media), (Other), (Pictures and photos), (Politics), (Psychology), (Quotes), (Religion), (Sales), (Shock content), (Software & Applications), (Sport), (Technologies), (Telegram), (Transport), (Travel), (Video and films)].
You answer only exact category name (from list above) and nothing else.
<</SYS>>
Keywords: [війна, осбстріли, шахеди, окупанти, зіткнення, танки, бмп]
Category: 
[/INST]War
[INST]
Keywords: [{', '.join(keywords)}]
Category: 
[/INST]
'''
        gen_text = gen(topic_prompt)[0]['generated_text']
        topic = gen_text.replace(topic_prompt, '') + '\n\n'
        topic = topic.partition('\n\n')[0]
        topic = topic.strip()
        return topic

    def get_topics_and_subtopics(self, topic_df, representation_column='Representation', llm_column='Llama2'):
        gen = self.topic_model.representation_model['Llama2'][1].model
        result_topics = defaultdict(list)

        for index, row in topic_df.iterrows():
            big_topic = self.get_topic(gen, row[representation_column])
            sub_topic = row[llm_column][0]
            sub_topic = sub_topic.replace('#', '')
            sub_topic = sub_topic.removeprefix('"').removeprefix("'")
            sub_topic = sub_topic.removesuffix('"').removesuffix("'")

            result_topics[big_topic].append(sub_topic)
        return result_topics

    def init_llama(
            self,
            model_id='meta-llama/Llama-2-7b-chat-hf',
            pipeline_kwargs: dict = None,
    ):

        if pipeline_kwargs is None:
            pipeline_kwargs = {
                'temperature': 0.1,
                'max_new_tokens': 15,
                'repetition_penalty': 1.1,
                'top_p': 0.9
            }

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

        # Llama 2 Model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        model.eval()

        generator = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            **pipeline_kwargs
        )

        return generator

    def visualize_documents(self, docs):
        docs = list(docs)
        llama2_labels = [label[0][0].split("\n")[0] for label in
                         self.topic_model.get_topics(full=True)["Llama2"].values()]
        self.topic_model.set_topic_labels(llama2_labels)

        embeddings = self.topic_model.embedding_model.embedding_model.encode(docs)
        reduced_embeddings = UMAP(
            n_neighbors=int(0.02 * len(docs)), n_components=2, min_dist=0.0, metric='cosine', random_state=42
        ).fit_transform(
            embeddings
        )

        return self.topic_model.visualize_documents(
            docs, reduced_embeddings=reduced_embeddings, hide_annotations=True, hide_document_hover=False,
            custom_labels=True
        )


if __name__ == '__main__':
    df = pd.read_csv('./data/processed_dataset.csv', usecols=['channelname', 'token_text', 'content'])
    h = TopicModeling()
    # grouped = df.groupby('channelname')
    # sorted_groups = sorted(grouped, key=lambda x: len(x[1]), reverse=True)
    # first_key = sorted_groups[0][0]  # get the first key in the groups
    # print(first_key)
    # first_group = sorted_groups[0][1]['token_text']
    first_group = df[df['channelname'] == 'novynylive']['token_text']

    pred = h.fit_model(first_group)
    pprint(pred[0])
    pprint(pred[1])
    # h.visualize_documents(first_group['token_text']).show()
