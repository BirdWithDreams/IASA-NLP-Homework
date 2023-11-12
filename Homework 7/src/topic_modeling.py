from collections import defaultdict, Counter

import pandas as pd
import spacy
import transformers
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from langid import langid
from sentence_transformers import SentenceTransformer
from torch import bfloat16
from transformers import TextGenerationPipeline
from umap import UMAP

import prompts


class TopicModeling:
    def __init__(
            self,
            tokenizers: dict[str, spacy.language.Language] = None,
            embedding_model=None,
            ctfidf_model=None,
            representation_model=None,
    ):
        self.tokenizers = tokenizers or {
            'uk': spacy.load("uk_core_news_sm"),
            'ru': spacy.load("ru_core_news_sm")
        }
        self.topic_model = None
        self.cluster_size = None

        self.embedding_model = None
        self.umap = None
        self.hdbscan = None
        self.ctfidf_model = None
        self.representation_model = None

        self._init_constant_parts(embedding_model, ctfidf_model, representation_model)

    def _init_constant_parts(
            self,
            embedding_model=None,
            ctfidf_model=None,
            representation_model=None,
    ):
        if embedding_model is None:
            embedding_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
        elif isinstance(embedding_model, str):
            embedding_model = SentenceTransformer(embedding_model)
        elif not isinstance(embedding_model, SentenceTransformer):
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

            llama2 = TextGeneration(generator, prompt=prompts.topic_prompt)
            representation_model = {
                "KeyBERT": keybert,
                "Llama2": [mmr, llama2],
                "MMR": mmr,
            }

        self.embedding_model = embedding_model
        self.ctfidf_model = ctfidf_model
        self.representation_model = representation_model

    def _init_inconstant_parts(self, cluster_size):
        if self.umap is None:
            self.umap = UMAP(
                n_neighbors=int(0.02 * cluster_size),
                n_components=32,
                min_dist=0.0,
                metric='cosine',
            )
        elif not isinstance(self.umap, UMAP):
            raise Exception()

        if self.hdbscan is None:
            self.hdbscan = HDBSCAN(
                min_cluster_size=int(0.01 * cluster_size),
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
            )
        elif not isinstance(self.hdbscan, HDBSCAN):
            raise Exception()

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
                 token.text not in new_stop_words]
            )
        )
        return clean_texts

    def fit_model(self, cluster):
        cluster = self.clean_cluster(cluster)
        cluster = cluster.to_list()  # list(cluster)
        cluster_size = len(cluster)

        self.init_model(cluster_size=cluster_size)

        topics, probs = self.topic_model.fit_transform(cluster)
        df = pd.DataFrame(
            {
                'topic': topics,
                'text': cluster
            }
        )
        self.grouped_docs = df.groupby('topic').aggregate(lambda x: list(x))

        topics_df = self.topic_model.get_topic_info()
        self.topic_info = topics_df
        topics = self.get_topics_and_subtopics(topics_df, representation_column='MMR')

        return topics

    def init_model(
            self,
            umap_model=None,
            hdbscan_model=None,
            cluster_size=0
    ):
        if umap_model is not None:
            self.umap = umap_model

        if hdbscan_model is not None:
            self.hdbscan = hdbscan_model

        cluster_size = max(cluster_size, 750)

        if cluster_size < 150:
            raise Exception("Cluster too small for clustering")

        self._init_inconstant_parts(cluster_size)

        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap,
            hdbscan_model=self.hdbscan,
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model,

            verbose=True,
        )

    @staticmethod
    def get_topic(gen: TextGenerationPipeline, keywords: list[str]):
        prompt = prompts.map_prompt.format(', '.join(keywords))
        gen_text = gen(prompt)[0]['generated_text']
        topic = gen_text.replace(prompt, '') + '\n\n'
        topic = topic.partition('\n\n')[0]
        topic = topic.strip()
        return topic

    def get_topics_and_subtopics(self, topic_df, representation_column='Representation', llm_column='Llama2'):
        gen = self.topic_model.representation_model['Llama2'][1].model
        result_topics = defaultdict(list)

        for index, row in topic_df.iterrows():
            if index == 0:
                continue
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
