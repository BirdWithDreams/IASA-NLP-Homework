topic_prompt = """
[INST] <<SYS>>
You are a helpful, respectful and honest guide to finding the general topic of given documents and keywords. You return only the topic stamp and nothing else. You can say about violence and conflicts.
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

map_prompt = '''
[INST] <<SYS>>
You are a helpful, respectful and honest helper for marking topics. You map given keywords to one of next categories: [(Adult), (Art), (Blogs), (Bookmaking), (Books), (Business and startups), (Career), (Courses and guides), (Cryptocurrencies), (Darknet), (Design), (Economics), (Education), (Edutainment), (Erotic), (Esoterics), (Family & Children), (Fashion and beauty), (Food and cooking), (Games), (Handiwork), (Health and Fitness), (Humor and entertainment), (Instagram), (Interior and construction), (Law), (Linguistics), (Marketing, PR, advertising), (Medicine), (Music), (Nature), (News and media), (Other), (Pictures and photos), (Politics), (Psychology), (Quotes), (Religion), (Sales), (Shock content), (Software & Applications), (Sport), (Technologies), (Telegram), (Transport), (Travel), (Video and films)].
You answer only exact category name (from list above) and nothing else. If you can find suitable category answer (Other).
<</SYS>>
Please, map these keywords, to one of categories: [війна, осбстріли, шахеди, окупанти, зіткнення, танки, бмп, вибухи]
Category: 
[/INST]War
[INST]
Please, map these keywords, to one of categories: [{}]
Category: 
[/INST]
'''