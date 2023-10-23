import re


def hard_processing(text):
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def remove_unbalanced_brackets(text):
    # Remove unopened or unclosed brackets
    while '(' in text and ')' not in text[text.index('('):]:
        text = text.replace('(', '', 1)
    while ')' in text and '(' not in text[:text.rindex(')')]:
        text = text[:text.rindex(')')] + text[text.rindex(')') + 1:]
    return text


def preprocess_text(text):
    # Your existing preprocessing steps
    text = re.sub(r'http\S+|www.\S+', '', text)
    phone_regex = r'\(?\+?\d{0,3}\)?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{2}[-.\s]?\d{2}'
    text = re.sub(phone_regex, '', text)

    chars_to_remove = ('+;°#̂,ี¡”\x03ֹ≠้‑".‘ા·:⁉*->˗’√Ⓑ\u202c§ื}¿؛⏺≈̀|∩„\xad¸®¬ิ¦;“¥–€̆ึ⋅Ⓔ\u202d≥→←҆¯'
                       '⏳̶™…©»№\x9d℃\x96\x08?«็−ุ،‰~ู̧∆‼•\\^¨\u2060᾿ְ※<₽\x7f\u200f่[\x04]⃣\x0f\xa0'
                       '⏰\u200c%±″†(÷′Ⓕ́ु$՚`/\'∙{ा)₴£\u202aั⁾≡×&¤⊕↓⁽\x97̈؟\x98๊≤@↑˚ો્´!›‚∞‛\u200b\x01')
    text = re.sub('[' + re.escape(chars_to_remove) + ']', '', text)

    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r'@\w+', '', text)
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:t\.me\/\S+|telegram\.me\/\S+|telegram\.dog\/\S+)"
    text = re.sub(pattern, '', text)
    emoji_pattern = re.compile(
        pattern="["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r' +', ' ', text)
    text = remove_unbalanced_brackets(text)

    return text


def get_locations(row, with_text=False):
    text, locs = row[0], row[1]
    locs = [text[loc[0]:loc[1]] for loc in locs]
    return (text, locs) if with_text else locs


def update_locations(row, text_processor):
    text, locations = row
    location_texts = get_locations((text, locations))
    text = text_processor(text)

    new_locations = []
    offset = 0
    for loc_text in location_texts:
        clean_loc_text = text_processor(loc_text)
        start_idx = text.find(clean_loc_text, offset)
        if start_idx != -1:
            end_idx = start_idx + len(clean_loc_text)
            new_locations.append((start_idx, end_idx))
            offset = end_idx

    return text, new_locations


def clean_text_and_locations(row, text_processor=preprocess_text):
    index = 0
    text, locations = update_locations(row, text_processor)
    result: list[tuple[int, int]] = []

    while True:
        if index >= len(locations):
            break

        location = locations[index]
        location_text = text[location[0]:location[1]]
        try:
            if location_text.startswith('-'):
                if len(location_text) > 1:
                    result[-1] = (result[-1][0], location[1])
                else:
                    result[-1] = (result[-1][0], locations[index + 1][1])
                    index += 1
            elif location_text.endswith(('.', '. ', '—', '— ')):
                if location_text != '.' and location_text != 'вул.':
                    new_end = location_text.rfind('.')
                    new_end = len(location_text[:new_end].strip())
                    result.append((location[0], location[0] + new_end))
            elif ' і ' in location_text:
                loc1, loc2 = location_text.split(' і ')
                result.extend(
                    [
                        (location[0], location[0] + len(loc1)),
                        (location[1] - len(loc2), location[1])
                    ]
                )
            else:
                result.append(location)
        except Exception:
            result = [-1]

        index += 1
    return text, result
