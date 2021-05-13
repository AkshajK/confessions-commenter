
import json 
import random
import requests
import os
# from nltk import word_tokenize, pos_tag, download

class SHAREAPI:
    def __init__(self):
        self.root_link = " https://07oyvkdcgg.execute-api.us-west-1.amazonaws.com/Prod" 
    def get_generatable_memes_info(self): 
        with open(os.path.join(os.path.dirname(__file__), "meme_data.json")) as f: #TODO: Maybe make an API call here?
            res = json.load(f)
        return res
    def get_initial_word_from_text(self, text, method="long_words"): #TODO: Maybe make an API call here?
        if method == "long_words":
            words = text.split(" ")
            important_words = [word for word in words if len(word) > 4]
            if len(important_words) > 0:
                return random.choice(important_words)
            else:    
                return ""
        elif method == "ntlk_verbs":
            # From https://stackoverflow.com/questions/5404243/extracting-english-verbs-from-a-given-text/5410074
            from nltk import word_tokenize, pos_tag, download
            # try:
                # nltk.data.find("tokenizers/punkt")
            # except LookupError:
            download("punkt", quiet=True)
            download("averaged_perceptron_tagger", quiet=True)
            tokens = word_tokenize(text)
            pos_tagged_tokens = pos_tag(tokens)
            verbs = [word[0] for word in pos_tagged_tokens if word[1].startswith('V')]
            if len(verbs) > 0:
                return random.choice(verbs)
            else:
                return "" 
        else:
            return ""
    def predict_meme_text(self, template_id, num_boxes, init_text = "", beam_width=1, max_output_length=140):
        params = {
            'template_id': template_id, 
            'num_boxes': num_boxes,  
            'init_text': init_text, 
            'beam_width': beam_width, 
            'max_output_length': max_output_length
        } 
        response = requests.get(f"{self.root_link}/predict", params=params).json()
        # print("Response from api call", response)
        if 'outputs' in response:
            return response['outputs']
        return 'Error'

if __name__ == '__main__':
    x = SHAREAPI()
    print(x.predict_meme_text(3218037, 2))