
import json 
import random
import requests
import os
from nltk import word_tokenize, pos_tag, download
#Download necessary tokenizers
download("punkt", quiet=True)
download("averaged_perceptron_tagger", quiet=True)

from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
class SHAREAPI:
    def __init__(self):
        self.root_link = "https://07oyvkdcgg.execute-api.us-west-1.amazonaws.com/Prod"
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
            'api_method': 'predict_meme', #REQUIRED
            'template_id': template_id, 
            'num_boxes': num_boxes,  
            'init_text': init_text, 
            'beam_width': beam_width, 
            'max_output_length': max_output_length
        } 
        response = requests.get(f"{self.root_link}/predict", params=params).json()
        # print("Response from api call", response)
        return response['outputs']
    def generate_gpt2_comments(self, msg, num=1):
        global generator
        q = msg[(msg.index(" ")+1):]
        prompt = f"{q}\n RESPONSE: " 
        text = generator(prompt, max_length=len(prompt.split(" "))+100, num_return_sequences=num)
        return [text[i]['generated_text'][len(prompt):] for i in range(num)]
    # def _generate_gpt2_comments(self, prompt, max_length, num_return_sequences):
    #     """STIL NOT READY. DO NOT USE YET"""
    #     params = {
    #         'api_method': 'predict_gpt2', #REQUIRED
    #         'prompt': prompt, 
    #         'max_length': max_length,  
    #         'num_return_sequences': num_return_sequences, 
    #     } 
    #     response = requests.get(f"{self.root_link}/predict", params=params).json() #TODO: Handle when the api call times out
    #     print(response)
    #     return response['outputs']
if __name__ == '__main__':
    x = SHAREAPI()
    print(x.predict_meme_text(3218037, 2))