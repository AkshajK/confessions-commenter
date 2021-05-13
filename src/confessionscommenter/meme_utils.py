import requests
import urllib 
import shutil
import os
from rich import print
import json
from confessionscommenter.api_utils import SHAREAPI
from confessionscommenter.image_clipboard import copy_meme_to_clipboard, save_image_locally
share_api = SHAREAPI()
class MemeGenerator:
    def __init__(self, username="mit_meme_creator", password="mit_meme_password"):
        self.username = username
        self.password = password
        self.api_root = "https://api.imgflip.com"
    def get_generatable_memes_info(self): 
        return share_api.get_generatable_memes_info()
    def get_popular_memes(self):
        data = requests.get(f"{self.api_root}/get_memes").json()
        return data['data']['memes']
    def create_meme(self, meme_id, text_list, save_to_clipboard=True):
        boxes = {f"boxes[{i}][text]": text for i, text in enumerate(text_list)}
        data = {
            'template_id': meme_id, #drake hotline bling
            'username': self.username,
            'password': self.password,
            **boxes
        }
        created_meme = requests.post(f"{self.api_root}/caption_image", data=data).json()
        if save_to_clipboard:
            copied = copy_meme_to_clipboard(created_meme['data']['url'])
        return created_meme, copied
    def save_local_meme(self, meme_data, filename):
        """Save locally the image created from the data returned """
        save_image_locally(meme_data['data']['url'], filename)
        return
    def get_initial_word_from_text(self, text, method):
        return share_api.get_initial_word_from_text(text, method)
    def predict_meme_text(self, template_id, num_boxes, init_text = "", beam_width=1, max_output_length=140): #TODO: Should make num_boxes an input, or just infer it from template_id? 
        return share_api.predict_meme_text(template_id, num_boxes, init_text, beam_width, max_output_length)
    def generate_captions(self, meme_id, input_text, num_boxes, save_to_clipboard=True):      
        """Uses MaCHinE LEarNiNG to create a caption *hopefully* related to it"""
        #STEP 1: Find an important word (for now lets just try a verb)
        initial_word = self.get_initial_word_from_text(input_text, method="long_words") + " "
        #STEP 2: Generate caption!
        generated_captions = self.predict_meme_text(
            template_id = meme_id, 
            num_boxes = num_boxes,  
            init_text = initial_word, 
            beam_width = 1, 
            max_output_length = 140
        )
        print(f"Confession: [#f5a6ff]{input_text}[/#f5a6ff]")
        print(f"Generated captions: [#03c6fc]{generated_captions}.[/#03c6fc]".replace("||.", ""))
        if not (generated_captions == "Error"):
            captions = generated_captions.split("|")[:-1] #last one is empty
            meme_info, copied = self.create_meme(meme_id, captions, save_to_clipboard)
            return meme_info, copied
        return -1, -1