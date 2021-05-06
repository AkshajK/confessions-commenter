import requests
import urllib 
import shutil


class MemeGenerator:
    def __init__(self, username, password, graph=-1):
        self.username = username
        self.password = password
        self.graph = graph
        self.api_root = "https://api.imgflip.com"
    def get_all_memes(self):
        data = requests.get(f"{self.api_root}/get_memes").json()
        return data['data']['memes']
    def create_and_post_meme(self, meme_id, post_id, top_text="", bottom_text="", extra_message=""):
        res = self.create_meme(meme_id, [top_text, bottom_text]) #create meme in the imgflip server
        local_image = "images/aaaaa.jpg" # TODO: Make it random, different for different pictures
        res = self.save_local_meme(res, local_image) # race condition?
        res = self.post_local_meme_facebook(post_id, local_image, extra_message)
        # print(res)
        return res
    def create_meme(self, meme_id, text_list):
        text_template = {f"text{i}": text for i, text in enumerate(text_list)}
        data = {
            'template_id': meme_id, #drake hotline bling
            'username': self.username,
            'password': self.password,
            **text_template
        }
        created_meme = requests.post(f"{self.api_root}/caption_image", data=data).json()
        return created_meme
    def save_local_meme(self, meme_data, filename):
        # Save the meme locally
        r = requests.get(meme_data['data']['url'], stream=True)
        if r.status_code == 200:
            with open(filename, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        return 
    def post_local_meme_facebook(self, post_id, meme_location, extra_message=""):
        comment= "here an image should goooo" #this comment will be edited
        res = self.graph.put_comment(object_id= post_id, message= comment ) #create comment
        comment_info = self.graph.put_photo( #put an image (and a new comment) in the created comment
            image=open(meme_location, 'rb'), 
            album_path=res['id'], 
            message= extra_message
        )
        story_fbid, comment_id = res['id'].split("_")
        meme_link = f"https://facebook.com/permalink.php?story_fbid={story_fbid}&id={post_id}&comment_id={comment_id}"
        return meme_link #link to the fb post