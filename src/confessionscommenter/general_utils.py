import pyperclip
from facebook_scraper import get_posts
import string
from rich import print
import sys, os

def copyComment(post, comment):
    pyperclip.copy(comment)

def getPosts(title='beaverconfessions', pages=5):
    posts = []
    for post in get_posts(title, pages=pages):
        new_post = {}
        if post['post_text'] is None: #sometimes it can return None. facebook-scraper does not guarantee 
            new_post['message'] = ""
        else:
            new_post['message'] = convert(post['post_text'])
        new_post['link'] = post['post_url']
        new_post['id'] = post['post_id']
        posts.append(new_post)
    return posts

def sanitize(s): #Taken from https://stackoverflow.com/a/8689826/14127936
    """Sanitize the input s by only keeping punctuation, digits, ascii letters and whitespace"""
    return ''.join(filter(lambda x: x in string.printable, s))

def convert(input): #From answer https://stackoverflow.com/questions/13101653/python-convert-complex-dictionary-of-strings-from-unicode-to-ascii
    """Recursively sanitize keys and elements of lists, tuples and dictionaries"""
    #TODO: Is this necessary? can we assume that the only place that needs to be sanitized is the 'message' parameter?
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.items()}
    elif isinstance(input, list) or isinstance(input, tuple):
        return [convert(element) for element in input]
    else: #assumed to be a string
        return sanitize(input)

#Taken from https://stackoverflow.com/a/45669280/14127936
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout