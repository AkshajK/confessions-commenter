import pyperclip
from facebook_scraper import get_posts
import string
from rich import print
importedGenerator = False
generator = -1
def importModel():
    global importedGenerator
    global generator 
    if not importedGenerator:
        print("Importing model...")
        from transformers import pipeline
        generator = pipeline('text-generation', model='gpt2')
        importedGenerator = True

def generateCommentsGPT2(msg, num=1):
    q = msg[(msg.index(" ")+1):]
    prompt = f"{q}\n RESPONSE: " 
    importModel()
    text = generator(prompt, max_length=len(prompt.split(" "))+100, num_return_sequences=num)
    return [text[i]['generated_text'][len(prompt):] for i in range(num)]

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
