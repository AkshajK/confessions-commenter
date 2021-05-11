import platform
import random
import requests
import shutil
import os
import string
def get_temporary_filename(file_dir="."):
    os.makedirs(file_dir, exist_ok=True)
    letters = string.ascii_lowercase
    while True: #Not entirely correct, breaks if directory has all the 10-digits images (tho I think it's a safe assumption)
        random_string = ''.join(random.choice(letters) for i in range(10)) 
        filename = os.path.join(file_dir, f"meme_{random_string}.jpg")
        if not os.path.exists(filename):
            return filename
def save_image_locally(url, filename):
    """Saves the image from an http(s) link locally"""
    # Save the meme locally
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(filename, 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

def copy_meme_to_clipboard(url, remove_local=True):
    directory = "images/"
    local_path = get_temporary_filename(directory)
    save_image_locally(url, local_path)
    copied = copy_local_image_to_clipboard(local_path)
    if remove_local:
        os.remove(local_path)
    if len([f for f in os.listdir(directory) if not f.startswith(".")]) == 0: #empty directory
        try:
            os.rmdir(directory)
        except:
            print(f"Couldn't remove directory {directory}")
    return copied

def windows_copy_local_image_to_clipboard(path):
    """
    Copy a jpg image to the clipboard. Windows only. 
    Taken from https://clay-atlas.com/us/blog/2020/10/30/python-en-pillow-screenshot-copy-clipboard/
    Returns True if it was successfully copied, False otherwise
    """
    system = platform.system()
    assert system == "Windows", "Only can run this clipboard function in Windows OS"
    try:
        import win32clipboard as clip #need pywin32
        import win32con
        from io import BytesIO
        from PIL import ImageGrab, Image
        image = Image.open(path)
        output = BytesIO()
        image.convert('RGB').save(output, 'BMP')
        data = output.getvalue()[14:] #TODO: why 14?
        output.close()
        clip.OpenClipboard()
        clip.EmptyClipboard()
        clip.SetClipboardData(win32con.CF_DIB, data)
        clip.CloseClipboard()
        print("Copied image to clipboard!")
        return True
    except:
        print("There was an error while copying the image to the clipboard. You might need to download it yourself. Sorry!")
        return False
def copy_local_image_to_clipboard(path):
    system = platform.system()
    if system == "Windows":
        return windows_copy_local_image_to_clipboard(path)
    elif system == "Java":
        pass
    elif system == "Linux":
        pass 
    else:
        print("Lol what system do you have") 
if __name__ == "__main__":
    windows_copy_local_image_to_clipboard("images/testxd.jpg")