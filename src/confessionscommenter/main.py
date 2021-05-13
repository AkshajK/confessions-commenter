import os
from rich import print
from rich.console import Console
from confessionscommenter.general_utils import *
from confessionscommenter.commandline_utils import *
from confessionscommenter.meme_utils import *
console = Console()
def main():
    os.system("clear")
    console.rule("[bold blue]Welcome to SHARE, the MIT Confessions Bot")
    print()
    memer = MemeGenerator("mit_meme_creator", "mit_meme_password") #Put your imgflip info here
    posts = getPosts()
    while True:
        index = choosePost(posts)
        if index == 0:
            break
        post = posts[index-1]
        print(f"CONFESSION: [#f5a6ff]{post['message']}[/#f5a6ff]")
        index = options(
            "[bold]What type of Comment?[/bold] (Enter number to continue)", 
            [
                "GPT2 Generated Comment", 
                "Automatically generate meme for this post", 
                "Generate meme for post with input text"
            ])
        if index == 0:
            generateComments(post)
            print(f"[bold]Comment copied to clipboard, paste here: {post['link']} [/bold]")
        elif index == 1:
            generatable_memes = memer.get_generatable_memes_info()['memes']
            names = [f"{meme['name']}. See examples at https://imgflip.com/meme/{meme['id']}" for meme in generatable_memes]
            i = options("[#03c6fc]Choose from the meme list:[/#03c6fc]", names)
            meme_info, copied = memer.generate_captions(generatable_memes[i]['id'], post['message'], generatable_memes[i]['box_count'])
            if copied == -1:
                pass
            elif copied:
                print(f"[bold]Meme copied to clipboard, paste here: {post['link']} [/bold]. You can also find it for download here {meme_info['data']['url']}")
            else: 
                print(f"[bold]Couldn't be copied to clipboard. Download it from {meme_info['data']['url']} and post it here {post['link']} [/bold]")

        elif index == 2:
            memes = memer.get_popular_memes()
            names = [f"{meme['name']}. See examples at https://imgflip.com/meme/{meme['id']}" for meme in memes]
            i = options("[#03c6fc]Choose from the meme list:[/#03c6fc]", names)
            captions = []
            print(f"[#f5a6ff]This meme requires [red]{memes[i]['box_count']} texts[/red][/#f5a6ff]")
            for j in range(1, memes[i]['box_count']+1):
                print(f"[#03c6fc]Type meme text #{j}[/#03c6fc]: ", end="")   
                text = input()
                captions.append(text)
            meme_info, copied = memer.create_meme(memes[i]['id'], captions)
            if copied:
                print(f"[bold]Meme copied to clipboard, paste here: {post['link']} [/bold]. You can also find it for download here {meme_info['data']['url']}")
            else: 
                print(f"[bold]Couldn't be copied to clipboard. Download it from {meme_info['data']['url']} and post it here {post['link']} [/bold]")
        else:
            pass

if __name__ == '__main__':
    main()