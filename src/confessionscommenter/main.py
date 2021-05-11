import os
from rich import print
from rich.console import Console
from confessionscommenter.general_utils import *
from confessionscommenter.commandline_utils import *
console = Console()
def main():
    os.system("clear")
    console.rule("[bold blue]Welcome to SHARE, the MIT Confessions Bot")
    print()
    # memer = MemeGenerator("mit_meme_creator", "mit_meme_password", graph)
    posts = getPosts()
    while True:
        index = choosePost(posts)
        if index == 0:
            break
        post = posts[index-1]
        print(f"CONFESSION: [#f5a6ff]{post['message']}[/#f5a6ff]")
        index = options("[bold]What type of Comment?[/bold] (Enter number to continue)", ["GPT2 Generated Comment"])#, "Write custom message with meme"])
        if index == 0:
            generateComments(post)
        else:
            pass 
            # add meme here

if __name__ == '__main__':
    main()