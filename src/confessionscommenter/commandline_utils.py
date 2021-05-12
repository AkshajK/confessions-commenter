from confessionscommenter.general_utils import *
from rich import print
import shutil
import pyperclip
size = int(shutil.get_terminal_size()[0])
def options(headerText, choices, start=0): 
    print(headerText)
    i = start
    for index in range(start, min(start+10, len(choices))):
        i += 1
        print(f"[blue]({i})[/blue] [white]{choices[index]}[/white]") 
    i += 1
    if(i != len(choices)+1):
        print(f"({i}) See More")
    ans = ""
    while True:
        try: 
            num = int(input())-1
            ans = choices[num]
            break
        except:
            pass
    
    if(num+1 == i):
        print("START: "+str(start))
        return options(headerText, choices, start+10)
    print(f"[bold]You chose[/bold] {ans}")
    return num

def choosePost(posts):
    choices = []
    choices.append(f"[#FFB6C1]Quit[/#FFB6C1]")
    for i in range(len(posts)):
        clipmsg = posts[i]['message'].split('\n')[0]
        choices.append(f"[white]{(clipmsg)[:(size-6)]}[/white]")
    index = options("[bold]Pick a post:[/bold]", choices)
    return index

def generateComments(post, numOptions=5, makeCommentFunc=copyComment):
    """Returns generated comment if chose, and None if decided not to comment"""
    comments = generateCommentsGPT2(post['message'], num=numOptions)
    for i in range(len(comments)):
        print(f"COMMENT {i+1}: [#03c6fc]{comments[i]}[/#03c6fc]\n----")
    print(f"Which comment to post? (respond number or NO)")
    ans = input()
    done = False 
    # comment_link = None
    for i in range(len(comments)):
        if ans ==str(i+1)+"":
            makeCommentFunc(post, comments[i])
            done = True 
            return comments[i]
    if not done:
        print("Okay, will not comment.")