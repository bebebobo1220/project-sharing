from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
import random
import Secret

updater = Updater(token=Secret.access_token, use_context=True)
dispatcher = updater.dispatcher
board = None
revealed = None
started=False
R, C, k=8, 9, 2


def init_board():
    global board
    global revealed

    board=[[0] * C for i in range(R)]
    #print(board)
    revealed=[[False] * C for i in range(R)]
    mines=random.sample([(x, y) for x in range(R) for y in range(C)], k)
    for x, y in mines:
        board[x][y]='*'
        for X in range(x - 1, x + 2):
            for Y in range(y - 1, y + 2):
                if 0 <= X < R and 0 <= Y < C and board[X][Y] != '*':
                    board[X][Y]+=1
init_board()


def show_board():
    lst = ''
    for board_r, real_r in zip(board, revealed):
        columns=[board_r[i] if real_r[i] else '@' for i in range(C)]
        lst += "".join(map(str, columns))
        lst += '\n'
    return lst


def real_board():
    lst = ''
    for board_r, real_r in zip(board, revealed):
        columns=[board_r[i]  for i in range(C)]
        lst += "".join(map(str, columns))
        lst += '\n'
    return lst


def open_cell(r, c):
    global started
    if r <= 0 or r > R or c <= 0 or c > C:
        return
    r, c=r - 1, c - 1
    if revealed[r][c]:
        return
    revealed[r][c]=True
    if board[r][c] == '*':
        started=False
        print('Boooooom!!!')
    elif board[r][c] == 0:
        for i in range(r - 1, r + 2):
            for j in range(c - 1, c + 2):
                open_cell(i + 1, j + 1)
    #print(f'open_cell({r + 1}, {c + 1}) is done.')
    #show_board()


#update
def play(update, context):
    while True: 
        global started
        global R, C, k
        global board
        global revealed
        cmd=update.message.text
        if cmd.startswith('/start'):
            tokens=cmd.split()
            if cmd == '/start':
                R, C, k=8, 9, 2
                started= True
                pass
            elif len(tokens) == 4 and tokens[0] == '/start' and all(x.isdigit() for x in tokens[1:]):
                R, C, k=[int(x) for x in tokens[1:]]
                if 0 <= R < 16 and 0 <= C < 16 and 2 <= k < R * C + 1:
                    pass
                else:
                    context.bot.send_message(chat_id=update.effective_chat.id,
                    text=f"參數設定出錯!!!\nR:必須在8~15之間\nC:必須在9~15之間\nk:必須在{(R * C)}之間")
                    break

                started= True
                init_board() 
        elif cmd.startswith('/open '):
            if not started:
                context.bot.send_message(chat_id=update.effective_chat.id,
                text='遊戲還沒開始')
                break
            try:
                r, c=eval(cmd[6:])
            except:
                context.bot.send_message(chat_id=update.effective_chat.id,
                    text='1. 無法執行你想要的指示')
                break

            open_cell(r, c)
        elif cmd.startswith('/restart'):
            if started == True:  
                revealed=[[False] * C for i in range(R)]
            else:
                context.bot.send_message(chat_id=update.effective_chat.id,
                    text= '遊戲還沒開始')
                break
        elif cmd.startswith('/quit'):
            break
        else:
            context.bot.send_message(chat_id=update.effective_chat.id,
                    text='遊戲結束')
            break  ## done
        context.bot.send_message(chat_id=update.effective_chat.id,
                    text= show_board())

        #context.bot.send_message(chat_id=update.effective_chat.id,
                    #text= real_board())
        break



def reply(update, context):
    update.callback_query.data == 'a'
    context.bot.edit_message_text('/start 直接開始遊戲 \n設指定的格子，列(R)的上限為8~15，欄(C)的上限為9~15，💣地雷數目最少為5個以上，上限為R*C個 \n範例: /start 10 10 8 \n/open 為輸入欲打開盤面之座標 : \n範例：/open (8, 1)；\n/restart 可重玩這一局（不重新 Random 地雷座標）\n 如果想重新開設一局新的，直接輸入/start ',
        chat_id=update.callback_query.message.chat_id,
                                      message_id=update.callback_query.message.message_id)


def surface (update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text='💣來玩踩地雷遊戲',
    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton('開始遊戲', callback_data='a')]]))



surface_handler = CommandHandler('read', surface)
start_handler = CommandHandler('start',play) # done 
restart_handler = CommandHandler('restart',play) # done 
open_handler = CommandHandler('open',play) # done

dispatcher.add_handler(surface_handler)
dispatcher.add_handler(CallbackQueryHandler(reply))
dispatcher.add_handler(start_handler)
dispatcher.add_handler(restart_handler)
dispatcher.add_handler(open_handler)
updater.start_polling()
updater.idle()
