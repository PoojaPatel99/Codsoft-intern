import random

# Define the board size
BOARD_SIZE = 3

# Symbols for the player and AI
PLAYER = 'X'
AI = 'O'

# Function to print the board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * (2 * BOARD_SIZE - 1))

# Function to check if the current player has won
def check_win(board, player):
    # Check rows, columns and diagonals
    for i in range(BOARD_SIZE):
        if all([cell == player for cell in board[i]]):  # Row check
            return True
        if all([board[j][i] == player for j in range(BOARD_SIZE)]):  # Column check
            return True
    
    # Diagonal checks
    if all([board[i][i] == player for i in range(BOARD_SIZE)]):  # Main diagonal
        return True
    if all([board[i][BOARD_SIZE - 1 - i] == player for i in range(BOARD_SIZE)]):  # Anti-diagonal
        return True
    
    return False

# Function to check if the board is full (draw condition)
def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

# Function to get all available moves
def get_available_moves(board):
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == ' ']

# Minimax algorithm to evaluate the best move for AI
def minimax(board, depth, is_maximizing_player):
    if check_win(board, PLAYER):  # Player wins
        return -1
    if check_win(board, AI):  # AI wins
        return 1
    if is_board_full(board):  # Draw
        return 0

    if is_maximizing_player:
        best_score = float('-inf')
        for move in get_available_moves(board):
            board[move[0]][move[1]] = AI
            score = minimax(board, depth + 1, False)
            board[move[0]][move[1]] = ' '
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for move in get_available_moves(board):
            board[move[0]][move[1]] = PLAYER
            score = minimax(board, depth + 1, True)
            board[move[0]][move[1]] = ' '
            best_score = min(score, best_score)
        return best_score

# Function to find the best move for the AI using Minimax
def best_move(board):
    best_score = float('-inf')
    move = None
    for possible_move in get_available_moves(board):
        board[possible_move[0]][possible_move[1]] = AI
        score = minimax(board, 0, False)
        board[possible_move[0]][possible_move[1]] = ' '
        if score > best_score:
            best_score = score
            move = possible_move
    return move

# Function to play the game
def play_game():
    # Create an empty board
    board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    print("Welcome to Tic-Tac-Toe!")
    print("You are X, and the AI is O.")
    print_board(board)
    
    while True:
        # Player's move
        while True:
            try:
                row, col = map(int, input("Enter your move (row and column) as 'row col' (0-based index): ").split())
                if board[row][col] == ' ':
                    board[row][col] = PLAYER
                    break
                else:
                    print("This cell is already taken. Choose another one.")
            except (ValueError, IndexError):
                print("Invalid input. Please enter two numbers between 0 and 2.")

        print_board(board)

        if check_win(board, PLAYER):
            print("You win!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

        # AI's move
        print("AI is making its move...")
        ai_move = best_move(board)
        board[ai_move[0]][ai_move[1]] = AI
        print_board(board)

        if check_win(board, AI):
            print("AI wins!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

# Run the game
if __name__ == "__main__":
    play_game()
