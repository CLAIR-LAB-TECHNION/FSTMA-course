import numpy as np

def super_render(env):
  m, n, _, _, _, _, _ = env.vector_observation_dims
  board_size_rows = m + 2
  board_size_columns = 2 * n + 1
  board = np.zeros((board_size_rows, board_size_columns),dtype='U2')
  taxis_locations, _, passengers_start_locations, destinations, passengers_status = env.state
  print(env.state)
  for i in range(board_size_rows):
    for j in range(board_size_columns):
      if i==0 or i==board_size_rows-1:
        if j==0 or j==board_size_columns-1:
          board[i,j] = '+'
        else:
          if j%2==1:
            board[i,j] = '--'
          else:
            board[i,j] = '-'
      else:
        if j==0 or j==board_size_columns-1:
          board[i,j] = '|'
        else:
          if j%2==1:
            board[i,j] = '  '
          else:
            board[i,j] = ':'
  wall = {1: [4,8,12,16,20], 2: [4,12,16,20], 4: [12], 5: [12], 7: [2,6,8,10,14,16,18,22]}
  for k in wall.keys():
    for v in wall[k]:
      board[k,v] = '|'
    
  for i, location in enumerate(taxis_locations):
      board[location[0] + 1, location[1] * 2 + 1] = 'T{}'.format(i + 1)

  for i, location in enumerate(passengers_start_locations):
    if location not in taxis_locations:
      if passengers_status[i] > 2 or passengers_status[i] == 1:
        board[location[0] + 1, location[1] * 2 + 1] = ''
      else:
        board[location[0] + 1, location[1] * 2 + 1] = 'P{}'.format(i)
  for i, location in enumerate(destinations):
    if location not in taxis_locations:
      if passengers_status[i] == 1:
        board[location[0] + 1, location[1] * 2 + 1] = 'P{}'.format(i)
      else:
        board[location[0] + 1, location[1] * 2 + 1] = 'D{}'.format(i)
  # print(board)

  for i in range(board_size_rows):
    for j in range(board_size_columns):
      if not j==board_size_columns-1:
        if board[i,j][0]=='T':
          print('\x1b[0;30;43m' + '\033[1m' + board[i,j] + '\033[0m' + '\x1b[0m', end='')
        else:
          print(board[i,j], end='')
      else:
        print(board[i,j])