"""Dataset construction/evaluation only. Never imported by the inference engine."""
from itertools import product
from schema import ACTIONS

TASKS = [(c,p,e) for c,p,e in product((None,'red','blue'), (None,'odd','even'), (None,'min','max'))
         if (c,p,e) != (None,None,None)]
HELD_OUT = {('red','even','max'), ('blue','odd','min')}

def task_name(task):
    return '_'.join(x or 'any' for x in task)

def answer(digits, colors, task):
    color, parity, extreme = task
    indices = [i for i,d in enumerate(digits)
               if (color is None or colors[i] == color)
               and (parity is None or d % 2 == (parity == 'odd'))]
    if not indices:
        return 'NONE'
    if extreme:
        value = (min if extreme == 'min' else max)(digits[i] for i in indices)
        indices = [i for i in indices if digits[i] == value]
    return ACTIONS[indices[0]] if len(indices) == 1 else None

def instruction(task, language, variant):
    color, parity, extreme = task
    if language == 'en':
        noun = ('odd ' if parity == 'odd' else 'even ' if parity else '') + 'digit'
        location = f' in a {color} box' if color else ''
        adjective = 'smallest ' if extreme == 'min' else 'largest ' if extreme else ''
        forms = [f'Choose the {adjective}{noun}{location}.',
                 f'Select the {adjective}{noun}{location}.',
                 f'Pick the {adjective}{noun}{location}, or choose NONE if there is none.',
                 f'Find the {adjective}{noun}{location}. Which position is it in?',
                 f'Look for {"the " + adjective if extreme else "an " if parity else "a "}{noun}{location}; indicate its position, or NONE if absent.',
                 f'Which box contains the {adjective}{noun}{location}? If no box qualifies, select NONE.']
    else:
        noun = '홀수' if parity == 'odd' else '짝수' if parity else '숫자'
        location = ('빨간' if color == 'red' else '파란') + ' 상자에 있는 ' if color else ''
        adjective = '가장 작은 ' if extreme == 'min' else '가장 큰 ' if extreme else ''
        forms = [f'{location}{adjective}{noun}를 골라.',
                 f'{location}{adjective}{noun}를 선택해 줘.',
                 f'{location}{adjective}{noun}를 찾아. 없으면 NONE을 선택해.',
                 f'{location}{adjective}{noun}는 어느 위치에 있어?',
                 f'선택할 대상은 {location}{adjective}{noun}야. 해당하는 상자의 위치로 답해 줘. 없으면 NONE이야.',
                 f'{location}{noun} 중 {"최솟값" if extreme == "min" else "최댓값" if extreme else "조건에 맞는 것"}의 위치를 알려 줘. 없으면 NONE으로 답해.']
        forms = [x.replace('홀수를','홀수를').replace('짝수를','짝수를') for x in forms]
    return forms[variant]
