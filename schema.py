"""Public inference schema: no labels, metadata, or oracle imports."""
ACTIONS = ('LEFT', 'CENTER', 'RIGHT', 'NONE')

def questions():
    return {'select': {'type': 'choice', 'instructions':
        'Follow the user instruction using the three handwritten digits and colored borders in the image. '
        'Select the matching position. NONE means no digit satisfies the condition.',
        'criteria': {'LEFT': 'Choose the left digit.', 'CENTER': 'Choose the center digit.',
                     'RIGHT': 'Choose the right digit.', 'NONE': 'No digit satisfies the instruction.'}}}

def model_input(image, instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Enter an instruction.')
    return {'state': instruction, 'images': [image], 'questions': questions()}
