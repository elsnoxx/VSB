def brackets_depth(input):
    """
    Given a string consisting of matched nested parentheses (<{[]}>), write a Python program to compute
    the depth of each leaf and return it in a list.
    
    Examples:
        '' = [0]
        '()' = [1]
        '[](()){}' = [1,2,1]
        '[]<(()){}>' = [1,3,2]
    
    Don't validate input for errors. You will always get a correct input.
    
    Hint: You can count the sequence of opening (+=1) and closing (-=1) brackets, or you can use a stack.
    """
    
    depth = 0
    restult = []
    stack = []
    add_flag = True

    if not input:
        return [0]
    
    for char in input:
        if char in '({<[':
            depth += 1
            stack.append(char)
            add_flag = True

        elif char in ')]>}':
            stack.pop()

            if add_flag:
                restult.append(depth)
                add_flag = False
            
            depth -= 1
    
        print(stack)

    if restult == []:
        return [0]
    else:
        return restult

    
def create_stack():
    return []

def check_empty(stack):
    print(f'Check empty: {len(stack)}')
    return len(stack) == 0

def push(stack, item):
    stack.append(item)

def pop(stack):
    if check_empty(stack):
        return None
    return stack.pop()

def validate(input):
    """
    Validate an input consisting of matched nested parentheses and strings (<{["\'\""]}>) containing
    any character and escape sequences. Strings and escape sequences follow python rules, that is
    any double quoted strings can use single quotes without escaping and any single quoted strings
    can use double quotes without escaping.
    
    Examples:
        ()   = True
        (<>) = True
        (<)> = False # ) does not close <
        ("") = True
        "(") = False # no opening (
        "'" = True
        "\"" = True
    
    Return True if the input string is in a valid format, otherwise return False.
    
    Hint: Use a stack.
    """
    if not input:
        return True
    if not any(c in '({[<)]}>\'"' for c in input):
        return False
    cnt = 0
    stack = create_stack()
    while cnt < len(input):
        curent = input[cnt]
        print(f'Current: {curent}, Stack: {stack}, Count: {cnt}')
        # Pokud jsme uvnitř stringu (na stacku je quote)
        if not check_empty(stack) and stack[-1] in "\'\"":
            quote = stack[-1]
            if curent == '\\' and cnt + 1 < len(input):
                cnt += 2
                continue
            elif curent == quote:
                pop(stack)
                cnt += 1
                continue
            else:
                cnt += 1
                continue

        if curent in "\"":
            push(stack, "\"")
            cnt += 1
            continue
        elif curent == '\'':
            push(stack, "'")
            cnt += 1
            continue

        if curent in '({<[':
            push(stack, curent)
            cnt += 1
            continue

        if curent in ')}]>':
            if check_empty(stack):
                return False
            if curent == ')' and stack[-1] == '(':
                pop(stack)
            elif curent == ']' and stack[-1] == '[':
                pop(stack)
            elif curent == '}' and stack[-1] == '{':
                pop(stack)
            elif curent == '>' and stack[-1] == '<':
                pop(stack)
            else:
                return False
            cnt += 1
            continue

        cnt += 1

    return check_empty(stack)
            
            


input = '"as\'df"()\'<{[(\'{asdf}' # "as'df"()'<{[('{asdf}
print(input)
print(validate(input))
