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
    
    stack = []
    for char in input:
        if char not in '({<[\'\")]}>':
            continue
        # print(stack)
        # Zpracování uvozovek
        if char in "'\"":
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        elif char in '({<[':
            stack.append(char)
        elif char in ')}]>':
            if char == ')' and stack and stack[-1] == '(':
                stack.pop()
            elif char == ']' and stack and stack[-1] == '[':
                stack.pop()
            elif char == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif char == '>' and stack and stack[-1] == '<':
                stack.pop()
            else:
                return False

    return not stack

input = "'a'"
print(validate(input))
