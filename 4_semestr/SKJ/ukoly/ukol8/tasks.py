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
    pass

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
    pass