grammar PLCProject;

prog : statement+ EOF ;

statement 
    : command? SEMI                                   # CMD
    | LBRACE statement* RBRACE                        # BLOCK
    | IF LPAREN expr RPAREN statement (ELSE statement)? # IF_ELSE
    | WHILE LPAREN expr RPAREN statement              # WHILE
    | FOR LPAREN expr? SEMI expr? SEMI expr? RPAREN statement # FOR
    ;

command 
    : vartype VARID (COMMA VARID)*                    # CMDVAR
    | expr                                            # CMDEXPR
    | READ VARID (COMMA VARID)*                       # CMDREAD
    | WRITE expr (COMMA expr)*                        # CMDWRITE
    ; 

expr 
    : LPAREN expr RPAREN                               # PAREN
    | INT                                              # INT
    | FLOAT                                            # FLOAT
    | BOOL                                             # BOOL   
    | STR                                              # STR
    | VARID                                            # VARID
    | SUB expr                                         # MINUS
    | NOT expr                                         # NOT
    | expr op=(MUL | DIV | MOD) expr                   # MUL_DIV_MOD
    | expr op=(ADD | SUB | CONCAT) expr                # ADD_SUB_CONCAT
    | expr op=(LT | GT) expr                           # REL
    | expr op=(EQ | NEQ) expr                          # EQUAL
    | expr AND expr                                    # AND
    | expr OR expr                                     # OR
    | <assoc=right> VARID ASSIGN expr                  # ASSIGN
    ;

vartype 
    : INT_KW
    | FLOAT_KW
    | BOOL_KW
    | STRING_KW
    ;

// LEXER - Klíčová slova
IF        : 'if';
ELSE      : 'else';
WHILE     : 'while';
FOR       : 'for';
READ      : 'read';
WRITE     : 'write';
INT_KW    : 'int';
FLOAT_KW  : 'float';
BOOL_KW   : 'bool';
STRING_KW : 'string';

// LEXER - Operátory a symboly
ASSIGN : '=' ;
SEMI   : ';' ;
COMMA  : ',' ;
LPAREN : '(' ;
RPAREN : ')' ;
LBRACE : '{' ;
RBRACE : '}' ;
LBRACKET : '[' ;
RBRACKET : ']' ;
MUL    : '*' ;
DIV    : '/' ;
MOD    : '%' ;
ADD    : '+' ;
SUB    : '-' ;
CONCAT : '.' ;
LT     : '<' ;
GT     : '>' ;
EQ     : '==' ;
NEQ    : '!=' ;
AND    : '&&' ;
OR     : '||' ;
NOT    : '!' ;

// LEXER - Literály a identifikátory
BOOL  : 'true' | 'false' ;
INT   : [0-9]+ ;
FLOAT : [0-9]+ '.' [0-9]* | '.' [0-9]+ ;
STR   : '"' ( '\\' . | ~[\\"\r\n] )* '"' ;
VARID : [a-zA-Z][0-9a-zA-Z]* ;

WS      : [ \t\r\n]+ -> skip ;
COMMENT : '//' ~[\r\n]* -> skip ;