;***************************************************************************
;
; Program for education in subject "Assembly Languages" and "APPS"
; petr.olivka@vsb.cz, Department of Computer Science, VSB-TUO
;
; Usage of variables in Assembly language
;
;***************************************************************************

    bits 64

    section .data

    ; external variables declared in C
    extern zvire, xnum, pass, g_c_array, array_size, g_c_int_array, cisla, cisla2

    ; list of public variables
    global g_a_zvire, g_a_bajt0, g_a_bajt1, g_a_bajt2, g_a_bajt3, text, vysledek, suma,suma2

g_a_zvire      dd      0,0,0,0,0
g_a_bajt0      db      0
g_a_bajt1      db      0
g_a_bajt2      db      0
g_a_bajt3      db      0
text           db      "my empty string", 0
vysledek       dd      0
suma           dd      0
suma2           dd      0

    section .text


;***************************************************************************
    ; Function that will reverse string kocka
    global access_zvire
access_zvire:
    mov eax, 0 
    mov al, [zvire + 4]
    mov [g_a_zvire + 0], eax
    mov al, [zvire + 3]
    mov [g_a_zvire + 1], eax
    mov al, [zvire + 2]
    mov [g_a_zvire + 2], eax
    mov al, [zvire + 1]
    mov [g_a_zvire + 3], eax
    mov al, [zvire + 0]
    mov [g_a_zvire + 4], eax
    ret


;***************************************************************************
    ; Function that will divide variable int xnum = 0x1F2E3D4C into char bajt0, bajt1, bajt2, bajt3
    global  access_bajts
access_bajts:
    ; Content of variable g_c_int will be moved to variable g_a_int.
    mov al, [xnum + 3]
    mov [g_a_bajt3 + 0], al
    mov al, [xnum + 2]
    mov [g_a_bajt2], al
    mov al, [xnum + 1]
    mov [g_a_bajt1], al
    mov al, [xnum + 0]
    mov [g_a_bajt0], al
    ret


;***************************************************************************
    ; Function that will transtalte string of hex to string
    global  access_pass
access_pass:
    ; Content of variable g_c_int will be moved to variable g_a_int.
    mov al, [pass + 0]
    mov [text + 0], al
    mov al, [pass + 1]
    mov [text + 1], al
    mov al, [pass + 2]
    mov [text + 2], al
    mov al, [pass + 3]
    mov [text + 3], al
    mov al, [pass + 4]
    mov [text + 4], al
    mov al, [pass + 5]
    mov [text + 5], al
    mov al, [pass + 6]
    mov [text + 6], al
    mov al, [pass + 7]
    mov [text + 7], al
    mov al, [pass + 8]
    mov [text + 8], al
    mov al, [pass + 9]
    mov [text + 9], al
    mov al, [pass + 10]
    mov [text + 10], al
    mov al, [pass + 11]
    mov [text + 11], al
    mov byte [text + 12 ], 0 
    
    ret
;***************************************************************************
    ; Function that will null upper bits
     global  access_pass_null
access_pass_null:
    mov [pass + 3], byte 0
    mov [pass + 7], byte 0
    mov [pass + 11], byte 0
    ret

;***************************************************************************
    ; Function that will odd given numbers
global  access_array_odd
access_array_odd:
    mov     ecx, 0
    mov rdx, 0

.loop:
    cmp     rdx, [array_size]
    jnl     .endfort

    mov     rax, [g_c_array + rdx * 8]
    and     rax, ~1 
    mov     [g_c_array + rdx * 8], rax
    inc     rdx
    jmp     .loop

.endfort:
    ret
;***************************************************************************
    ; Function that will change sing
global  access_array_neg
access_array_neg:
    mov ecx, 0
    mov rdx, 0
    
.loop:
    cmp rdx, [array_size]
    jnl .endfort
    mov rax, [g_c_array + rdx * 8]
    not rax
    inc rax
    mov [g_c_array + rdx * 8], rax
    inc rdx
    jmp .loop

.endfort:
    ret

;***************************************************************************
; Function that will remove lowest number
global  access_array_min
access_array_min:
    mov ecx, 0
    mov rdx, 0
    mov r8, 0
    mov r10d, [g_c_int_array + 0 * 4]

.loop:
    cmp rdx, 10
    je .endfort

    mov eax, [g_c_int_array + rdx * 4]

    cmp r10d, eax  
    jg .update_min

.next_element:
    inc rdx
    jmp .loop

.update_min:
    mov r10d, eax  
    mov r8, rdx
    jmp .next_element

.endfort:
    mov [vysledek], r10
    mov [g_c_int_array + r8 * 4], dword 0
    ret


;***************************************************************************
    ; Function that will add a value to the sum of an array with negate num
    global  access_suma
access_suma:
    mov eax, 0
    mov rdx, 0 
.loop:
    cmp rdx, 10
    je .endfort
    movsx ecx, byte [cisla + rdx]
    add eax, ecx
    inc rdx
    jmp .loop

.endfort:
    mov [suma], eax
    ret

;***************************************************************************
    ; Function that will add a value to the sum of an array with positiv num
    global  access_suma2
access_suma2:
    mov eax, 0
    mov rdx, 0 
.loop:
    cmp rdx, 10
    je .endfort
    movsx ecx, byte [cisla2 + rdx]  ; Load a signed byte from cisla into ecx
    add eax, ecx            ; Add value from cisla to suma
    inc rdx
    jmp .loop

.endfort:
    mov [suma2], eax          ; Store the updated value of suma
    ret