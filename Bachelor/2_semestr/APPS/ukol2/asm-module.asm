    bits 64

    section .data
        g_tabulka db "0123456789ABCDEF"

    section .text

;***************************************************************************
    ; zadani 1 - Najděte v poli čísel long nejmenší číslo, které má dolní bajt nulový.
    global nejmensi_56bit
nejmensi_56bit:
    xor rax, rax
    mov r10, -1
    mov rcx, 0

.while:
    cmp rcx, rsi
    je .konec

    mov rax, [rdi + 8*rcx]
    and eax, 0xFF
    test eax, eax
    jnz .nenulovy

    cmp r10, -1
    jne .nenulovy

    mov r10, rcx

.nenulovy:
    inc rcx
    jmp .while

.konec:
    cmp r10, -1
    je .end
    mov rax, [rdi + 8*r10]
.end:
    ret


;***************************************************************************
    ; zadani 2 - Převeďte číslo long na hex string.
    global long2hexstr
long2hexstr:
    mov r8, rdi
    mov r9, rsi
    mov rdx, r8
    mov rcx, r9
    xor rax, rax
    mov r10, 16

.loop:
    test rdx, rdx
    jz .end

    mov rax, rdx
    and rax, 0xF
    mov al, [g_tabulka + rax]

    mov [rcx], al
    inc rcx

    shr rdx, 4
    dec r10
    jnz .loop 

.end:
    mov byte [rcx], 0 
    ret

;***************************************************************************
    ; zadani 3 - Zjistěte, zda je v řetězci více malý či velkým písmen.
    global pismena
pismena:
    xor rax, rax
    mov rcx, 0

.back:
    cmp byte [rdi + rcx], byte 0
    je .konec

    cmp byte [rdi + rcx], byte 65
    jle .velke
    cmp byte [rdi + rcx], byte 90
    jge .velke


    inc rax
    jmp .next

.velke:
    dec rax

.next:
    inc rcx
    jmp .back

.konec:
    ret


;***************************************************************************
    ; zadani 4 - Spočítejte faktoriál čísla int a výsledek vraťte jako hodnotu long. Pokud dojde k přetečení při výpočtu, bude výsledek 0.
    global faktorial
faktorial:
    mov rax, 1
    mov rcx, 1
    
.loop:
    cmp rcx, rdi
    jg .end
    imul rax, rcx
    jo .overflow
    inc rcx
    jmp .loop


.overflow:
    xor rax, rax
.end:
    ret 
    
    
;***************************************************************************
    ; zadani 5 - Které číslo v poli čísel int má nejvyšší zbytek po dělení číslem K? Vynulujte v poli všechna čísla, která mají zbytek po dělení menší, než ten nejvyšší.
    global nejvetsi_modulo

nejvetsi_modulo:
    mov rcx, 0
    mov r11, 0
    mov r10, rdx

.back:
    cmp rcx, rsi
    je .end
    mov eax, dword [rdi + 4*rcx]
    cdq
    idiv r10
    cmp rdx, r11
    jg .vyssi
    inc rcx
    jmp .back

.vyssi:
    mov r11, rdx
    inc rcx
    jmp .back
.end:
    mov rax, r11
    mov rcx, 0

.while:
    cmp rcx, rsi
    je .konec
    mov eax, dword [rdi + 4*rcx]
    cdq
    idiv r10
    cmp rdx, r11
    jl .nulovani
    inc rcx
    jmp .while
.nulovani:
    mov dword [rdi + 4*rcx], 0
    inc rcx
    jmp .while

.konec:
    mov rax, r11
    ret


;***************************************************************************
    ; zadani 6 a - Implementujte si funkci pro převod řetězce na velká či malá písmena. Podmíněný skok využijte jen pro cyklus, pro převod znaků se snažte využít jen instrukce CMOVxx.
    global pismenaNaVelke
pismenaNaVelke:
    xor rax, rax
    mov rcx, 0

.back:
    cmp byte [rdi + rcx], byte 0
    je .konec

    mov edx, [rdi + rcx]
    xor edx, 32
    cmp edx, [rdi + rcx]
    cmovge edx, [rdi + rcx]
    mov [rdi + rcx], edx

    inc rcx
    jmp .back

.konec:
    ret

;***************************************************************************
    ; zadani 6 b - Implementujte si funkci pro převod řetězce na velká či malá písmena. Podmíněný skok využijte jen pro cyklus, pro převod znaků se snažte využít jen instrukce CMOVxx.
    global pismenaNaMale
pismenaNaMale:
    xor rax, rax
    mov rcx, 0

.back:
    cmp byte [rdi + rcx], byte 0
    je .konec

    mov edx, [rdi + rcx]
    xor edx, 32
    cmp edx, [rdi + rcx]
    cmovbe edx, [rdi + rcx]
    mov [rdi + rcx], edx

    inc rcx
    jmp .back

.konec:
    ret

;***************************************************************************
; zadani 7 - Ověřte, zda je zadané číslo long prvočíslem.
    global ukol7
ukol7:
    xor r11, r11
    cmp rdi, 0
    je .not_prime
    cmp rdi, 1
    je .not_prime

    mov rcx, 1
.loop:
    cmp rcx, rdi
    jge .end

    mov rdx, 0
    mov rax, rdi
    idiv rcx
    cmp rdx, 0
    je .delitel

    inc rcx
    jmp .loop 
    
.delitel:
    inc r11
    inc rcx
    jmp .loop 

.end:
    cmp r11, 2
    jb .prime

.not_prime:
    mov rax, 0
    ret

.prime:
    mov rax, 1
    ret
