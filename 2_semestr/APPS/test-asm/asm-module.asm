    bits 64

    section .data

    section .text

;***************************************************************************
    ; zadani 3 - Ověřte, zda je zadané číslo X mocnicnou čísla M.  Výsledek bude -1 nebo příslušná mocnina.
    global je_mocnina
je_mocnina:
    xor rax, rax
    mov rax, 1

.loop: 
    imul rsi
    cmp rax, rdi
    jg .not
    cmp rax, rdi
    je .ano
    cmp rax, rdi
    jb .loop

.ano:
    mov rax, rsi
    ret

.not:   
    mov rax, -1
    ret


; zadani 2 - Vyplňte pole mocnimami čísla X. Při přetečení budou další výsledky 0.
    global mocniny
mocniny:
    mov r8, rdi
    mov r9, rdx
    xor rcx, rcx

.next:
    mov rax, r9
    mov rbx, rcx
    mul r9
    jc .overflow
    mov qword [r8], rax

    inc rcx
    add r8, 8
    cmp rcx, rsi
    mov r9, rax
    jl .next

    ret  
     
.overflow:
    mov rcx, rsi
    mov qword [r8], 0

    inc rcx
    add r8, 8
    cmp rcx, rsi
    jl .next

    ret 

; zadani 1 - Spočítejte, kolik čísel v poli je v zadaném rozsahu.
   global in_range
in_range:
    mov r10, rcx
    xor rcx, rcx
    mov ecx, esi            
    xor r9d, r9d 

.loop:          
    mov rax, [rdi + (rcx - 1) * 8]
    cmp rax, rdx
    jl  .next

    cmp rax, r10
    jg  .next

    inc r9d
.next:          
    loop .loop
    mov eax, r9d
    ret