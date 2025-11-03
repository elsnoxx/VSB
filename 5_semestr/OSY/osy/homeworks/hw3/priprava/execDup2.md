
Stručně: ano — chápeš to správně. Doplnění (stručně):

dup2(oldfd, newfd) přesměruje/newfd tak, že newfd bude ukazovat na stejný otevřený soubor/socket jako oldfd. Pokud už newfd byl otevřený, dup2 ho zavře a nahradí. Vrací newfd nebo -1 při chybě.
To se dělá před exec, aby nový program (po execlp) už měl standardní vstup/výstup přesměrovaný (např. stdout → pipe nebo socket).
execlp(...) nahradí aktuální proces novým programem; pokud uspěje, už se nikdy nevrátí. Pokud selže, vrátí -1 a je potřeba volat _exit(1) (v childu).
Po fork() vždy zavírat nepotřebné fd v rodiči i v childu — jinak nebude fungovat EOF a mohou zůstávat visící popisovače.
Krátká ukázka pořadí v childu (schematicky):

// prepare fds: pipefd[0]=read, pipefd[1]=write, client_sock je socket
// convert child: stdout -> pipe write
dup2(pipefd[1], STDOUT_FILENO);
close(pipefd[0]); close(pipefd[1]);
close(client_sock);
execlp("convert", "convert", "-resize", resize_arg, "podzim.png", "-", (char*)NULL);
_exit(1); // exec selhal

// xz child: stdin <- pipe read, stdout -> client socket
dup2(pipefd[0], STDIN_FILENO);
dup2(client_sock, STDOUT_FILENO);
close(pipefd[0]); close(pipefd[1]);
close(client_sock);
execlp("xz", "xz", "-", "--stdout", (char*)NULL);
_exit(1); // exec selhal