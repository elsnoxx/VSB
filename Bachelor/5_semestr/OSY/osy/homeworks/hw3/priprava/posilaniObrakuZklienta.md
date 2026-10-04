Níže stručně ukázka jak upravit chování — klient pošle řádek s rozlišením a hned za ním binární data obrázku; server přijme řádek, uloží případný přebytek (část obrázku, která byla načtena při čtení řádku) a následně předá celý obraz do convert přes interní pipe; convert->pipe->xz->socket pošle výsledek zpět klientovi. Klient po odeslání obsahu obraz souboru zavolá shutdown(sock, SHUT_WR) aby server dostal EOF a mohl konvertovat.

Server — nahraďte funkci handle_client takto:

```c++
// ...existing code...
void handle_client( int t_sock_client )
{
    log_msg(LOG_INFO,"Child process handling client.");

    char l_buf[4096];
    // read initial chunk (may contain resolution + start of image)
    int l_len = read(t_sock_client, l_buf, sizeof(l_buf));
    if (l_len <= 0) { log_msg(LOG_ERROR,"Read failed"); close(t_sock_client); exit(1); }

    // find newline -> resolution line
    int i;
    for (i = 0; i < l_len && l_buf[i] != '\n'; ++i);
    if (i == l_len) { log_msg(LOG_ERROR,"No resolution line"); close(t_sock_client); exit(1); }

    // extract resolution
    char res[64];
    int rlen = (i < (int)sizeof(res)-1) ? i : (int)sizeof(res)-1;
    memcpy(res, l_buf, rlen); res[rlen] = '\0';
    log_msg(LOG_INFO,"Resolution: %s", res);

    // prepare resize argument (with '!')
    char resize_arg[64];
    snprintf(resize_arg, sizeof(resize_arg), "%s!", res);

    // create pipe for convert stdout -> xz stdin
    int pipe_conv[2];
    if (pipe(pipe_conv) < 0) { log_msg(LOG_ERROR,"pipe failed"); close(t_sock_client); exit(1); }

    // create pipe to feed convert stdin (server will write image data here)
    int pipe_data[2];
    if (pipe(pipe_data) < 0) { log_msg(LOG_ERROR,"pipe failed"); close(pipe_conv[0]); close(pipe_conv[1]); close(t_sock_client); exit(1); }

    // fork convert
    pid_t pid_convert = fork();
    if (pid_convert == 0)
    {
        // convert child: stdin <- pipe_data[0], stdout -> pipe_conv[1]
        dup2(pipe_data[0], STDIN_FILENO);
        dup2(pipe_conv[1], STDOUT_FILENO);
        close(pipe_data[0]); close(pipe_data[1]);
        close(pipe_conv[0]); close(pipe_conv[1]);
        close(t_sock_client);
        execlp("convert","convert","-resize", resize_arg, "-", "-", (char*)NULL);
        _exit(1);
    }
    else if (pid_convert < 0) { log_msg(LOG_ERROR,"fork convert failed"); close(pipe_data[0]); close(pipe_data[1]); close(pipe_conv[0]); close(pipe_conv[1]); close(t_sock_client); exit(1); }

    // fork xz
    pid_t pid_xz = fork();
    if (pid_xz == 0)
    {
        // xz child: stdin <- pipe_conv[0], stdout -> client socket
        dup2(pipe_conv[0], STDIN_FILENO);
        dup2(t_sock_client, STDOUT_FILENO);
        close(pipe_data[0]); close(pipe_data[1]);
        close(pipe_conv[0]); close(pipe_conv[1]);
        close(t_sock_client);
        execlp("xz","xz","-","--stdout",(char*)NULL);
        _exit(1);
    }
    else if (pid_xz < 0) { log_msg(LOG_ERROR,"fork xz failed"); kill(pid_convert,SIGTERM); close(pipe_data[0]); close(pipe_data[1]); close(pipe_conv[0]); close(pipe_conv[1]); close(t_sock_client); exit(1); }

    // parent (per-client handler) — close unused read ends we won't use
    close(pipe_data[0]);
    close(pipe_conv[0]);
    close(pipe_conv[1]); // xz has stdin duped, we don't need write end of conv pipe

    // write any leftover bytes (after newline) into pipe_data
    int leftover = l_len - (i + 1);
    if (leftover > 0)
    {
        int wrote = write(pipe_data[1], l_buf + i + 1, leftover);
        if (wrote != leftover) log_msg(LOG_ERROR,"Partial write of leftover");
    }

    // now forward rest of data from client socket into pipe_data
    while (1)
    {
        int n = read(t_sock_client, l_buf, sizeof(l_buf));
        if (n < 0) { if (errno == EINTR) continue; log_msg(LOG_ERROR,"Read error from client"); break; }
        if (n == 0) break; // client finished upload
        int off = 0;
        while (off < n)
        {
            int w = write(pipe_data[1], l_buf + off, n - off);
            if (w < 0) { if (errno == EINTR) continue; log_msg(LOG_ERROR,"Write to convert pipe failed"); break; }
            off += w;
        }
    }

    // close write end to signal EOF to convert
    close(pipe_data[1]);
    // close client socket write-side is still open for xz to send back, but parent can keep socket open; xz writes directly to socket
    // wait for children
    int st;
    waitpid(pid_convert, &st, 0);
    waitpid(pid_xz, &st, 0);
    close(t_sock_client);
    log_msg(LOG_INFO,"Finished handling client");
    exit(0);
}
// ...existing code...
```

Klient — po odeslání rozlišení pošlete obsah lokálního souboru (např. podzim.png) a potom zavolejte shutdown(sock, SHUT_WR); poté čtěte odpověď a uložíte do image.img (zbytek logiky zůstává). Vložte/nahraďte část po odeslání resolution takto:


```c++
// ...existing code...
    // Send resolution to server immediately after connection
    char l_resolution_msg[256];
    snprintf( l_resolution_msg, sizeof(l_resolution_msg), "%s\n", l_resolution );
    if ( write( l_sock_server, l_resolution_msg, strlen(l_resolution_msg) ) < 0 ) { /* error */ }

    // open local image file and send it to server
    int fd_img = open("podzim.png", O_RDONLY);
    if ( fd_img >= 0 )
    {
        char buf[4096];
        int r;
        while ( (r = read(fd_img, buf, sizeof(buf))) > 0 )
        {
            int off = 0;
            while ( off < r )
            {
                int w = write(l_sock_server, buf + off, r - off);
                if ( w < 0 ) { if (errno == EINTR) continue; /* handle error */ break; }
                off += w;
            }
        }
        close(fd_img);
    }

    // signal end of upload (so server's convert gets EOF) but keep socket open to receive result
    shutdown(l_sock_server, SHUT_WR);

// ...then continue reading from socket as already implemented and save into image.img...
// ...existing code...
```