// ...existing code...
    /* create pipe for convert stdin (we will write image bytes into pipe_in[1]) */
    int pipe_in[2];
    if ( pipe(pipe_in) < 0 )
    {
        log_msg( LOG_ERROR, "pipe() failed for convert stdin" );
        close(pipefd[0]); close(pipefd[1]);
        close( t_sock_client );
        exit(1);
    }

    // First child: convert -> reads from pipe_in[0], writes to pipefd[1]
    pid_t pid_convert = fork();
    if ( pid_convert < 0 )
    {
        log_msg( LOG_ERROR, "fork() failed for convert" );
        close(pipefd[0]); close(pipefd[1]);
        close(pipe_in[0]); close(pipe_in[1]);
        close( t_sock_client );
        exit(1);
    }

    if ( pid_convert == 0 )
    {   // convert child
        // stdin <- pipe_in[0]
        dup2(pipe_in[0], STDIN_FILENO);
        // stdout -> pipe write (to xz)
        dup2(pipefd[1], STDOUT_FILENO);
        // close unused fds
        close(pipe_in[0]); close(pipe_in[1]);
        close(pipefd[0]); close(pipefd[1]);
        close(t_sock_client);
        // exec convert: read input from stdin ("-") and write output to stdout ("-")
        execlp("convert", "convert", "-resize", resize_arg, "-", "-", (char*)NULL);
        _exit(1);
    }

    // parent (per-client handler) — we will write the image bytes into pipe_in[1]
    close(pipe_in[0]); // parent doesn't read from convert stdin

    /* If you already have image in memory: */
    // unsigned char *img; size_t img_len;  // assume filled earlier
    // const unsigned char *p = img; size_t rem = img_len;
    // while ( rem > 0 ) { ssize_t w = write(pipe_in[1], p, rem); if (w <= 0) break; p += w; rem -= w; }

    /* Otherwise read local file and stream it into convert stdin: */
    int fd_img = open("podzim.png", O_RDONLY);
    if ( fd_img >= 0 )
    {
        char ibuf[4096];
        ssize_t r;
        while ( (r = read(fd_img, ibuf, sizeof(ibuf))) > 0 )
        {
            char *wp = ibuf;
            ssize_t towrite = r;
            while ( towrite > 0 )
            {
                ssize_t w = write(pipe_in[1], wp, towrite);
                if ( w <= 0 ) { if (errno == EINTR) continue; break; }
                wp += w; towrite -= w;
            }
        }
        close(fd_img);
    }
    /* finished feeding convert stdin */
    close(pipe_in[1]);
// ...existing code continues (fork xz, waitpid, etc.) ...