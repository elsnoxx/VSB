//***************************************************************************
//
// Program example for labs in subject Operating Systems
//
// Petr Olivka, Dept. of Computer Science, petr.olivka@vsb.cz, 2017
//
// Example of socket server.
//
// This program is example of socket server and it allows to connect and serve
// the only one client.
// The mandatory argument of program is port number for listening.
//
//***************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/select.h>
#include <poll.h>

#define STR_CLOSE   "close"
#define STR_QUIT    "quit"

//***************************************************************************
// log messages

#define LOG_ERROR               0       // errors
#define LOG_INFO                1       // information and notifications
#define LOG_DEBUG               2       // debug messages

// debug flag
int g_debug = LOG_INFO;

void log_msg( int t_log_level, const char *t_form, ... )
{
    const char *out_fmt[] = {
            "ERR: (%d-%s) %s\n",
            "INF: %s\n",
            "DEB: %s\n" };

    if ( t_log_level && t_log_level > g_debug ) return;

    char l_buf[ 1024 ];
    va_list l_arg;
    va_start( l_arg, t_form );
    vsprintf( l_buf, t_form, l_arg );
    va_end( l_arg );

    switch ( t_log_level )
    {
    case LOG_INFO:
    case LOG_DEBUG:
        fprintf( stdout, out_fmt[ t_log_level ], l_buf );
        break;

    case LOG_ERROR:
        fprintf( stderr, out_fmt[ t_log_level ], errno, strerror( errno ), l_buf );
        break;
    }
}

//***************************************************************************
// help

void help( int t_narg, char **t_args )
{
    if ( t_narg <= 1 || !strcmp( t_args[ 1 ], "-h" ) )
    {
        printf(
            "\n"
            "  Socket server example.\n"
            "\n"
            "  Use: %s [-h -d] port_number\n"
            "\n"
            "    -d  debug mode \n"
            "    -h  this help\n"
            "\n", t_args[ 0 ] );

        exit( 0 );
    }

    if ( !strcmp( t_args[ 1 ], "-d" ) )
        g_debug = LOG_DEBUG;
}

//***************************************************************************
void handle_client( int t_sock_client )
{
    log_msg( LOG_INFO, "Child process handling client communication." );
    
    char l_buf[ 256 ];
    
    // Read resolution from client (first message)
    int l_len = read( t_sock_client, l_buf, sizeof( l_buf ) - 1 );
    if ( l_len <= 0 )
    {
        log_msg( LOG_ERROR, "Unable to read resolution from client." );
        close( t_sock_client );
        exit( 1 );
    }
    
    // Null terminate the string
    l_buf[l_len] = '\0';
    
    // Remove newline if present
    char* newline = strchr(l_buf, '\n');
    if (newline) *newline = '\0';
    
    log_msg( LOG_INFO, "Received resolution request: '%s'", l_buf );
    
    // Create resize argument string (with '!' as required)
    char resize_arg[64];
    snprintf(resize_arg, sizeof(resize_arg), "%.*s!", (int)(sizeof(resize_arg) - 2), l_buf);
    
    log_msg( LOG_INFO, "Starting convert | xz with resolution %s", resize_arg );

    int pipefd[2];
    if ( pipe(pipefd) < 0 )
    {
        log_msg( LOG_ERROR, "pipe() failed" );
        close( t_sock_client );
        exit(1);
    }

    // First child: convert -> writes to pipefd[1]
    pid_t pid_convert = fork();
    if ( pid_convert < 0 )
    {
        log_msg( LOG_ERROR, "fork() failed for convert" );
        close(pipefd[0]); close(pipefd[1]);
        close( t_sock_client );
        exit(1);
    }

    if ( pid_convert == 0 )
    {   // convert child
        // stdout -> pipe write
        dup2(pipefd[1], STDOUT_FILENO);
        // close unused fds
        // close(pipefd[0]);
        // close(pipefd[1]);
        close(t_sock_client);
        // exec convert
        // g++ -D NAME=jmeno pozdrav.cpp out.bin
        // execlp("convert", "convert", "-resize", resize_arg, "podzim.png", "-", (char*)NULL);
        char def[128];
        snprintf(def, sizeof(def), "NAME=%s", resize_arg);
        execlp("g++", "g++", "-D", def, "pozdrav.cpp", "-o" ,"out.bin", (char*)NULL);
        // if exec fails
        log_msg( LOG_ERROR, "exec convert failed" );
        _exit(1);
    }

    

    // parent (the original per-client child): close pipe ends and socket, wait for children
    close(pipefd[0]);
    close(pipefd[1]);

    int status;
    waitpid(pid_convert, &status, 0);
    if ( WIFEXITED(status) )
        log_msg( LOG_INFO, "convert exited %d", WEXITSTATUS(status) );
    else
        log_msg( LOG_ERROR, "convert terminated abnormally" );

    // Second child: xz reads from pipefd[0] and writes to client socket
    pid_t pid_xz = fork();
    if ( pid_xz < 0 )
    {
        log_msg( LOG_ERROR, "fork() failed for xz" );
        // try to clean up
        kill(pid_convert, SIGTERM);
        close(pipefd[0]); close(pipefd[1]);
        close( t_sock_client );
        exit(1);
    }

    if ( pid_xz == 0 )
    {   // xz child
        // stdin <- pipe read
        // dup2(pipefd[0], STDIN_FILENO);
        // stdout -> client socket
        dup2(t_sock_client, STDOUT_FILENO);
        // close unused fds
        // close(pipefd[0]);
        // close(pipefd[1]);
        close(t_sock_client);
        // exec xz: xz - --stdout
        // execlp("xz", "xz", "-", "--stdout", (char*)NULL);
        // xz out.bin --stdout
        execlp("xz", "xz", "out.bin", "--stdout", (char*)NULL);
        log_msg( LOG_ERROR, "exec xz failed" );
        _exit(1);
    }

    waitpid(pid_xz, &status, 0);
    if ( WIFEXITED(status) )
        log_msg( LOG_INFO, "xz exited %d", WEXITSTATUS(status) );
    else
        log_msg( LOG_ERROR, "xz terminated abnormally" );

    // close client socket (xz child already duplicated it, but close here too)
    close( t_sock_client );

    log_msg( LOG_INFO, "Child process finished handling client." );
    exit( 0 );
}

//***************************************************************************

int main( int t_narg, char **t_args )
{
    if ( t_narg <= 1 ) help( t_narg, t_args );

    int l_port = 0;

    // parsing arguments
    for ( int i = 1; i < t_narg; i++ )
    {
        if ( !strcmp( t_args[ i ], "-d" ) )
            g_debug = LOG_DEBUG;

        if ( !strcmp( t_args[ i ], "-h" ) )
            help( t_narg, t_args );

        if ( *t_args[ i ] != '-' && !l_port )
        {
            l_port = atoi( t_args[ i ] );
            break;
        }
    }

    if ( l_port <= 0 )
    {
        log_msg( LOG_INFO, "Bad or missing port number %d!", l_port );
        help( t_narg, t_args );
    }

    log_msg( LOG_INFO, "Server will listen on port: %d.", l_port );

    // socket creation
    int l_sock_listen = socket( AF_INET, SOCK_STREAM, 0 );
    if ( l_sock_listen == -1 )
    {
        log_msg( LOG_ERROR, "Unable to create socket.");
        exit( 1 );
    }

    in_addr l_addr_any = { INADDR_ANY };
    sockaddr_in l_srv_addr;
    l_srv_addr.sin_family = AF_INET;
    l_srv_addr.sin_port = htons( l_port );
    l_srv_addr.sin_addr = l_addr_any;

    // Enable the port number reusing
    int l_opt = 1;
    if ( setsockopt( l_sock_listen, SOL_SOCKET, SO_REUSEADDR, &l_opt, sizeof( l_opt ) ) < 0 )
      log_msg( LOG_ERROR, "Unable to set socket option!" );

    // assign port number to socket
    if ( bind( l_sock_listen, (const sockaddr * ) &l_srv_addr, sizeof( l_srv_addr ) ) < 0 )
    {
        log_msg( LOG_ERROR, "Bind failed!" );
        close( l_sock_listen );
        exit( 1 );
    }

    // listenig on set port
    if ( listen( l_sock_listen, 1 ) < 0 )
    {
        log_msg( LOG_ERROR, "Unable to listen on given port!" );
        close( l_sock_listen );
        exit( 1 );
    }

    log_msg( LOG_INFO, "Enter 'quit' to quit server." );

    // go!
    while ( 1 )
    {
        struct pollfd fds[2];
        fds[0].fd = STDIN_FILENO;
        fds[0].events = POLLIN;
        fds[1].fd = l_sock_listen;
        fds[1].events = POLLIN;

        int l_poll = poll(fds, 2, -1);
        if ( l_poll < 0 )
        {
            log_msg( LOG_ERROR, "Function poll failed!" );
            exit( 1 );
        }

        if ( fds[0].revents & POLLIN )
        { // data on stdin
            char buf[ 128 ];
            int l_len = read( STDIN_FILENO, buf, sizeof( buf) );
            if ( l_len == 0 )
            {
                log_msg( LOG_DEBUG, "Stdin closed." );
                exit( 0 );
            }
            if ( l_len < 0 )
            {
                log_msg( LOG_DEBUG, "Unable to read from stdin!" );
                exit( 1 );
            }

            log_msg( LOG_DEBUG, "Read %d bytes from stdin", l_len );
            // request to quit?
            if ( !strncmp( buf, STR_QUIT, strlen( STR_QUIT ) ) )
            {
                log_msg( LOG_INFO, "Request to 'quit' entered.");
                close( l_sock_listen );
                exit( 0 );
            }
        }

        if ( fds[1].revents & POLLIN )
        { // new client connection
            sockaddr_in l_rsa;
            int l_rsa_size = sizeof( l_rsa );

            // accept new connection
            int l_sock_client = accept( l_sock_listen, ( sockaddr * ) &l_rsa, ( socklen_t * ) &l_rsa_size );
            if ( l_sock_client == -1 )
            {
                log_msg( LOG_ERROR, "Unable to accept new client." );
                continue;
            }

            uint l_lsa = sizeof( l_srv_addr );
            // my IP
            getsockname( l_sock_client, ( sockaddr * ) &l_srv_addr, &l_lsa );
            log_msg( LOG_INFO, "My IP: '%s'  port: %d",
                             inet_ntoa( l_srv_addr.sin_addr ), ntohs( l_srv_addr.sin_port ) );
            // client IP
            getpeername( l_sock_client, ( sockaddr * ) &l_srv_addr, &l_lsa );
            log_msg( LOG_INFO, "Client IP: '%s'  port: %d",
                             inet_ntoa( l_srv_addr.sin_addr ), ntohs( l_srv_addr.sin_port ) );

            // fork to handle client
            pid_t pid = fork();

            if (pid == 0)
            { // child process
                close( l_sock_listen ); // child doesn't need listen socket
                handle_client( l_sock_client );
                // handle_client will close l_sock_client and exit
            }
            else if (pid > 0)
            { // parent process
                close( l_sock_client ); // parent doesn't need client socket
                log_msg( LOG_INFO, "Forked process %d to handle client.", pid );
            }
            else
            { // fork failed
                log_msg( LOG_ERROR, "Fork failed!" );
                close( l_sock_client );
            }
        }
    } // while ( 1 )

    return 0;
}
