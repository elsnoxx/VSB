//***************************************************************************
//
// Program example for subject Operating Systems
//
// Petr Olivka, Dept. of Computer Science, petr.olivka@vsb.cz, 2021
//
// Example of socket server/client.
//
// This program is example of socket client.
// The mandatory arguments of program is IP adress or name of server and
// a port number.
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
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/wait.h>

#define STR_CLOSE               "close"

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
            "  Socket client example.\n"
            "\n"
            "  Use: %s [-h -d] ip_or_name port_number resolution\n"
            "\n"
            "    -d  debug mode \n"
            "    -h  this help\n"
            "    resolution format: WIDTHxHEIGHT (e.g., 1400x700)\n"
            "\n", t_args[ 0 ] );

        exit( 0 );
    }

    if ( !strcmp( t_args[ 1 ], "-d" ) )
        g_debug = LOG_DEBUG;
}

//***************************************************************************

int main( int t_narg, char **t_args )
{

    if ( t_narg <= 3 ) help( t_narg, t_args );

    int l_port = 0;
    char *l_host = nullptr;
    char *l_resolution = nullptr;

    // parsing arguments
    for ( int i = 1; i < t_narg; i++ )
    {
        if ( !strcmp( t_args[ i ], "-d" ) )
            g_debug = LOG_DEBUG;

        if ( !strcmp( t_args[ i ], "-h" ) )
            help( t_narg, t_args );

        if ( *t_args[ i ] != '-' )
        {
            if ( !l_host )
                l_host = t_args[ i ];
            else if ( !l_port )
                l_port = atoi( t_args[ i ] );
            else if ( !l_resolution )
                l_resolution = t_args[ i ];
        }
    }

    if ( !l_host || !l_port || !l_resolution )
    {
        log_msg( LOG_INFO, "Host, port or resolution is missing!" );
        help( t_narg, t_args );
        exit( 1 );
    }

    log_msg( LOG_INFO, "Connection to '%s':%d with resolution %s.", l_host, l_port, l_resolution );

    addrinfo l_ai_req, *l_ai_ans;
    bzero( &l_ai_req, sizeof( l_ai_req ) );
    l_ai_req.ai_family = AF_INET;
    l_ai_req.ai_socktype = SOCK_STREAM;

    int l_get_ai = getaddrinfo( l_host, nullptr, &l_ai_req, &l_ai_ans );
    if ( l_get_ai )
    {
        log_msg( LOG_ERROR, "Unknown host name!" );
        exit( 1 );
    }

    sockaddr_in l_cl_addr =  *( sockaddr_in * ) l_ai_ans->ai_addr;
    l_cl_addr.sin_port = htons( l_port );
    freeaddrinfo( l_ai_ans );

    // socket creation
    int l_sock_server = socket( AF_INET, SOCK_STREAM, 0 );
    if ( l_sock_server == -1 )
    {
        log_msg( LOG_ERROR, "Unable to create socket.");
        exit( 1 );
    }

    // connect to server
    if ( connect( l_sock_server, ( sockaddr * ) &l_cl_addr, sizeof( l_cl_addr ) ) < 0 )
    {
        log_msg( LOG_ERROR, "Unable to connect server." );
        exit( 1 );
    }

    uint l_lsa = sizeof( l_cl_addr );
    // my IP
    getsockname( l_sock_server, ( sockaddr * ) &l_cl_addr, &l_lsa );
    log_msg( LOG_INFO, "My IP: '%s'  port: %d",
             inet_ntoa( l_cl_addr.sin_addr ), ntohs( l_cl_addr.sin_port ) );
    // server IP
    getpeername( l_sock_server, ( sockaddr * ) &l_cl_addr, &l_lsa );
    log_msg( LOG_INFO, "Server IP: '%s'  port: %d",
             inet_ntoa( l_cl_addr.sin_addr ), ntohs( l_cl_addr.sin_port ) );

    // Open image file for writing
    int l_img_fd = open( "image.img", O_WRONLY | O_CREAT | O_TRUNC, 0644 );
    if ( l_img_fd < 0 )
    {
        log_msg( LOG_ERROR, "Unable to create image.img file." );
        close( l_sock_server );
        exit( 1 );
    }

    int l_img_fd2 = open( "runme.xz", O_WRONLY | O_CREAT | O_TRUNC, 0644 );
    if ( l_img_fd2 < 0 )
    {
        log_msg( LOG_ERROR, "Unable to create runme.xz file." );
        close( l_sock_server );
        exit( 1 );
    }

    // Send resolution to server immediately after connection
    char l_resolution_msg[256];
    snprintf( l_resolution_msg, sizeof(l_resolution_msg), "%s\n", l_resolution );
    int l_res_len = write( l_sock_server, l_resolution_msg, strlen(l_resolution_msg) );
    if ( l_res_len < 0 )
    {
        log_msg( LOG_ERROR, "Unable to send resolution to server." );
        close( l_sock_server );
        close( l_img_fd );
        exit( 1 );
    }
    else
        log_msg( LOG_INFO, "Sent resolution '%s' to server.", l_resolution );

    log_msg( LOG_INFO, "Waiting for image data from server..." );
    log_msg( LOG_INFO, "Enter 'close' to close application." );

    // Replace poll with select
    int total_bytes = 0;
    while ( 1 )
    {
        char l_buf[ 4096 ];  // Larger buffer for image data
        fd_set read_fds;
        int max_fd = (l_sock_server > STDIN_FILENO) ? l_sock_server : STDIN_FILENO;

        // select from fds
        FD_ZERO(&read_fds);
        FD_SET(STDIN_FILENO, &read_fds);
        FD_SET(l_sock_server, &read_fds);

        int l_select = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if ( l_select < 0 )
        {
            log_msg( LOG_ERROR, "Function select failed!" );
            break;
        }

        // data on stdin?
        if ( FD_ISSET(STDIN_FILENO, &read_fds) )
        {
            //  read from stdin
            int l_len = read( STDIN_FILENO, l_buf, sizeof( l_buf ) );
            if ( l_len == 0 )
            {
                log_msg( LOG_DEBUG, "Stdin closed." );
                break;
            }
            if ( l_len < 0 )
            {
                log_msg( LOG_ERROR, "Unable to read from stdin." );
                break;
            }
            else
                log_msg( LOG_DEBUG, "Read %d bytes from stdin.", l_len );

            // send data to server
            l_len = write( l_sock_server, l_buf, l_len );
            if ( l_len < 0 )
            {
                log_msg( LOG_ERROR, "Unable to send data to server." );
                break;
            }
            else
                log_msg( LOG_DEBUG, "Sent %d bytes to server.", l_len );
        }

        // data from server?
        if ( FD_ISSET(l_sock_server, &read_fds) )
        {
            // read data from server
            int l_len = read( l_sock_server, l_buf, sizeof( l_buf ) );
            if ( l_len == 0 )
            {
                log_msg( LOG_INFO, "Server closed socket. Image transfer complete." );
                log_msg( LOG_INFO, "Total bytes received: %d", total_bytes );
                log_msg( LOG_INFO, "Image saved as image.img" );
                break;
            }
            else if ( l_len < 0 )
            {
                log_msg( LOG_ERROR, "Unable to read data from server." );
                break;
            }
            else
            {
                log_msg( LOG_DEBUG, "Read %d bytes from server.", l_len );
                total_bytes += l_len;
            }

            // Save received data to image file
            // int l_write_len = write( l_img_fd, l_buf, l_len );
            int l_write_len = write( l_img_fd2, l_buf, l_len );
            if ( l_write_len < 0 )
            {
                log_msg( LOG_ERROR, "Unable to write data to image file." );
                break;
            }
            else if ( l_write_len != l_len )
            {
                log_msg( LOG_ERROR, "Partial write to image file." );
                break;
            }

            // Optional: display progress for large files
            if ( total_bytes % 10240 == 0 )  // Every 10KB
            {
                log_msg( LOG_INFO, "Received %d bytes so far...", total_bytes );
            }
        }
    }

    // close sockets and files
    close( l_sock_server );
    close( l_img_fd );
    close( l_img_fd2 );

    if ( total_bytes > 0 )
    {
        log_msg( LOG_INFO, "Image saved as image.img, launching viewer..." );

        int pd[2];
        if ( pipe(pd) < 0 )
        {
            log_msg( LOG_ERROR, "pipe() failed for viewer" );
            log_msg( LOG_INFO, "You can view with: xz -d image.img --stdout | display -" );
        }
        else
        {
            pid_t p1 = fork();
            if ( p1 < 0 )
            {
                log_msg( LOG_ERROR, "fork() failed for decompressor" );
                close(pd[0]); close(pd[1]);
                log_msg( LOG_INFO, "You can view with: xz -d image.img --stdout | display -" );
            }
            else if ( p1 == 0 )
            {
                // child 1: xz -d image.img --stdout  -> writes to pd[1]
                dup2(pd[1], STDOUT_FILENO);
                // close(pd[0]); close(pd[1]);
                // execlp("xz", "xz", "-d", "image.img", "--stdout", (char*)NULL);
                // runme.xz
                execlp("xz", "xz", "-d", "runme.xz", (char*)NULL);
                _exit(1);
            }
            else
            {
                int st;
                waitpid(p1, &st, 0);
                pid_t p2 = fork();
                if ( p2 < 0 )
                {
                    log_msg( LOG_ERROR, "fork() failed for viewer" );
                    // try to clean up
                    kill(p1, SIGTERM);
                    close(pd[0]); close(pd[1]);
                    waitpid(p1, NULL, 0);
                    log_msg( LOG_INFO, "You can view with: xz -d image.img --stdout | display -" );
                }
                else if ( p2 == 0 )
                {
                    // child 2: display -  <- reads from pd[0]
                    dup2(pd[0], STDIN_FILENO);
                    close(pd[0]); close(pd[1]);
                    execlp("chmod", "chmod", "+x" ,"runme", (char*)NULL);
                    _exit(1);
                }
                else
                {
                    // parent closes pipe ends and waits for both children
                    close(pd[0]); close(pd[1]);

                    int st;
                    // waitpid(p1, &st, 0);
                    waitpid(p2, &st, 0);

                    log_msg( LOG_INFO, "Viewer closed, exiting client." );
                }


                pid_t p3 = fork();
                if ( p3 < 0 )
                {
                    log_msg( LOG_ERROR, "fork() failed for viewer" );
                    log_msg( LOG_INFO, "You can view with: xz -d image.img --stdout | display -" );
                }
                else if ( p3 == 0 )
                {
                    // child 3: run 
                    execlp("./runme", "./runme", (char*)NULL);
                    _exit(1);
                }
                else
                {
                    // parent closes pipe ends and waits for both children
                    close(pd[0]); close(pd[1]);

                    
                    waitpid(p1, &st, 0);
                    waitpid(p2, &st, 0);
                    waitpid(p3, &st, 0);

                    log_msg( LOG_INFO, "Viewer closed, exiting client." );
                }
            }
        }
    }

    return 0;
  }
