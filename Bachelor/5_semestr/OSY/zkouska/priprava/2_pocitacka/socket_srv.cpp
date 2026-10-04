//***************************************************************************
//
// Program example for labs in subject Operating Systems
//
// Petr Olivka, Dept. of Computer Science, petr.olivka@vsb.cz, 2026
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
#include <poll.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/wait.h>
#include <netdb.h>
#include <pthread.h>

#define STR_CLOSE   "close"
#define STR_QUIT    "quit"
#define N 4

typedef struct {
    int sock;
    int clientsCount;
} ClientArg;

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

    char l_buf[ 4096 ];
    va_list l_arg;
    va_start( l_arg, t_form );
    vsnprintf( l_buf, sizeof( l_buf ), t_form, l_arg );
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

void* childProcess(void* arg){
    ClientArg* clientArg = (ClientArg*)arg;
    int l_sock_client = clientArg->sock;
    int clientsCount = clientArg->clientsCount;
    for (int i = 0; i < 10; i++){
        printf("%d\n", i);
        char l_buf[ 128 ];
    
        int l_len = read(l_sock_client, l_buf, sizeof( l_buf ));

        l_buf[l_len] = '\0';

        char c_buff[128];

        strncpy(c_buff, l_buf, l_len);    
        

        char* token = strtok(l_buf, "+-*/");
        int number[2];
        int cnt = 0;
        while(token != NULL){
            log_msg( LOG_INFO, "Received token: '%s'", token );
            number[cnt] = atoi(token);
            cnt++;
            token = strtok(NULL, "+-*/");
        }

        
        char* op = strpbrk(c_buff, "+-*/");
        char result[128];
        switch  (*op) {
            case '+':
                log_msg( LOG_INFO, "Addition operation." );
                sprintf(result, "%d", number[0] + number[1]);
                printf("Result: %s\n", result);
                l_len = write(l_sock_client, result, sizeof(result));
                break;
            case '-':
                log_msg( LOG_INFO, "Subtraction operation." );
                sprintf(result, "%d", number[0] - number[1]);
                l_len = write(l_sock_client, result, sizeof(result));
                break;
            case '*':
                log_msg( LOG_INFO, "Multiplication operation." );
                sprintf(result, "%d", number[0] * number[1]);
                l_len = write(l_sock_client, result, sizeof(result));
                break;
            case '/':
                log_msg( LOG_INFO, "Division operation." );
                if( number[1] == 0 ){
                    char err[] = "Error: Division by zero!";
                    l_len = write(l_sock_client, err, sizeof(err));
                    close(l_sock_client);
                    exit(0);
                }

                sprintf(result, "%d", number[0] / number[1]);
                l_len = write(l_sock_client, result, sizeof(result));
                break;
            default:
                log_msg( LOG_ERROR, "Unknown operation." );
                close(l_sock_client);
                exit(1);
        }
    }
    


    close(l_sock_client);
    exit(0);
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

    int client_count = 0;

    // go!
    while ( 1 )
    {
        int l_sock_client = -1;

        sockaddr_in l_rsa;
        int l_rsa_size = sizeof( l_rsa );
        // new connection
        l_sock_client = accept( l_sock_listen, ( sockaddr * ) &l_rsa, ( socklen_t * ) &l_rsa_size );
        if ( l_sock_client == -1 )
        {
            log_msg( LOG_ERROR, "Unable to accept new client." );
            close( l_sock_listen );
            exit( 1 );
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


        client_count++;

        if(client_count > N){
            log_msg( LOG_INFO, "Maximum number of clients reached. Connection refused." );
            int l_len = write(l_sock_client, "Maximum number of clients reached. Connection refused.\n", sizeof("Maximum number of clients reached. Connection refused.\n") + 1);
            close(l_sock_client);
            client_count--;
            continue;
        }

        pthread_t thread;
        ClientArg args = {l_sock_client, client_count};
        if(pthread_create(&thread, NULL,childProcess, &args) != 0){
            log_msg( LOG_ERROR, "Unable to create thread." );
            close(l_sock_client);
            client_count--;
            continue;
        }


    } // while ( 1 )

    return 0;
}