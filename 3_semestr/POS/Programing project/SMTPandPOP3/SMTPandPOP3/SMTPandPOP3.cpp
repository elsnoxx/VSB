#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

// Jednoduchý nástroj na odesílání p?íkaz? a ?tení odpov?dí
void send_command(int socket_fd, const std::string& command) {
    send(socket_fd, command.c_str(), command.size(), 0);
    std::cout << "Sent: " << command;

    char buffer[1024] = { 0 };
    recv(socket_fd, buffer, sizeof(buffer), 0);
    std::cout << "Response: " << buffer << std::endl;
}

int main() {
    const std::string smtp_server = "smtp.example.com"; // Zm?? na vlastní SMTP server
    const int smtp_port = 25; // Port 25 pro nešifrované p?ipojení
    const std::string from_email = "sender@example.com";
    const std::string to_email = "recipient@example.com";
    const std::string email_subject = "Test Email";
    const std::string email_body = "This is a test email sent using raw sockets.";

    // Vytvo?ení socketu
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        std::cerr << "Failed to create socket." << std::endl;
        return 1;
    }

    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(smtp_port);

    // P?eklad IP adresy SMTP serveru
    if (inet_pton(AF_INET, smtp_server.c_str(), &server_address.sin_addr) <= 0) {
        std::cerr << "Invalid address/ Address not supported." << std::endl;
        close(socket_fd);
        return 1;
    }

    // P?ipojení k serveru
    if (connect(socket_fd, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Connection failed." << std::endl;
        close(socket_fd);
        return 1;
    }

    char buffer[1024] = { 0 };
    recv(socket_fd, buffer, sizeof(buffer), 0); // ?tení úvodní zprávy
    std::cout << "Server: " << buffer << std::endl;

    // SMTP handshake
    send_command(socket_fd, "HELO client.example.com\r\n");

    // Odeslání e-mailu
    send_command(socket_fd, "MAIL FROM:<" + from_email + ">\r\n");
    send_command(socket_fd, "RCPT TO:<" + to_email + ">\r\n");
    send_command(socket_fd, "DATA\r\n");
    send_command(socket_fd, "Subject: " + email_subject + "\r\n\r\n" + email_body + "\r\n.\r\n");

    // Ukon?ení spojení
    send_command(socket_fd, "QUIT\r\n");

    close(socket_fd);
    return 0;
}
