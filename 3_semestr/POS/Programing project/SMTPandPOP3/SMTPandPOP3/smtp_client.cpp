#include "smtp_client.h"
#include <iostream>
#include <string>
#include <sstream>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iomanip>
#pragma comment(lib, "ws2_32.lib")


void send_command(SOCKET socket_fd, const std::string& command) {
    send(socket_fd, command.c_str(), command.size(), 0);
    std::cout << "Sent: " << command;

    char buffer[1024] = { 0 };
    recv(socket_fd, buffer, sizeof(buffer), 0);
    std::cout << "Response: " << buffer << std::endl;
}

//// Funkce pro pøevod øetìzce na Base64
//std::string encode_base64(const std::string& input) {
//    static const char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
//    std::string output;
//    int val = 0, valb = -6;
//    for (unsigned char c : input) {
//        val = (val << 8) + c;
//        valb += 8;
//        while (valb >= 0) {
//            output.push_back(table[(val >> valb) & 0x3F]);
//            valb -= 6;
//        }
//    }
//    if (valb > -6) output.push_back(table[((val << 8) >> (valb + 8)) & 0x3F]);
//    while (output.size() % 4) output.push_back('=');
//    return output;
//}

//std::string get_current_datetime() {
//    std::time_t now = std::time(nullptr);
//    std::tm local_time;
//    localtime_s(&local_time, &now);
//
//    std::ostringstream oss;
//    oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S");
//    return oss.str();
//}
// Hlavní funkce pro odeslání e-mailu
void send_email(const std::string& smtp_server, int smtp_port, const std::string& from_email,
    const std::string& to_email, const std::string& username, const std::string& password) {
    std::string email_subject;
    std::string email_body;
    // Inicializace Winsock
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        throw std::runtime_error("Failed to initialize Winsock.");
    }

    // Vytvoøení socketu
    SOCKET socket_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_fd == INVALID_SOCKET) {
        WSACleanup();
        throw std::runtime_error("Failed to create socket.");
    }

    sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(smtp_port);

    // Z url na IP
    struct addrinfo hints = {}, * res = nullptr;
    hints.ai_family = AF_INET; // IPv4
    hints.ai_socktype = SOCK_STREAM; // TCP

    if (getaddrinfo(smtp_server.c_str(), nullptr, &hints, &res) != 0) {
        closesocket(socket_fd);
        WSACleanup();
        throw std::runtime_error("Failed to resolve server address.");
    }

    server_address.sin_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr;
    freeaddrinfo(res);


    // Pøipojení k serveru
    if (connect(socket_fd, (struct sockaddr*)&server_address, sizeof(server_address)) == SOCKET_ERROR) {
        closesocket(socket_fd);
        WSACleanup();
        throw std::runtime_error("Connection failed.");
    }

    char buffer[1024] = { 0 };
    recv(socket_fd, buffer, sizeof(buffer), 0);
    std::cout << "Server: " << buffer << std::endl;

    // SMTP handshake - vse na vpn jen
    send_command(socket_fd, "HELO client.example.com\r\n");
    //send_command(socket_fd, "AUTH LOGIN\r\n");
    //send_command(socket_fd, encode_base64(username) + "\r\n");
    //send_command(socket_fd, encode_base64(password) + "\r\n");

    // Odeslání e-mailu
    send_command(socket_fd, "MAIL FROM:<" + from_email + ">\r\n");
    send_command(socket_fd, "RCPT TO:<" + to_email + ">\r\n");
    send_command(socket_fd, "DATA\r\n");

    std::cout << "Enter email subject: ";
    std::getline(std::cin, email_subject); // Pøedmìt zprávy

    std::cout << "Enter email body: ";
    std::getline(std::cin, email_body); // Tìlo zprávy

    send_command(socket_fd, "Subject: " + email_subject + "\r\n\r\n" + email_body + "\r\n.\r\n");

    // End process
    send_command(socket_fd, "QUIT\r\n");

    closesocket(socket_fd);
    WSACleanup();
}
