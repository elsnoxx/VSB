#include "pop3_client.h"
#include <iostream>
#include <string>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <fstream>
#include <sstream>
#pragma comment(lib, "ws2_32.lib")


void send_pop3_command(SOCKET socket_fd, const std::string& command, std::string& response) {
    send(socket_fd, command.c_str(), command.size(), 0);
    std::cout << "Sent: " << command;

    char buffer[1024] = { 0 };
    int bytes_received = recv(socket_fd, buffer, sizeof(buffer), 0);
    if (bytes_received <= 0) {
        throw std::runtime_error("Failed to receive response from POP3 server.");
    }
    response = std::string(buffer, bytes_received);
    std::cout << "Response: " << response << std::endl;


    if (response.rfind("-ERR", 0) == 0) {
        throw std::runtime_error("POP3 server returned an error: " + response);
    }
}


bool is_message_number_valid(int message_number, const std::string& list_response) {
    std::istringstream response_stream(list_response);
    std::string line;
    std::string message_number_str = std::to_string(message_number);

    while (std::getline(response_stream, line)) {
        if (line.find(message_number_str) == 0) {
            return true;
        }
    }
    return false;
}


// Hlavní funkce pro stahování e-mailù
void retrieve_email(const std::string& pop3_server, int pop3_port, const std::string& username, const std::string& password, int message_number) {
    std::string filename = "downloaded_email.txt";
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
    server_address.sin_port = htons(pop3_port);

    // URL to IP
    struct addrinfo hints = {}, * res = nullptr;
    hints.ai_family = AF_INET; // IPv4
    hints.ai_socktype = SOCK_STREAM; // TCP

    if (getaddrinfo(pop3_server.c_str(), nullptr, &hints, &res) != 0) {
        closesocket(socket_fd);
        WSACleanup();
        throw std::runtime_error("Failed to resolve server address.");
    }

    server_address.sin_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr;
    freeaddrinfo(res);

    // Connection to server
    if (connect(socket_fd, (struct sockaddr*)&server_address, sizeof(server_address)) == SOCKET_ERROR) {
        closesocket(socket_fd);
        WSACleanup();
        throw std::runtime_error("Connection failed.");
    }

    char buffer[1024] = { 0 };
    recv(socket_fd, buffer, sizeof(buffer), 0);
    std::cout << "Server: " << buffer << std::endl;


    // login
    std::string response;
    send_pop3_command(socket_fd, "USER " + username + "\r\n", response);
    send_pop3_command(socket_fd, "PASS " + password + "\r\n", response);

    // List of emails
    send_pop3_command(socket_fd, "LIST\r\n", response);

    if (!is_message_number_valid(message_number, response)) {
        std::cerr << "Error: Invalid email number. The message does not exist.\n";
        send_pop3_command(socket_fd, "QUIT\r\n", response);
        closesocket(socket_fd);
        WSACleanup();
        return;
    }

    // email download
    send_pop3_command(socket_fd, "RETR " + std::to_string(message_number) + "\r\n", response);
    std::cout << "Email content:\n" << response << std::endl;

    // ulozeni do souboru
    std::ofstream output_file(filename, std::ios::out | std::ios::binary);
    if (!output_file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    output_file << response;
    output_file.close();

    // Logout
    send_pop3_command(socket_fd, "QUIT\r\n", response);

    closesocket(socket_fd);
    WSACleanup();
}
