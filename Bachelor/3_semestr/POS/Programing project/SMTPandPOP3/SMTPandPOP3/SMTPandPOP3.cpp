#include <iostream>
#include <ctime>
#include <sstream>
#include <string>
#include "smtp_client.h"
#include "pop3_client.h"
#include <iomanip>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode: smtp|pop3> [params...]" << std::endl;
        return 1;
    }

    
    std::string mode = argv[1];
    if (mode == "smtp") {
        if (argc != 8) {
            std::cerr << "Usage: " << argv[0] << " smtp <smtp_server> <smtp_port> <from_email> <to_email> <username> <password>" << std::endl;
            return 1;
        }

        // SMTP parametry
        std::string smtp_server = argv[2];
        int smtp_port = std::stoi(argv[3]);
        std::string from_email = argv[4];
        std::string to_email = argv[5];
        std::string username = argv[6];
        std::string password = argv[7];
        //std::string email_subject = "Test Email - " + get_current_datetime();
        //std::string email_body = "This is a test email sent using raw sockets.";

        try {
            // Odesílání e-mailu pomocí SMTP
            send_email(smtp_server, smtp_port, from_email, to_email, username, password);
        }
        catch (const std::exception& e) {
            std::cerr << "Error (SMTP): " << e.what() << std::endl;
            return 1;
        }
    }
    else if (mode == "pop3") {
        if (argc != 7) {
            std::cerr << "Usage: " << argv[0] << " pop3 <pop3_server> <pop3_port> <username> <password> <message_number>" << std::endl;
            return 1;
        }

        // POP3 parametry
        std::string pop3_server = argv[2];
        int pop3_port = std::stoi(argv[3]);
        std::string username = argv[4];
        std::string password = argv[5];
        int message_number = std::stoi(argv[6]);

        try {
            // Stahování jedné zprávy pomocí POP3
            retrieve_email(pop3_server, pop3_port, username, password, message_number);
        }
        catch (const std::exception& e) {
            std::cerr << "Error (POP3): " << e.what() << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << "Invalid mode. Use 'smtp' or 'pop3'." << std::endl;
        return 1;
    }

    return 0;
}
