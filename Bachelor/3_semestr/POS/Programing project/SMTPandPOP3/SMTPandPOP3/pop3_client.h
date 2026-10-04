#pragma once
#ifndef POP3_CLIENT_H
#define POP3_CLIENT_H

#include <string>

// Deklarace funkcí
void retrieve_email(const std::string& pop3_server, int pop3_port, const std::string& username, const std::string& password, int message_number);



#endif // POP3_CLIENT_H
