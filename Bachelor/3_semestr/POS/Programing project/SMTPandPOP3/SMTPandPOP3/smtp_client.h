#pragma once
#ifndef SMTP_CLIENT_H
#define SMTP_CLIENT_H

#include <string>

// Deklarace funkcí
void send_email(const std::string& smtp_server, int smtp_port, const std::string& from_email,
    const std::string& to_email, const std::string& username, const std::string& password);

std::string encode_base64(const std::string& input);

#endif // SMTP_CLIENT_H
