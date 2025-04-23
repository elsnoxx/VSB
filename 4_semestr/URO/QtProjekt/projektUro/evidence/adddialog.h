#ifndef ADDDIALOG_H
#define ADDDIALOG_H

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QComboBox>

class AddDialog : public QDialog {
    Q_OBJECT

public:
    AddDialog(QWidget *parent = nullptr,
              const QString &id = "",
              const QString &type = "",
              const QString &manufacturer = "",
              const QString &price = "",
              const QString &serial = "",
              const QString &status = "",
              const QString &location = "",
              const QString &size = "",
              const QString &purchaseDate = "",
              const QString &note = "");

    QString getId() const;
    QString getType() const;
    QString getManufacturer() const;
    QString getPrice() const;
    QString getSerial() const;
    QString getStatus() const;
    QString getLocation() const;
    QString getSize() const;
    QString getPurchaseDate() const;
    QString getNote() const;

private:
    QLineEdit *idEdit, *priceEdit, *serialEdit, *sizeEdit, *purchaseDateEdit;
    QComboBox *typeCombo, *manufacturerCombo, *statusCombo, *locationCombo;
    QPlainTextEdit *noteEdit;
    QPushButton *okButton, *cancelButton;
};

#endif // ADDDIALOG_H
