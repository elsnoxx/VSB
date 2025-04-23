#include "adddialog.h"
#include "data.h"
#include <QVBoxLayout>
#include <QFormLayout>
#include <QHBoxLayout>

AddDialog::AddDialog(QWidget *parent,
                     const QString &id,
                     const QString &type,
                     const QString &manufacturer,
                     const QString &price,
                     const QString &serial,
                     const QString &status,
                     const QString &location,
                     const QString &size,
                     const QString &purchaseDate,
                     const QString &note)
    : QDialog(parent)
{
    setWindowTitle("Záznam");

    idEdit = new QLineEdit(id);

    typeCombo = new QComboBox;
    typeCombo->addItems(typeList);
    typeCombo->setCurrentText(type);

    manufacturerCombo = new QComboBox;
    manufacturerCombo->addItems(manufacturerList);
    manufacturerCombo->setCurrentText(manufacturer);

    priceEdit = new QLineEdit(price);
    serialEdit = new QLineEdit(serial);

    statusCombo = new QComboBox;
    statusCombo->addItems(statusList);
    statusCombo->setCurrentText(status);

    locationCombo = new QComboBox;
    locationCombo->addItems(locationList);
    locationCombo->setCurrentText(location);

    sizeEdit = new QLineEdit(size);
    purchaseDateEdit = new QLineEdit(purchaseDate);
    noteEdit = new QPlainTextEdit(note);

    QFormLayout *form = new QFormLayout;
    form->addRow("ID:", idEdit);
    form->addRow("Type:", typeCombo);
    form->addRow("Manufacturer:", manufacturerCombo);
    form->addRow("Price:", priceEdit);
    form->addRow("Serial Number:", serialEdit);
    form->addRow("Status:", statusCombo);
    form->addRow("Location:", locationCombo);
    form->addRow("Size:", sizeEdit);
    form->addRow("Purchase Date:", purchaseDateEdit);
    form->addRow("Poznámka:", noteEdit);

    okButton = new QPushButton("OK");
    cancelButton = new QPushButton("Zrušit");

    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addStretch();
    buttonLayout->addWidget(okButton);
    buttonLayout->addWidget(cancelButton);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addLayout(form);
    mainLayout->addLayout(buttonLayout);

    setLayout(mainLayout);

    connect(okButton, &QPushButton::clicked, this, &AddDialog::accept);
    connect(cancelButton, &QPushButton::clicked, this, &AddDialog::reject);
}

QString AddDialog::getId() const { return idEdit->text(); }
QString AddDialog::getType() const { return typeCombo->currentText(); }
QString AddDialog::getManufacturer() const { return manufacturerCombo->currentText(); }
QString AddDialog::getPrice() const { return priceEdit->text(); }
QString AddDialog::getSerial() const { return serialEdit->text(); }
QString AddDialog::getStatus() const { return statusCombo->currentText(); }
QString AddDialog::getLocation() const { return locationCombo->currentText(); }
QString AddDialog::getSize() const { return sizeEdit->text(); }
QString AddDialog::getPurchaseDate() const { return purchaseDateEdit->text(); }
QString AddDialog::getNote() const { return noteEdit->toPlainText(); }
