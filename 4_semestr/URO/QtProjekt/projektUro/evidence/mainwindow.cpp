#include "mainwindow.h"
#include "adddialog.h"
#include "data.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QWidget>
#include <QMenu>
#include <QMenuBar>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>
#include <QPlainTextEdit>
#include <QMap>

// Statická data
struct Record {
    QString id, type, manufacturer, price, serial, status, location, size, purchaseDate;
};
QVector<Record> staticData = {
    {"1", "Laptop", "Dell", "25000", "SN123", "Used", "Brno", "15\"", "2021-01-01"},
    {"2", "Monitor", "HP", "5000", "SN456", "New", "Praha", "24\"", "2023-03-15"},
    {"3", "Printer", "Canon", "3000", "SN789", "Damaged", "Ostrava", "-", "2020-11-20"},
    {"4", "PC", "Lenovo", "18000", "SN321", "Refurbished", "Plzeň", "-", "2022-06-10"},
    {"5", "Scanner", "HP", "4000", "SN654", "Used", "Olomouc", "A4", "2021-09-12"},
    {"6", "Laptop", "Asus", "22000", "SN987", "New", "Brno", "14\"", "2024-02-01"},
    {"7", "Monitor", "Dell", "6000", "SN852", "Used", "Praha", "27\"", "2020-05-20"},
    {"8", "PC", "Adwantech", "20000", "SN741", "Damaged", "Ostrava", "-", "2019-12-10"},
    {"9", "Printer", "HP", "3500", "SN963", "Refurbished", "Plzeň", "-", "2022-08-18"},
    {"10", "Scanner", "Canon", "4500", "SN159", "New", "Olomouc", "A3", "2023-11-05"},
    {"11", "Laptop", "Lenovo", "27000", "SN357", "Used", "Brno", "13\"", "2021-03-22"},
    {"12", "Monitor", "HP", "5200", "SN258", "New", "Praha", "22\"", "2024-01-10"},
    {"13", "PC", "Dell", "19500", "SN753", "Used", "Ostrava", "-", "2022-04-14"},
    {"14", "Printer", "Canon", "3200", "SN4567", "Damaged", "Plzeň", "-", "2020-07-30"},
    {"15", "Scanner", "Adwantech", "4800", "SN8520", "Refurbished", "Olomouc", "A4", "2023-05-19"}
};

QStringList statusList = {"", "Used", "Damaged", "New", "Refurbished"};
QStringList typeList = {"", "Monitor", "PC", "Scanner", "Printer", "Laptop"};
QStringList locationList = {"", "Brno", "Prague", "Ostrava", "Plzen", "Olomouc"};
QStringList manufacturerList = {"", "Dell", "Adwantech", "HP", "Lenovo", "Asus"};

QMap<QString, QString> notesMap; // ID -> poznámka

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    QWidget *central = new QWidget;
    QVBoxLayout *mainLayout = new QVBoxLayout;

    // Menu bar
    QMenuBar *menuBar = new QMenuBar(this);

    // Menu "Soubor"
    QMenu *souborMenu = new QMenu("Soubor", this);
    QAction *novaAkce = new QAction("Nový", this);
    QAction *smazatVseAkce = new QAction("Smazat vše", this);
    QAction *ukoncitAkce = new QAction("Ukončit", this);

    // Přidání akcí do menu
    souborMenu->addAction(novaAkce);
    souborMenu->addAction(smazatVseAkce);
    souborMenu->addSeparator();
    souborMenu->addAction(ukoncitAkce);

    // Přidání menu do menuBaru
    menuBar->addMenu(souborMenu);
    setMenuBar(menuBar);

    // Funkce pro akce
    connect(novaAkce, &QAction::triggered, this, &MainWindow::addRecord);
    connect(smazatVseAkce, &QAction::triggered, this, &MainWindow::clearAllRecords);
    connect(ukoncitAkce, &QAction::triggered, this, &MainWindow::close);

    // Vyhledávací pole
    QLabel *serialLabel = new QLabel("Serial Number:");
    QLineEdit *serialEdit = new QLineEdit;
    QLabel *statusLabel = new QLabel("Status:");
    QComboBox *statusCombo = new QComboBox;
    statusCombo->addItems(statusList);

    QLabel *typeLabel = new QLabel("Type:");
    QComboBox *typeCombo = new QComboBox;
    typeCombo->addItems(typeList);

    QLabel *locationLabel = new QLabel("Location:");
    QComboBox *locationCombo = new QComboBox;
    locationCombo->addItems(locationList);

    QLabel *manufacturerLabel = new QLabel("Manufacturer:");
    QComboBox *manufacturerCombo = new QComboBox;
    manufacturerCombo->addItems(manufacturerList);

    QHBoxLayout *searchLayout = new QHBoxLayout;
    searchLayout->addWidget(serialLabel);
    searchLayout->addWidget(serialEdit);
    searchLayout->addWidget(statusLabel);
    searchLayout->addWidget(statusCombo);
    searchLayout->addWidget(typeLabel);
    searchLayout->addWidget(typeCombo);
    searchLayout->addWidget(locationLabel);
    searchLayout->addWidget(locationCombo);
    searchLayout->addWidget(manufacturerLabel);
    searchLayout->addWidget(manufacturerCombo);
    mainLayout->addLayout(searchLayout);

    // Tabulka
    table = new QTableWidget(0, 9);
    table->setHorizontalHeaderLabels({"ID", "Type", "Manufacturer", "Price", "Serial Number", "Status", "Location", "Size", "Purchase Date"});
    mainLayout->addWidget(table);
    setFixedSize(950, 600);

    // Naplnění statickými daty
    for (const auto &rec : staticData) {
        int row = table->rowCount();
        table->insertRow(row);
        table->setItem(row, 0, new QTableWidgetItem(rec.id));
        table->setItem(row, 1, new QTableWidgetItem(rec.type));
        table->setItem(row, 2, new QTableWidgetItem(rec.manufacturer));
        table->setItem(row, 3, new QTableWidgetItem(rec.price));
        table->setItem(row, 4, new QTableWidgetItem(rec.serial));
        table->setItem(row, 5, new QTableWidgetItem(rec.status));
        table->setItem(row, 6, new QTableWidgetItem(rec.location));
        table->setItem(row, 7, new QTableWidgetItem(rec.size));
        table->setItem(row, 8, new QTableWidgetItem(rec.purchaseDate));
    }

    // Tab widget
    tabWidget = new QTabWidget;

    // První tab: detail záznamu
    QWidget *detailTab = new QWidget;
    QGridLayout *detailLayout = new QGridLayout;

    QLineEdit *idEdit = new QLineEdit; idEdit->setReadOnly(true);
    QLineEdit *typeEdit = new QLineEdit; typeEdit->setReadOnly(true);
    QLineEdit *manufacturerEdit = new QLineEdit; manufacturerEdit->setReadOnly(true);
    QLineEdit *priceEdit = new QLineEdit; priceEdit->setReadOnly(true);
    QLineEdit *serialEdit2 = new QLineEdit; serialEdit2->setReadOnly(true);
    QLineEdit *statusEdit = new QLineEdit; statusEdit->setReadOnly(true);
    QLineEdit *locationEdit = new QLineEdit; locationEdit->setReadOnly(true);
    QLineEdit *sizeEdit = new QLineEdit; sizeEdit->setReadOnly(true);
    QLineEdit *purchaseDateEdit = new QLineEdit; purchaseDateEdit->setReadOnly(true);

    // Levý sloupec
    detailLayout->addWidget(new QLabel("ID:"), 0, 0);
    detailLayout->addWidget(idEdit, 0, 1);
    detailLayout->addWidget(new QLabel("Type:"), 1, 0);
    detailLayout->addWidget(typeEdit, 1, 1);
    detailLayout->addWidget(new QLabel("Manufacturer:"), 2, 0);
    detailLayout->addWidget(manufacturerEdit, 2, 1);
    detailLayout->addWidget(new QLabel("Price:"), 3, 0);
    detailLayout->addWidget(priceEdit, 3, 1);
    detailLayout->addWidget(new QLabel("Serial Number:"), 4, 0);
    detailLayout->addWidget(serialEdit2, 4, 1);

    // Pravý sloupec
    detailLayout->addWidget(new QLabel("Status:"), 0, 2);
    detailLayout->addWidget(statusEdit, 0, 3);
    detailLayout->addWidget(new QLabel("Location:"), 1, 2);
    detailLayout->addWidget(locationEdit, 1, 3);
    detailLayout->addWidget(new QLabel("Size:"), 2, 2);
    detailLayout->addWidget(sizeEdit, 2, 3);
    detailLayout->addWidget(new QLabel("Purchase Date:"), 3, 2);
    detailLayout->addWidget(purchaseDateEdit, 3, 3);

    detailTab->setLayout(detailLayout);

    // Druhý tab: poznámka
    QWidget *noteTab = new QWidget;
    QVBoxLayout *noteLayout = new QVBoxLayout;
    QPlainTextEdit *noteEdit = new QPlainTextEdit;
    noteEdit->setReadOnly(true); // Poznámka pouze pro čtení
    QPushButton *editNoteButton = new QPushButton("Editovat poznámku");
    noteLayout->addWidget(noteEdit);
    noteLayout->addWidget(editNoteButton);
    noteTab->setLayout(noteLayout);

    tabWidget->addTab(detailTab, "Detail");
    tabWidget->addTab(noteTab, "Poznámka");
    mainLayout->addWidget(tabWidget);

    // Tlačítka
    QHBoxLayout *buttonLayout = new QHBoxLayout;
    editButton = new QPushButton("Editovat záznam");
    addButton = new QPushButton("Přidat nový záznam");
    deleteButton = new QPushButton("Odstranit záznam");
    clearButton = new QPushButton("Vyčistění záznamů");
    buttonLayout->addWidget(editButton);
    buttonLayout->addWidget(addButton);
    buttonLayout->addWidget(deleteButton);
    buttonLayout->addWidget(clearButton);

    mainLayout->addLayout(buttonLayout);

    // Akce
    connect(addButton, &QPushButton::clicked, this, &MainWindow::addRecord);
    connect(editButton, &QPushButton::clicked, this, &MainWindow::editRecord);
    connect(deleteButton, &QPushButton::clicked, this, &MainWindow::deleteRecord);
    connect(clearButton, &QPushButton::clicked, this, &MainWindow::clearRecord);
    connect(table, &QTableWidget::cellClicked, this, [=](int row, int) {
        idEdit->setText(table->item(row, 0)->text());
        typeEdit->setText(table->item(row, 1)->text());
        manufacturerEdit->setText(table->item(row, 2)->text());
        priceEdit->setText(table->item(row, 3)->text());
        serialEdit2->setText(table->item(row, 4)->text());
        statusEdit->setText(table->item(row, 5)->text());
        locationEdit->setText(table->item(row, 6)->text());
        sizeEdit->setText(table->item(row, 7)->text());
        purchaseDateEdit->setText(table->item(row, 8)->text());
        // Poznámka podle ID
        noteEdit->setPlainText(notesMap.value(table->item(row, 0)->text()));
    });

    // Vyhledávání podle Serial Number, Status, Type, Location, Manufacturer
    auto filterFunc = [=]() {
        QString serialText = serialEdit->text();
        QString statusText = statusCombo->currentText();
        QString typeText = typeCombo->currentText();
        QString locationText = locationCombo->currentText();
        QString manufacturerText = manufacturerCombo->currentText();
        for (int i = 0; i < table->rowCount(); ++i) {
            bool serialMatch = table->item(i, 4)->text().contains(serialText, Qt::CaseInsensitive) || serialText.isEmpty();
            bool statusMatch = (statusText.isEmpty() || table->item(i, 5)->text() == statusText);
            bool typeMatch = (typeText.isEmpty() || table->item(i, 1)->text() == typeText);
            bool locationMatch = (locationText.isEmpty() || table->item(i, 6)->text() == locationText);
            bool manufacturerMatch = (manufacturerText.isEmpty() || table->item(i, 2)->text() == manufacturerText);
            table->setRowHidden(i, !(serialMatch && statusMatch && typeMatch && locationMatch && manufacturerMatch));
        }
    };

    connect(serialEdit, &QLineEdit::textChanged, this, filterFunc);
    connect(statusCombo, &QComboBox::currentTextChanged, this, filterFunc);
    connect(typeCombo, &QComboBox::currentTextChanged, this, filterFunc);
    connect(locationCombo, &QComboBox::currentTextChanged, this, filterFunc);
    connect(manufacturerCombo, &QComboBox::currentTextChanged, this, filterFunc);

    // Uložení poznámky při změně
    connect(noteEdit, &QPlainTextEdit::textChanged, this, [=]() {
        int row = table->currentRow();
        if (row >= 0) {
            QString id = table->item(row, 0)->text();
            notesMap[id] = noteEdit->toPlainText();
        }
    });

    // Otevření popup dialogu pro editaci poznámky
    connect(editNoteButton, &QPushButton::clicked, this, [=]() {
        int row = table->currentRow();
        if (row < 0) return;
        QString id = table->item(row, 0)->text();
        QString currentNote = notesMap.value(id);

        // Jednoduchý dialog pro editaci poznámky
        QDialog dialog(this);
        dialog.setWindowTitle("Editace poznámky");
        QVBoxLayout layout;
        QPlainTextEdit edit;
        edit.setPlainText(currentNote);
        layout.addWidget(&edit);
        QHBoxLayout buttons;
        QPushButton ok("OK"), cancel("Zrušit");
        buttons.addWidget(&ok);
        buttons.addWidget(&cancel);
        layout.addLayout(&buttons);
        dialog.setLayout(&layout);

        connect(&ok, &QPushButton::clicked, &dialog, &QDialog::accept);
        connect(&cancel, &QPushButton::clicked, &dialog, &QDialog::reject);

        if (dialog.exec() == QDialog::Accepted) {
            notesMap[id] = edit.toPlainText();
            noteEdit->setPlainText(edit.toPlainText());
        }
    });

    central->setLayout(mainLayout);
    setCentralWidget(central);
    setWindowTitle("cv07_evidence");
}

MainWindow::~MainWindow() {}

void MainWindow::editRecord() {
    int row = table->currentRow();
    if (row < 0) return;

    QString id = table->item(row, 0)->text();
    AddDialog dialog(this,
                     table->item(row, 0)->text(),
                     table->item(row, 1)->text(),
                     table->item(row, 2)->text(),
                     table->item(row, 3)->text(),
                     table->item(row, 4)->text(),
                     table->item(row, 5)->text(),
                     table->item(row, 6)->text(),
                     table->item(row, 7)->text(),
                     table->item(row, 8)->text(),
                     notesMap.value(id)
                     );

    if (dialog.exec() == QDialog::Accepted) {
        table->setItem(row, 0, new QTableWidgetItem(dialog.getId()));
        table->setItem(row, 1, new QTableWidgetItem(dialog.getType()));
        table->setItem(row, 2, new QTableWidgetItem(dialog.getManufacturer()));
        table->setItem(row, 3, new QTableWidgetItem(dialog.getPrice()));
        table->setItem(row, 4, new QTableWidgetItem(dialog.getSerial()));
        table->setItem(row, 5, new QTableWidgetItem(dialog.getStatus()));
        table->setItem(row, 6, new QTableWidgetItem(dialog.getLocation()));
        table->setItem(row, 7, new QTableWidgetItem(dialog.getSize()));
        table->setItem(row, 8, new QTableWidgetItem(dialog.getPurchaseDate()));
        notesMap[dialog.getId()] = dialog.getNote();
    }
}

void MainWindow::deleteRecord() {
    int row = table->currentRow();
    if (row >= 0) {
        table->removeRow(row);
    }
}

void MainWindow::clearRecord() {
    idEdit->clear();
    typeEdit->clear();
    manufacturerEdit->clear();
    priceEdit->clear();
    serialEdit2->clear();
    statusEdit->clear();
    locationEdit->clear();
    sizeEdit->clear();
    purchaseDateEdit->clear();
    noteEdit->clear();
}

void MainWindow::addRecord() {
    AddDialog dialog(this);
    if (dialog.exec() == QDialog::Accepted) {
        int row = table->rowCount();
        table->insertRow(row);
        table->setItem(row, 0, new QTableWidgetItem(dialog.getId()));
        table->setItem(row, 1, new QTableWidgetItem(dialog.getType()));
        table->setItem(row, 2, new QTableWidgetItem(dialog.getManufacturer()));
        table->setItem(row, 3, new QTableWidgetItem(dialog.getPrice()));
        table->setItem(row, 4, new QTableWidgetItem(dialog.getSerial()));
        table->setItem(row, 5, new QTableWidgetItem(dialog.getStatus()));
        table->setItem(row, 6, new QTableWidgetItem(dialog.getLocation()));
        table->setItem(row, 7, new QTableWidgetItem(dialog.getSize()));
        table->setItem(row, 8, new QTableWidgetItem(dialog.getPurchaseDate()));
        notesMap[dialog.getId()] = dialog.getNote();
    }
}

void MainWindow::clearAllRecords() {
    table->setRowCount(0);
    clearRecord();
}

