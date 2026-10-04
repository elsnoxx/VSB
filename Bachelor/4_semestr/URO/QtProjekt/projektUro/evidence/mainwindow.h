#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTableWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QTabWidget>
#include <QPlainTextEdit>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void addRecord();
    void editRecord();
    void deleteRecord();
    void clearRecord();
    void clearAllRecords();


private:
    QLineEdit *searchEdit;
    QTableWidget *table;
    QTabWidget *tabWidget;
    QLineEdit *jmenoEdit;
    QLineEdit *prijmeniEdit;
    QLineEdit *vekEdit;
    QLineEdit *adresaEdit;
    QPushButton *editButton;
    QPushButton *addButton;
    QPushButton *deleteButton;
    QPushButton *clearButton;
    QLineEdit *idEdit, *typeEdit, *manufacturerEdit, *priceEdit, *serialEdit2, *statusEdit, *locationEdit, *sizeEdit, *purchaseDateEdit;
    QPlainTextEdit *noteEdit;

};

#endif // MAINWINDOW_H
