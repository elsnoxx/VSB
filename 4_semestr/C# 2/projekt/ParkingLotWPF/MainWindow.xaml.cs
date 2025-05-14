using ApiCalls.Model;
using ApiCalls;
using System;
using System.Collections.ObjectModel;
using System.Windows;

namespace ParkingLotWPF
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainWindowViewModel();
        }
    }
}
